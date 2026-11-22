"""Position sizing, stop/target calculation and portfolio guards."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import pandas as pd

from trading_framework.config.settings import ExchangeConfig, RiskConfig
from trading_framework.strategy.signals import Direction, SignalEvent

logger = logging.getLogger(__name__)

_EPS = 1e-12


@dataclass(frozen=True)
class MarketLimits:
    """Exchange precision and minimums for a symbol.

    Attributes:
        amount_step: Order size increment.
        price_tick: Price increment.
        min_amount: Minimum order size.
        min_notional: Minimum order value in quote currency.
    """

    amount_step: float
    price_tick: float
    min_amount: float
    min_notional: float

    @classmethod
    def from_config(cls, cfg: ExchangeConfig) -> "MarketLimits":
        """Build limits from the offline fallbacks in :class:`ExchangeConfig`."""
        return cls(cfg.amount_step, cfg.price_tick, cfg.min_amount, cfg.min_notional)


def _decimals(step: float) -> int:
    return max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0


def floor_to_step(value: float, step: float) -> float:
    """Round ``value`` down to a multiple of ``step`` (avoids over-sizing)."""
    return round(math.floor(value / step + 1e-9) * step, _decimals(step) + 2)


def round_to_tick(value: float, tick: float) -> float:
    """Round ``value`` to the nearest multiple of ``tick``."""
    return round(round(value / tick) * tick, _decimals(tick) + 2)


@dataclass(frozen=True)
class OrderPlan:
    """A fully sized trade ready for execution.

    Attributes:
        symbol: Market symbol.
        direction: LONG or SHORT.
        entry_price: Expected entry price (after slippage).
        stop_price: Protective stop.
        take_profit: Target at ``reward_risk`` x risk distance.
        amount: Base-currency quantity (exchange-rounded).
        risk_amount: Quote-currency loss if stopped at ``stop_price`` (ex-fees).
        notional: ``amount * entry_price``.
    """

    symbol: str
    direction: Direction
    entry_price: float
    stop_price: float
    take_profit: float
    amount: float
    risk_amount: float
    notional: float


class RiskManager:
    """Implements the strategy's money-management rules.

    Args:
        config: Risk settings.
        reward_risk: Target multiple of the risk distance (``2.0`` = 1:2).
        limits: Exchange precision/minimums.
    """

    def __init__(self, config: RiskConfig, reward_risk: float, limits: MarketLimits) -> None:
        self.cfg = config
        self.reward_risk = reward_risk
        self.limits = limits
        self._day: pd.Timestamp | None = None
        self._day_start_equity: float | None = None
        self._halted = False

    # --------------------------------------------------------------- pricing
    def take_profit(self, entry: float, stop: float, direction: Direction) -> float:
        """Fixed R:R target. Long: ``entry + rr * (entry - stop)``; short mirrored."""
        risk = abs(entry - stop)
        tp = entry + direction.value * self.reward_risk * risk
        return round_to_tick(tp, self.limits.price_tick)

    def position_size(self, equity: float, entry: float, stop: float) -> float:
        """Size so that hitting the stop loses ``risk_per_trade`` of equity.

        ``size = equity * risk% / |entry - stop|``, capped by ``max_leverage`` and
        floored to the exchange amount step.

        Returns:
            Quantity in base currency, or ``0.0`` if below exchange minimums.
        """
        distance = abs(entry - stop)
        if equity <= 0 or distance <= _EPS or entry <= 0:
            return 0.0
        raw = equity * self.cfg.risk_per_trade / distance
        cap = equity * self.cfg.max_leverage / entry
        if raw > cap:
            logger.debug("Size capped by leverage: %.6f -> %.6f", raw, cap)
        size = floor_to_step(min(raw, cap), self.limits.amount_step)
        if size < self.limits.min_amount or size * entry < self.limits.min_notional:
            return 0.0
        return size

    def plan_trade(self, signal: SignalEvent, entry_price: float, equity: float) -> tuple[OrderPlan | None, str]:
        """Turn a signal into a sized order using the expected fill price.

        Args:
            signal: Entry signal with a structural stop.
            entry_price: Expected fill price.
            equity: Current account equity.

        Returns:
            ``(plan, "ok")`` or ``(None, reason)``.
        """
        if not signal.is_entry or signal.stop_price is None:
            return None, "not an entry signal"
        d = signal.direction
        stop = round_to_tick(signal.stop_price, self.limits.price_tick)
        if d is Direction.LONG and stop >= entry_price:
            return None, "entry at/below stop (gapped through)"
        if d is Direction.SHORT and stop <= entry_price:
            return None, "entry at/above stop (gapped through)"
        amount = self.position_size(equity, entry_price, stop)
        if amount <= 0:
            return None, "size below exchange minimum"
        tp = self.take_profit(entry_price, stop, d)
        return OrderPlan(
            symbol=signal.symbol, direction=d, entry_price=entry_price, stop_price=stop,
            take_profit=tp, amount=amount, risk_amount=amount * abs(entry_price - stop),
            notional=amount * entry_price,
        ), "ok"

    # ---------------------------------------------------------------- guards
    def on_equity(self, timestamp: pd.Timestamp, equity: float) -> None:
        """Update the UTC-day loss breaker with the latest marked equity."""
        day = timestamp.tz_convert("UTC").normalize() if timestamp.tzinfo else timestamp.normalize()
        if self._day is None or day != self._day:
            if self._halted:
                logger.info("New UTC day %s: daily loss breaker reset", day.date())
            self._day, self._day_start_equity, self._halted = day, equity, False
            return
        limit = self.cfg.daily_loss_limit
        if limit is not None and not self._halted and self._day_start_equity:
            if equity <= self._day_start_equity * (1 - limit):
                self._halted = True
                logger.warning("Daily loss limit hit (%.2f -> %.2f); halting new entries until next UTC day",
                               self._day_start_equity, equity)

    @property
    def halted(self) -> bool:
        """True when the daily loss breaker is active."""
        return self._halted

    def can_open(self, symbol: str, direction: Direction, open_for_symbol: int, open_total: int) -> tuple[bool, str]:
        """Portfolio-level exposure checks.

        Args:
            symbol: Symbol to trade.
            direction: Requested direction.
            open_for_symbol: Currently open positions in ``symbol``.
            open_total: Currently open positions across all symbols.

        Returns:
            ``(allowed, reason)``.
        """
        if self._halted:
            return False, "daily loss limit reached"
        if direction is Direction.SHORT and not self.cfg.allow_shorts:
            return False, "shorts disabled"
        if open_for_symbol >= self.cfg.max_positions_per_symbol:
            return False, f"max positions for {symbol} reached"
        if open_total >= self.cfg.max_total_positions:
            return False, "max total positions reached"
        return True, "ok"
