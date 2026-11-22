"""Bar-processing core shared by the backtester and the paper/live runner.

Event order for each newly closed M5 candle ``t``:

1. (Caller) If a signal is pending from candle ``t-1``, enter at ``t``'s open
   via :meth:`TradingEngine.execute_pending`.
2. :meth:`TradingEngine.on_bar_close` checks open positions' stops/targets
   against ``t``'s range, marks equity at ``t``'s close, updates the daily loss
   breaker and asks the strategy for a new signal using data up to ``t``.

Intrabar ambiguity (both stop and target inside one candle) is resolved
pessimistically: the stop is assumed to fill first.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from trading_framework.analytics.trade_log import TradeLogger
from trading_framework.execution.base import Broker
from trading_framework.portfolio.portfolio import ClosedTrade, Portfolio, Position
from trading_framework.risk.manager import RiskManager
from trading_framework.strategy.signals import Direction, SignalEvent
from trading_framework.strategy.two_step import TwoStepStrategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Bar:
    """A closed candle."""

    time: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class TradingEngine:
    """Coordinates strategy, risk, broker and portfolio for one symbol.

    Args:
        symbol: Market symbol.
        strategy: Signal generator.
        risk: Risk manager.
        portfolio: Portfolio ledger.
        broker: Execution venue (paper or live).
        max_spread: Maximum spread fraction allowed at entry.
        trade_logger: Optional trade journal.
    """

    def __init__(self, symbol: str, strategy: TwoStepStrategy, risk: RiskManager, portfolio: Portfolio,
                 broker: Broker, max_spread: float, trade_logger: TradeLogger | None = None) -> None:
        self.symbol = symbol
        self.strategy = strategy
        self.risk = risk
        self.portfolio = portfolio
        self.broker = broker
        self.max_spread = max_spread
        self.trade_logger = trade_logger
        self.pending: SignalEvent | None = None
        self.signals: list[SignalEvent] = []
        self.rejections: list[tuple[pd.Timestamp, str]] = []

    # ------------------------------------------------------------- entries
    def execute_pending(self, reference_price: float, time: pd.Timestamp) -> Position | None:
        """Execute the pending signal (if any) at ``reference_price``.

        The pending signal is always consumed: an entry that cannot be taken at
        the next open is discarded rather than chased.
        """
        signal, self.pending = self.pending, None
        if signal is None:
            return None
        d = signal.direction
        spread = self.broker.get_spread(self.symbol)
        if spread > self.max_spread:
            return self._reject(time, f"spread {spread:.4%} > max {self.max_spread:.4%}")
        ok, why = self.risk.can_open(self.symbol, d, len(self.portfolio.open_positions(self.symbol)),
                                     len(self.portfolio.open_positions()))
        if not ok:
            return self._reject(time, why)
        expected = self.broker.expected_fill_price(d.entry_side, reference_price)
        plan, why = self.risk.plan_trade(signal, expected, self.portfolio.equity({self.symbol: reference_price}))
        if plan is None:
            return self._reject(time, why)
        try:
            fill = self.broker.submit_entry(plan, reference_price, time)
        except Exception as exc:  # noqa: BLE001 - broker failures must not crash the loop
            logger.error("Entry failed: %s", exc)
            return self._reject(time, f"broker error: {exc}")
        # Recompute target from the actual fill so R:R stays exactly 1:rr.
        tp = self.risk.take_profit(fill.price, plan.stop_price, d)
        pos = self.portfolio.open_position(
            self.symbol, d, fill.amount, fill.price, time, plan.stop_price, tp, fill.fee,
            risk_amount=fill.amount * abs(fill.price - plan.stop_price), reason=signal.reason,
            metadata={"zone": signal.zone_source, "atr": signal.atr},
        )
        logger.info("ENTRY %s %.6f %s @ %.2f | SL %.2f | TP %.2f | %s", d.name, fill.amount, self.symbol,
                    fill.price, plan.stop_price, tp, signal.reason)
        return pos

    def _reject(self, time: pd.Timestamp, reason: str) -> None:
        logger.info("Entry skipped at %s: %s", time, reason)
        self.rejections.append((time, reason))
        return None

    # --------------------------------------------------------------- exits
    def _exit(self, pos: Position, price: float, time: pd.Timestamp, order_type: str, reason: str) -> ClosedTrade:
        fill = self.broker.submit_exit(pos.symbol, pos.direction.exit_side, pos.amount, price, time,
                                       order_type)  # type: ignore[arg-type]
        trade = self.portfolio.close_position(pos.id, fill.price, time, fill.fee, reason)
        if self.trade_logger:
            self.trade_logger.log(trade)
        return trade

    def check_exits(self, bar: Bar) -> list[ClosedTrade]:
        """Resolve stop/target hits for open positions within ``bar``."""
        closed: list[ClosedTrade] = []
        for pos in list(self.portfolio.open_positions(self.symbol)):
            if pos.direction is Direction.LONG:
                if bar.open <= pos.stop_price:
                    closed.append(self._exit(pos, bar.open, bar.time, "stop", "stop_gap"))
                elif bar.low <= pos.stop_price:
                    closed.append(self._exit(pos, pos.stop_price, bar.time, "stop", "stop_loss"))
                elif bar.open >= pos.take_profit:
                    closed.append(self._exit(pos, bar.open, bar.time, "take_profit", "take_profit_gap"))
                elif bar.high >= pos.take_profit:
                    closed.append(self._exit(pos, pos.take_profit, bar.time, "take_profit", "take_profit"))
            else:
                if bar.open >= pos.stop_price:
                    closed.append(self._exit(pos, bar.open, bar.time, "stop", "stop_gap"))
                elif bar.high >= pos.stop_price:
                    closed.append(self._exit(pos, pos.stop_price, bar.time, "stop", "stop_loss"))
                elif bar.open <= pos.take_profit:
                    closed.append(self._exit(pos, bar.open, bar.time, "take_profit", "take_profit_gap"))
                elif bar.low <= pos.take_profit:
                    closed.append(self._exit(pos, pos.take_profit, bar.time, "take_profit", "take_profit"))
        return closed

    def close_all(self, price: float, time: pd.Timestamp, reason: str = "end_of_data") -> list[ClosedTrade]:
        """Flatten every open position at ``price`` (market)."""
        return [self._exit(p, price, time, "market", reason) for p in list(self.portfolio.open_positions(self.symbol))]

    # ------------------------------------------------------------ bar close
    def on_bar_close(self, bar: Bar, m15_window: pd.DataFrame, m5_window: pd.DataFrame) -> SignalEvent | None:
        """Process a closed M5 candle.

        Args:
            bar: The candle that just closed (must equal ``m5_window``'s last row).
            m15_window: Recent M15 candles (unclosed rows are ignored by the strategy).
            m5_window: Recent closed M5 candles ending with ``bar``.

        Returns:
            The new entry signal, if one was queued for the next open.
        """
        self.check_exits(bar)
        close_time = bar.time + self.strategy.entry_td
        equity = self.portfolio.mark(close_time, {self.symbol: bar.close})
        self.risk.on_equity(close_time, equity)

        signal = self.strategy.generate_signals(m15_window, m5_window, self.symbol)
        if not signal.is_entry:
            return None
        self.signals.append(signal)
        if self.risk.halted:
            self._reject(close_time, "daily loss limit reached")
            return None
        if len(self.portfolio.open_positions(self.symbol)) >= self.risk.cfg.max_positions_per_symbol:
            self._reject(close_time, "position already open")
            return None
        logger.debug("Signal %s at %s: %s", signal.direction.name, signal.timestamp, signal.reason)
        self.pending = signal
        return signal
