"""Margin-style portfolio accounting (works for longs and shorts).

``cash`` is the realised balance: it changes only by fees and realised PnL.
``equity = cash + unrealised PnL``. This mirrors a USDT-margined derivatives
account; for spot longs the equity figures are identical.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from trading_framework.strategy.signals import Direction


@dataclass
class Position:
    """An open position with its bracket levels."""

    id: int
    symbol: str
    direction: Direction
    amount: float
    entry_price: float
    entry_time: pd.Timestamp
    stop_price: float
    take_profit: float
    entry_fee: float
    risk_amount: float
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def unrealized_pnl(self, price: float) -> float:
        """Mark-to-market PnL before exit fees."""
        return (price - self.entry_price) * self.amount * self.direction.value


@dataclass(frozen=True)
class ClosedTrade:
    """A completed round trip."""

    id: int
    symbol: str
    direction: str
    amount: float
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    stop_price: float
    take_profit: float
    exit_reason: str
    entry_reason: str
    gross_pnl: float
    fees: float
    net_pnl: float
    risk_amount: float
    r_multiple: float
    holding_minutes: float

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict representation for logging/serialisation."""
        return asdict(self)


class Portfolio:
    """Tracks cash, open positions, closed trades and the equity curve.

    Args:
        initial_equity: Starting account balance in quote currency.
    """

    def __init__(self, initial_equity: float) -> None:
        self.initial_equity = initial_equity
        self.cash = initial_equity
        self.positions: dict[int, Position] = {}
        self.closed_trades: list[ClosedTrade] = []
        self._equity_points: list[tuple[pd.Timestamp, float]] = []
        self._ids = itertools.count(1)
        self._last_prices: dict[str, float] = {}

    # ------------------------------------------------------------- queries
    def open_positions(self, symbol: str | None = None) -> list[Position]:
        """Open positions, optionally filtered by symbol."""
        return [p for p in self.positions.values() if symbol is None or p.symbol == symbol]

    def equity(self, prices: dict[str, float] | None = None) -> float:
        """Cash plus unrealised PnL at ``prices`` (or last marked prices)."""
        px = {**self._last_prices, **(prices or {})}
        return self.cash + sum(p.unrealized_pnl(px.get(p.symbol, p.entry_price)) for p in self.positions.values())

    # ----------------------------------------------------------- mutations
    def open_position(
        self, symbol: str, direction: Direction, amount: float, price: float, time: pd.Timestamp,
        stop_price: float, take_profit: float, fee: float, risk_amount: float,
        reason: str = "", metadata: dict[str, Any] | None = None,
    ) -> Position:
        """Record a filled entry and deduct its fee."""
        pos = Position(next(self._ids), symbol, direction, amount, price, time, stop_price,
                       take_profit, fee, risk_amount, reason, metadata or {})
        self.positions[pos.id] = pos
        self.cash -= fee
        return pos

    def close_position(self, position_id: int, price: float, time: pd.Timestamp, fee: float, reason: str) -> ClosedTrade:
        """Record a filled exit, realise PnL and return the trade record."""
        pos = self.positions.pop(position_id)
        gross = pos.unrealized_pnl(price)
        self.cash += gross - fee
        fees = pos.entry_fee + fee
        net = gross - fees
        trade = ClosedTrade(
            id=pos.id, symbol=pos.symbol, direction=pos.direction.name, amount=pos.amount,
            entry_time=pos.entry_time, entry_price=pos.entry_price, exit_time=time, exit_price=price,
            stop_price=pos.stop_price, take_profit=pos.take_profit, exit_reason=reason,
            entry_reason=pos.reason, gross_pnl=gross, fees=fees, net_pnl=net,
            risk_amount=pos.risk_amount,
            r_multiple=net / pos.risk_amount if pos.risk_amount > 0 else 0.0,
            holding_minutes=(time - pos.entry_time).total_seconds() / 60.0,
        )
        self.closed_trades.append(trade)
        return trade

    def mark(self, time: pd.Timestamp, prices: dict[str, float]) -> float:
        """Mark open positions to market, append to the equity curve and return equity."""
        self._last_prices.update(prices)
        eq = self.equity()
        self._equity_points.append((time, eq))
        return eq

    # ------------------------------------------------------------- exports
    def equity_curve(self) -> pd.Series:
        """Equity indexed by mark time."""
        if not self._equity_points:
            return pd.Series(dtype=float, name="equity")
        t, v = zip(*self._equity_points)
        return pd.Series(v, index=pd.DatetimeIndex(t), name="equity")

    def trades_frame(self) -> pd.DataFrame:
        """Closed trades as a DataFrame."""
        return pd.DataFrame([t.to_dict() for t in self.closed_trades])
