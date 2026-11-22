"""Signal and trend value objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class Direction(int, Enum):
    """Trade direction. The integer value is the PnL sign."""

    LONG = 1
    SHORT = -1
    NONE = 0

    @property
    def entry_side(self) -> str:
        """Order side that opens a position in this direction."""
        if self is Direction.NONE:
            raise ValueError("Direction.NONE has no order side")
        return "buy" if self is Direction.LONG else "sell"

    @property
    def exit_side(self) -> str:
        """Order side that closes a position in this direction."""
        return "sell" if self is Direction.LONG else "buy"


class Trend(str, Enum):
    """M15 market structure classification."""

    UP = "up"
    DOWN = "down"
    NONE = "none"


@dataclass(frozen=True)
class SignalEvent:
    """Output of :meth:`TwoStepStrategy.generate_signals`.

    A signal with ``direction == Direction.NONE`` is a "no trade" decision; its
    ``reason`` explains why, which is useful for research and debugging.

    Attributes:
        symbol: Market symbol.
        timestamp: Close time of the confirmation candle — the earliest moment
            the signal is knowable. The entry is the open of the next candle,
            which begins at exactly this timestamp.
        direction: LONG, SHORT or NONE.
        reason: Human-readable explanation.
        stop_price: Structural stop (pullback extreme -/+ k x ATR).
        reference_price: Confirmation candle close (pre-fill price estimate).
        atr: M15 ATR used for the stop.
        trend: M15 structure at the time of the rejection.
        zone_source: Which support/resistance level was retested.
        zone_level: Price of that level.
        pullback_extreme: Lowest low (long) / highest high (short) of the pullback leg.
        rejection_time: Open time of the rejection candle.
        metadata: Free-form diagnostics.
    """

    symbol: str
    timestamp: pd.Timestamp
    direction: Direction
    reason: str
    stop_price: float | None = None
    reference_price: float | None = None
    atr: float | None = None
    trend: Trend = Trend.NONE
    zone_source: str | None = None
    zone_level: float | None = None
    pullback_extreme: float | None = None
    rejection_time: pd.Timestamp | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_entry(self) -> bool:
        """True when the signal asks to open a position."""
        return self.direction is not Direction.NONE
