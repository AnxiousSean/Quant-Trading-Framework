"""Broker interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from trading_framework.risk.manager import MarketLimits, OrderPlan

OrderType = Literal["market", "stop", "take_profit"]


@dataclass(frozen=True)
class Fill:
    """An executed order.

    Attributes:
        symbol: Market symbol.
        side: ``"buy"`` or ``"sell"``.
        amount: Filled quantity.
        price: Average fill price.
        fee: Fee paid in quote currency.
        timestamp: Fill time.
        order_type: ``market``, ``stop`` or ``take_profit``.
        reference_price: Price the order was based on (for slippage stats).
        order_id: Exchange order id, if any.
    """

    symbol: str
    side: str
    amount: float
    price: float
    fee: float
    timestamp: pd.Timestamp
    order_type: OrderType
    reference_price: float
    order_id: str | None = None

    @property
    def slippage(self) -> float:
        """Signed fractional deviation from the reference price (positive = adverse)."""
        sign = 1 if self.side == "buy" else -1
        return sign * (self.price - self.reference_price) / self.reference_price


class Broker(ABC):
    """Common interface for paper and live brokers."""

    @abstractmethod
    def market_limits(self, symbol: str) -> MarketLimits:
        """Exchange precision and minimums for ``symbol``."""

    @abstractmethod
    def get_spread(self, symbol: str) -> float:
        """Current bid/ask spread as a fraction of mid price."""

    @abstractmethod
    def expected_fill_price(self, side: str, reference_price: float) -> float:
        """Price a market order is expected to fill at (used for sizing)."""

    @abstractmethod
    def submit_entry(self, plan: OrderPlan, reference_price: float, timestamp: pd.Timestamp) -> Fill:
        """Open a position (and, for live brokers, attach protective orders)."""

    @abstractmethod
    def submit_exit(
        self, symbol: str, side: str, amount: float, price: float, timestamp: pd.Timestamp,
        order_type: OrderType,
    ) -> Fill:
        """Close (part of) a position at ``price`` for the given order type."""
