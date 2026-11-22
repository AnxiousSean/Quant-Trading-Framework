"""Simulated broker with deterministic slippage and fees."""

from __future__ import annotations

import logging

import pandas as pd

from trading_framework.config.settings import ExecutionConfig
from trading_framework.execution.base import Broker, Fill, OrderType
from trading_framework.risk.manager import MarketLimits, OrderPlan, round_to_tick

logger = logging.getLogger(__name__)


class PaperBroker(Broker):
    """Fills orders instantly with configurable adverse slippage and taker fees.

    Slippage model:
        * Market and stop orders fill ``slippage`` worse than the reference price.
        * Take-profit orders are modelled as resting limit orders: no slippage.

    Args:
        config: Execution settings (fees, slippage, assumed spread).
        limits: Market precision/minimums.
    """

    def __init__(self, config: ExecutionConfig, limits: MarketLimits) -> None:
        self.cfg = config
        self.limits = limits

    def market_limits(self, symbol: str) -> MarketLimits:
        return self.limits

    def get_spread(self, symbol: str) -> float:
        return self.cfg.assumed_spread

    def _slipped(self, side: str, price: float) -> float:
        sign = 1 if side == "buy" else -1
        return round_to_tick(price * (1 + sign * self.cfg.slippage), self.limits.price_tick)

    def expected_fill_price(self, side: str, reference_price: float) -> float:
        return self._slipped(side, reference_price)

    def _fill(self, symbol: str, side: str, amount: float, price: float, ts: pd.Timestamp,
              order_type: OrderType, ref: float) -> Fill:
        fee = amount * price * self.cfg.taker_fee
        fill = Fill(symbol, side, amount, price, fee, ts, order_type, ref)
        logger.debug("PAPER %s %s %.6f @ %.2f fee=%.4f", order_type, side, amount, price, fee)
        return fill

    def submit_entry(self, plan: OrderPlan, reference_price: float, timestamp: pd.Timestamp) -> Fill:
        side = plan.direction.entry_side
        return self._fill(plan.symbol, side, plan.amount, self._slipped(side, reference_price),
                          timestamp, "market", reference_price)

    def submit_exit(self, symbol: str, side: str, amount: float, price: float, timestamp: pd.Timestamp,
                    order_type: OrderType) -> Fill:
        fill_price = price if order_type == "take_profit" else self._slipped(side, price)
        return self._fill(symbol, side, amount, fill_price, timestamp, order_type, price)
