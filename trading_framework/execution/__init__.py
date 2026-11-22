"""Broker abstraction: paper (simulated) and live (ccxt, dry-run by default)."""

from trading_framework.execution.base import Broker, Fill
from trading_framework.execution.live import LiveBroker, LiveTradingNotAuthorized
from trading_framework.execution.paper import PaperBroker

__all__ = ["Broker", "Fill", "LiveBroker", "LiveTradingNotAuthorized", "PaperBroker"]
