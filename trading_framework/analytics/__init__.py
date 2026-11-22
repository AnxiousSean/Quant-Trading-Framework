"""Trade logging, performance metrics and reports."""

from trading_framework.analytics.metrics import compute_metrics, drawdown_series
from trading_framework.analytics.report import write_report
from trading_framework.analytics.trade_log import TradeLogger

__all__ = ["TradeLogger", "compute_metrics", "drawdown_series", "write_report"]
