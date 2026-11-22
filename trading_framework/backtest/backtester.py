"""Transparent bar-by-bar backtester over M5 candles with M15 context."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from trading_framework.analytics.metrics import compute_metrics
from trading_framework.analytics.trade_log import TradeLogger
from trading_framework.backtest.engine import Bar, TradingEngine
from trading_framework.config.settings import AppConfig
from trading_framework.execution.paper import PaperBroker
from trading_framework.portfolio.portfolio import Portfolio
from trading_framework.risk.manager import MarketLimits, RiskManager
from trading_framework.strategy.signals import SignalEvent
from trading_framework.strategy.two_step import TwoStepStrategy

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Outputs of a backtest run."""

    metrics: dict[str, Any]
    equity: pd.Series
    trades: pd.DataFrame
    signals: list[SignalEvent] = field(default_factory=list)
    rejections: list[tuple[pd.Timestamp, str]] = field(default_factory=list)


class Backtester:
    """Replays history candle by candle through :class:`TradingEngine`.

    Only rows whose close time is at or before the current M5 close are handed
    to the strategy, and the strategy re-applies that filter itself, so there
    are two independent guards against look-ahead.

    Args:
        config: Full application config.
        trade_logger: Optional trade journal.
        limits: Market limits (defaults to config fallbacks).
    """

    def __init__(self, config: AppConfig, trade_logger: TradeLogger | None = None,
                 limits: MarketLimits | None = None) -> None:
        self.cfg = config
        self.limits = limits or MarketLimits.from_config(config.exchange)
        self.trade_logger = trade_logger

    def _build_engine(self) -> TradingEngine:
        c = self.cfg
        strategy = TwoStepStrategy(c.strategy, c.data.trend_timeframe, c.data.entry_timeframe, c.exchange.symbol)
        risk = RiskManager(c.risk, c.strategy.reward_risk, self.limits)
        broker = PaperBroker(c.execution, self.limits)
        portfolio = Portfolio(c.backtest.initial_equity)
        return TradingEngine(c.exchange.symbol, strategy, risk, portfolio, broker, c.execution.max_spread,
                             self.trade_logger)

    def run(self, data_m15: pd.DataFrame, data_m5: pd.DataFrame) -> BacktestResult:
        """Run the backtest.

        Args:
            data_m15: Clean M15 candles.
            data_m5: Clean M5 candles.

        Returns:
            :class:`BacktestResult` with metrics, equity curve and trades.
        """
        engine = self._build_engine()
        strat = engine.strategy
        m15_close = (data_m15.index + strat.trend_td).asi8
        m5_close = (data_m5.index + strat.entry_td).asi8
        o, h, lo, c, v = (data_m5[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close", "volume"))
        w15, w5 = strat.trend_window, strat.entry_window

        first = int(np.searchsorted(m5_close, m15_close[min(w15, len(m15_close) - 1)], side="left"))
        logger.info("Backtesting %d M5 candles (warm-up %d)", len(data_m5) - first, first)

        for i in range(first, len(data_m5)):
            t = data_m5.index[i]
            if engine.pending is not None:
                engine.execute_pending(o[i], t)
            j = int(np.searchsorted(m15_close, m5_close[i], side="right"))
            engine.on_bar_close(
                Bar(t, o[i], h[i], lo[i], c[i], v[i]),
                data_m15.iloc[max(0, j - w15): j],
                data_m5.iloc[max(0, i + 1 - w5): i + 1],
            )

        if len(data_m5):
            last_close_t = data_m5.index[-1] + strat.entry_td
            engine.close_all(c[-1], last_close_t)
            engine.portfolio.mark(last_close_t, {engine.symbol: c[-1]})

        equity = engine.portfolio.equity_curve()
        trades = engine.portfolio.trades_frame()
        metrics = compute_metrics(equity, trades, self.cfg.backtest.initial_equity,
                                  self.cfg.backtest.periods_per_year, self.cfg.backtest.risk_free_rate)
        metrics["signals"] = len(engine.signals)
        metrics["skipped_entries"] = len(engine.rejections)
        return BacktestResult(metrics, equity, trades, engine.signals, engine.rejections)
