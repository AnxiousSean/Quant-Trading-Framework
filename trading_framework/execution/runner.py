"""Polling loop for paper trading (and dry-run / live trading).

Each poll fetches the latest closed M5 candles from the exchange's public API,
derives M15 candles, and feeds any newly closed M5 candle to the shared
:class:`TradingEngine`. When a signal is produced it is executed immediately at
the current ticker price — the real-time equivalent of "next candle's open".
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from trading_framework.analytics.trade_log import TradeLogger
from trading_framework.backtest.engine import Bar, TradingEngine
from trading_framework.config.settings import AppConfig
from trading_framework.data.handler import CCXTDataHandler, DataUnavailableError
from trading_framework.data.loaders import resample_ohlcv
from trading_framework.execution.base import Broker
from trading_framework.portfolio.portfolio import Portfolio
from trading_framework.risk.manager import RiskManager
from trading_framework.strategy.two_step import TwoStepStrategy

logger = logging.getLogger(__name__)


class PaperTrader:
    """Real-time loop driving a :class:`TradingEngine` with exchange data.

    Args:
        config: Application config.
        broker: :class:`PaperBroker` or :class:`LiveBroker`.
        data_handler: Market data source.
        trade_logger: Optional trade journal.
    """

    def __init__(self, config: AppConfig, broker: Broker, data_handler: CCXTDataHandler,
                 trade_logger: TradeLogger | None = None) -> None:
        self.cfg = config
        self.data = data_handler
        c = config
        strategy = TwoStepStrategy(c.strategy, c.data.trend_timeframe, c.data.entry_timeframe, c.exchange.symbol)
        limits = broker.market_limits(c.exchange.symbol)
        risk = RiskManager(c.risk, c.strategy.reward_risk, limits)
        self.engine = TradingEngine(c.exchange.symbol, strategy, risk, Portfolio(c.backtest.initial_equity),
                                    broker, c.execution.max_spread, trade_logger)
        self.last_processed: pd.Timestamp | None = None
        ratio = int(strategy.trend_td / strategy.entry_td)
        self.m5_limit = min(1500, max(strategy.entry_window, (strategy.trend_window + 2) * ratio))

    def _reference_price(self, fallback: float) -> float:
        last_price = getattr(self.engine.broker, "last_price", None)
        if last_price is None:
            try:
                return float(self.data.exchange.fetch_ticker(self.cfg.exchange.symbol)["last"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Ticker unavailable (%s); using last close", exc)
                return fallback
        try:
            return float(last_price(self.cfg.exchange.symbol))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ticker unavailable (%s); using last close", exc)
            return fallback

    def step(self) -> None:
        """Run one poll: process newly closed candles and act on signals."""
        sym, d = self.cfg.exchange.symbol, self.cfg.data
        m5 = self.data.fetch_recent(sym, d.entry_timeframe, self.m5_limit)
        if m5.empty:
            logger.warning("No candles returned")
            return
        m15 = resample_ohlcv(m5, d.entry_timeframe, d.trend_timeframe)
        new = m5.index if self.last_processed is None else m5.index[m5.index > self.last_processed]
        if self.last_processed is None:
            new = new[-1:]  # start live: only evaluate the latest closed candle
        for t in new:
            i = m5.index.get_loc(t)
            row = m5.iloc[i]
            bar = Bar(t, float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]),
                      float(row["volume"]))
            signal = self.engine.on_bar_close(bar, m15, m5.iloc[: i + 1])
            self.last_processed = t
            if signal is not None and t == m5.index[-1]:
                ref = self._reference_price(bar.close)
                self.engine.execute_pending(ref, pd.Timestamp.now(tz="UTC"))
            elif signal is not None:
                logger.info("Stale signal at %s discarded (catch-up)", t)
                self.engine.pending = None
        eq = self.engine.portfolio.equity()
        logger.info("Processed up to %s | equity %.2f | open positions %d", self.last_processed, eq,
                    len(self.engine.portfolio.open_positions()))

    def run(self, max_iterations: int | None = None) -> None:
        """Poll forever (or ``max_iterations`` times). Ctrl+C to stop."""
        n = 0
        while max_iterations is None or n < max_iterations:
            try:
                self.step()
            except DataUnavailableError as exc:
                logger.error("Data error: %s", exc)
            except KeyboardInterrupt:
                raise
            except Exception:  # noqa: BLE001 - keep the loop alive, but log loudly
                logger.exception("Unexpected error in trading loop")
            n += 1
            if max_iterations is None or n < max_iterations:
                time.sleep(self.cfg.execution.poll_interval_seconds)
