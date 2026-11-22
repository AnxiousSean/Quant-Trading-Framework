import ccxt
import pandas as pd
import pytest

from trading_framework.config.settings import DataConfig, ExchangeConfig
from trading_framework.data import CCXTDataHandler, DataUnavailableError

STEP_MS = 5 * 60 * 1000
T0 = pd.Timestamp("2024-01-01", tz="UTC")


class FakeExchange:
    """Serves deterministic 5m candles; can fail transiently."""

    def __init__(self, n_bars: int = 2500, fail_first: int = 0):
        base = int(T0.timestamp() * 1000)
        self.rows = [[base + i * STEP_MS, 100 + i, 101 + i, 99 + i, 100.5 + i, 1.0] for i in range(n_bars)]
        self.calls = 0
        self.fail_first = fail_first

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=500):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise ccxt.NetworkError("transient")
        if since is None:
            return self.rows[-limit:]
        return [r for r in self.rows if r[0] >= since][:limit]


def handler(tmp_path, ex, **kw):
    return CCXTDataHandler(ExchangeConfig(retry_backoff_seconds=0.001, **kw),
                           DataConfig(cache_dir=str(tmp_path), fetch_limit=1000), exchange=ex)


def test_pagination_fetches_full_range(tmp_path):
    ex = FakeExchange()
    end = T0 + pd.Timedelta(minutes=5 * 2500)
    df = handler(tmp_path, ex).fetch_ohlcv("X", "5m", T0, end)
    assert len(df) == 2500 and df.index.is_monotonic_increasing
    assert ex.calls >= 3  # 1000-candle pages


def test_cache_avoids_refetch(tmp_path):
    ex = FakeExchange()
    end = T0 + pd.Timedelta(minutes=5 * 1200)
    h = handler(tmp_path, ex)
    first = h.load("X", "5m", T0, end)
    calls = ex.calls
    second = h.load("X", "5m", T0, end)
    assert ex.calls == calls  # served from SQLite
    pd.testing.assert_frame_equal(first, second, check_freq=False)


def test_retry_then_success(tmp_path):
    ex = FakeExchange(n_bars=10, fail_first=2)
    df = handler(tmp_path, ex, max_retries=3).fetch_ohlcv("X", "5m", T0, T0 + pd.Timedelta(minutes=50))
    assert len(df) == 10


def test_retry_exhausted_raises(tmp_path):
    ex = FakeExchange(fail_first=99)
    with pytest.raises(DataUnavailableError):
        handler(tmp_path, ex, max_retries=1).fetch_ohlcv("X", "5m", T0, T0 + pd.Timedelta(hours=1))


def test_forming_candle_is_dropped(tmp_path):
    now = pd.Timestamp.now(tz="UTC").floor("5min")
    ex = FakeExchange(n_bars=0)
    ex.rows = [[int((now - pd.Timedelta(minutes=5 * k)).timestamp() * 1000), 1, 2, 0.5, 1.5, 1] for k in (2, 1, 0)]
    df = handler(tmp_path, ex).fetch_recent("X", "5m", 10)
    assert df.index[-1] == now - pd.Timedelta(minutes=5)  # the candle opened at `now` is still forming


def test_paper_trader_step_runs_offline(tmp_path):
    from trading_framework.config.settings import AppConfig
    from trading_framework.execution import PaperBroker
    from trading_framework.execution.runner import PaperTrader
    from trading_framework.risk import MarketLimits

    now = pd.Timestamp.now(tz="UTC").floor("15min")
    ex = FakeExchange(n_bars=600)
    start = now - pd.Timedelta(minutes=5 * 600)
    ex.rows = [[int((start + pd.Timedelta(minutes=5 * i)).timestamp() * 1000), *r[1:]] for i, r in enumerate(ex.rows)]
    ex.fetch_ticker = lambda symbol: {"last": ex.rows[-1][4]}
    cfg = AppConfig.model_validate({"data": {"cache_dir": str(tmp_path)}, "logging": {"file": None},
                                    "execution": {"poll_interval_seconds": 0.001}})
    trader = PaperTrader(cfg, PaperBroker(cfg.execution, MarketLimits.from_config(cfg.exchange)),
                         CCXTDataHandler(cfg.exchange, cfg.data, exchange=ex))
    trader.run(max_iterations=2)
    assert trader.last_processed == now - pd.Timedelta(minutes=5)
    assert trader.engine.portfolio.equity() == pytest.approx(cfg.backtest.initial_equity)
