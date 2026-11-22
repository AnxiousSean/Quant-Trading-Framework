import json

import numpy as np
import pandas as pd
import pytest

from tests.conftest import M5, M15, m5_retest, uptrend_m15
from trading_framework.analytics import compute_metrics, drawdown_series, write_report
from trading_framework.analytics.trade_log import TradeLogger
from trading_framework.backtest import Backtester, TradingEngine
from trading_framework.backtest.engine import Bar
from trading_framework.config.settings import ExecutionConfig, RiskConfig, StrategyConfig, load_config
from trading_framework.data import DataValidator, generate_synthetic_ohlcv, load_market_data, resample_ohlcv
from trading_framework.execution import LiveBroker, LiveTradingNotAuthorized, PaperBroker
from trading_framework.portfolio import Portfolio
from trading_framework.risk import MarketLimits, RiskManager
from trading_framework.strategy import Direction, SignalEvent, TwoStepStrategy

LIMITS = MarketLimits(0.001, 0.01, 0.001, 5.0)
T0 = pd.Timestamp("2024-01-01", tz="UTC")


def make_engine(slippage=0.0, fee=0.0) -> TradingEngine:
    ex = ExecutionConfig(slippage=slippage, taker_fee=fee)
    return TradingEngine("X", TwoStepStrategy(StrategyConfig()), RiskManager(RiskConfig(max_leverage=100), 2.0, LIMITS),
                         Portfolio(10_000), PaperBroker(ex, LIMITS), ex.max_spread)


def open_long(engine, stop=98.0):
    engine.pending = SignalEvent("X", T0, Direction.LONG, "t", stop_price=stop, reference_price=100.0)
    return engine.execute_pending(100.0, T0)


# ---------------------------------------------------------------- execution
def test_paper_broker_slippage_and_fees():
    b = PaperBroker(ExecutionConfig(slippage=0.001, taker_fee=0.0004), LIMITS)
    assert b.expected_fill_price("buy", 100.0) == pytest.approx(100.1)
    assert b.expected_fill_price("sell", 100.0) == pytest.approx(99.9)
    f = b.submit_exit("X", "sell", 2.0, 100.0, T0, "take_profit")
    assert f.price == 100.0 and f.fee == pytest.approx(2 * 100 * 0.0004)  # TP = limit, no slippage


def test_stop_and_target_exits():
    e = make_engine()
    pos = open_long(e)
    assert pos.take_profit == pytest.approx(104.0)
    e.check_exits(Bar(T0, 100.0, 103.0, 99.0, 102.0))
    assert e.portfolio.open_positions()  # neither level touched
    trades = e.check_exits(Bar(T0 + M5, 102.0, 104.5, 101.0, 104.0))
    assert trades[0].exit_reason == "take_profit"
    assert trades[0].r_multiple == pytest.approx(2.0)


def test_stop_assumed_first_when_both_hit():
    e = make_engine()
    open_long(e)
    t = e.check_exits(Bar(T0, 100.0, 105.0, 97.0, 101.0))[0]
    assert t.exit_reason == "stop_loss" and t.r_multiple == pytest.approx(-1.0)


def test_gap_through_stop_fills_at_open():
    e = make_engine()
    open_long(e)
    t = e.check_exits(Bar(T0, 96.0, 96.5, 95.0, 96.0))[0]
    assert t.exit_reason == "stop_gap" and t.exit_price == pytest.approx(96.0)


def test_fees_and_slippage_reduce_pnl():
    e = make_engine(slippage=0.0005, fee=0.0004)
    pos = open_long(e)
    assert pos.entry_price == pytest.approx(100.05)
    t = e.check_exits(Bar(T0, 100.0, 100.5, 90.0, 95.0))[0]
    assert t.net_pnl < -pos.risk_amount  # slipped stop + fees exceed 1R
    assert t.fees > 0


def test_spread_filter_blocks_entry():
    ex = ExecutionConfig(assumed_spread=0.01, max_spread=0.001)
    e = TradingEngine("X", TwoStepStrategy(StrategyConfig()), RiskManager(RiskConfig(), 2.0, LIMITS),
                      Portfolio(10_000), PaperBroker(ex, LIMITS), ex.max_spread)
    assert open_long(e) is None and "spread" in e.rejections[0][1]


def test_engine_queues_signal_and_enters_next_open():
    m15, swing_low = uptrend_m15()
    m5 = m5_retest(m15, swing_low)
    e = make_engine()
    last = m5.iloc[-1]
    sig = e.on_bar_close(Bar(m5.index[-1], *last[["open", "high", "low", "close", "volume"]]), m15, m5)
    assert sig is not None and e.pending is sig
    pos = e.execute_pending(float(last["close"]), sig.timestamp)
    assert pos.direction is Direction.LONG and pos.stop_price < pos.entry_price < pos.take_profit
    assert e.pending is None


# --------------------------------------------------------------------- live
def test_live_broker_requires_keys(monkeypatch):
    from trading_framework.config.settings import ExchangeConfig

    monkeypatch.delenv("EXCHANGE_API_KEY", raising=False)
    monkeypatch.delenv("EXCHANGE_API_SECRET", raising=False)
    with pytest.raises(LiveTradingNotAuthorized):
        LiveBroker(ExchangeConfig(), ExecutionConfig(dry_run=False), live_flag=True, exchange=object())


def test_live_broker_dry_run_by_default():
    from trading_framework.config.settings import ExchangeConfig
    from trading_framework.risk.manager import OrderPlan

    class Boom:
        def create_order(self, *a, **k):  # pragma: no cover - must never be called
            raise AssertionError("order sent in dry-run")

        def load_markets(self):
            raise ConnectionError

    b = LiveBroker(ExchangeConfig(), ExecutionConfig(), live_flag=True, exchange=Boom())
    assert b.dry_run
    plan = OrderPlan("X", Direction.LONG, 100.0, 98.0, 104.0, 1.0, 2.0, 100.0)
    assert b.submit_entry(plan, 100.0, T0).order_id == "dry-run"
    # No --live flag => always dry-run, even if config disables it.
    assert LiveBroker(ExchangeConfig(), ExecutionConfig(dry_run=False), live_flag=False, exchange=Boom()).dry_run


# ------------------------------------------------------------ full backtest
def test_backtest_on_synthetic_data(app_cfg, tmp_path):
    m15, m5 = load_market_data(app_cfg)
    log = TradeLogger(str(tmp_path / "t.sqlite"), str(tmp_path / "t.csv"))
    res = Backtester(app_cfg, log).run(m15, m5)
    assert res.equity.iloc[0] == pytest.approx(app_cfg.backtest.initial_equity, rel=0.05)
    assert {"sharpe", "max_drawdown_pct", "win_rate_pct", "profit_factor", "trades"} <= res.metrics.keys()
    if not res.trades.empty:
        # Losses are ~1R (+costs); wins cap near 2R.
        assert res.trades["r_multiple"].max() <= 2.05
        assert res.trades["r_multiple"].min() >= -1.6
        assert (res.trades["entry_time"] <= res.trades["exit_time"]).all()
        assert len(pd.read_csv(tmp_path / "t.csv")) == len(res.trades)
    paths = write_report(str(tmp_path / "rep"), res.metrics, res.equity, res.trades)
    assert json.loads(paths["metrics_json"].read_text())["trades"] == res.metrics["trades"]
    assert "<svg" in paths["html"].read_text()


def test_backtest_has_no_lookahead(app_cfg):
    """Truncating future data must not change trades that closed before the cut."""
    m15, m5 = load_market_data(app_cfg)
    full = Backtester(app_cfg).run(m15, m5).trades
    cut = m5.index[len(m5) * 2 // 3]
    part = Backtester(app_cfg).run(m15[m15.index + M15 <= cut], m5[m5.index < cut]).trades
    if full.empty:
        pytest.skip("no trades on this seed")
    done_full = full[full["exit_time"] < cut - M5].reset_index(drop=True)
    done_part = part[part["exit_reason"] != "end_of_data"].reset_index(drop=True)
    pd.testing.assert_frame_equal(done_full, done_part)


# ------------------------------------------------------------- data/metrics
def test_resample_drops_incomplete_buckets():
    m5 = generate_synthetic_ohlcv(10, seed=1)  # 10 x 5m = 3 full 15m buckets + 1 partial
    m15 = resample_ohlcv(m5, "5m", "15m")
    assert len(m15) == 3
    assert m15["high"].iloc[0] == pytest.approx(m5["high"].iloc[:3].max())
    assert m15["close"].iloc[0] == pytest.approx(m5["close"].iloc[2])


def test_validator_flags_gaps_and_bad_rows():
    df = generate_synthetic_ohlcv(500, seed=2)
    df = df.drop(df.index[100:150])
    df.iloc[10, df.columns.get_loc("high")] = df["low"].iloc[10] - 1  # invalid candle
    v = DataValidator("5m", min_bars=100, outlier_threshold=0.15, max_gap_ratio=0.02)
    rep = v.validate(df)
    assert rep.missing_bars == 50 and rep.invalid_ohlc == 1 and not rep.ok
    assert len(v.clean(df)) == len(df) - 1
    assert not v.validate(df.head(10)).sufficient_history


def test_metrics_basic():
    idx = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
    eq = pd.Series([100, 110, 99, 120, 120], index=idx, dtype=float)
    assert drawdown_series(eq).min() == pytest.approx(99 / 110 - 1)
    trades = pd.DataFrame({"net_pnl": [10.0, -5.0, 20.0], "r_multiple": [1, -1, 2], "fees": [0.1] * 3,
                           "holding_minutes": [5, 10, 15], "direction": ["LONG", "SHORT", "LONG"]})
    m = compute_metrics(eq, trades, 100.0)
    assert m["profit_factor"] == pytest.approx(6.0)
    assert m["win_rate_pct"] == pytest.approx(66.6667, rel=1e-3)
    assert m["total_return_pct"] == pytest.approx(20.0)
    assert np.isfinite(m["sharpe"])


def test_config_yaml_and_env_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("risk:\n  risk_per_trade: 0.02\nstrategy:\n  atr_period: 10\n")
    cfg = load_config(p, environ={"TSF__RISK__RISK_PER_TRADE": "0.005", "TSF__EXECUTION__DRY_RUN": "true"})
    assert cfg.risk.risk_per_trade == 0.005 and cfg.strategy.atr_period == 10
    with pytest.raises(ValueError):
        load_config(p, environ={"TSF__STRATEGY__ATR_PERIOD": "1"})
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_sample_config_loads():
    cfg = load_config("config/btc_usdt.yaml", environ={})
    assert cfg.exchange.id == "binanceusdm" and cfg.execution.dry_run is True
    assert cfg.execution.taker_fee == 0.0004 and cfg.execution.slippage == 0.0005
