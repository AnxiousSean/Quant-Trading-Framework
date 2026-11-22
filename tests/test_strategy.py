import pandas as pd
import pytest

from tests.conftest import M5, M15, bars_from_path, m5_retest, mirror, uptrend_m15
from trading_framework.config.settings import StrategyConfig
from trading_framework.strategy import Direction, Trend, TwoStepStrategy

K = 400.0  # mirror constant


@pytest.fixture
def strat(strategy_cfg):
    return TwoStepStrategy(strategy_cfg, "15m", "5m", "BTC/USDT")


# ----------------------------------------------------------------- structure
def test_uptrend_structure(strat):
    m15, swing_low = uptrend_m15()
    s = strat.analyze_structure(m15)
    assert s.trend is Trend.UP
    assert s.swing_lows[-1].price == pytest.approx(swing_low)
    assert s.atr > 0 and s.lower_band < s.last_close < s.upper_band


def test_downtrend_structure(strat):
    m15, _ = uptrend_m15()
    assert strat.analyze_structure(mirror(m15, K)).trend is Trend.DOWN


def test_range_is_no_trade_zone(strat):
    pts = [100.0]
    for k in range(24):  # equal highs/lows oscillation
        target = 110.0 if k % 2 == 0 else 100.0
        step = (target - pts[-1]) / 4
        pts += [pts[-1] + step * (i + 1) for i in range(4)]
    s = strat.analyze_structure(bars_from_path(pts))
    assert s.trend is Trend.NONE
    assert "ambiguous" in s.reason or "swings" in s.reason


def test_broken_structure_is_no_trade(strat):
    m15, swing_low = uptrend_m15()
    crash = m15.iloc[[-1]].copy()
    crash.index = crash.index + M15
    crash[["open", "high", "low", "close"]] = [m15["close"].iloc[-1], m15["close"].iloc[-1] + 0.1,
                                               swing_low - 3, swing_low - 2]
    s = strat.analyze_structure(pd.concat([m15, crash]))
    assert s.trend is Trend.NONE


def test_insufficient_history(strat):
    m15, _ = uptrend_m15()
    s = strat.analyze_structure(m15.tail(30))
    assert s.trend is Trend.NONE and "insufficient" in s.reason


# ------------------------------------------------------------------- signals
def test_long_signal_on_swing_low_retest(strat, strategy_cfg):
    m15, swing_low = uptrend_m15()
    m5 = m5_retest(m15, swing_low)
    sig = strat.generate_signals(m15, m5)
    assert sig.direction is Direction.LONG, sig.reason
    assert sig.is_entry and sig.trend is Trend.UP
    assert sig.zone_source == "swing_low"
    assert sig.zone_level == pytest.approx(swing_low)
    # Entry is at the open of the candle after confirmation.
    assert sig.timestamp == m5.index[-1] + M5
    assert sig.reference_price == pytest.approx(m5["close"].iloc[-1])
    # Stop = lowest low of pullback leg - 2 x ATR.
    atr_v = strat.analyze_structure(m15).atr
    assert sig.pullback_extreme == pytest.approx(m5["low"].min())
    assert sig.stop_price == pytest.approx(m5["low"].min() - strategy_cfg.stop_atr_multiplier * atr_v)
    assert sig.stop_price < sig.reference_price


def test_short_signal_on_swing_high_retest(strat):
    m15, swing_low = uptrend_m15()
    m15_dn = mirror(m15, K)
    m5_dn = mirror(m5_retest(m15, swing_low), K)
    sig = strat.generate_signals(m15_dn, m5_dn)
    assert sig.direction is Direction.SHORT, sig.reason
    assert sig.zone_source == "swing_high"
    assert sig.zone_level == pytest.approx(K - swing_low)
    assert sig.pullback_extreme == pytest.approx(m5_dn["high"].max())
    assert sig.stop_price > sig.reference_price


def test_long_signal_on_atr_band_only(strategy_cfg):
    cfg = strategy_cfg.model_copy(update={"zone_sources": ["atr_band"]})
    strat = TwoStepStrategy(cfg, "15m", "5m")
    m15, _ = uptrend_m15()
    lower = strat.analyze_structure(m15).lower_band
    sig = strat.generate_signals(m15, m5_retest(m15, lower))
    assert sig.direction is Direction.LONG, sig.reason
    assert sig.zone_source == "lower_atr_band"


def test_no_signal_without_confirmation(strat):
    m15, swing_low = uptrend_m15()
    sig = strat.generate_signals(m15, m5_retest(m15, swing_low, confirm=False))
    assert sig.direction is Direction.NONE


def test_no_signal_when_support_not_retested(strat):
    m15, swing_low = uptrend_m15()
    sig = strat.generate_signals(m15, m5_retest(m15, swing_low, reach=False))
    assert sig.direction is Direction.NONE
    assert "did not retest" in sig.reason


def test_countertrend_pattern_ignored(strat):
    m15, swing_low = uptrend_m15()
    # A bearish rejection/confirmation inside an uptrend must not produce a short.
    bearish = mirror(m5_retest(m15, swing_low), K)
    bearish[["open", "high", "low", "close"]] -= (K - 2 * float(m15["close"].iloc[-1]))
    sig = strat.generate_signals(m15, bearish)
    assert sig.direction is not Direction.SHORT


def test_no_signal_in_range(strat):
    pts = [100.0]
    for k in range(24):
        target = 110.0 if k % 2 == 0 else 100.0
        step = (target - pts[-1]) / 4
        pts += [pts[-1] + step * (i + 1) for i in range(4)]
    m15 = bars_from_path(pts)
    sig = strat.generate_signals(m15, m5_retest(m15, 99.8))
    assert sig.direction is Direction.NONE
    assert "no trend" in sig.reason


def test_spike_filter(strategy_cfg):
    cfg = strategy_cfg.model_copy(update={"max_signal_bar_range_atr": 0.1})
    m15, swing_low = uptrend_m15()
    sig = TwoStepStrategy(cfg).generate_signals(m15, m5_retest(m15, swing_low))
    assert sig.direction is Direction.NONE and "spike" in sig.reason


def test_volume_filter(strategy_cfg):
    cfg = strategy_cfg.model_copy(update={"min_volume": 1_000.0})
    m15, swing_low = uptrend_m15()
    sig = TwoStepStrategy(cfg).generate_signals(m15, m5_retest(m15, swing_low))
    assert sig.direction is Direction.NONE and "volume" in sig.reason


def test_insufficient_m5(strat):
    m15, swing_low = uptrend_m15()
    assert strat.generate_signals(m15, m5_retest(m15, swing_low).tail(2)).direction is Direction.NONE


# --------------------------------------------------------------- no leakage
def test_future_m15_rows_are_ignored(strat):
    """M15 candles that close after the rejection candle must not affect the signal."""
    m15, swing_low = uptrend_m15()
    m5 = m5_retest(m15, swing_low)
    baseline = strat.generate_signals(m15, m5)
    # Append an M15 candle that is still open when the rejection candle closes and crashes support.
    rej_close = m5.index[-2] + M5
    future = pd.DataFrame(
        {"open": [m15["close"].iloc[-1]], "high": [m15["close"].iloc[-1]], "low": [swing_low - 30],
         "close": [swing_low - 25], "volume": [1e6]},
        index=[m15.index[-1] + 2 * M15],
    )
    assert future.index[0] + M15 > rej_close
    leaked = strat.generate_signals(pd.concat([m15, future]), m5)
    assert leaked == baseline
    assert leaked.direction is Direction.LONG


def test_signal_depends_only_on_closed_m5(strat):
    m15, swing_low = uptrend_m15()
    m5 = m5_retest(m15, swing_low)
    # Evaluated one candle earlier (confirmation not yet closed) there is no trade.
    assert strat.generate_signals(m15, m5.iloc[:-1]).direction is Direction.NONE


def test_config_validation():
    with pytest.raises(ValueError):
        StrategyConfig(trend_lookback=4, swing_left=2, swing_right=2)
    with pytest.raises(ValueError):
        StrategyConfig(zone_sources=[])
    with pytest.raises(ValueError):
        TwoStepStrategy(StrategyConfig(), "5m", "15m")
