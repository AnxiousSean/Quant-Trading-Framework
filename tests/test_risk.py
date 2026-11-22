import pandas as pd
import pytest

from trading_framework.config.settings import RiskConfig
from trading_framework.risk import MarketLimits, RiskManager
from trading_framework.risk.manager import floor_to_step, round_to_tick
from trading_framework.strategy import Direction, SignalEvent

LIMITS = MarketLimits(amount_step=0.001, price_tick=0.01, min_amount=0.001, min_notional=5.0)
T0 = pd.Timestamp("2024-01-01 00:00", tz="UTC")


def rm(**kw) -> RiskManager:
    return RiskManager(RiskConfig(**{"max_leverage": 100.0, **kw}), reward_risk=2.0, limits=LIMITS)


def signal(direction=Direction.LONG, stop=98.0) -> SignalEvent:
    return SignalEvent("X", T0, direction, "test", stop_price=stop, reference_price=100.0, atr=1.0)


def test_position_size_risks_fixed_fraction():
    # (10_000 * 1%) / (100 - 98) = 50 units
    assert rm().position_size(10_000, 100.0, 98.0) == pytest.approx(50.0)
    assert rm(risk_per_trade=0.02).position_size(10_000, 100.0, 98.0) == pytest.approx(100.0)


def test_position_size_symmetric_for_shorts():
    assert rm().position_size(10_000, 100.0, 102.0) == pytest.approx(50.0)


def test_position_size_leverage_cap():
    # Uncapped: 100 / 0.5 = 200 units = 20,000 notional; cap 1x equity -> 100 units
    assert rm(max_leverage=1.0).position_size(10_000, 100.0, 99.5) == pytest.approx(100.0)


def test_position_size_respects_precision_and_minimums():
    size = rm().position_size(1_000, 30_123.0, 29_000.0)  # raw = 0.0089047...
    assert size == pytest.approx(0.008)
    tiny = RiskManager(RiskConfig(), 2.0, MarketLimits(0.001, 0.1, 0.001, 100.0))
    assert tiny.position_size(100, 30_000.0, 29_000.0) == 0.0  # below min amount / notional


def test_take_profit_is_two_r():
    r = rm()
    assert r.take_profit(100.0, 98.0, Direction.LONG) == pytest.approx(104.0)   # 100 + 2*(100-98)
    assert r.take_profit(100.0, 102.0, Direction.SHORT) == pytest.approx(96.0)  # 100 - 2*(102-100)


def test_plan_trade_long_and_short():
    plan, why = rm().plan_trade(signal(), 100.0, 10_000)
    assert why == "ok" and plan.amount == pytest.approx(50.0)
    assert plan.take_profit == pytest.approx(104.0)
    assert plan.risk_amount == pytest.approx(100.0)
    plan, _ = rm().plan_trade(signal(Direction.SHORT, 102.0), 100.0, 10_000)
    assert plan.take_profit == pytest.approx(96.0)


def test_plan_trade_rejects_gap_through_stop():
    plan, why = rm().plan_trade(signal(stop=98.0), 97.5, 10_000)
    assert plan is None and "stop" in why
    plan, why = rm().plan_trade(signal(Direction.SHORT, 102.0), 102.5, 10_000)
    assert plan is None


def test_daily_loss_limit_halts_and_resets():
    r = rm(daily_loss_limit=0.03)
    r.on_equity(T0, 10_000)
    r.on_equity(T0 + pd.Timedelta(hours=1), 9_800)
    assert not r.halted
    r.on_equity(T0 + pd.Timedelta(hours=2), 9_699)
    assert r.halted
    assert r.can_open("X", Direction.LONG, 0, 0) == (False, "daily loss limit reached")
    r.on_equity(T0 + pd.Timedelta(days=1), 9_699)
    assert not r.halted


def test_daily_loss_limit_disabled():
    r = rm(daily_loss_limit=None)
    r.on_equity(T0, 10_000)
    r.on_equity(T0 + pd.Timedelta(hours=1), 1_000)
    assert not r.halted


def test_exposure_limits():
    r = rm(max_positions_per_symbol=1, max_total_positions=2)
    assert r.can_open("X", Direction.LONG, 0, 0)[0]
    assert not r.can_open("X", Direction.LONG, 1, 1)[0]
    assert not r.can_open("Y", Direction.LONG, 0, 2)[0]
    assert not rm(allow_shorts=False).can_open("X", Direction.SHORT, 0, 0)[0]


def test_rounding_helpers():
    assert floor_to_step(0.0129999, 0.001) == pytest.approx(0.012)
    assert floor_to_step(0.003, 0.001) == pytest.approx(0.003)  # no float floor error
    assert round_to_tick(100.126, 0.01) == pytest.approx(100.13)


def test_risk_config_validation():
    with pytest.raises(ValueError):
        RiskConfig(risk_per_trade=0.5)
    with pytest.raises(ValueError):
        RiskConfig(max_positions_per_symbol=2, max_total_positions=1)
