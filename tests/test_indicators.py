import numpy as np
import pandas as pd
import pytest

from tests.conftest import bars_from_path, zigzag
from trading_framework.indicators import atr, atr_bands, find_swing_points, swing_points_known_at, true_range


def _frame(rows):
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="15min", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def test_true_range_uses_previous_close_gaps():
    df = _frame([(10, 11, 9, 10), (14, 15, 13.5, 14)])  # gap up: |15 - 10| = 5
    tr = true_range(df["high"], df["low"], df["close"])
    assert tr.iloc[0] == pytest.approx(2.0)
    assert tr.iloc[1] == pytest.approx(5.0)


def test_atr_wilder_values():
    rows = [(10, 11, 9, 10)] * 3 + [(10, 14, 10, 12)]
    df = _frame(rows)
    a = atr(df["high"], df["low"], df["close"], period=3)
    assert np.isnan(a.iloc[0]) and np.isnan(a.iloc[1])
    assert a.iloc[2] == pytest.approx(2.0)              # seed = mean(2,2,2)
    assert a.iloc[3] == pytest.approx((2.0 * 2 + 4.0) / 3)  # Wilder smoothing with TR=4


def test_atr_rejects_bad_period():
    df = _frame([(1, 2, 0.5, 1)] * 5)
    with pytest.raises(ValueError):
        atr(df["high"], df["low"], df["close"], 1)


def test_atr_is_causal():
    df = bars_from_path(zigzag(100, 10, 8, 4))
    full = atr(df["high"], df["low"], df["close"], 14)
    for k in (20, 30, len(df) - 1):
        part = atr(df["high"].iloc[:k], df["low"].iloc[:k], df["close"].iloc[:k], 14)
        pd.testing.assert_series_equal(part, full.iloc[:k])


def test_atr_bands_envelope():
    df = bars_from_path(zigzag(100, 10, 8, 4))
    b = atr_bands(df, 14, 2.0)
    valid = b.dropna()
    np.testing.assert_allclose(valid["upper"], df.loc[valid.index, "close"] + 2 * valid["atr"])
    np.testing.assert_allclose(valid["lower"], df.loc[valid.index, "close"] - 2 * valid["atr"])
    hl2 = atr_bands(df, 14, 2.0, source="hl2")
    np.testing.assert_allclose(hl2["mid"], (df["high"] + df["low"]) / 2)
    with pytest.raises(ValueError):
        atr_bands(df, 14, 2.0, source="nope")


def test_swings_found_at_turning_points():
    pts = zigzag(100, legs=4, up=10, down=5)  # turning points at bars 4, 8, 12
    df = bars_from_path(pts)
    swings = find_swing_points(df, 2, 2)
    highs = [(s.pivot_pos, s.price) for s in swings if s.kind == "high"]
    lows = [(s.pivot_pos, s.price) for s in swings if s.kind == "low"]
    # bars_from_path drops the first point, so turning point k sits at row k-1
    assert highs == [(3, pytest.approx(110.2)), (11, pytest.approx(115.2))]
    assert lows == [(7, pytest.approx(104.8))]
    assert all(s.confirm_pos == s.pivot_pos + 2 for s in swings)


def test_swings_never_use_unconfirmed_bars():
    df = bars_from_path(zigzag(100, legs=6, up=10, down=5))
    swings = find_swing_points(df, 2, 2)
    assert all(s.confirm_pos <= len(df) - 1 for s in swings)
    # Truncated data yields exactly the swings that were confirmed by then.
    for k in range(10, len(df)):
        part = find_swing_points(df.iloc[:k], 2, 2)
        assert part == swing_points_known_at(swings, k - 1)
