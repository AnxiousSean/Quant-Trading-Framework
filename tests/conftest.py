"""Synthetic market fixtures with hand-designed structure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_framework.config.settings import AppConfig, StrategyConfig

M15 = pd.Timedelta(minutes=15)
M5 = pd.Timedelta(minutes=5)
START = pd.Timestamp("2024-01-01", tz="UTC")


def bars_from_path(points: list[float], start: pd.Timestamp = START, freq: pd.Timedelta = M15,
                   wick: float = 0.2) -> pd.DataFrame:
    """Build candles whose closes follow ``points``.

    Opens sit 30% of the way from the previous close toward the new close so
    turning-point candles are strict fractal pivots (no tied highs/lows).
    """
    rows = []
    prev = points[0]
    for p in points[1:]:
        o = prev + 0.3 * (p - prev)
        rows.append((o, max(o, p) + wick, min(o, p) - wick, p, 100.0))
        prev = p
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC", name="timestamp")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)


def zigzag(start: float, legs: int, up: float, down: float, bars_per_leg: int = 4) -> list[float]:
    """Alternating up/down legs (``down`` may be negative for a downtrend start)."""
    pts = [start]
    for k in range(legs):
        delta = up if k % 2 == 0 else -down
        for _ in range(bars_per_leg):
            pts.append(pts[-1] + delta / bars_per_leg)
    return pts


def mirror(df: pd.DataFrame, k: float) -> pd.DataFrame:
    """Reflect prices around ``k / 2`` turning an uptrend into a downtrend."""
    return pd.DataFrame({"open": k - df["open"], "high": k - df["low"], "low": k - df["high"],
                         "close": k - df["close"], "volume": df["volume"]}, index=df.index)


def uptrend_m15() -> tuple[pd.DataFrame, float]:
    """~90 M15 candles of HH/HL structure ending in a 3-candle pullback.

    Returns:
        ``(data_m15, last_swing_low_price)``.
    """
    pts = zigzag(50.0, legs=23, up=10.0, down=5.0)  # ends after an up leg (23 odd -> last leg up)
    pullback = [pts[-1] - 1.25 * i for i in (1, 2, 3)]
    df = bars_from_path(pts + pullback)
    last_trough = pts[-1] - 10.0  # trough before the final up leg
    return df, last_trough - 0.2  # pivot low = close - wick


def m5_retest(m15: pd.DataFrame, level: float, *, pierce: float = 0.8, confirm: bool = True,
              reach: bool = True) -> pd.DataFrame:
    """M5 candles after ``m15`` that drift down to ``level`` then reject it.

    Args:
        m15: M15 frame; M5 candles start when its last candle closes.
        level: Support level to retest.
        pierce: How far the rejection wick dips below ``level``.
        confirm: If False, the confirmation candle fails to close above the rejection high.
        reach: If False, the whole sequence stays 5 points above ``level``.
    """
    base = level if reach else level + 5.0
    last_close = float(m15["close"].iloc[-1])
    drift = np.linspace(last_close, base + 1.5, 4)
    rows = []
    for a, b in zip(drift[:-1], drift[1:]):
        rows.append((a, a + 0.1, b - 0.1, b, 100.0))
    o = base + 1.0
    rows.append((o, base + 1.3, base - pierce, base + 1.2, 100.0))  # rejection: long lower wick
    if confirm:
        rows.append((base + 1.2, base + 2.1, base + 1.1, base + 2.0, 100.0))  # bullish, > rej high
    else:
        rows.append((base + 1.2, base + 1.25, base + 0.9, base + 1.0, 100.0))  # bearish, fails
    start = m15.index[-1] + M15
    idx = pd.date_range(start, periods=len(rows), freq=M5, tz="UTC", name="timestamp")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)


@pytest.fixture
def strategy_cfg() -> StrategyConfig:
    return StrategyConfig()


@pytest.fixture
def app_cfg() -> AppConfig:
    return AppConfig.model_validate({
        "data": {"source": "synthetic", "synthetic_bars": 6000, "synthetic_seed": 3},
        "logging": {"file": None},
    })
