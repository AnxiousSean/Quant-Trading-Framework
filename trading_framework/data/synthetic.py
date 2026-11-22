"""Synthetic OHLCV generator for offline demos and tests.

Produces a regime-switching random walk (trending up, trending down, ranging)
with realistic-looking wicks. It is NOT a model of real markets and results
on it say nothing about live performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_ohlcv(
    bars: int,
    timeframe_minutes: int = 5,
    start: str = "2024-01-01",
    start_price: float = 40_000.0,
    bar_vol: float = 0.0015,
    regime_length: int = 600,
    trend_drift: float = 0.0003,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV candles.

    Args:
        bars: Number of candles.
        timeframe_minutes: Candle duration.
        start: First candle open time (UTC).
        start_price: Initial price.
        bar_vol: Per-bar log-return volatility.
        regime_length: Mean bars per regime.
        trend_drift: Per-bar drift magnitude in trending regimes.
        seed: RNG seed for reproducibility.

    Returns:
        UTC-indexed OHLCV frame.
    """
    rng = np.random.default_rng(seed)
    drift = np.empty(bars)
    i = 0
    while i < bars:
        length = max(50, int(rng.exponential(regime_length)))
        drift[i : i + length] = rng.choice([trend_drift, -trend_drift, 0.0])
        i += length
    # Add mean-reverting oscillation so trends contain pullbacks (swing structure).
    t = np.arange(bars)
    wave = 0.6 * bar_vol * np.sin(2 * np.pi * t / rng.integers(24, 48))
    rets = drift + wave + rng.normal(0, bar_vol, bars)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    body_hi = np.maximum(open_, close)
    body_lo = np.minimum(open_, close)
    wick_scale = bar_vol * close
    upper = body_hi + np.abs(rng.normal(0, 0.8, bars)) * wick_scale
    lower = body_lo - np.abs(rng.normal(0, 0.8, bars)) * wick_scale
    volume = rng.lognormal(mean=3.0, sigma=0.5, size=bars)
    idx = pd.date_range(start=start, periods=bars, freq=f"{timeframe_minutes}min", tz="UTC", name="timestamp")
    return pd.DataFrame({"open": open_, "high": upper, "low": lower, "close": close, "volume": volume}, index=idx)
