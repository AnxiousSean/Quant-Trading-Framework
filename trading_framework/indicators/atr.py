"""Average True Range and ATR Bands.

All functions are pure and causal: the value at row ``t`` only depends on rows
``<= t``. Callers are responsible for passing only *closed* candles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Compute the True Range.

    ``TR = max(high - low, |high - prev_close|, |low - prev_close|)``. The first
    row (no previous close) falls back to ``high - low``.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.

    Returns:
        True range series aligned with the inputs.
    """
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    return ranges.max(axis=1, skipna=True).rename("tr")


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Wilder's Average True Range.

    Seeded with the simple mean of the first ``period`` true ranges, then
    smoothed with ``ATR_t = (ATR_{t-1} * (period - 1) + TR_t) / period``.
    Rows before the seed are ``NaN``.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        period: Smoothing period (must be > 1).

    Returns:
        ATR series aligned with the inputs.

    Raises:
        ValueError: If ``period`` is not > 1.
    """
    if period <= 1:
        raise ValueError("ATR period must be > 1")
    tr = true_range(high, low, close).to_numpy(dtype=float)
    out = np.full(tr.shape, np.nan)
    if len(tr) >= period:
        out[period - 1] = tr[:period].mean()
        for i in range(period, len(tr)):
            out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return pd.Series(out, index=high.index, name="atr")


def atr_bands(
    df: pd.DataFrame,
    period: int,
    multiplier: float,
    source: str = "close",
) -> pd.DataFrame:
    """ATR volatility envelope around a price source.

    ``upper = src + multiplier * ATR`` and ``lower = src - multiplier * ATR``.

    Args:
        df: OHLC frame with ``open, high, low, close`` columns.
        period: ATR period.
        multiplier: Band width in ATRs.
        source: ``"close"``, ``"hl2"`` or ``"hlc3"``.

    Returns:
        DataFrame with ``atr``, ``mid``, ``upper`` and ``lower`` columns.

    Raises:
        ValueError: On an unknown ``source``.
    """
    if source == "close":
        mid = df["close"]
    elif source == "hl2":
        mid = (df["high"] + df["low"]) / 2.0
    elif source == "hlc3":
        mid = (df["high"] + df["low"] + df["close"]) / 3.0
    else:
        raise ValueError(f"Unknown band source: {source!r}")
    a = atr(df["high"], df["low"], df["close"], period)
    return pd.DataFrame(
        {"atr": a, "mid": mid, "upper": mid + multiplier * a, "lower": mid - multiplier * a},
        index=df.index,
    )
