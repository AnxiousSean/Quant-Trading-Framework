"""Fractal (pivot) swing high / swing low detection without look-ahead.

A swing high at bar ``i`` requires ``high[i]`` to be strictly greater than the
``left`` highs before it and the ``right`` highs after it. Because the right-hand
bars must exist, a pivot at ``i`` is only *known* once bar ``i + right`` has
closed. Every :class:`SwingPoint` carries both indices so consumers can filter
by what was knowable at a given moment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SwingPoint:
    """A confirmed swing pivot.

    Attributes:
        kind: ``"high"`` or ``"low"``.
        pivot_pos: Integer position of the pivot candle.
        confirm_pos: Integer position of the candle whose close confirms the pivot.
        time: Timestamp (index label) of the pivot candle.
        price: Pivot price (the candle's high or low).
    """

    kind: Literal["high", "low"]
    pivot_pos: int
    confirm_pos: int
    time: pd.Timestamp
    price: float


def _pivots(values: np.ndarray, left: int, right: int, is_high: bool) -> list[int]:
    out: list[int] = []
    n = len(values)
    for i in range(left, n - right):
        v = values[i]
        if np.isnan(v):
            continue
        window_l = values[i - left : i]
        window_r = values[i + 1 : i + 1 + right]
        if is_high:
            if np.all(v > window_l) and np.all(v > window_r):
                out.append(i)
        elif np.all(v < window_l) and np.all(v < window_r):
            out.append(i)
    return out


def find_swing_points(df: pd.DataFrame, left: int, right: int) -> list[SwingPoint]:
    """Detect all fractal swing points in ``df``.

    Only pivots whose confirmation candle exists inside ``df`` are returned, so
    the result never depends on data beyond the last row.

    Args:
        df: Frame with ``high`` and ``low`` columns (closed candles only).
        left: Bars to the left that must be lower (highs) / higher (lows).
        right: Bars to the right; also the confirmation lag.

    Returns:
        Swing points sorted by pivot position.
    """
    if left < 1 or right < 1:
        raise ValueError("left and right must be >= 1")
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    idx = df.index
    points = [
        SwingPoint("high", i, i + right, idx[i], float(highs[i]))
        for i in _pivots(highs, left, right, is_high=True)
    ] + [
        SwingPoint("low", i, i + right, idx[i], float(lows[i]))
        for i in _pivots(lows, left, right, is_high=False)
    ]
    return sorted(points, key=lambda p: (p.pivot_pos, p.kind))


def swing_points_known_at(points: list[SwingPoint], pos: int) -> list[SwingPoint]:
    """Filter swing points to those confirmed at or before integer position ``pos``."""
    return [p for p in points if p.confirm_pos <= pos]
