"""OHLCV data quality checks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trading_framework.timeframes import timeframe_to_timedelta

logger = logging.getLogger(__name__)


class InsufficientDataError(ValueError):
    """Raised when data fails validation in a way that makes it unusable."""


@dataclass
class ValidationReport:
    """Result of :meth:`DataValidator.validate`."""

    rows: int
    duplicates: int = 0
    unsorted: bool = False
    nan_rows: int = 0
    invalid_ohlc: int = 0
    non_positive: int = 0
    missing_bars: int = 0
    gap_ratio: float = 0.0
    largest_gap: pd.Timedelta | None = None
    outliers: int = 0
    sufficient_history: bool = True
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True if no blocking issue was found."""
        return not self.issues


class DataValidator:
    """Checks for gaps, outliers, malformed candles and insufficient history.

    Args:
        timeframe: Expected candle spacing.
        min_bars: Minimum candles required.
        outlier_threshold: Absolute log-return above which a candle is flagged.
        max_gap_ratio: Maximum tolerated fraction of missing candles.
    """

    def __init__(self, timeframe: str, min_bars: int, outlier_threshold: float, max_gap_ratio: float) -> None:
        self.timeframe = timeframe
        self.td = timeframe_to_timedelta(timeframe)
        self.min_bars = min_bars
        self.outlier_threshold = outlier_threshold
        self.max_gap_ratio = max_gap_ratio

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """Inspect ``df`` and report problems (does not modify it)."""
        r = ValidationReport(rows=len(df))
        if df.empty:
            r.sufficient_history = False
            r.issues.append("no data")
            return r
        r.duplicates = int(df.index.duplicated().sum())
        r.unsorted = not df.index.is_monotonic_increasing
        r.nan_rows = int(df[["open", "high", "low", "close"]].isna().any(axis=1).sum())
        r.invalid_ohlc = int(((df["high"] < df[["open", "close"]].max(axis=1))
                              | (df["low"] > df[["open", "close"]].min(axis=1))
                              | (df["high"] < df["low"])).sum())
        r.non_positive = int((df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())

        idx = df.index.unique().sort_values()
        expected = int((idx[-1] - idx[0]) / self.td) + 1
        r.missing_bars = max(0, expected - len(idx))
        r.gap_ratio = r.missing_bars / expected if expected else 0.0
        diffs = idx.to_series().diff().dropna()
        r.largest_gap = diffs.max() if not diffs.empty else None

        log_ret = np.log(df["close"]).diff().abs()
        r.outliers = int((log_ret > self.outlier_threshold).sum())
        r.sufficient_history = len(idx) >= self.min_bars

        if r.nan_rows or r.invalid_ohlc or r.non_positive:
            logger.warning("%s: %d NaN, %d invalid OHLC, %d non-positive rows (cleanable)",
                           self.timeframe, r.nan_rows, r.invalid_ohlc, r.non_positive)
        if r.outliers:
            logger.warning("%s: %d outlier candles (|log return| > %.2f)", self.timeframe, r.outliers,
                           self.outlier_threshold)
        if r.missing_bars:
            logger.warning("%s: %d missing candles (%.2f%%), largest gap %s", self.timeframe, r.missing_bars,
                           100 * r.gap_ratio, r.largest_gap)
        if not r.sufficient_history:
            r.issues.append(f"insufficient history: {len(idx)} < {self.min_bars}")
        if r.gap_ratio > self.max_gap_ratio:
            r.issues.append(f"gap ratio {r.gap_ratio:.2%} exceeds {self.max_gap_ratio:.2%}")
        return r

    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        """Sort, de-duplicate and drop NaN / malformed / non-positive candles.

        Gaps are intentionally *not* filled: synthetic candles would create fake
        wicks and swing points.
        """
        out = df[~df.index.duplicated(keep="last")].sort_index()
        out = out.dropna(subset=["open", "high", "low", "close"])
        bad = ((out["high"] < out[["open", "close"]].max(axis=1))
               | (out["low"] > out[["open", "close"]].min(axis=1))
               | (out[["open", "high", "low", "close"]] <= 0).any(axis=1))
        if bad.any():
            logger.warning("Dropping %d malformed candles", int(bad.sum()))
        out = out[~bad]
        if "volume" in out:
            out = out.assign(volume=out["volume"].fillna(0.0))
        return out

    def validate_or_raise(self, df: pd.DataFrame) -> ValidationReport:
        """Validate and raise :class:`InsufficientDataError` on blocking issues."""
        report = self.validate(df)
        if not report.ok:
            raise InsufficientDataError(f"{self.timeframe} data invalid: {'; '.join(report.issues)}")
        return report
