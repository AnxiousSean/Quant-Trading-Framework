"""Helpers to assemble the M5/M15 data pair from any source."""

from __future__ import annotations

import logging

import pandas as pd

from trading_framework.config.settings import AppConfig
from trading_framework.data.handler import CCXTDataHandler
from trading_framework.data.synthetic import generate_synthetic_ohlcv
from trading_framework.data.validator import DataValidator
from trading_framework.timeframes import timeframe_to_pandas_rule, timeframe_to_timedelta

logger = logging.getLogger(__name__)


def resample_ohlcv(df: pd.DataFrame, source_timeframe: str, target_timeframe: str) -> pd.DataFrame:
    """Aggregate lower-timeframe candles into complete higher-timeframe candles.

    Higher-timeframe candles with missing constituents (e.g. the still-forming
    last bucket, or exchange gaps) are dropped so no partial candle is used.

    Args:
        df: Source OHLCV, UTC-indexed by open time.
        source_timeframe: e.g. ``"5m"``.
        target_timeframe: e.g. ``"15m"``.

    Returns:
        Resampled OHLCV indexed by the higher-timeframe open time.
    """
    ratio = timeframe_to_timedelta(target_timeframe) / timeframe_to_timedelta(source_timeframe)
    if ratio < 1 or ratio != int(ratio):
        raise ValueError("target timeframe must be an integer multiple of the source timeframe")
    rule = timeframe_to_pandas_rule(target_timeframe)
    agg = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    counts = df["close"].resample(rule, label="left", closed="left").count()
    return agg[counts == int(ratio)].dropna()


def load_csv(path: str) -> pd.DataFrame:
    """Load OHLCV from CSV with a ``timestamp`` column (ISO string or epoch ms)."""
    raw = pd.read_csv(path)
    ts = raw["timestamp"]
    raw["timestamp"] = (pd.to_datetime(ts, unit="ms", utc=True) if pd.api.types.is_numeric_dtype(ts)
                        else pd.to_datetime(ts, utc=True))
    return raw.set_index("timestamp")[["open", "high", "low", "close", "volume"]].astype(float).sort_index()


def load_market_data(cfg: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, clean and validate the (M15, M5) pair according to ``cfg.data``.

    Returns:
        ``(data_m15, data_m5)``.

    Raises:
        InsufficientDataError: If validation fails.
        DataUnavailableError: If the exchange returns nothing.
    """
    d = cfg.data
    if d.source == "synthetic":
        minutes = int(timeframe_to_timedelta(d.entry_timeframe).total_seconds() // 60)
        m5 = generate_synthetic_ohlcv(d.synthetic_bars, minutes, start=d.start or "2024-01-01",
                                      seed=d.synthetic_seed)
        m15 = None
    elif d.source == "csv":
        if not d.csv_m5_path:
            raise ValueError("data.csv_m5_path is required for source=csv")
        m5 = load_csv(d.csv_m5_path)
        m15 = load_csv(d.csv_m15_path) if (d.csv_m15_path and not d.derive_trend_from_entry) else None
    else:
        handler = CCXTDataHandler(cfg.exchange, d)
        start = pd.Timestamp(d.start or "2024-01-01", tz="UTC")
        end = pd.Timestamp(d.end, tz="UTC") if d.end else None
        m5 = handler.load(cfg.exchange.symbol, d.entry_timeframe, start, end)
        m15 = None if d.derive_trend_from_entry else handler.load(cfg.exchange.symbol, d.trend_timeframe, start, end)

    m5 = DataValidator.clean(m5)
    m15 = resample_ohlcv(m5, d.entry_timeframe, d.trend_timeframe) if m15 is None else DataValidator.clean(m15)

    DataValidator(d.entry_timeframe, d.min_history_bars * 3, d.outlier_return_threshold,
                  d.max_gap_ratio).validate_or_raise(m5)
    DataValidator(d.trend_timeframe, d.min_history_bars, d.outlier_return_threshold,
                  d.max_gap_ratio).validate_or_raise(m15)
    logger.info("Data ready: %d M5 / %d M15 candles", len(m5), len(m15))
    return m15, m5
