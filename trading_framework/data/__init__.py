"""Data ingestion, caching, validation and synthetic generation."""

from trading_framework.data.handler import CCXTDataHandler, DataUnavailableError
from trading_framework.data.loaders import load_csv, load_market_data, resample_ohlcv
from trading_framework.data.synthetic import generate_synthetic_ohlcv
from trading_framework.data.validator import DataValidator, InsufficientDataError, ValidationReport

__all__ = [
    "CCXTDataHandler",
    "DataUnavailableError",
    "DataValidator",
    "InsufficientDataError",
    "ValidationReport",
    "generate_synthetic_ohlcv",
    "load_csv",
    "load_market_data",
    "resample_ohlcv",
]
