"""Timeframe helpers shared across modules."""

from __future__ import annotations

import pandas as pd

_UNIT_TO_PANDAS = {"m": "min", "h": "h", "d": "D", "w": "W"}


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    """Convert a ccxt-style timeframe string (e.g. ``"5m"``) to a Timedelta.

    Args:
        timeframe: Timeframe such as ``"1m"``, ``"5m"``, ``"15m"``, ``"1h"``, ``"1d"``.

    Returns:
        The duration of one candle.

    Raises:
        ValueError: If the timeframe string cannot be parsed.
    """
    if len(timeframe) < 2 or timeframe[-1] not in _UNIT_TO_PANDAS:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}")
    try:
        value = int(timeframe[:-1])
    except ValueError as exc:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}") from exc
    return pd.Timedelta(value, unit=_UNIT_TO_PANDAS[timeframe[-1]])


def timeframe_to_pandas_rule(timeframe: str) -> str:
    """Convert a ccxt timeframe to a pandas resample rule (``"15m"`` -> ``"15min"``)."""
    timeframe_to_timedelta(timeframe)  # validate
    return f"{int(timeframe[:-1])}{_UNIT_TO_PANDAS[timeframe[-1]]}"
