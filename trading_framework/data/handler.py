"""ccxt OHLCV ingestion with pagination, retries and a SQLite cache."""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import pandas as pd

from trading_framework.config.settings import DataConfig, ExchangeConfig
from trading_framework.timeframes import timeframe_to_timedelta

logger = logging.getLogger(__name__)
T = TypeVar("T")

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataUnavailableError(RuntimeError):
    """Raised when data cannot be fetched after all retries."""


def ohlcv_to_frame(rows: list[list[float]]) -> pd.DataFrame:
    """Convert ccxt ``[[ts_ms, o, h, l, c, v], ...]`` to a UTC-indexed DataFrame."""
    if not rows:
        return pd.DataFrame(columns=OHLCV_COLUMNS, index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))
    df = pd.DataFrame(rows, columns=["timestamp", *OHLCV_COLUMNS])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").astype(float)
    return df[~df.index.duplicated(keep="last")].sort_index()


class CCXTDataHandler:
    """Fetch historical and recent candles from a ccxt exchange.

    Candles are cached in ``<cache_dir>/ohlcv.sqlite`` so repeated backtests do
    not hit the exchange. Only *closed* candles are ever returned or cached.

    Args:
        exchange_cfg: Exchange settings.
        data_cfg: Data settings.
        exchange: Optional ccxt exchange instance (dependency injection for tests).
    """

    def __init__(self, exchange_cfg: ExchangeConfig, data_cfg: DataConfig, exchange: Any | None = None) -> None:
        self.ex_cfg = exchange_cfg
        self.cfg = data_cfg
        self._exchange = exchange
        self.cache_path = Path(data_cfg.cache_dir) / "ohlcv.sqlite"

    # ---------------------------------------------------------------- infra
    @property
    def exchange(self) -> Any:
        """Lazily constructed public (unauthenticated) ccxt client."""
        if self._exchange is None:
            import ccxt

            cls = getattr(ccxt, self.ex_cfg.id)
            self._exchange = cls({
                "enableRateLimit": self.ex_cfg.enable_rate_limit,
                "timeout": self.ex_cfg.request_timeout_ms,
                "options": {"defaultType": self.ex_cfg.market_type},
            })
            if self.ex_cfg.sandbox:
                self._exchange.set_sandbox_mode(True)
        return self._exchange

    def _retry(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Call ``fn`` with exponential backoff on transient ccxt errors."""
        try:
            import ccxt

            transient: tuple[type[BaseException], ...] = (ccxt.NetworkError, ccxt.ExchangeNotAvailable,
                                                         ccxt.RequestTimeout, ccxt.DDoSProtection)
        except ImportError:  # pragma: no cover - ccxt is a hard dependency for this class
            transient = (ConnectionError, TimeoutError)
        delay = self.ex_cfg.retry_backoff_seconds
        for attempt in range(self.ex_cfg.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except transient as exc:
                if attempt >= self.ex_cfg.max_retries:
                    raise DataUnavailableError(f"{fn.__name__} failed after {attempt + 1} attempts: {exc}") from exc
                logger.warning("Transient error (%s); retry %d in %.1fs", exc, attempt + 1, delay)
                time.sleep(delay)
                delay *= 2
        raise DataUnavailableError("unreachable")  # pragma: no cover

    # ---------------------------------------------------------------- fetch
    def fetch_ohlcv(self, symbol: str, timeframe: str, start: pd.Timestamp,
                    end: pd.Timestamp | None = None) -> pd.DataFrame:
        """Fetch ``[start, end)`` candles with forward pagination.

        Args:
            symbol: ccxt symbol.
            timeframe: ccxt timeframe.
            start: Inclusive UTC start.
            end: Exclusive UTC end (defaults to now).

        Returns:
            Closed candles only.
        """
        tf = timeframe_to_timedelta(timeframe)
        now = pd.Timestamp.now(tz="UTC")
        end = min(end or now, now)
        since = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        rows: list[list[float]] = []
        while since < end_ms:
            batch = self._retry(self.exchange.fetch_ohlcv, symbol, timeframe, since, self.cfg.fetch_limit)
            if not batch:
                break
            rows.extend(batch)
            last_ts = int(batch[-1][0])
            if last_ts < since:  # defensive: exchange returned older data
                break
            since = last_ts + int(tf.total_seconds() * 1000)
            logger.debug("Fetched %d %s candles up to %s", len(batch), timeframe,
                         pd.Timestamp(last_ts, unit="ms", tz="UTC"))
            if len(batch) < 2:
                break
        df = ohlcv_to_frame(rows)
        df = df[(df.index >= start) & (df.index < end)]
        return df[df.index + tf <= now]  # drop the still-forming candle

    def fetch_recent(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Fetch the most recent ``limit`` closed candles (for paper/live loops)."""
        tf = timeframe_to_timedelta(timeframe)
        rows = self._retry(self.exchange.fetch_ohlcv, symbol, timeframe, None, limit + 1)
        df = ohlcv_to_frame(rows)
        return df[df.index + tf <= pd.Timestamp.now(tz="UTC")].tail(limit)

    # ---------------------------------------------------------------- cache
    def _connect(self) -> sqlite3.Connection:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.cache_path)
        con.execute(
            "CREATE TABLE IF NOT EXISTS ohlcv (exchange TEXT, symbol TEXT, timeframe TEXT, ts INTEGER,"
            " open REAL, high REAL, low REAL, close REAL, volume REAL,"
            " PRIMARY KEY (exchange, symbol, timeframe, ts))"
        )
        return con

    def _read_cache(self, symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        with self._connect() as con:
            rows = con.execute(
                "SELECT ts, open, high, low, close, volume FROM ohlcv WHERE exchange=? AND symbol=?"
                " AND timeframe=? AND ts>=? AND ts<? ORDER BY ts",
                (self.ex_cfg.id, symbol, timeframe, int(start.timestamp() * 1000), int(end.timestamp() * 1000)),
            ).fetchall()
        return ohlcv_to_frame([list(r) for r in rows])

    def _write_cache(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        records = [
            (self.ex_cfg.id, symbol, timeframe, int(ts.timestamp() * 1000), *map(float, row))
            for ts, row in zip(df.index, df[OHLCV_COLUMNS].to_numpy())
        ]
        with self._connect() as con:
            con.executemany("INSERT OR REPLACE INTO ohlcv VALUES (?,?,?,?,?,?,?,?,?)", records)

    def load(self, symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp | None = None,
             use_cache: bool = True) -> pd.DataFrame:
        """Load candles for ``[start, end)``, fetching only what the cache lacks.

        Raises:
            DataUnavailableError: If nothing could be loaded.
        """
        tf = timeframe_to_timedelta(timeframe)
        end = min(end or pd.Timestamp.now(tz="UTC"), pd.Timestamp.now(tz="UTC"))
        cached = self._read_cache(symbol, timeframe, start, end) if use_cache else ohlcv_to_frame([])
        parts = [cached]
        if cached.empty:
            parts.append(self.fetch_ohlcv(symbol, timeframe, start, end))
        else:
            if cached.index[0] - start >= tf:
                parts.append(self.fetch_ohlcv(symbol, timeframe, start, cached.index[0]))
            if end - (cached.index[-1] + tf) >= tf:
                parts.append(self.fetch_ohlcv(symbol, timeframe, cached.index[-1] + tf, end))
        fetched = [p for p in parts[1:] if not p.empty]
        df = pd.concat([p for p in parts if not p.empty]) if any(not p.empty for p in parts) else cached
        df = df[~df.index.duplicated(keep="last")].sort_index()
        if use_cache:
            for p in fetched:
                self._write_cache(symbol, timeframe, p)
        if df.empty:
            raise DataUnavailableError(f"No {timeframe} data for {symbol} between {start} and {end}")
        logger.info("Loaded %d %s candles for %s (%s -> %s)", len(df), timeframe, symbol,
                    df.index[0], df.index[-1])
        return df
