"""Pydantic configuration models, YAML loading and environment overrides.

Every tunable number in the framework lives here so that nothing is hard-coded
in the strategy, risk or execution layers.

Environment overrides use the ``TSF__`` prefix with ``__`` as a nesting
separator, e.g.::

    TSF__RISK__RISK_PER_TRADE=0.005
    TSF__STRATEGY__ATR_MULTIPLIER=2.5

API secrets are never stored in YAML: the YAML only names the environment
variables that hold them (``exchange.api_key_env`` / ``exchange.api_secret_env``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ENV_PREFIX = "TSF__"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ExchangeConfig(_Base):
    """Exchange connection settings."""

    id: str = Field("binanceusdm", description="ccxt exchange id")
    symbol: str = "BTC/USDT:USDT"
    market_type: Literal["spot", "future", "swap"] = "future"
    api_key_env: str = "EXCHANGE_API_KEY"
    api_secret_env: str = "EXCHANGE_API_SECRET"
    sandbox: bool = False
    enable_rate_limit: bool = True
    request_timeout_ms: int = Field(30_000, gt=0)
    max_retries: int = Field(5, ge=0)
    retry_backoff_seconds: float = Field(2.0, gt=0)
    # Fallback market limits used when exchange metadata is unavailable (backtests offline).
    amount_step: float = Field(0.001, gt=0, description="Order size precision step")
    price_tick: float = Field(0.1, gt=0, description="Price precision tick")
    min_amount: float = Field(0.001, ge=0)
    min_notional: float = Field(5.0, ge=0)

    def api_credentials(self) -> tuple[str | None, str | None]:
        """Read API key/secret from the configured environment variables."""
        return os.getenv(self.api_key_env), os.getenv(self.api_secret_env)


class DataConfig(_Base):
    """Data ingestion and validation settings."""

    trend_timeframe: str = "15m"
    entry_timeframe: str = "5m"
    source: Literal["ccxt", "csv", "synthetic"] = "ccxt"
    start: str | None = Field("2024-01-01", description="ISO date for backtest start (UTC)")
    end: str | None = Field(None, description="ISO date for backtest end (UTC); None = now")
    cache_dir: str = "cache"
    csv_m5_path: str | None = None
    csv_m15_path: str | None = None
    derive_trend_from_entry: bool = Field(
        True, description="Resample M15 from M5 so both timeframes are perfectly consistent"
    )
    fetch_limit: int = Field(1000, gt=0, le=1500, description="Candles per ccxt request")
    min_history_bars: int = Field(200, gt=0)
    outlier_return_threshold: float = Field(0.15, gt=0, description="Abs log-return flagged as outlier")
    max_gap_ratio: float = Field(0.02, ge=0, le=1, description="Max fraction of missing candles tolerated")
    synthetic_bars: int = Field(20_000, gt=0)
    synthetic_seed: int = 42


class StrategyConfig(_Base):
    """Two-Step Trader's Landing parameters."""

    atr_period: int = Field(14, gt=1)
    atr_multiplier: float = Field(2.0, gt=0, description="ATR band width multiplier")
    band_source: Literal["close", "hl2", "hlc3"] = "close"
    trend_lookback: int = Field(20, gt=4, description="M15 candles scanned for HH/HL structure")
    swing_left: int = Field(2, ge=1, description="Bars left of a fractal pivot")
    swing_right: int = Field(2, ge=1, description="Bars right of a fractal pivot (confirmation lag)")
    min_swings: int = Field(2, ge=2, description="Swing highs AND lows required to define a trend")
    require_structure_intact: bool = Field(
        True, description="Uptrend void if last M15 close < last swing low (mirror for downtrend)"
    )
    zone_sources: list[Literal["swing", "flip", "atr_band"]] = Field(
        default_factory=lambda: ["swing", "atr_band"],
        description="swing=last HL/LH, flip=broken prior swing high/low, atr_band=lower/upper band",
    )
    zone_tolerance_atr: float = Field(
        0.1, ge=0, description="Zone half-width in ATRs; wick must reach level + tolerance"
    )
    min_wick_to_range: float = Field(0.5, gt=0, le=1, description="Rejection wick / candle range")
    min_wick_to_body: float = Field(1.5, ge=0, description="Rejection wick / candle body")
    max_signal_bar_range_atr: float = Field(
        3.0, gt=0, description="Skip if rejection/confirmation candle range exceeds N x ATR (spike filter)"
    )
    min_volume: float = Field(0.0, ge=0, description="Minimum M5 volume on signal candles (illiquidity filter)")
    stop_atr_multiplier: float = Field(2.0, gt=0, description="Stop distance beyond pullback extreme in ATRs")
    reward_risk: float = Field(2.0, gt=0)
    pullback_max_m5_bars: int = Field(288, gt=2, description="Max M5 bars scanned for the pullback extreme")

    @field_validator("zone_sources")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("zone_sources must contain at least one source")
        return v

    @model_validator(mode="after")
    def _lookback_vs_swings(self) -> "StrategyConfig":
        if self.trend_lookback <= self.swing_left + self.swing_right:
            raise ValueError("trend_lookback must exceed swing_left + swing_right")
        return self

    @property
    def required_trend_bars(self) -> int:
        """M15 history needed for warmed-up indicators plus the structure window."""
        return max(self.atr_period * 5, self.trend_lookback + self.swing_left + self.swing_right) + 5


class RiskConfig(_Base):
    """Position sizing and portfolio guards."""

    risk_per_trade: float = Field(0.01, gt=0, le=0.1, description="Fraction of equity risked per trade")
    max_positions_per_symbol: int = Field(1, ge=1)
    max_total_positions: int = Field(1, ge=1)
    max_leverage: float = Field(3.0, gt=0, description="Max notional / equity for a single position")
    daily_loss_limit: float | None = Field(0.03, gt=0, le=1, description="None disables the breaker")
    allow_shorts: bool = True

    @model_validator(mode="after")
    def _positions(self) -> "RiskConfig":
        if self.max_total_positions < self.max_positions_per_symbol:
            raise ValueError("max_total_positions must be >= max_positions_per_symbol")
        return self


class ExecutionConfig(_Base):
    """Fill simulation and live safeguards."""

    taker_fee: float = Field(0.0004, ge=0)
    slippage: float = Field(0.0005, ge=0, description="Fraction of price applied adversely to market/stop fills")
    assumed_spread: float = Field(0.0001, ge=0, description="Spread assumed by the paper broker")
    max_spread: float = Field(0.001, gt=0, description="Skip entries if spread exceeds this fraction")
    max_slippage: float = Field(0.002, gt=0, description="Abort live entries if fill deviates more than this")
    dry_run: bool = Field(True, description="LiveBroker logs orders instead of sending them")
    poll_interval_seconds: float = Field(10.0, gt=0)


class BacktestConfig(_Base):
    """Backtest settings."""

    initial_equity: float = Field(10_000.0, gt=0)
    output_dir: str = "reports"
    trade_db: str = "reports/trades.sqlite"
    periods_per_year: int = Field(365, gt=0, description="Daily periods per year for Sharpe (crypto = 365)")
    risk_free_rate: float = 0.0


class LoggingConfig(_Base):
    """Logging settings."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file: str | None = "logs/framework.log"


class AppConfig(_Base):
    """Root configuration object."""

    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _coerce(value: str) -> Any:
    """Parse an env-var string as YAML so numbers/bools/lists get proper types."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def apply_env_overrides(raw: dict[str, Any], environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Overlay ``TSF__SECTION__FIELD`` environment variables onto a raw config dict.

    Args:
        raw: Configuration dictionary loaded from YAML.
        environ: Environment mapping (defaults to ``os.environ``).

    Returns:
        The updated dictionary (mutated in place and returned).
    """
    env = os.environ if environ is None else environ
    for key, value in env.items():
        if not key.startswith(ENV_PREFIX):
            continue
        path = [p.lower() for p in key[len(ENV_PREFIX):].split("__") if p]
        if not path:
            continue
        node = raw
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = _coerce(value)
    return raw


def load_config(path: str | Path | None = None, environ: dict[str, str] | None = None) -> AppConfig:
    """Load and validate configuration.

    Args:
        path: YAML file path. ``None`` uses defaults only.
        environ: Optional environment mapping for overrides (testing hook).

    Returns:
        Validated :class:`AppConfig`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        pydantic.ValidationError: If any value is invalid.
    """
    raw: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        raw = yaml.safe_load(p.read_text()) or {}
    apply_env_overrides(raw, environ)
    return AppConfig.model_validate(raw)
