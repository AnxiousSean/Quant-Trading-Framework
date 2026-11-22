"""The "Two-Step Trader's Landing" strategy.

Step 1 (M15): identify trend structure (HH/HL or LH/LL) and the zone a pullback
should land on (last swing low / high, a flipped prior swing, or the ATR band).

Step 2 (M5): wait for a rejection candle whose wick pierces the zone and whose
body closes back on the trend side, then a confirmation candle that closes
beyond the rejection candle's extreme. Enter at the next M5 open.

Look-ahead protection:
    * Every M5 row passed in is treated as a closed candle; the last row is the
      confirmation candle and the one before it the rejection candle.
    * M15 rows are filtered to those whose *close time* is at or before the
      rejection candle's close, regardless of what the caller passes.
    * Swing pivots are only used once their right-hand confirmation bars exist.

This class performs no I/O and holds no mutable trading state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trading_framework.config.settings import StrategyConfig
from trading_framework.indicators.atr import atr_bands
from trading_framework.indicators.swings import SwingPoint, find_swing_points
from trading_framework.strategy.signals import Direction, SignalEvent, Trend
from trading_framework.timeframes import timeframe_to_timedelta

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class MarketStructure:
    """Snapshot of M15 structure as of the last closed M15 candle.

    Attributes:
        trend: UP, DOWN or NONE.
        swing_highs: Confirmed swing highs inside the lookback window, oldest first.
        swing_lows: Confirmed swing lows inside the lookback window, oldest first.
        atr: Last ATR value.
        upper_band: Last upper ATR band.
        lower_band: Last lower ATR band.
        last_close: Last M15 close.
        as_of: Open time of the last M15 candle used.
        reason: Why the trend was classified as it was.
    """

    trend: Trend
    swing_highs: list[SwingPoint] = field(default_factory=list)
    swing_lows: list[SwingPoint] = field(default_factory=list)
    atr: float = float("nan")
    upper_band: float = float("nan")
    lower_band: float = float("nan")
    last_close: float = float("nan")
    as_of: pd.Timestamp | None = None
    reason: str = ""


@dataclass(frozen=True)
class _Candle:
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)


class TwoStepStrategy:
    """Signal generator for the Two-Step Trader's Landing setup.

    Args:
        config: Strategy parameters.
        trend_timeframe: Structure timeframe (default ``"15m"``).
        entry_timeframe: Trigger timeframe (default ``"5m"``).
        symbol: Symbol stamped on emitted signals.
    """

    def __init__(
        self,
        config: StrategyConfig,
        trend_timeframe: str = "15m",
        entry_timeframe: str = "5m",
        symbol: str = "",
    ) -> None:
        self.cfg = config
        self.symbol = symbol
        self.trend_td = timeframe_to_timedelta(trend_timeframe)
        self.entry_td = timeframe_to_timedelta(entry_timeframe)
        if self.entry_td >= self.trend_td:
            raise ValueError("entry timeframe must be shorter than trend timeframe")

    # ------------------------------------------------------------------ sizes
    @property
    def trend_window(self) -> int:
        """Number of trailing M15 candles the strategy needs."""
        return self.cfg.required_trend_bars

    @property
    def entry_window(self) -> int:
        """Number of trailing M5 candles the strategy needs."""
        return self.cfg.pullback_max_m5_bars + 2

    # -------------------------------------------------------------- structure
    def analyze_structure(self, data_m15: pd.DataFrame) -> MarketStructure:
        """Classify M15 structure using only the candles provided.

        Args:
            data_m15: Closed M15 candles, ascending, indexed by open time.

        Returns:
            :class:`MarketStructure` for the last row.
        """
        n = len(data_m15)
        if n < self.cfg.required_trend_bars:
            return MarketStructure(Trend.NONE, reason=f"insufficient M15 history ({n})")

        bands = atr_bands(data_m15, self.cfg.atr_period, self.cfg.atr_multiplier, self.cfg.band_source)
        last = bands.iloc[-1]
        last_close = float(data_m15["close"].iloc[-1])
        as_of = data_m15.index[-1]

        window_start = n - self.cfg.trend_lookback
        swings = [p for p in find_swing_points(data_m15, self.cfg.swing_left, self.cfg.swing_right)
                  if p.pivot_pos >= window_start]
        highs = [p for p in swings if p.kind == "high"]
        lows = [p for p in swings if p.kind == "low"]
        base = dict(
            swing_highs=highs, swing_lows=lows, atr=float(last["atr"]),
            upper_band=float(last["upper"]), lower_band=float(last["lower"]),
            last_close=last_close, as_of=as_of,
        )

        if np.isnan(base["atr"]):
            return MarketStructure(Trend.NONE, reason="ATR not warmed up", **base)
        k = self.cfg.min_swings
        if len(highs) < k or len(lows) < k:
            return MarketStructure(Trend.NONE, reason=f"not enough swings ({len(highs)}H/{len(lows)}L)", **base)

        h = [p.price for p in highs[-k:]]
        lo = [p.price for p in lows[-k:]]
        rising = all(b > a for a, b in zip(h, h[1:])) and all(b > a for a, b in zip(lo, lo[1:]))
        falling = all(b < a for a, b in zip(h, h[1:])) and all(b < a for a, b in zip(lo, lo[1:]))

        if rising:
            if self.cfg.require_structure_intact and last_close < lows[-1].price:
                return MarketStructure(Trend.NONE, reason="uptrend broken: close below last HL", **base)
            return MarketStructure(Trend.UP, reason="HH/HL", **base)
        if falling:
            if self.cfg.require_structure_intact and last_close > highs[-1].price:
                return MarketStructure(Trend.NONE, reason="downtrend broken: close above last LH", **base)
            return MarketStructure(Trend.DOWN, reason="LH/LL", **base)
        return MarketStructure(Trend.NONE, reason="ambiguous structure (range)", **base)

    # ---------------------------------------------------------------- candles
    def _is_bullish_rejection(self, c: _Candle) -> bool:
        if c.range <= 0:
            return False
        return (c.lower_wick / c.range >= self.cfg.min_wick_to_range
                and c.lower_wick >= self.cfg.min_wick_to_body * c.body)

    def _is_bearish_rejection(self, c: _Candle) -> bool:
        if c.range <= 0:
            return False
        return (c.upper_wick / c.range >= self.cfg.min_wick_to_range
                and c.upper_wick >= self.cfg.min_wick_to_body * c.body)

    @staticmethod
    def _candle(row: pd.Series) -> _Candle:
        return _Candle(float(row["open"]), float(row["high"]), float(row["low"]),
                       float(row["close"]), float(row["volume"]))

    # ----------------------------------------------------------------- zones
    def _support_levels(self, s: MarketStructure) -> list[tuple[str, float]]:
        levels: list[tuple[str, float]] = []
        for src in self.cfg.zone_sources:
            if src == "swing" and s.swing_lows:
                levels.append(("swing_low", s.swing_lows[-1].price))
            elif src == "flip" and len(s.swing_highs) >= 2:
                levels.append(("flipped_swing_high", s.swing_highs[-2].price))
            elif src == "atr_band":
                levels.append(("lower_atr_band", s.lower_band))
        return levels

    def _resistance_levels(self, s: MarketStructure) -> list[tuple[str, float]]:
        levels: list[tuple[str, float]] = []
        for src in self.cfg.zone_sources:
            if src == "swing" and s.swing_highs:
                levels.append(("swing_high", s.swing_highs[-1].price))
            elif src == "flip" and len(s.swing_lows) >= 2:
                levels.append(("flipped_swing_low", s.swing_lows[-2].price))
            elif src == "atr_band":
                levels.append(("upper_atr_band", s.upper_band))
        return levels

    # ---------------------------------------------------------------- signals
    def generate_signals(
        self, data_m15: pd.DataFrame, data_m5: pd.DataFrame, symbol: str | None = None
    ) -> SignalEvent:
        """Evaluate the setup on the latest closed M5 candle.

        Args:
            data_m15: M15 OHLCV (index = UTC open time). Rows not yet closed at the
                rejection candle's close are ignored automatically.
            data_m5: Closed M5 OHLCV, ascending. ``[-2]`` is the rejection candidate,
                ``[-1]`` the confirmation candidate.
            symbol: Optional override of the configured symbol.

        Returns:
            A :class:`SignalEvent`; ``direction`` is ``NONE`` when no trade.
        """
        sym = symbol or self.symbol
        if len(data_m5) < 3:
            return SignalEvent(sym, pd.Timestamp(0, tz="UTC"), Direction.NONE, "insufficient M5 history")
        for col in REQUIRED_COLUMNS:
            if col not in data_m5.columns or col not in data_m15.columns:
                raise KeyError(f"missing column {col!r}")

        rej_time = data_m5.index[-2]
        conf_time = data_m5.index[-1]
        signal_time = conf_time + self.entry_td
        rej = self._candle(data_m5.iloc[-2])
        conf = self._candle(data_m5.iloc[-1])

        def none(reason: str, trend: Trend = Trend.NONE) -> SignalEvent:
            return SignalEvent(sym, signal_time, Direction.NONE, reason, trend=trend, rejection_time=rej_time)

        # Cheap M5 pattern pre-check before any M15 work.
        long_pattern = (self._is_bullish_rejection(rej) and conf.close > conf.open and conf.close > rej.high)
        short_pattern = (self._is_bearish_rejection(rej) and conf.close < conf.open and conf.close < rej.low)
        if not (long_pattern or short_pattern):
            return none("no rejection/confirmation pattern")

        if min(rej.volume, conf.volume) < self.cfg.min_volume:
            return none("volume below minimum (illiquid)")

        # M15 context strictly as of the rejection candle's close.
        rej_close = rej_time + self.entry_td
        m15 = data_m15[data_m15.index + self.trend_td <= rej_close]
        structure = self.analyze_structure(m15)
        if structure.trend is Trend.NONE:
            return none(f"no trend: {structure.reason}")

        atr_v = structure.atr
        max_range = self.cfg.max_signal_bar_range_atr * atr_v
        if rej.range > max_range or conf.range > max_range:
            return none("signal candle range too large vs ATR (spike filter)", structure.trend)

        tol = self.cfg.zone_tolerance_atr * atr_v
        if structure.trend is Trend.UP:
            if not long_pattern:
                return none("pattern opposes uptrend", structure.trend)
            hit = next(((n, lvl) for n, lvl in self._support_levels(structure)
                        if rej.low <= lvl + tol and rej.close > lvl), None)
            if hit is None:
                return none("rejection wick did not retest support", structure.trend)
            leg_start = structure.swing_highs[-1].time if structure.swing_highs else rej_time
            leg = data_m5[data_m5.index >= leg_start].tail(self.cfg.pullback_max_m5_bars)
            extreme = float(leg["low"].min())
            stop = extreme - self.cfg.stop_atr_multiplier * atr_v
            direction = Direction.LONG
            if stop >= conf.close:
                return none("invalid stop above price", structure.trend)
        else:
            if not short_pattern:
                return none("pattern opposes downtrend", structure.trend)
            hit = next(((n, lvl) for n, lvl in self._resistance_levels(structure)
                        if rej.high >= lvl - tol and rej.close < lvl), None)
            if hit is None:
                return none("rejection wick did not retest resistance", structure.trend)
            leg_start = structure.swing_lows[-1].time if structure.swing_lows else rej_time
            leg = data_m5[data_m5.index >= leg_start].tail(self.cfg.pullback_max_m5_bars)
            extreme = float(leg["high"].max())
            stop = extreme + self.cfg.stop_atr_multiplier * atr_v
            direction = Direction.SHORT
            if stop <= conf.close:
                return none("invalid stop below price", structure.trend)

        zone_name, zone_level = hit
        return SignalEvent(
            symbol=sym,
            timestamp=signal_time,
            direction=direction,
            reason=f"{structure.trend.value}trend retest of {zone_name} @ {zone_level:.2f}",
            stop_price=stop,
            reference_price=conf.close,
            atr=atr_v,
            trend=structure.trend,
            zone_source=zone_name,
            zone_level=zone_level,
            pullback_extreme=extreme,
            rejection_time=rej_time,
            metadata={"m15_as_of": structure.as_of, "structure": structure.reason},
        )
