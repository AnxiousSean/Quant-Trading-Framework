"""Performance statistics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak (0 at highs, negative below)."""
    peak = equity.cummax()
    return (equity / peak - 1.0).rename("drawdown")


def compute_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    initial_equity: float,
    periods_per_year: int = 365,
    risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    """Compute summary statistics.

    Sharpe and Sortino use **daily** returns of the equity curve (resampled at
    UTC day ends) annualised with ``periods_per_year`` (365 for 24/7 crypto).

    Args:
        equity: Equity curve indexed by time.
        trades: Closed trades (``net_pnl``, ``r_multiple``, ``holding_minutes`` ...).
        initial_equity: Starting balance.
        periods_per_year: Annualisation factor for daily returns.
        risk_free_rate: Annual risk-free rate.

    Returns:
        Ordered dictionary of metrics.
    """
    m: dict[str, Any] = {}
    final = float(equity.iloc[-1]) if not equity.empty else initial_equity
    m["initial_equity"] = initial_equity
    m["final_equity"] = final
    m["total_return_pct"] = 100 * (final / initial_equity - 1)
    m["max_drawdown_pct"] = 100 * float(drawdown_series(equity).min()) if not equity.empty else 0.0

    if len(equity) > 1:
        days = (equity.index[-1] - equity.index[0]).total_seconds() / 86_400
        m["period_days"] = round(days, 2)
        m["cagr_pct"] = 100 * ((final / initial_equity) ** (365 / days) - 1) if days > 0 and final > 0 else float("nan")
        daily = equity.resample("1D").last().dropna()
        rets = daily.pct_change().dropna()
        excess = rets - risk_free_rate / periods_per_year
        std = excess.std(ddof=1)
        m["sharpe"] = float(excess.mean() / std * math.sqrt(periods_per_year)) if std and std > 0 else float("nan")
        downside = excess[excess < 0].std(ddof=1)
        m["sortino"] = (float(excess.mean() / downside * math.sqrt(periods_per_year))
                        if downside and downside > 0 else float("nan"))
        m["calmar"] = (m["cagr_pct"] / abs(m["max_drawdown_pct"])
                       if m["max_drawdown_pct"] < 0 and not math.isnan(m["cagr_pct"]) else float("nan"))
    else:
        m.update(period_days=0.0, cagr_pct=float("nan"), sharpe=float("nan"), sortino=float("nan"),
                 calmar=float("nan"))

    n = len(trades)
    m["trades"] = n
    if n:
        pnl = trades["net_pnl"]
        wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
        m["win_rate_pct"] = 100 * len(wins) / n
        gross_loss = -losses.sum()
        m["profit_factor"] = float(wins.sum() / gross_loss) if gross_loss > 0 else float("inf")
        m["avg_win"] = float(wins.mean()) if len(wins) else 0.0
        m["avg_loss"] = float(losses.mean()) if len(losses) else 0.0
        m["expectancy"] = float(pnl.mean())
        m["avg_r_multiple"] = float(trades["r_multiple"].mean())
        m["total_fees"] = float(trades["fees"].sum())
        m["avg_holding_minutes"] = float(trades["holding_minutes"].mean())
        m["longs"] = int((trades["direction"] == "LONG").sum())
        m["shorts"] = int((trades["direction"] == "SHORT").sum())
        streak = best = 0
        for v in pnl:
            streak = streak + 1 if v <= 0 else 0
            best = max(best, streak)
        m["max_consecutive_losses"] = best
    else:
        m.update(win_rate_pct=0.0, profit_factor=float("nan"), avg_win=0.0, avg_loss=0.0, expectancy=0.0,
                 avg_r_multiple=0.0, total_fees=0.0, avg_holding_minutes=0.0, longs=0, shorts=0,
                 max_consecutive_losses=0)
    return {k: (round(v, 4) if isinstance(v, float) and np.isfinite(v) else v) for k, v in m.items()}
