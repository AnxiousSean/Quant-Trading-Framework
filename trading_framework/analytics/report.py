"""CSV / JSON / HTML performance report writer (no plotting dependencies)."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from trading_framework.analytics.metrics import drawdown_series


def _svg_line(series: pd.Series, width: int = 900, height: int = 220, color: str = "#2563eb") -> str:
    s = series.dropna()
    if len(s) < 2:
        return "<p>Not enough data to plot.</p>"
    if len(s) > 2000:
        s = s.iloc[:: math.ceil(len(s) / 2000)]
    lo, hi = float(s.min()), float(s.max())
    span = hi - lo or 1.0
    xs = [i * width / (len(s) - 1) for i in range(len(s))]
    ys = [height - (v - lo) / span * height for v in s]
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" preserveAspectRatio="none" '
            f'style="background:#f8fafc;border:1px solid #e2e8f0">'
            f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{pts}"/></svg>'
            f'<div style="font:12px monospace;color:#64748b">min {lo:,.2f} · max {hi:,.2f} · '
            f'{s.index[0]} → {s.index[-1]}</div>')


def write_report(output_dir: str, metrics: dict[str, Any], equity: pd.Series, trades: pd.DataFrame,
                 title: str = "Backtest report") -> dict[str, Path]:
    """Write ``metrics.json``, ``metrics.csv``, ``equity.csv``, ``trades.csv`` and ``report.html``.

    Returns:
        Mapping of artefact name to path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics_json": out / "metrics.json",
        "metrics_csv": out / "metrics.csv",
        "equity_csv": out / "equity.csv",
        "trades_csv": out / "trades.csv",
        "html": out / "report.html",
    }
    paths["metrics_json"].write_text(json.dumps(metrics, indent=2, default=str))
    pd.Series(metrics, name="value").to_csv(paths["metrics_csv"], header=True)
    pd.DataFrame({"equity": equity, "drawdown": drawdown_series(equity)}).to_csv(paths["equity_csv"])
    trades.to_csv(paths["trades_csv"], index=False)

    rows = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
                   for k, v in metrics.items())
    trade_html = (trades.tail(200).to_html(index=False, float_format=lambda x: f"{x:,.4f}", border=0)
                  if not trades.empty else "<p>No trades.</p>")
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#0f172a}}
table{{border-collapse:collapse;font-size:13px}}th,td{{padding:4px 10px;border-bottom:1px solid #e2e8f0;text-align:right}}
th{{text-align:left}}.note{{color:#b45309}}</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="note">Educational use only. Not financial advice. Past/simulated performance does not predict future results.</p>
<h2>Summary</h2><table>{rows}</table>
<h2>Equity</h2>{_svg_line(equity)}
<h2>Drawdown</h2>{_svg_line(drawdown_series(equity), color="#dc2626")}
<h2>Trades (last 200)</h2>{trade_html}
</body></html>"""
    paths["html"].write_text(doc)
    return paths
