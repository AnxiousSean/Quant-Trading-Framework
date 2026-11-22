"""Persistent trade journal (SQLite, optionally mirrored to CSV)."""

from __future__ import annotations

import csv
import logging
import sqlite3
from pathlib import Path

from trading_framework.portfolio.portfolio import ClosedTrade

logger = logging.getLogger(__name__)

_COLUMNS = [
    "id", "symbol", "direction", "amount", "entry_time", "entry_price", "exit_time", "exit_price",
    "stop_price", "take_profit", "exit_reason", "entry_reason", "gross_pnl", "fees", "net_pnl",
    "risk_amount", "r_multiple", "holding_minutes",
]


class TradeLogger:
    """Append every closed trade to SQLite and (optionally) CSV.

    Args:
        db_path: SQLite file path.
        csv_path: Optional CSV mirror.
        run_id: Identifier stored with each row to separate runs.
    """

    def __init__(self, db_path: str, csv_path: str | None = None, run_id: str = "default") -> None:
        self.db_path = Path(db_path)
        self.csv_path = Path(csv_path) if csv_path else None
        self.run_id = run_id
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        cols = ", ".join(f"{c} TEXT" if c in {"symbol", "direction", "entry_time", "exit_time", "exit_reason",
                                               "entry_reason"} else f"{c} REAL" for c in _COLUMNS)
        with sqlite3.connect(self.db_path) as con:
            con.execute(f"CREATE TABLE IF NOT EXISTS trades (run_id TEXT, {cols})")
        if self.csv_path:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with self.csv_path.open("w", newline="") as fh:
                csv.writer(fh).writerow(["run_id", *_COLUMNS])

    def log(self, trade: ClosedTrade) -> None:
        """Persist one trade."""
        d = trade.to_dict()
        row = [self.run_id] + [str(d[c]) if c.endswith("_time") else d[c] for c in _COLUMNS]
        with sqlite3.connect(self.db_path) as con:
            con.execute(f"INSERT INTO trades VALUES ({','.join('?' * len(row))})", row)
        if self.csv_path:
            with self.csv_path.open("a", newline="") as fh:
                csv.writer(fh).writerow(row)
        logger.info("TRADE %s %s %.6f | %s -> %s | %.2f -> %.2f | %s | net %.2f (%.2fR)",
                    trade.direction, trade.symbol, trade.amount, trade.entry_time, trade.exit_time,
                    trade.entry_price, trade.exit_price, trade.exit_reason, trade.net_pnl, trade.r_multiple)
