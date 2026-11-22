"""CLI entry point.

Examples:
    python main.py backtest --config config/btc_usdt.yaml
    python main.py backtest --config config/synthetic_demo.yaml
    python main.py fetch    --config config/btc_usdt.yaml
    python main.py paper    --config config/btc_usdt.yaml
    python main.py paper    --config config/btc_usdt.yaml --live   # LiveBroker (dry-run unless config says otherwise)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

import pandas as pd

from trading_framework.analytics.report import write_report
from trading_framework.analytics.trade_log import TradeLogger
from trading_framework.backtest.backtester import Backtester
from trading_framework.config.settings import AppConfig, load_config
from trading_framework.data.handler import CCXTDataHandler, DataUnavailableError
from trading_framework.data.loaders import load_market_data
from trading_framework.data.validator import DataValidator, InsufficientDataError
from trading_framework.execution.live import LiveBroker, LiveTradingNotAuthorized
from trading_framework.execution.paper import PaperBroker
from trading_framework.execution.runner import PaperTrader
from trading_framework.logging_setup import setup_logging
from trading_framework.risk.manager import MarketLimits

logger = logging.getLogger("main")

DISCLAIMER = "Educational software only — not financial advice. Trading crypto involves substantial risk of loss."


def cmd_backtest(cfg: AppConfig, args: argparse.Namespace) -> int:
    m15, m5 = load_market_data(cfg)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.output or cfg.backtest.output_dir
    tlog = TradeLogger(cfg.backtest.trade_db, csv_path=f"{out}/trade_log.csv", run_id=run_id)
    result = Backtester(cfg, tlog).run(m15, m5)
    paths = write_report(out, result.metrics, result.equity, result.trades,
                         title=f"Two-Step Trader's Landing — {cfg.exchange.symbol} ({run_id})")
    print("\n=== Backtest summary ===")
    for k, v in result.metrics.items():
        print(f"{k:>24}: {v}")
    print(f"\nReport: {paths['html']}\n{DISCLAIMER}")
    return 0


def cmd_fetch(cfg: AppConfig, args: argparse.Namespace) -> int:
    handler = CCXTDataHandler(cfg.exchange, cfg.data)
    start = pd.Timestamp(cfg.data.start or "2024-01-01", tz="UTC")
    end = pd.Timestamp(cfg.data.end, tz="UTC") if cfg.data.end else None
    for tf in {cfg.data.entry_timeframe, cfg.data.trend_timeframe}:
        df = handler.load(cfg.exchange.symbol, tf, start, end)
        report = DataValidator(tf, cfg.data.min_history_bars, cfg.data.outlier_return_threshold,
                               cfg.data.max_gap_ratio).validate(df)
        print(f"{tf}: {report}")
    return 0


def cmd_paper(cfg: AppConfig, args: argparse.Namespace) -> int:
    if args.live:
        broker = LiveBroker(cfg.exchange, cfg.execution, live_flag=True)
    else:
        broker = PaperBroker(cfg.execution, MarketLimits.from_config(cfg.exchange))
    mode = "live" if args.live else "paper"
    tlog = TradeLogger(cfg.backtest.trade_db.replace(".sqlite", f"_{mode}.sqlite"), run_id=mode)
    trader = PaperTrader(cfg, broker, CCXTDataHandler(cfg.exchange, cfg.data), tlog)
    logger.warning("%s | mode=%s", DISCLAIMER, mode)
    try:
        trader.run(max_iterations=args.iterations)
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Two-Step Trader's Landing crypto framework. " + DISCLAIMER)
    sub = p.add_subparsers(dest="command", required=True)
    for name, helptext in (("backtest", "run an event-driven backtest"),
                           ("paper", "run the real-time paper trader"),
                           ("fetch", "download and validate historical data into the cache")):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("--config", required=True, help="path to YAML config")
        sp.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
        if name == "backtest":
            sp.add_argument("--output", default=None, help="report directory (overrides config)")
        if name == "paper":
            sp.add_argument("--live", action="store_true",
                            help="use LiveBroker (still dry-run unless execution.dry_run=false and API keys set)")
            sp.add_argument("--iterations", type=int, default=None, help="stop after N polls")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg.logging, args.log_level)
    handlers = {"backtest": cmd_backtest, "paper": cmd_paper, "fetch": cmd_fetch}
    try:
        return handlers[args.command](cfg, args)
    except (DataUnavailableError, InsufficientDataError, LiveTradingNotAuthorized) as exc:
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
