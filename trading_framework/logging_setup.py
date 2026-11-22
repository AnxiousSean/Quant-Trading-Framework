"""Standard-library logging configuration."""

from __future__ import annotations

import logging
from pathlib import Path

from trading_framework.config.settings import LoggingConfig

FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(cfg: LoggingConfig, level_override: str | None = None) -> None:
    """Configure root logging to stderr and, optionally, a file."""
    level = getattr(logging, (level_override or cfg.level).upper())
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if cfg.file:
        Path(cfg.file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(cfg.file))
    logging.basicConfig(level=level, format=FORMAT, handlers=handlers, force=True)
    for noisy in ("ccxt", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
