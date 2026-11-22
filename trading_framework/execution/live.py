"""Live ccxt broker. **Dry-run by default.**

Safety layers (all must be passed to send a real order):

1. The CLI must be invoked with ``--live``.
2. ``execution.dry_run`` must be ``false`` in the config.
3. API key and secret must be present in the configured environment variables.

In dry-run mode, orders are logged and simulated at the reference price plus
configured slippage; nothing is sent to the exchange. Market data (spreads,
precision) is still read from the exchange's public endpoints when reachable.

.. warning::
   This is an interface/stub implementation. Order placement uses Binance
   USDⓈ-M style bracket params (``STOP_MARKET`` / ``TAKE_PROFIT_MARKET`` with
   ``reduceOnly``). Position reconciliation, partial fills, order-status
   polling and exchange-side bracket cancellation are NOT implemented. Review
   and test thoroughly on a testnet before risking capital.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from trading_framework.config.settings import ExchangeConfig, ExecutionConfig
from trading_framework.execution.base import Broker, Fill, OrderType
from trading_framework.risk.manager import MarketLimits, OrderPlan, round_to_tick

logger = logging.getLogger(__name__)


class LiveTradingNotAuthorized(RuntimeError):
    """Raised when live trading is requested without all safeguards satisfied."""


class SlippageExceeded(RuntimeError):
    """Raised when a live fill deviates more than ``max_slippage``."""


class LiveBroker(Broker):
    """ccxt-backed broker.

    Args:
        exchange_cfg: Exchange settings (id, symbol, credential env var names).
        exec_cfg: Execution settings (``dry_run``, fees, slippage guard).
        live_flag: Whether ``--live`` was passed on the command line.
        exchange: Optional pre-built ccxt exchange (dependency injection for tests).
    """

    def __init__(self, exchange_cfg: ExchangeConfig, exec_cfg: ExecutionConfig, live_flag: bool,
                 exchange: Any | None = None) -> None:
        self.ex_cfg = exchange_cfg
        self.cfg = exec_cfg
        self.dry_run = exec_cfg.dry_run or not live_flag
        key, secret = exchange_cfg.api_credentials()
        if not self.dry_run and not (key and secret):
            raise LiveTradingNotAuthorized(
                f"Live trading requires {exchange_cfg.api_key_env} and {exchange_cfg.api_secret_env} env vars")
        if live_flag and exec_cfg.dry_run:
            logger.warning("--live given but execution.dry_run=true: orders will NOT be sent")
        self.exchange = exchange or self._build_exchange(key, secret)
        self._limits: dict[str, MarketLimits] = {}
        mode = "DRY-RUN" if self.dry_run else "LIVE — REAL ORDERS"
        logger.warning("LiveBroker initialised in %s mode on %s", mode, exchange_cfg.id)

    def _build_exchange(self, key: str | None, secret: str | None) -> Any:
        import ccxt  # local import keeps ccxt optional for offline backtests

        cls = getattr(ccxt, self.ex_cfg.id)
        params: dict[str, Any] = {
            "enableRateLimit": self.ex_cfg.enable_rate_limit,
            "timeout": self.ex_cfg.request_timeout_ms,
            "options": {"defaultType": self.ex_cfg.market_type},
        }
        if key and secret:
            params.update(apiKey=key, secret=secret)
        ex = cls(params)
        if self.ex_cfg.sandbox:
            ex.set_sandbox_mode(True)
        return ex

    # --------------------------------------------------------------- market
    def market_limits(self, symbol: str) -> MarketLimits:
        if symbol in self._limits:
            return self._limits[symbol]
        fallback = MarketLimits.from_config(self.ex_cfg)
        try:
            self.exchange.load_markets()
            m = self.exchange.market(symbol)
            amount_step = float(m["precision"].get("amount") or fallback.amount_step)
            price_tick = float(m["precision"].get("price") or fallback.price_tick)
            limits = MarketLimits(
                amount_step=amount_step,
                price_tick=price_tick,
                min_amount=float((m["limits"].get("amount") or {}).get("min") or fallback.min_amount),
                min_notional=float((m["limits"].get("cost") or {}).get("min") or fallback.min_notional),
            )
        except Exception as exc:  # noqa: BLE001 - network/metadata issues fall back to config
            logger.warning("Could not load market limits (%s); using config fallbacks", exc)
            limits = fallback
        self._limits[symbol] = limits
        return limits

    def get_spread(self, symbol: str) -> float:
        try:
            ob = self.exchange.fetch_order_book(symbol, limit=5)
            bid, ask = ob["bids"][0][0], ob["asks"][0][0]
            return (ask - bid) / ((ask + bid) / 2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Spread unavailable (%s); treating as infinite", exc)
            return float("inf")

    def last_price(self, symbol: str) -> float:
        """Last traded price from the public ticker."""
        return float(self.exchange.fetch_ticker(symbol)["last"])

    def expected_fill_price(self, side: str, reference_price: float) -> float:
        sign = 1 if side == "buy" else -1
        return reference_price * (1 + sign * self.cfg.slippage)

    # --------------------------------------------------------------- orders
    def submit_entry(self, plan: OrderPlan, reference_price: float, timestamp: pd.Timestamp) -> Fill:
        side = plan.direction.entry_side
        exit_side = plan.direction.exit_side
        if self.dry_run:
            price = round_to_tick(self.expected_fill_price(side, reference_price),
                                  self.market_limits(plan.symbol).price_tick)
            logger.info("[DRY-RUN] %s %s %.6f @ ~%.2f | SL %.2f | TP %.2f",
                        side.upper(), plan.symbol, plan.amount, price, plan.stop_price, plan.take_profit)
            return Fill(plan.symbol, side, plan.amount, price, plan.amount * price * self.cfg.taker_fee,
                        timestamp, "market", reference_price, "dry-run")

        order = self.exchange.create_order(plan.symbol, "market", side, plan.amount)
        price = float(order.get("average") or order.get("price") or reference_price)
        fee = float((order.get("fee") or {}).get("cost") or plan.amount * price * self.cfg.taker_fee)
        fill = Fill(plan.symbol, side, float(order.get("filled") or plan.amount), price, fee, timestamp,
                    "market", reference_price, str(order.get("id")))
        if fill.slippage > self.cfg.max_slippage:
            logger.error("Slippage %.4f exceeds limit; flattening position", fill.slippage)
            self.exchange.create_order(plan.symbol, "market", exit_side, fill.amount, params={"reduceOnly": True})
            raise SlippageExceeded(f"slippage {fill.slippage:.4%} > {self.cfg.max_slippage:.4%}")
        bracket = {"reduceOnly": True}
        self.exchange.create_order(plan.symbol, "STOP_MARKET", exit_side, fill.amount,
                                   params={**bracket, "stopPrice": plan.stop_price})
        self.exchange.create_order(plan.symbol, "TAKE_PROFIT_MARKET", exit_side, fill.amount,
                                   params={**bracket, "stopPrice": plan.take_profit})
        logger.info("LIVE entry %s %.6f @ %.2f with SL %.2f / TP %.2f", side, fill.amount, price,
                    plan.stop_price, plan.take_profit)
        return fill

    def submit_exit(self, symbol: str, side: str, amount: float, price: float, timestamp: pd.Timestamp,
                    order_type: OrderType) -> Fill:
        if self.dry_run:
            logger.info("[DRY-RUN] exit %s %s %.6f @ %.2f (%s)", side, symbol, amount, price, order_type)
            return Fill(symbol, side, amount, price, amount * price * self.cfg.taker_fee, timestamp,
                        order_type, price, "dry-run")
        # Live exits are expected to be handled by the exchange-side bracket orders placed at entry.
        # Reconciliation (fetch_positions / fetch_my_trades) is not implemented in this stub.
        raise NotImplementedError("Live exit reconciliation is not implemented; brackets live on the exchange")
