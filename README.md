# Quant-Trading-Framework — Two-Step Trader's Landing (Crypto)

A modular, event-driven Python framework for researching, backtesting and paper-trading the
**"Two-Step Trader's Landing"** price-action strategy (as presented by Steven from The Trading
Channel), adapted from Forex to 24/7 crypto markets.

> ⚠️ **Disclaimer.** This software is for **educational and research purposes only** and is
> **not financial advice**. Backtests (especially on synthetic data) do not predict future
> results. Crypto derivatives trading carries a high risk of loss. The `LiveBroker` is an
> interface/stub — do not trade real money with it without your own review and testnet testing.

---

## Project structure

```
.
├── main.py                       # CLI: backtest | paper [--live] | fetch
├── requirements.txt              # pinned dependencies
├── config/
│   ├── btc_usdt.yaml             # BTC/USDT perpetual on Binance USDⓈ-M (defaults)
│   └── synthetic_demo.yaml       # offline demo, no network/API keys
├── trading_framework/
│   ├── config/settings.py        # pydantic models, YAML loader, TSF__ env overrides
│   ├── data/
│   │   ├── handler.py            # CCXTDataHandler: pagination, retries, SQLite cache
│   │   ├── validator.py          # DataValidator: gaps, outliers, bad candles, history
│   │   ├── loaders.py            # M5 -> M15 resampling, CSV loading, pipeline
│   │   └── synthetic.py          # regime-switching synthetic OHLCV
│   ├── indicators/
│   │   ├── atr.py                # true_range, Wilder ATR, ATR bands
│   │   └── swings.py             # fractal swing points with confirmation index
│   ├── strategy/
│   │   ├── signals.py            # Direction, Trend, SignalEvent
│   │   └── two_step.py           # TwoStepStrategy (pure, no I/O)
│   ├── risk/manager.py           # sizing, SL/TP, exchange precision, daily breaker
│   ├── portfolio/portfolio.py    # positions, PnL, equity curve
│   ├── execution/
│   │   ├── base.py               # Broker interface, Fill
│   │   ├── paper.py              # PaperBroker: slippage + fees
│   │   ├── live.py               # LiveBroker (ccxt) — dry-run by default
│   │   └── runner.py             # PaperTrader real-time polling loop
│   ├── backtest/
│   │   ├── engine.py             # TradingEngine: shared bar-processing core
│   │   └── backtester.py         # bar-by-bar replay
│   ├── analytics/
│   │   ├── metrics.py            # Sharpe, Sortino, drawdown, PF, win rate…
│   │   ├── trade_log.py          # SQLite + CSV trade journal
│   │   └── report.py             # metrics.json/csv, equity.csv, trades.csv, report.html
│   ├── logging_setup.py
│   └── timeframes.py
└── tests/                        # pytest suite (59 tests, synthetic data)
```

## Setup

Python 3.10+.

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
pytest
```

## Quick start

Offline demo on synthetic data (no network, no keys):

```bash
python main.py backtest --config config/synthetic_demo.yaml
```

Real BTC/USDT data from Binance (public endpoints, cached to `cache/ohlcv.sqlite`):

```bash
python main.py fetch --config config/btc_usdt.yaml
```

```bash
python main.py backtest --config config/btc_usdt.yaml
```

Real-time paper trading (public market data, simulated fills):

```bash
python main.py paper --config config/btc_usdt.yaml
```

LiveBroker in **dry-run** (orders are logged, never sent):

```bash
python main.py paper --config config/btc_usdt.yaml --live
```

Real orders require **all three**: the `--live` flag, `execution.dry_run: false`, and API keys in
the env vars named by `exchange.api_key_env` / `exchange.api_secret_env`
(`BINANCE_API_KEY` / `BINANCE_API_SECRET` in the sample config). Keys are never read from YAML.

## Configuration

Everything tunable is in YAML and validated by pydantic (bad values fail fast at start-up).
Any field can be overridden with an environment variable `TSF__<SECTION>__<FIELD>`:

```bash
TSF__RISK__RISK_PER_TRADE=0.005 TSF__STRATEGY__ATR_MULTIPLIER=2.5 python main.py backtest --config config/btc_usdt.yaml
```

Key parameters (defaults):

| Section | Field | Default | Meaning |
|---|---|---|---|
| strategy | `atr_period` / `atr_multiplier` | 14 / 2.0 | ATR bands on M15 |
| strategy | `trend_lookback` | 20 | M15 candles scanned for HH/HL or LH/LL |
| strategy | `swing_left` / `swing_right` | 2 / 2 | fractal pivot width (right = confirmation lag) |
| strategy | `min_swings` | 2 | swing highs **and** lows needed to call a trend |
| strategy | `zone_sources` | `[swing, atr_band]` | support/resistance used for the retest (`flip` also available) |
| strategy | `min_wick_to_range` / `min_wick_to_body` | 0.5 / 1.5 | what counts as a "long" rejection wick |
| strategy | `stop_atr_multiplier` / `reward_risk` | 2.0 / 2.0 | SL = pullback extreme ∓ 2×ATR, TP = 1:2 |
| strategy | `max_signal_bar_range_atr`, `min_volume` | 3.0, 0 | spike / illiquidity filters |
| risk | `risk_per_trade` | 0.01 | 1% of equity per trade |
| risk | `max_positions_per_symbol` / `max_total_positions` | 1 / 1 | exposure caps |
| risk | `max_leverage` | 3.0 | notional cap (size is reduced if the 1% size would exceed it) |
| risk | `daily_loss_limit` | 0.03 | halt new entries for the UTC day after −3% (null disables) |
| execution | `taker_fee` / `slippage` | 0.0004 / 0.0005 | 0.04% fee, 0.05% adverse slippage |
| execution | `max_spread` / `max_slippage` | 0.001 / 0.002 | skip entries on wide spreads; abort live fills beyond slippage |
| execution | `dry_run` | true | LiveBroker never sends orders unless false **and** `--live` |

For **spot** markets set `exchange.id: binance`, `symbol: BTC/USDT`, `market_type: spot`,
`risk.allow_shorts: false`, `risk.max_leverage: 1.0`.

## Strategy implementation

| Rule | Implementation |
|---|---|
| Trend (M15) | Fractal swings inside the last `trend_lookback` candles; last `min_swings` highs **and** lows strictly rising ⇒ UP, falling ⇒ DOWN; otherwise no-trade. Uptrend void if the last M15 close is below the last HL (mirror for shorts). |
| Retest zone | Most recent swing low (HL) and/or lower ATR band (`close − 2×ATR`) for longs; swing high (LH) / upper band for shorts. Optional `flip`: the prior swing high broken by the latest HH. |
| Rejection wick (M5) | Wick ≥ 50% of range and ≥ 1.5× body; low reaches `level + tolerance×ATR` and close is back above `level`. |
| Confirmation (M5) | Next candle bullish and closes above the rejection high (mirror for shorts). |
| Trigger | Enter at the **open of the following M5 candle**. |
| Stop | Lowest low of the pullback leg (M5 lows since the last M15 swing high) − 2×M15 ATR. |
| Target | `entry + 2 × (entry − stop)` computed from the **actual fill**. |
| Size | `equity × risk% / |entry − stop|`, floored to exchange step, capped by leverage, rejected below min amount/notional. |

### Look-ahead protection

* Indicators are causal (tested: values on truncated data equal the full-history prefix).
* Swing pivots carry a `confirm_pos`; a pivot is only usable after its right-hand bars close.
* The strategy itself drops any M15 candle that hadn't closed when the rejection candle closed,
  even if the caller passes it (tested by injecting a "future" crash candle).
* The backtester only passes rows closed at the current M5 close, and fills entries at the next
  candle's open.
* M15 is resampled from M5 by default (`derive_trend_from_entry`), dropping incomplete buckets.
* Test `test_backtest_has_no_lookahead` checks trades closed before a cut-off are identical
  whether or not later data exists.

### Simulation assumptions (be aware when interpreting results)

* If a candle touches both stop and target, the **stop is assumed first** (conservative).
* Gaps through the stop fill at the open; stops and market orders pay slippage; take-profits are
  treated as resting limits (no slippage). All fills pay the taker fee.
* Spread is not in OHLCV data: the paper broker uses `assumed_spread`; live/paper-live use the
  order book. A spike filter (`max_signal_bar_range_atr`) and `min_volume` proxy for illiquidity.
* Account model is USDT-margined (equity = cash + unrealised PnL); funding rates are **not** modelled.
* A signal not fillable at the next open is discarded, never chased.

## Reading the results

`backtest` prints a summary and writes to `backtest.output_dir` (default `reports/`):

* `report.html` — summary table, equity and drawdown charts, last 200 trades
* `metrics.json` / `metrics.csv`, `equity.csv` (equity + drawdown), `trades.csv`
* `trade_log.csv` and `trades.sqlite` — the persistent journal (with `run_id`)

| Metric | How to read it |
|---|---|
| `total_return_pct`, `cagr_pct` | Net of fees and slippage. |
| `max_drawdown_pct` | Worst peak-to-trough on marked equity — size your expectations by this. |
| `sharpe`, `sortino` | Daily returns annualised with 365 days (crypto trades 24/7). |
| `win_rate_pct` | With 1:2 R:R, break-even is ≈ 33% before costs (higher after fees). |
| `profit_factor` | Gross profit / gross loss; > 1 is profitable. |
| `avg_r_multiple`, `expectancy` | Average outcome per trade in R and in quote currency. |
| `signals`, `skipped_entries` | How often the setup fired, and how many were blocked by risk/filters. |

A small number of trades means the statistics are noisy — this setup is selective by design.
Test across several periods/symbols and vary parameters before drawing conclusions; beware of
overfitting.

## Extending

* New indicator → pure function in `trading_framework/indicators/` + test.
* New strategy → class exposing `generate_signals(data_m15, data_m5) -> SignalEvent`,
  `trend_window`, `entry_window`, `entry_td`.
* New venue → implement `execution.base.Broker`.

## Known limitations / stubs

* `LiveBroker`: places market entry + reduce-only `STOP_MARKET`/`TAKE_PROFIT_MARKET` brackets
  (Binance USDⓈ-M params) but has **no** position reconciliation, partial-fill handling,
  order-status polling or bracket cancellation. In real-live mode local exit tracking is an
  approximation of the exchange-side brackets.
* Single symbol per process; multi-symbol portfolios would run one engine per symbol sharing a
  `Portfolio`/`RiskManager`.
* No funding-rate or liquidation modelling.
