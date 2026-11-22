#!/usr/bin/env bash
# Set up the Two-Step Trader's Landing framework on a Linux machine/VM.
#
# Usage: scripts/setup_linux.sh [options]
#
#   --python PATH         Python interpreter to use (default: newest python3.10+ found)
#   --install-deps        Install python3-venv/pip via apt or dnf (uses sudo)
#   --skip-tests          Don't run the pytest suite after installing
#   --check-exchange      Check that the exchange in --config is reachable (public API)
#   --install-service     Install a systemd *user* service for paper trading (not started)
#   --config PATH         Config used by --check-exchange / --install-service
#                         (default: config/btc_usdt.yaml)
#   -h, --help            Show this help
#
# Safe to re-run. Nothing here places orders; the service runs `paper` mode
# (simulated fills) only.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_DIR/.venv"
CONFIG="config/btc_usdt.yaml"
PYTHON=""
INSTALL_DEPS=0
RUN_TESTS=1
CHECK_EXCHANGE=0
INSTALL_SERVICE=0
SERVICE_NAME="twostep-paper"
ENV_FILE="$HOME/.config/twostep/env"

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m ok\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarn\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror\033[0m %s\n' "$*" >&2; exit 1; }

usage() { sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON="${2:?--python needs a path}"; shift 2 ;;
    --install-deps) INSTALL_DEPS=1; shift ;;
    --skip-tests) RUN_TESTS=0; shift ;;
    --check-exchange) CHECK_EXCHANGE=1; shift ;;
    --install-service) INSTALL_SERVICE=1; shift ;;
    --config) CONFIG="${2:?--config needs a path}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; die "unknown option: $1" ;;
  esac
done

[[ "$(uname -s)" == "Linux" ]] || die "this script is for Linux"
cd "$REPO_DIR"
[[ -f requirements.txt && -f main.py ]] || die "run from inside the repository checkout"
[[ -f "$CONFIG" ]] || die "config not found: $CONFIG"

# --------------------------------------------------------------- system deps
if [[ $INSTALL_DEPS -eq 1 ]]; then
  info "Installing system packages"
  if command -v apt-get >/dev/null; then
    sudo apt-get update -y
    sudo apt-get install -y python3 python3-venv python3-pip
  elif command -v dnf >/dev/null; then
    sudo dnf install -y python3 python3-pip
  else
    warn "no apt-get or dnf; install Python 3.10+ with venv support yourself"
  fi
fi

# ------------------------------------------------------------------- python
py_ok() { "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; }

if [[ -z "$PYTHON" ]]; then
  for c in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null && py_ok "$c"; then PYTHON="$(command -v "$c")"; break; fi
  done
fi
[[ -n "$PYTHON" ]] || die "Python 3.10+ not found (try --install-deps, or install python3.10+ and pass --python)"
py_ok "$PYTHON" || die "$PYTHON is older than 3.10"
ok "Python: $PYTHON ($("$PYTHON" --version 2>&1))"

# --------------------------------------------------------------------- venv
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  info "Creating virtual environment in .venv"
  "$PYTHON" -m venv "$VENV_DIR" \
    || die "venv creation failed; on Debian/Ubuntu run: sudo apt-get install python3-venv (or re-run with --install-deps)"
else
  ok "Reusing existing .venv"
fi
VPY="$VENV_DIR/bin/python"

info "Installing pinned dependencies"
"$VPY" -m pip install --quiet --upgrade pip
"$VPY" -m pip install --quiet -r requirements.txt
ok "Dependencies installed"

mkdir -p cache reports logs

# -------------------------------------------------------------------- tests
if [[ $RUN_TESTS -eq 1 ]]; then
  info "Running test suite"
  "$VPY" -m pytest -q -p no:cacheprovider || die "tests failed; fix before trading"
  ok "Tests passed"
fi

# --------------------------------------------------------------- clock sync
info "Checking clock synchronisation"
if command -v timedatectl >/dev/null && timedatectl show >/dev/null 2>&1; then
  if [[ "$(timedatectl show -p NTPSynchronized --value)" == "yes" ]]; then
    ok "System clock is NTP-synchronised"
  else
    warn "clock is NOT synchronised. Candle-close logic and signed exchange requests need an accurate clock."
    warn "fix with: sudo timedatectl set-ntp true   (and check the VM's time sync with the host)"
  fi
else
  warn "timedatectl unavailable; make sure NTP (chrony/systemd-timesyncd) is running"
fi

# ----------------------------------------------------------------- exchange
if [[ $CHECK_EXCHANGE -eq 1 ]]; then
  info "Checking exchange connectivity for $CONFIG"
  if "$VPY" - "$CONFIG" <<'PY'
import sys, time
import ccxt
from trading_framework.config.settings import load_config

cfg = load_config(sys.argv[1])
ex = getattr(ccxt, cfg.exchange.id)({"timeout": 15000, "options": {"defaultType": cfg.exchange.market_type}})
try:
    server_ms = ex.fetch_time()
    drift = abs(server_ms - time.time() * 1000) / 1000
    candles = ex.fetch_ohlcv(cfg.exchange.symbol, cfg.data.entry_timeframe, limit=3)
except Exception as exc:  # report one line, not a traceback
    print(f"{cfg.exchange.id}: {type(exc).__name__}: {str(exc)[:200]}", file=sys.stderr)
    sys.exit(1)
print(f"{cfg.exchange.id} reachable; {cfg.exchange.symbol} returned {len(candles)} candles; clock drift {drift:.2f}s")
if drift > 1.0:
    print("WARNING: clock drift > 1s; signed (live) requests may be rejected", file=sys.stderr)
PY
  then ok "Exchange reachable"
  else warn "exchange check failed. The exchange may block your region/network; try another exchange.id in the config."
  fi
fi

# ------------------------------------------------------------------ service
if [[ $INSTALL_SERVICE -eq 1 ]]; then
  command -v systemctl >/dev/null || die "systemctl not found; cannot install a systemd service"
  info "Installing systemd user service '$SERVICE_NAME' (paper mode)"
  mkdir -p "$(dirname "$ENV_FILE")" "$HOME/.config/systemd/user"
  if [[ ! -f "$ENV_FILE" ]]; then
    (umask 077; cat > "$ENV_FILE") <<'ENV'
# Environment for the twostep-paper service. Keep this file private (chmod 600).
# Paper mode needs no API keys. Only fill these in for --live use, and never commit them.
# BINANCE_API_KEY=
# BINANCE_API_SECRET=
# Config overrides, e.g.:
# TSF__RISK__RISK_PER_TRADE=0.005
ENV
    ok "Created $ENV_FILE (mode 600)"
  fi
  chmod 600 "$ENV_FILE"
  cat > "$HOME/.config/systemd/user/$SERVICE_NAME.service" <<UNIT
[Unit]
Description=Two-Step Trader's Landing paper trader
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$REPO_DIR
EnvironmentFile=-$ENV_FILE
ExecStart=$VPY $REPO_DIR/main.py paper --config $CONFIG
Restart=on-failure
RestartSec=30

[Install]
WantedBy=default.target
UNIT
  systemctl --user daemon-reload 2>/dev/null || warn "could not reload the user systemd daemon (no user session?)"
  ok "Service file written to ~/.config/systemd/user/$SERVICE_NAME.service (not started)"
  cat <<NEXT

  Start it now and on boot:
    systemctl --user enable --now $SERVICE_NAME
  Keep it running after you log out:
    sudo loginctl enable-linger $(id -un)
  Follow the logs:
    journalctl --user -u $SERVICE_NAME -f
NEXT
fi

# ------------------------------------------------------------------ summary
cat <<DONE

Setup complete. Next steps:
  source .venv/bin/activate
  # offline demo:
  python main.py backtest --config config/synthetic_demo.yaml
  # download real data, backtest it, then paper trade (simulated fills, real time):
  python main.py fetch    --config $CONFIG
  python main.py backtest --config $CONFIG
  python main.py paper    --config $CONFIG

Educational software only, not financial advice.
DONE
