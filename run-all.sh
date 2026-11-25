#!/usr/bin/env bash
# Runs every uk-agents service on this machine, in the foreground. Ctrl-C stops them all.
#
#   ./run-all.sh                     # REST API, Telegram bot, Streamlit UI, property crawler
#   ./run-all.sh whatsapp mails      # those four plus the named optional services
#   ./run-all.sh --only rest streamlit   # just the named services
#   ./run-all.sh --no-setup          # skip uv sync / playwright install
#
# Services (each runs `uv run python -m <module>` from the project root):
#   rest       clients.rest.main          REST API           http://localhost:8000
#   telegram   clients.telegram.main      Telegram bot       (TELEGRAM_BOT_TOKEN)
#   streamlit  clients.streamlit.main     Streamlit UI       http://localhost:8501 (needs rest)
#   crawler    property_crawler.main      property crawler   every hour → Pinecone
#   cron       clients.cron.main          optional: CRON_SEARCH_PROMPT every 30 min
#   whatsapp   agents.whatsapp.main       optional: WhatsApp group listener (WHATSAPP_*)
#   mails      agents.mails.main          optional: IMAP mail digest monitor (MAIL_*)
#
# Each service's output goes to logs/run-<name>.log; the application log is logs/agent.log
# (the crawler's is logs/crawler.log).
#
# One-time setup on a new machine (Ubuntu):
#   curl -LsSf https://astral.sh/uv/install.sh | sh       # uv (installs Python itself)
#   sudo apt install -y git nodejs npm                    # npx starts the MCP servers in mcp.json
#   cp .env-template .env                                 # then fill in real values
#   uv sync && sudo .venv/bin/playwright install-deps chromium   # crawler's browser libraries
#
# On the server the services normally run under systemd instead (deployment/README.md):
#   systemctl --user status|restart|stop uk-agents.service
#   journalctl --user -u uk-agents.service -f
# This script refuses to start while that service is active: two Telegram bots on one
# token conflict, and the ports would clash.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1
LOG_DIR="$ROOT/logs"
STOP_TIMEOUT=20

DEFAULT_SERVICES="rest telegram streamlit crawler"
SETUP=true
ONLY=false
EXTRA=()

log() { printf '%s [run-all] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
die() { log "ERROR: $*"; exit 1; }

module_of() {
    case "$1" in
        rest | telegram | streamlit | cron) echo "clients.$1.main" ;;
        crawler) echo "property_crawler.main" ;;
        whatsapp | mails) echo "agents.$1.main" ;;
        *) return 1 ;;
    esac
}

while [ $# -gt 0 ]; do
    case "$1" in
        --no-setup) SETUP=false ;;
        --only) ONLY=true ;;
        -h | --help) sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) die "unknown option $1 (see --help)" ;;
        *) module_of "$1" >/dev/null || die "unknown service '$1' (see --help)"; EXTRA+=("$1") ;;
    esac
    shift
done

if [ "$ONLY" = true ]; then
    [ ${#EXTRA[@]} -gt 0 ] || die "--only needs at least one service name"
    SERVICES="${EXTRA[*]}"
else
    SERVICES="$DEFAULT_SERVICES ${EXTRA[*]:-}"
fi

# --- checks ----------------------------------------------------------------------
command -v uv >/dev/null || die "uv is not installed (https://docs.astral.sh/uv/)"
command -v npx >/dev/null || log "WARNING: npx not found — the MCP servers (finance/stocks tools) will not connect; install Node.js"
[ -f "$ROOT/.env" ] || die "no .env in $ROOT (cp .env-template .env and fill it in)"
if systemctl --user is-active --quiet uk-agents.service 2>/dev/null; then
    die "uk-agents.service is running these services already — stop it first: systemctl --user stop uk-agents.service"
fi

# --- setup -----------------------------------------------------------------------
mkdir -p "$LOG_DIR"
if [ "$SETUP" = true ]; then
    log "installing dependencies (uv sync)"
    uv sync --quiet || die "uv sync failed"
    case " $SERVICES " in
        *" crawler "*)
            log "installing the crawler's browser (playwright install chromium)"
            uv run --no-sync playwright install chromium >"$LOG_DIR/run-playwright.log" 2>&1 ||
                log "WARNING: playwright install failed — see logs/run-playwright.log"
            ;;
    esac
fi

# --- run -------------------------------------------------------------------------
declare -A PIDS=()

start() {
    : >"$LOG_DIR/run-$1.log"
    set -m   # own process group, so stopping a service stops its children (MCP servers, Chromium)
    uv run --no-sync python -m "$(module_of "$1")" >>"$LOG_DIR/run-$1.log" 2>&1 &
    set +m
    PIDS[$1]=$!
    log "started $1 (pid $!, output: logs/run-$1.log)"
}

stop_all() {
    trap - INT TERM
    log "stopping"
    local name pid waited
    for name in "${!PIDS[@]}"; do kill -TERM -- "-${PIDS[$name]}" 2>/dev/null; done
    for name in "${!PIDS[@]}"; do
        pid=${PIDS[$name]}; waited=0
        while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt "$STOP_TIMEOUT" ]; do sleep 1; waited=$((waited + 1)); done
        kill -KILL -- "-$pid" 2>/dev/null
        log "stopped $name"
    done
    exit 0
}
trap stop_all INT TERM

for name in $SERVICES; do
    start "$name"
    [ "$name" = rest ] && sleep 2   # let the REST API bind before Streamlit starts calling it
done
log "running: $SERVICES — Ctrl-C to stop"

# Report a service that exits; the others keep running.
while true; do
    for name in "${!PIDS[@]}"; do
        if ! kill -0 "${PIDS[$name]}" 2>/dev/null; then
            log "$name exited — last lines of logs/run-$name.log:"
            tail -n 5 "$LOG_DIR/run-$name.log" | sed 's/^/    /'
            unset "PIDS[$name]"
        fi
    done
    [ ${#PIDS[@]} -gt 0 ] || { log "every service has exited"; exit 1; }
    sleep 5
done
