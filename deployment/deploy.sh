#!/usr/bin/env bash
#
# deployment/deploy.sh — run uk-agents and keep it on the latest main.
#
# Starts the services (REST API, Telegram bot, Streamlit UI, property crawler), then every
# DEPLOY_POLL_SECONDS:
#   - restarts any service that has died;
#   - asks GitHub whether main has moved; if it has: pulls it, builds (uv sync),
#     runs the tests, and restarts the services on the new code.
# A commit that fails to build or fails its tests is rolled back and skipped;
# the services keep running the last good version.
#
# Runs in the foreground until stopped (Ctrl-C / SIGTERM stops the services too).
# See deployment/README.md.

# Everything lives in main() so bash has read the whole file before running any of
# it — the script may be replaced on disk by the very pull it performs.
main() {
    set -uo pipefail
    # Nothing here reads the keyboard. Without this, a command that touches the terminal
    # while the script runs in the background (nohup … &) is suspended by the terminal.
    exec </dev/null

    REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    SELF="$REPO_DIR/deployment/deploy.sh"
    REMOTE="${DEPLOY_REMOTE:-origin}"
    BRANCH="${DEPLOY_BRANCH:-main}"
    POLL_SECONDS="${DEPLOY_POLL_SECONDS:-60}"
    SERVICES="${DEPLOY_SERVICES:-rest telegram streamlit crawler}"   # see module_of
    RUN_TESTS="${DEPLOY_RUN_TESTS:-true}"
    STOP_TIMEOUT="${DEPLOY_STOP_TIMEOUT:-20}"
    RUN_DIR="$REPO_DIR/deployment/.run"
    LOG_DIR="$REPO_DIR/logs"

    export PATH="$HOME/.local/bin:$PATH"                 # where the uv installer puts uv
    export STREAMLIT_SERVER_HEADLESS=true                # no browser, no first-run prompt
    export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

    cd "$REPO_DIR" || exit 1
    mkdir -p "$RUN_DIR" "$LOG_DIR"

    preflight
    trap 'log "stopping"; stop_all; exit 0' INT TERM

    if fetch && behind; then
        log "checkout is behind $REMOTE/$BRANCH — updating before the first start"
        pull || die "could not fast-forward to $REMOTE/$BRANCH (see above)"
    fi
    build || die "build failed"
    start_all
    local deployed   # the commit the services are running
    deployed="$(git rev-parse HEAD)"
    log "running $(git rev-parse --short HEAD); checking $REMOTE/$BRANCH every ${POLL_SECONDS}s"

    local skipped=""   # a commit that could not be deployed — not retried until main moves again
    while true; do
        sleep "$POLL_SECONDS" &
        wait $!        # an interruptible sleep: the trap runs at once
        restart_dead
        fetch || continue
        local target previous
        target="$(git rev-parse "$REMOTE/$BRANCH")"
        # Compare with what is running, not with the checkout: a commit that reached the
        # checkout some other way (a manual git pull) still has to be deployed.
        [ "$target" = "$deployed" ] && continue
        [ "$target" = "$skipped" ] && continue
        previous="$(git rev-parse HEAD)"
        log "new commit on $BRANCH: $(git log -1 --format='%h %s' "$target")"
        if [ "$previous" != "$target" ] && ! behind; then
            log "NOT deployed — this checkout has commits that $REMOTE/$BRANCH doesn't"
            skipped="$target"
            continue
        fi
        if pull && build && run_tests; then
            log "deploying $(git rev-parse --short HEAD)"
            stop_all
            exec "$SELF"   # start again as the new version of this script
        fi
        log "NOT deployed — staying on $(git rev-parse --short "$previous")"
        skipped="$target"
        git reset --quiet --keep "$previous" && build
    done
}

log() { printf '%s [deploy] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
die() { log "ERROR: $*"; exit 1; }

preflight() {
    command -v git >/dev/null || die "git is not installed"
    command -v uv >/dev/null || die "uv is not installed (https://docs.astral.sh/uv/)"
    local current
    current="$(git rev-parse --abbrev-ref HEAD)"
    [ "$current" = "$BRANCH" ] || die "this checkout is on '$current', not '$BRANCH' — use a clone that stays on $BRANCH"
    [ -f "$REPO_DIR/.env" ] || die "no .env in $REPO_DIR (cp .env-template .env and fill it in)"
    local name
    for name in $SERVICES; do
        [ -f "$REPO_DIR/$(module_of "$name" | tr . /).py" ] || die "unknown service '$name' (no $(module_of "$name" | tr . /).py)"
    done
}

# The Python module a service runs: a client is clients/<name>/main.py.
module_of() {
    case "$1" in
        crawler) echo "agents.property.crawler.main" ;;
        *) echo "clients.$1.main" ;;
    esac
}

# Ask the remote for the latest commit of the branch. A failed fetch (GitHub or the
# network is down) just means "nothing new right now".
fetch() {
    git fetch --quiet "$REMOTE" "+refs/heads/$BRANCH:refs/remotes/$REMOTE/$BRANCH" 2>/dev/null ||
        { log "fetch failed — will retry"; return 1; }
}

# True when the remote branch has commits this checkout doesn't, and nothing the other way.
behind() {
    [ "$(git rev-parse HEAD)" != "$(git rev-parse "$REMOTE/$BRANCH")" ] &&
        git merge-base --is-ancestor HEAD "$REMOTE/$BRANCH"
}

pull() { git merge --quiet --ff-only "$REMOTE/$BRANCH"; }

build() {
    log "building (uv sync)"
    uv sync --locked --quiet || return 1
    case " $SERVICES " in
        *" crawler "*)
            # The Chromium build that this Playwright version expects; downloads only when it changed.
            # A failure leaves the crawler failing its passes, not the other services undeployed.
            uv run --no-sync playwright install chromium >"$LOG_DIR/deploy-playwright.log" 2>&1 ||
                log "WARNING: playwright install chromium failed — see logs/deploy-playwright.log"
            ;;
    esac
    return 0
}

run_tests() {
    [ "$RUN_TESTS" = "true" ] || return 0
    log "running tests"
    uv run --no-sync python -m pytest -q >"$LOG_DIR/deploy-tests.log" 2>&1 && return 0
    log "tests failed — see logs/deploy-tests.log"
    return 1
}

pid_of() { cat "$RUN_DIR/$1.pid" 2>/dev/null; }

is_running() {
    local pid
    pid="$(pid_of "$1")"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

start() {
    # Output of the current run only; the application's own history is in logs/agent.log
    : >"$LOG_DIR/deploy-$1.log"
    set -m   # its own process group, so stopping the service stops its children too
    uv run --no-sync python -m "$(module_of "$1")" >>"$LOG_DIR/deploy-$1.log" 2>&1 &
    set +m
    echo $! >"$RUN_DIR/$1.pid"
    log "started $1 (pid $!)"
}

stop() {
    local pid waited=0
    pid="$(pid_of "$1")"
    rm -f "$RUN_DIR/$1.pid"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || return 0
    kill -TERM -- "-$pid" 2>/dev/null
    while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt "$STOP_TIMEOUT" ]; do
        sleep 1
        waited=$((waited + 1))
    done
    kill -KILL -- "-$pid" 2>/dev/null
    log "stopped $1"
}

start_all() { local name; for name in $SERVICES; do start "$name"; done; }
stop_all() { local name; for name in $SERVICES; do stop "$name"; done; }

restart_dead() {
    local name
    for name in $SERVICES; do
        is_running "$name" && continue
        log "$name is not running — restarting (its last output: logs/deploy-$name.log)"
        start "$name"
    done
}

main "$@"
exit
