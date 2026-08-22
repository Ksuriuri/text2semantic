#!/usr/bin/env bash
# Start (or restart) the dest WebUI from a pidfile. Do not `pkill -f webui.py`:
# that pattern also matches a remote ssh --command that happens to mention it.
#
# Usage, from the machine that hosts the checkpoint:
#   source /path/to/webui.env
#   bash scripts/deploy/start_webui.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PIDFILE="${WEBUI_PIDFILE:-$ROOT/webui.pid}"
LOG="${WEBUI_LOG:-$ROOT/webui.log}"

if [[ -f "$PIDFILE" ]]; then
  old="$(cat "$PIDFILE" || true)"
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null; then
    kill "$old"
    sleep 2
  fi
  rm -f "$PIDFILE"
fi

# shellcheck disable=SC1091
if [[ -f "${WEBUI_ENV:-}" ]]; then
  # explicit env file
  set -a
  # shellcheck disable=SC1090
  source "$WEBUI_ENV"
  set +a
fi

nohup bash "$ROOT/scripts/launch_webui.sh" \
  --host "${WEBUI_HOST:-0.0.0.0}" \
  --port "${WEBUI_PORT:-8080}" \
  >"$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "started webui pid $(cat "$PIDFILE") log $LOG"
