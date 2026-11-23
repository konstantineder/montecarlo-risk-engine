#!/bin/bash
# Idempotent. Installs, builds, and starts the desk. Restarts if it is already up.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PORT="${PORT:-3000}"

if [[ ! "$PORT" =~ ^[0-9]{1,5}$ ]]; then
  printf 'Invalid PORT: %s. Use a number from 1 to 65535.\n' "$PORT" >&2
  exit 2
fi
PORT=$((10#$PORT))
if ((PORT < 1 || PORT > 65535)); then
  printf 'Invalid PORT: %s. Use a number from 1 to 65535.\n' "$PORT" >&2
  exit 2
fi
if ! command -v lsof >/dev/null 2>&1; then
  printf 'lsof is required to find the process listening on port %s. Install lsof and rerun.\n' "$PORT" >&2
  exit 1
fi

if [ ! -d node_modules/@nestjs/core ] || [ ! -x node_modules/.bin/tsc ]; then
  npm ci
fi
# Build first so a failed build leaves the running desk available.
npm run build

listener_pids() {
  local found status
  if found="$(lsof -nP -t -iTCP:"$PORT" -sTCP:LISTEN)"; then
    printf '%s\n' "$found"
  else
    status=$?
    # lsof returns 1 when there are no matching listeners.
    if [ "$status" -ne 1 ]; then
      printf 'Could not inspect TCP port %s.\n' "$PORT" >&2
      return "$status"
    fi
  fi
}

wait_for_port() {
  local attempt found
  for ((attempt = 0; attempt < $1; attempt++)); do
    found="$(listener_pids)" || return "$?"
    if [ -z "$found" ]; then return 0; fi
    sleep 0.1
  done
  return 1
}

pids="$(listener_pids)"
if [ -n "$pids" ]; then
  printf 'Stopping listener(s) on TCP port %s.\n' "$PORT"
  for pid in $pids; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  if ! wait_for_port 50; then
    printf 'Port %s is still occupied; force-stopping its listener(s).\n' "$PORT"
    pids="$(listener_pids)"
    for pid in $pids; do
      kill -KILL "$pid" 2>/dev/null || true
    done
    if ! wait_for_port 20; then
      printf 'Cannot release TCP port %s. Check permission to stop its listener(s) and retry.\n' "$PORT" >&2
      exit 1
    fi
  fi
fi

export RUNS_DIR="${RUNS_DIR:-$ROOT/../runs}"
export PORT
exec node dist/main.js
