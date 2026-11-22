#!/bin/bash
# Idempotent. Installs, builds, and starts the desk. Restarts if it is already up.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PORT="${PORT:-3000}"

if [ ! -d node_modules/@nestjs/core ]; then
  npm install
fi
npm run build

if curl -sf -o /dev/null --max-time 2 "http://127.0.0.1:${PORT}/health"; then
  for p in /proc/[0-9]*; do
    tr "\0" " " 2>/dev/null < "$p/cmdline" | grep -q "[d]ist/main.js" && kill "$(basename "$p")" 2>/dev/null || true
  done
  for i in $(seq 1 20); do
    curl -sf -o /dev/null --max-time 1 "http://127.0.0.1:${PORT}/health" || break
    sleep 0.3
  done
fi

RUNS_DIR="${RUNS_DIR:-$ROOT/../runs}" PORT="$PORT" node dist/main.js
