#!/bin/bash
# Idempotent. Installs, builds, and starts the desk. Safe to run again.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PORT="${PORT:-3000}"

if [ ! -d node_modules/@nestjs/core ]; then
  npm install
fi
npm run build

if curl -sf -o /dev/null --max-time 2 "http://127.0.0.1:${PORT}/health"; then
  echo "TQQQ desk already running at http://127.0.0.1:${PORT}"
  exit 0
fi

RUNS_DIR="${RUNS_DIR:-$ROOT/../runs}" PORT="$PORT" node dist/main.js
