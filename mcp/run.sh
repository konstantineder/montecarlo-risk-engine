#!/bin/sh
# Stdio MCP. No port. Codex and ChatGPT Desktop start this file.
# First time, and after a pull, from the repo root:
#   git pull
#   python3.12 -m venv .venv
#   .venv/bin/python -m pip install -r requirements.txt
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT" || exit 1
PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "tqqq-book: $PY is missing. From the repo root run:" >&2
  echo "  git pull" >&2
  echo "  python3.12 -m venv .venv" >&2
  echo "  .venv/bin/python -m pip install -r requirements.txt" >&2
  exit 1
fi
if ! "$PY" -c "import numpy" >/dev/null 2>&1; then
  echo "tqqq-book: numpy is not installed in $PY. From the repo root run:" >&2
  echo "  git pull" >&2
  echo "  .venv/bin/python -m pip install -r requirements.txt" >&2
  exit 1
fi
exec "$PY" -m tqqq_book.mcp_server
