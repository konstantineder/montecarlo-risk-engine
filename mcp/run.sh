#!/bin/sh
# Stdio MCP. No port. ChatGPT Desktop and Codex start this file.
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT" || exit 1
exec python3 -m tqqq_book.mcp_server
