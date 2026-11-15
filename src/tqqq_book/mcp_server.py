"""Stdio MCP for the TQQQ book. No port. Nothing is ordered.

Same shape as the recovery-desk server: one JSON-RPC message per line.
"""

from __future__ import annotations

import json
import sys

from tqqq_book.desk import build_book, config_from_mapping, run
from tqqq_book.store import list_runs, load_path, load_summary

PROTOCOL = "2025-06-18"

CONFIG = {
    "type": "object",
    "properties": {
        "lots": {"type": "number", "description": "100-share lots. Default 100, which is 10,000 shares."},
        "spot": {"type": "number", "description": "TQQQ spot. Default 59.22."},
        "putDte": {"type": "number", "description": "Initial DTE of the protective puts. Alias of initialDte. Default 180. Independent of the path length."},
        "initialDte": {"type": "number", "description": "Initial DTE of the puts you buy. Same as putDte. Default 180."},
        "horizonDays": {"type": "number", "description": "Trading days to simulate. Default 100. The put is marked for whatever life is left."},
        "iv": {"type": "number", "description": "Opening implied vol. Default 0.71. Premiums use this, not the IBKR fill."},
        "vol": {"type": "number", "description": "TQQQ realized vol used for the path. Default 0.66."},
        "drift": {"type": "number", "description": "QQQ annual drift. Default 0.12."},
        "rate": {"type": "number", "description": "Risk-free rate. Default 0.04."},
        "otm": {"type": "array", "items": {"type": "number"}, "description": "1w, 2w, 3w, 4w call OTM as decimals. Default 0.03, 0.05, 0.08, 0.10."},
        "contracts": {"type": "array", "items": {"type": "number"}, "description": "Contracts in those four tenors at 100 lots. Default 12, 12, 12, 10."},
        "paths": {"type": "number", "description": "Monte Carlo paths. Default 400."},
        "seed": {"type": "number", "description": "Default 13."},
        "strategy": {"type": "string", "enum": ["wheel", "hold"]},
    },
    "additionalProperties": False,
}

TOOLS = [
    {
        "name": "tqqq_book",
        "description": "Price the opening book with this repo's analytic Black-Scholes. Returns each put and call premium, DTE, and how many days the put still has after the horizon. Does not run paths. Nothing is ordered.",
        "inputSchema": CONFIG,
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    },
    {
        "name": "tqqq_monte_carlo",
        "description": "Run the wheel or the hold on daily 3x TQQQ paths. Premiums are the engine prices from tqqq_book. Saves the run and returns each day of the median path: price, P&L, MTM, IV, and trades. Every path is in the saved paths file. Nothing is ordered.",
        "inputSchema": CONFIG,
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    },
    {
        "name": "tqqq_report",
        "description": "Run tqqq_monte_carlo, save it, and write desk.html in that run folder. The page has the median path day by day. Nothing is ordered.",
        "inputSchema": CONFIG,
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    },
    {
        "name": "tqqq_runs",
        "description": "List saved Monte Carlo runs, newest last. Nothing is ordered.",
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "number", "description": "How many runs to return. Default 20."}},
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    },
    {
        "name": "tqqq_run",
        "description": "Read one saved run. Pass id, or latest. Pass path to get that path's day-by-day price, P&L, MTM, and trades. Omit path to get the summary and the median path. Nothing is ordered.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Run id from tqqq_runs, or latest."},
                "path": {"type": "number", "description": "Path number inside that run. Omit for the summary."},
            },
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    },
]


def _send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _book_view(raw: dict) -> dict:
    cfg = config_from_mapping(raw)
    book = build_book(cfg)
    path_days = min(cfg.horizon_days, cfg.put_dte)
    legs = []
    for leg in book["legs"]:
        legs.append({
            "kind": leg.kind,
            "side": leg.side,
            "tag": leg.tag,
            "qty": leg.qty,
            "strike": leg.strike,
            "dte": leg.expiry,
            "premium": round(leg.entry, 4),
            "days_left_at_horizon": max(0, leg.expiry - path_days),
            "priced_by": "BlackScholesModel analytic",
        })
    return {
        "put_dte": cfg.put_dte,
        "initial_dte": cfg.put_dte,
        "horizon_days": cfg.horizon_days,
        "path_days": path_days,
        "put_days_left": max(0, cfg.put_dte - path_days),
        "lots": cfg.lots,
        "shares": book["shares"],
        "spot": cfg.spot,
        "iv": cfg.iv,
        "deployed": round(book["deployed"], 2),
        "put_premium": round(book["put_premium"], 2),
        "call_premium": round(book["call_premium"], 2),
        "floor": book["floor"],
        "model_gap_vs_fill": round(book["model_gap"], 2),
        "legs": legs,
        "note": "Premiums are the analytic Black-Scholes price at this spot, IV, and DTE. Not the IBKR fill.",
    }


def _call(name: str, args: dict) -> str:
    if name == "tqqq_book":
        return json.dumps(_book_view(args))
    if name == "tqqq_monte_carlo":
        result = run(config_from_mapping(args))
        return json.dumps(result)
    if name == "tqqq_report":
        result = run(config_from_mapping(args))
        shown = {key: result[key] for key in ("note", "config", "book", "monte_carlo", "vol", "saved", "median_path")}
        shown["ribbon_days"] = len(result["ribbon"])
        return json.dumps(shown)
    if name == "tqqq_runs":
        limit = int(args.get("limit") or 20)
        return json.dumps(list_runs(limit))
    if name == "tqqq_run":
        run_id = str(args.get("id") or "latest")
        if "path" in args and args["path"] is not None:
            return json.dumps(load_path(run_id, int(args["path"])))
        return json.dumps(load_summary(run_id))
    raise ValueError(f"Unknown tool {name}. Nothing was run.")


def _on_message(raw: str) -> None:
    try:
        message = json.loads(raw)
    except json.JSONDecodeError:
        _send({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
        return
    msg_id = message.get("id")
    if not isinstance(msg_id, (str, int)):
        return
    method = message.get("method")
    if method == "initialize":
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        requested = params.get("protocolVersion")
        _send({
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": requested if isinstance(requested, str) and requested else PROTOCOL,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "tqqq-book", "version": "1.0.0"},
                "instructions": "Read-only. tqqq_book prices the puts and calls. tqqq_monte_carlo runs the paths and saves every day. tqqq_run reads a saved run. Put initial DTE and horizon days are separate. Realized vol does not drift. A premium is the model, not a broker quote. Nothing is ordered.",
            },
        })
        return
    if method == "ping":
        _send({"jsonrpc": "2.0", "id": msg_id, "result": {}})
        return
    if method == "tools/list":
        _send({"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}})
        return
    if method == "tools/call":
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        name = params.get("name")
        args = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        try:
            _send({"jsonrpc": "2.0", "id": msg_id, "result": {"content": [{"type": "text", "text": _call(str(name), args)}], "isError": False}})
        except Exception as error:
            _send({"jsonrpc": "2.0", "id": msg_id, "result": {"content": [{"type": "text", "text": str(error)}], "isError": True}})
        return
    _send({"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Unknown method {method}"}})


def main() -> None:
    for line in sys.stdin:
        trimmed = line.strip()
        if trimmed:
            _on_message(trimmed)


if __name__ == "__main__":
    main()
