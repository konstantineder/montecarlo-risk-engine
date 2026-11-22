"""Write one simulation as a zip of CSVs on stdout."""
import csv
import io
import json
import sys
import zipfile
from pathlib import Path


def cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return value


def rows(header, data):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    writer.writerows(data)
    return buf.getvalue()


def main() -> None:
    folder = Path(sys.argv[1])
    summary = json.loads((folder / "summary.json").read_text())
    summary_rows = []
    for key, value in (summary.get("config") or {}).items():
        summary_rows.append(("config", key, json.dumps(value) if isinstance(value, (dict, list)) else value))
    for key, value in (summary.get("monte_carlo") or {}).items():
        summary_rows.append(("monte_carlo", key, value))
    for key, value in (summary.get("book") or {}).items():
        if isinstance(value, (dict, list)):
            continue
        summary_rows.append(("book", key, value))
    for rule in summary.get("rules") or []:
        summary_rows.append(("rule", rule.get("name"), json.dumps(rule)))

    paths, days, trades, positions, rule_rows = [], [], [], [], []
    with (folder / "paths.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            path = json.loads(line)
            index = path.get("path")
            trade_count = sum(len(day.get("trades") or []) for day in path.get("days") or [])
            paths.append([
                index, path.get("pnl"), path.get("ending_spot"), path.get("min_spot"),
                path.get("max_dd"), path.get("ending_shares"), path.get("assignments"),
                path.get("new_calls"), path.get("buybacks"), path.get("roll_cash"),
                path.get("floor_after"), trade_count,
            ])
            for day in path.get("days") or []:
                day_trades = day.get("trades") or []
                acted = ",".join(row.get("name", "") for row in day.get("rules") or [] if row.get("acted"))
                days.append([
                    index, day.get("day"), day.get("price"), day.get("pnl"), day.get("mtm"),
                    day.get("iv"), day.get("shares"), len(day_trades), acted,
                ])
                for trade in day_trades:
                    trades.append([
                        index, day.get("day"), trade.get("rule") or trade.get("tag") or "",
                        trade.get("action"), trade.get("kind"), trade.get("qty"),
                        trade.get("strike"), trade.get("premium"), trade.get("cash"),
                        trade.get("tag"), trade.get("reason"), trade.get("expiry"),
                    ])
                for leg in day.get("position") or []:
                    positions.append([
                        index, day.get("day"), leg.get("name"), leg.get("side"), leg.get("qty"),
                        leg.get("strike"), leg.get("expiry"), leg.get("cost"), leg.get("mark"),
                        leg.get("mtm"), leg.get("tag"),
                    ])
                for rule in day.get("rules") or []:
                    rule_rows.append([index, day.get("day"), rule.get("name"), cell(rule.get("acted"))])

    out = zipfile.ZipFile(sys.stdout.buffer, "w", zipfile.ZIP_DEFLATED)
    out.writestr("summary.csv", rows(["section", "key", "value"], summary_rows))
    out.writestr("paths.csv", rows(
        ["path", "pnl", "ending_spot", "min_spot", "max_dd", "ending_shares",
         "assignments", "new_calls", "buybacks", "roll_cash", "floor_after", "trades"],
        paths,
    ))
    out.writestr("days.csv", rows(
        ["path", "day", "spot", "pnl", "mtm", "iv", "shares", "trades", "rules_acted"],
        days,
    ))
    out.writestr("trades.csv", rows(
        ["path", "day", "rule", "action", "kind", "qty", "strike", "premium", "cash", "tag", "reason", "expiry"],
        trades,
    ))
    out.writestr("positions.csv", rows(
        ["path", "day", "name", "side", "qty", "strike", "expiry", "cost", "mark", "mtm", "tag"],
        positions,
    ))
    out.writestr("rules.csv", rows(["path", "day", "rule", "acted"], rule_rows))
    out.close()


if __name__ == "__main__":
    main()
