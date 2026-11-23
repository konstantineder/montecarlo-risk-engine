"""Pool parameter configurations into one saved Monte Carlo simulation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from tqqq_book.desk import BookConfig, _median_path, _ribbon, _summary, config_from_mapping, run


def run_combined(configurations: list[dict], progress: Callable[[dict], None] | None = None) -> dict:
    if not configurations or len(configurations) > 1000:
        raise ValueError("A combined simulation needs between 1 and 1,000 configurations.")
    configs: list[BookConfig] = [config_from_mapping(raw) for raw in configurations]
    if any(cfg.paths < 1 or cfg.horizon_days < 1 for cfg in configs):
        raise ValueError("paths and horizon days must be at least 1")
    total = sum(cfg.paths for cfg in configs)
    max_days = max(cfg.horizon_days for cfg in configs)
    paths: list[dict] = []
    groups = []

    def notify(phase: str, percent: float, completed: int, **extra) -> None:
        if progress:
            progress({"phase": phase, "percent": round(percent, 2), "completed": completed, "total": total, **extra})

    notify("Preparing combined simulation", 0, 0)
    for index, cfg in enumerate(configs, 1):
        offset = len(paths)

        def forward(event: dict) -> None:
            percent = 90 * (offset + cfg.paths * event["percent"] / 100) / total
            notify(f"Configuration {index}/{len(configs)} · {event['phase']}", percent, offset + event["completed"])

        result = run(cfg, progress=forward, persist=False)
        component_paths = result.pop("_paths")
        for local, path in enumerate(component_paths):
            paths.append({**path, "path": offset + local, "configuration": index, "configuration_path": local, "config": result["config"]})
        groups.append({"id": index, **{key: result[key] for key in ("config", "book", "monte_carlo", "vol", "rules")}})
        notify(f"Configuration {index}/{len(configs)} completed", 90 * len(paths) / total, len(paths))

    notify("Combining path distributions", 92, total)
    spots = np.full((total, max_days + 1), np.nan)
    pnls = np.full_like(spots, np.nan)
    mtms = np.full_like(spots, np.nan)
    trades = np.zeros_like(spots, dtype=int)
    for index, path in enumerate(paths):
        days = path["days"]
        spots[index, :len(days)] = [day["price"] for day in days]
        pnls[index, :len(days)] = [day["pnl"] for day in days]
        mtms[index, :len(days)] = [day["mtm"] for day in days]
        trades[index, :len(days)] = [len(day["trades"]) for day in days]
    ribbon = _ribbon(spots, pnls, trades)
    for day, row in enumerate(ribbon):
        for quantile, label in ((0.05, "p05"), (0.50, "p50"), (0.95, "p95")):
            row[f"mtm_{label}"] = float(np.nanquantile(mtms[:, day], quantile))

    # Dollar book figures describe the average starting book of a pooled path.
    book = {
        key: sum(group["book"][key] * group["config"]["paths"] for group in groups) / total
        for key in ("deployed", "put_premium", "call_premium", "floor", "model_gap_vs_fill")
    }
    book.update({"puts": [], "calls": [], "opening_trades": [], "basis": "path-weighted average"})
    first = groups[0]["config"]
    combined = {
        "note": (
            f"One combined simulation with {len(configs)} configurations and {total} paths. "
            "Each configuration contributes its requested paths to the pooled P&L distribution; P&Ls are observations, not added together. "
            "Each path retains its own configuration. Equal seeds reuse the same market shocks for comparisons. "
            "Final P&L uses each path's own horizon; daily bands include only paths still running. "
            "Opening book figures are path-weighted averages; individual books and rules are listed by configuration."
        ),
        "config": {
            **first, "mode": "combined", "configuration_count": len(configs), "paths": total,
            "horizon_days": max_days, "path_days": max_days,
            "horizon_range": [min(cfg.horizon_days for cfg in configs), max_days],
            "reference_configuration": 1,
        },
        "configurations": groups,
        "book": book,
        "monte_carlo": _summary("combined", paths),
        "vol": {"note": "Volatility and IV settings are stored separately for each configuration."},
        "rules": [],
        "ribbon": ribbon,
        "median_path": _median_path(paths),
    }
    from tqqq_book.report import write_report
    from tqqq_book.store import save_run
    notify("Saving combined paths", 94, total)
    combined["saved"] = save_run(combined, paths)
    folder = Path(combined["saved"]["dir"])
    notify("Writing combined report", 96, total)
    combined["saved"]["html"] = write_report(combined, folder / "desk.html")
    notify("Finishing combined report", 98, total)
    write_report(combined, folder.parent / "tqqq-desk.html")
    notify("Completed", 100, total, run=combined["saved"]["id"])
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    payload = json.load(sys.stdin)

    def emit(event: dict) -> None:
        print("TQQQ_PROGRESS " + json.dumps(event), flush=True)

    result = run_combined(payload["configurations"], progress=emit if args.progress else None)
    print(f"saved {result['saved']['id']} · {result['config']['configuration_count']} configurations · {result['config']['paths']} paths")


if __name__ == "__main__":
    main()
