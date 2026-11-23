"""Pool parameter configurations or saved results into one simulation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

import numpy as np

from tqqq_book.desk import _median_path, _ribbon, _summary, config_from_mapping, run


def _notify(progress, total, phase, percent, completed, **extra):
    if progress:
        progress({"phase": phase, "percent": round(percent, 2), "completed": completed, "total": total, **extra})


class _CombinedPaths:
    """Keep plot data in memory and stream full trade histories to disk."""

    def __init__(self, folder: Path, total: int, max_days: int):
        self.total = total
        self.max_days = max_days
        from tqqq_book.checkpoint import engine_folder, root
        checkpoint = engine_folder(root())
        self.source = (checkpoint / "combined.paths.jsonl") if checkpoint else folder / "paths.jsonl"
        self.source.parent.mkdir(parents=True, exist_ok=True)
        self.keys = set()
        self.handle = None
        self.rows = []
        self.spots = np.full((total, max_days + 1), np.nan)
        self.pnls = np.full_like(self.spots, np.nan)
        self.mtms = np.full_like(self.spots, np.nan)
        self.trades = np.zeros_like(self.spots, dtype=int)
        # Recover complete JSONL records and discard an interrupted final write.
        if checkpoint and self.source.exists():
            with self.source.open("rb+") as handle:
                valid = 0
                while line := handle.readline():
                    from tqqq_book.checkpoint import check_pause
                    check_pause()
                    try:
                        row = json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        handle.truncate(valid)
                        break
                    self._remember(row)
                    valid = handle.tell()
        self.handle = self.source.open("a" if checkpoint else "w")

    def _remember(self, row):
        index = len(self.rows)
        days = row["days"]
        if index >= self.total or not days or len(days) > self.max_days + 1:
            raise ValueError("Checkpoint path count or horizon does not match its configuration")
        self.keys.add((row["configuration"], row["configuration_path"]))
        self.rows.append({key: value for key, value in row.items() if key not in ("days", "config", "prices", "nav_values")})
        self.spots[index, :len(days)] = [day["price"] for day in days]
        self.pnls[index, :len(days)] = [day["pnl"] for day in days]
        self.mtms[index, :len(days)] = [day["mtm"] for day in days]
        self.trades[index, :len(days)] = [len(day["trades"]) for day in days]

    def add(self, path: dict, group: dict, local: int):
        from tqqq_book.checkpoint import check_pause
        check_pause()
        if (group["id"], local) in self.keys:
            return
        row = {**path, "path": len(self.rows), "configuration": group["id"], "configuration_path": local, "config": path.get("config", group["config"])}
        if "source_run" in group:
            row.update(source_run=group["source_run"], source_path=path["path"])
        self.handle.write(json.dumps(row) + "\n")
        self.handle.flush()
        self._remember(row)

    def close(self):
        if self.handle:
            self.handle.close()

    def save(self, groups: list[dict], progress=None, analysis: dict | None = None) -> dict:
        self.close()
        if len(self.rows) != self.total:
            raise ValueError("Saved path count does not match the declared number of paths.")
        notify = lambda phase, percent, **extra: _notify(progress, self.total, phase, percent, self.total, **extra)
        notify("Combining path distributions", 92)
        ribbon = _ribbon(self.spots, self.pnls, self.trades)
        for day, row in enumerate(ribbon):
            for quantile, label in ((0.05, "p05"), (0.50, "p50"), (0.95, "p95")):
                row[f"mtm_{label}"] = float(np.nanquantile(self.mtms[:, day], quantile))
        book = {
            key: sum(group["book"][key] * group["config"]["paths"] for group in groups) / self.total
            for key in ("deployed", "put_premium", "call_premium", "floor", "model_gap_vs_fill")
        }
        book.update({"puts": [], "calls": [], "opening_trades": [], "basis": "path-weighted average"})
        target = float(np.median([row["pnl"] for row in self.rows]))
        median_index = min(self.rows, key=lambda row: abs(row["pnl"] - target))["path"]
        with self.source.open() as handle:
            median = next(json.loads(line) for index, line in enumerate(handle) if index == median_index)
        combined = {
            "note": (
                f"One combined simulation with {len(groups)} configurations and {self.total} paths. "
                "Each configuration contributes its requested paths to the pooled P&L distribution; P&Ls are observations, not added together. "
                "Each path retains its own configuration. Equal seeds reuse the same market shocks for comparisons. "
                "Final P&L uses each path's own horizon; daily bands include only paths still running. "
                "Opening book figures are path-weighted averages; individual books and rules are listed by configuration."
            ),
            "config": {
                **groups[0]["config"], "mode": "combined", "configuration_count": len(groups), "paths": self.total,
                "horizon_days": self.max_days, "path_days": self.max_days,
                "horizon_range": [min(group["config"]["horizon_days"] for group in groups), self.max_days],
                "reference_configuration": 1,
            },
            "configurations": groups,
            "book": book,
            "monte_carlo": _summary("combined", self.rows),
            "vol": {"note": "Volatility and IV settings are stored separately for each configuration."},
            "rules": [], "ribbon": ribbon, "median_path": _median_path([median]),
        }
        if analysis:
            combined["analysis"] = analysis
        from tqqq_book.report import write_report
        from tqqq_book.store import save_run
        notify("Saving combined paths", 94)
        combined["saved"] = save_run(combined, self.source)
        folder = Path(combined["saved"]["dir"])
        notify("Writing combined report", 96)
        # Full histories remain on disk and in the app; the standalone report previews the median.
        combined["saved"]["html"] = write_report(combined, folder / "desk.html", preview_paths=[median])
        notify("Finishing combined report", 98)
        write_report(combined, folder.parent / "tqqq-desk.html", preview_paths=[median])
        notify("Completed", 100, run=combined["saved"]["id"])
        return combined


def run_combined(configurations: list[dict], progress: Callable[[dict], None] | None = None) -> dict:
    if not configurations or len(configurations) > 1000:
        raise ValueError("A combined simulation needs between 1 and 1,000 configurations.")
    configs = [config_from_mapping(raw) for raw in configurations]
    if any(cfg.paths < 1 or cfg.horizon_days < 1 for cfg in configs):
        raise ValueError("paths and horizon days must be at least 1")
    total = sum(cfg.paths for cfg in configs)
    _notify(progress, total, "Preparing combined simulation", 0, 0)
    with TemporaryDirectory(prefix="tqqq-combined-") as temp:
        writer = _CombinedPaths(Path(temp), total, max(cfg.horizon_days for cfg in configs))
        groups = []
        try:
            for index, cfg in enumerate(configs, 1):
                offset = sum(item.paths for item in configs[:index-1])

                def forward(event):
                    percent = 90 * (offset + cfg.paths * event["percent"] / 100) / total
                    _notify(progress, total, f"Configuration {index}/{len(configs)} · {event['phase']}", max(percent, 90 * len(writer.rows) / total), max(len(writer.rows), offset + event["completed"]))

                from tqqq_book.checkpoint import root
                checkpoint = root()
                result = run(cfg, progress=forward, persist=False, checkpoint_dir=checkpoint / f"configuration-{index}" if checkpoint else None)
                group = {"id": index, **{key: result[key] for key in ("config", "book", "monte_carlo", "vol", "rules")}}
                groups.append(group)
                for local, path in enumerate(result["_paths"]):
                    writer.add(path, group, local)
                del result, path
                _notify(progress, total, f"Configuration {index}/{len(configs)} completed", 90 * len(writer.rows) / total, len(writer.rows))
            return writer.save(groups, progress)
        finally:
            writer.close()


def combine_saved_runs(run_ids: list[str], progress: Callable[[dict], None] | None = None) -> dict:
    from tqqq_book.store import folder_for
    if not run_ids or len(run_ids) > 1000 or len(set(run_ids)) != len(run_ids):
        raise ValueError("Choose between 1 and 1,000 distinct saved runs.")
    if any(not isinstance(run_id, str) or not re.fullmatch(r"[\w-]+", run_id) for run_id in run_ids):
        raise ValueError("Invalid saved run ID.")
    sources = []
    groups = []
    for index, run_id in enumerate(run_ids, 1):
        folder = folder_for(run_id)
        summary = json.loads((folder / "summary.json").read_text())
        if summary["config"].get("mode") == "combined":
            raise ValueError("Select separate simulation results to combine.")
        groups.append({"id": index, "source_run": run_id, **{key: summary[key] for key in ("config", "book", "monte_carlo", "vol", "rules")}})
        sources.append(folder / "paths.jsonl")
    total = sum(group["config"]["paths"] for group in groups)
    _notify(progress, total, "Loading saved results", 0, 0)
    with TemporaryDirectory(prefix="tqqq-combine-saved-") as temp:
        writer = _CombinedPaths(Path(temp), total, max(group["config"]["horizon_days"] for group in groups))
        try:
            for group, source in zip(groups, sources):
                count = 0
                with source.open() as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        writer.add(json.loads(line), group, count)
                        count += 1
                        _notify(progress, total, f"Combining saved result {group['id']}/{len(groups)}", 90 * len(writer.rows) / total, len(writer.rows))
                if count != group["config"]["paths"]:
                    raise ValueError("Saved path count does not match its configuration.")
            return writer.save(groups, progress)
        finally:
            writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--saved-runs", action="store_true")
    args = parser.parse_args()
    payload = json.load(sys.stdin)

    def emit(event: dict) -> None:
        print("TQQQ_PROGRESS " + json.dumps(event), flush=True)

    operation = combine_saved_runs if args.saved_runs else run_combined
    result = operation(payload["runs" if args.saved_runs else "configurations"], progress=emit if args.progress else None)
    print(f"saved {result['saved']['id']} · {result['config']['configuration_count']} configurations · {result['config']['paths']} paths")


if __name__ == "__main__":
    from tqqq_book.checkpoint import guarded
    guarded(main)
