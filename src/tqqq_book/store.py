"""Persist each Monte Carlo run under runs/. Nothing is ordered."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

RUNS = Path(os.environ.get("RUNS_DIR") or Path(__file__).resolve().parents[2] / "runs").expanduser().resolve()


def save_run(result: dict, paths: list[dict]) -> dict:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = RUNS / f"{stamp}-s{result['config']['seed']}"
    extra = 2
    while folder.exists():
        folder = RUNS / f"{stamp}-s{result['config']['seed']}-{extra}"
        extra += 1
    folder.mkdir(parents=True)
    paths_file = folder / "paths.jsonl"
    with paths_file.open("w") as handle:
        for row in paths:
            handle.write(json.dumps(row) + "\n")
    summary = {
        "id": folder.name,
        "created": stamp,
        "note": result["note"],
        "config": result["config"],
        **({"configurations": result["configurations"]} if "configurations" in result else {}),
        "book": result["book"],
        "monte_carlo": result["monte_carlo"],
        "vol": result["vol"],
        "rules": result.get("rules") or [],
        "ribbon": result["ribbon"],
        "median_path": result["median_path"],
    }
    summary_file = folder / "summary.json"
    summary_file.write_text(json.dumps(summary))
    listed = {
        "id": folder.name,
        "created": stamp,
        "initial_dte": result["config"]["initial_dte"],
        "horizon_days": result["config"]["horizon_days"],
        "paths": result["config"]["paths"],
        "seed": result["config"]["seed"],
        "strategy": result["config"]["strategy"],
        "median": result["monte_carlo"]["median"],
        "dir": str(folder),
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    with (RUNS / "index.jsonl").open("a") as handle:
        handle.write(json.dumps(listed) + "\n")
    (RUNS / "latest.json").write_text(json.dumps(listed))
    return {
        "id": folder.name,
        "dir": str(folder),
        "summary": str(summary_file),
        "paths": str(paths_file),
    }


def list_runs(limit: int = 20) -> list[dict]:
    index = RUNS / "index.jsonl"
    if not index.exists():
        return []
    rows = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
    return rows[-limit:]


def _folder(run_id: str) -> Path:
    if run_id in ("", "latest"):
        pointer = RUNS / "latest.json"
        if not pointer.exists():
            raise ValueError("No saved run yet.")
        run_id = json.loads(pointer.read_text())["id"]
    folder = RUNS / run_id
    if not (folder / "summary.json").exists():
        raise ValueError(f"No saved run {run_id}.")
    return folder


def load_summary(run_id: str) -> dict:
    return json.loads((_folder(run_id) / "summary.json").read_text())


def folder_for(run_id: str = "latest") -> Path:
    return _folder(run_id)


def load_saved(folder: Path) -> dict:
    result = json.loads((folder / "summary.json").read_text())
    result["saved"] = {"paths": str(folder / "paths.jsonl"), "dir": str(folder), "id": folder.name}
    return result


def load_path(run_id: str, path_index: int) -> dict:
    with (_folder(run_id) / "paths.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["path"] == path_index:
                return row
    raise ValueError(f"Path {path_index} is not in this run.")
