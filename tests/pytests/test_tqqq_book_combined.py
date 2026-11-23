import csv
import io
import json
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

import context
import numpy as np

from tqqq_book import store
from tqqq_book.combined import run_combined


def test_combined_run_pools_paths_once_and_retains_each_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "RUNS", tmp_path)
    events = []
    result = run_combined([
        {"paths": 2, "horizon": 3, "lots": 100, "puts": "100:60:30", "otm": "0.03,0.05,0.08,0.10"},
        {"paths": 3, "horizon": 5, "lots": 50, "puts": "100:90:45", "otm": "0.08,0.05,0.08,0.10"},
    ], progress=events.append)
    folders = [folder for folder in tmp_path.iterdir() if folder.is_dir()]
    assert len(folders) == 1
    folder = folders[0]
    summary = json.loads((folder / "summary.json").read_text())
    paths = [json.loads(line) for line in (folder / "paths.jsonl").read_text().splitlines()]
    assert [path["path"] for path in paths] == list(range(5))
    assert Counter(path["configuration"] for path in paths) == {1: 2, 2: 3}
    assert [len(path["days"]) for path in paths] == [4, 4, 6, 6, 6]
    assert paths[0]["config"]["put_sleeves"][0]["dte"] == 60
    assert paths[-1]["config"]["put_sleeves"][0]["dte"] == 90
    assert paths[-1]["config"]["lots"] == 50
    assert summary["config"]["mode"] == "combined"
    assert summary["config"]["paths"] == summary["monte_carlo"]["paths"] == 5
    assert len(summary["configurations"]) == 2
    pnl = [path["pnl"] for path in paths]
    assert result["monte_carlo"]["mean"] == np.mean(pnl)
    assert result["monte_carlo"]["p05"] == np.quantile(pnl, 0.05)
    assert result["monte_carlo"]["prob_loss"] == np.mean(np.array(pnl) < 0)
    assert result["ribbon"][0]["active_paths"] == 5
    assert result["ribbon"][4]["active_paths"] == 3
    own_mtms = [path["days"][0]["mtm"] for path in paths]
    assert result["ribbon"][0]["mtm_p50"] == np.median(own_mtms)
    weighted_deployed = sum(group["book"]["deployed"] * group["config"]["paths"] for group in summary["configurations"]) / 5
    assert result["book"]["deployed"] == weighted_deployed
    assert events[-1]["completed"] == events[-1]["total"] == 5
    assert events[-1]["run"] == folder.name
    assert events[-1]["percent"] == 100
    assert [event["percent"] for event in events] == sorted(event["percent"] for event in events)
    assert (folder / "desk.html").is_file()
    assert "Combined simulation" in (folder / "desk.html").read_text()

    exporter = Path(__file__).resolve().parents[2] / "desk-app" / "export_run.py"
    archive = subprocess.run([sys.executable, str(exporter), str(folder)], check=True, capture_output=True).stdout
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        exported = list(csv.DictReader(io.StringIO(zipped.read("paths.csv").decode())))
        assert Counter(path["configuration"] for path in exported) == {"1": 2, "2": 3}
        assert "configurations.csv" in zipped.namelist()


def test_six_configurations_with_one_hundred_paths_save_one_simulation(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "RUNS", tmp_path)
    result = run_combined([
        {"paths": 100, "horizon": 1, "otm": f"{otm},0.05,0.08,0.10"}
        for otm in (0.03, 0.04, 0.05, 0.06, 0.07, 0.08)
    ])
    assert result["config"]["configuration_count"] == 6
    assert result["monte_carlo"]["paths"] == 600
    assert len(store.list_runs()) == 1
    paths = [json.loads(line) for line in Path(result["saved"]["paths"]).read_text().splitlines()]
    assert len(paths) == 600
    assert len({path["path"] for path in paths}) == 600
    assert Counter(path["configuration"] for path in paths) == {i: 100 for i in range(1, 7)}
