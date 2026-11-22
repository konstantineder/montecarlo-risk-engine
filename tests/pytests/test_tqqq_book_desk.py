import json
import sys

import context
import pytest

from tqqq_book import desk, store


@pytest.mark.parametrize("key", ["putSleeves", "put_sleeves"])
def test_config_parses_custom_put_sleeves(key):
    cfg = desk.config_from_mapping({key: "25:30:10,75:120:60"})

    assert cfg.put_sleeves == (
        desk.PutSleeve(0.25, 30, 10, "d30"),
        desk.PutSleeve(0.75, 120, 60, "d120"),
    )


@pytest.mark.parametrize(
    "sleeves, message",
    [
        ("100:60", "Each put sleeve is percent:dte:roll"),
        ("0:60:30,100:90:45", "Bad put sleeve"),
        ("100:0:1", "Bad put sleeve"),
        ("100:60:60", "Bad put sleeve"),
        ("50:60:30", "Put sleeve percents must sum to 100"),
    ],
)
def test_config_rejects_invalid_put_sleeves(sleeves, message):
    with pytest.raises(ValueError, match=message):
        desk.config_from_mapping({"put_sleeves": sleeves})


def test_simulation_command_saves_run_with_put_sleeves(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "RUNS", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tqqq_book",
            "--lots", "100",
            "--horizon", "3",
            "--paths", "2",
            "--seed", "13",
            "--spot", "59.22",
            "--iv", "0.71",
            "--strategy", "wheel",
            "--puts", "10:60:30,40:90:45,50:180:90",
            "--otm", "0.03,0.09,0.10,0.13",
            "--contracts", "12,10,8,8",
        ],
    )

    desk.main()

    latest = json.loads((tmp_path / "latest.json").read_text())
    folder = tmp_path / latest["id"]
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["config"]["put_sleeves"] == [
        {"weight": 0.10, "dte": 60, "roll_at": 30, "tag": "d60"},
        {"weight": 0.40, "dte": 90, "roll_at": 45, "tag": "d90"},
        {"weight": 0.50, "dte": 180, "roll_at": 90, "tag": "d180"},
    ]
    assert [tenor["contracts"] for tenor in summary["config"]["tenors"]] == [12, 10, 8, 8]
    paths = [json.loads(line) for line in (folder / "paths.jsonl").read_text().splitlines()]
    assert len(paths) == 2
    assert all(len(path["days"]) == 4 for path in paths)
    assert (folder / "desk.html").is_file()
