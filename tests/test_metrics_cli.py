"""`ask-or-act metrics` over saved runs (created locally from fakes)."""

import asyncio
import json
from pathlib import Path

import pytest

from ask_or_act.cli import main
from ask_or_act.core.runs import save_run
from tests.helpers import fake, make_examples


def make_run(tmp_path: Path, name: str, kind: str, seed: int = 1, n: int = 200) -> Path:
    examples = make_examples(n, k=3)
    decisions = asyncio.run(fake(kind, seed=seed).decide_batch(examples))
    return save_run(tmp_path, name, {"dataset": {"name": "toy"}}, decisions, examples)


def test_single_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = make_run(tmp_path, "r1", "calibrated")
    assert main(["metrics", str(run), "--n-boot", "100"]) == 0
    out = capsys.readouterr().out
    assert "ece_width" in out
    assert "feasibility" in out
    data = json.loads((run / "metrics.json").read_text())
    assert data["report"]["n"] == 200
    assert data["report"]["settings"]["n_boot"] == 100


def test_two_runs_compare(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a = make_run(tmp_path, "a", "calibrated")
    b = make_run(tmp_path, "b", "overconfident")
    out_file = tmp_path / "cmp.json"
    assert main(["metrics", str(a), str(b), "--n-boot", "100", "--out", str(out_file)]) == 0
    assert "accuracy-controlled calibration" in capsys.readouterr().out
    data = json.loads(out_file.read_text())
    assert data["comparison"]["mcnemar"]["p_value"] == 1.0
    assert "DA" in data["comparison"]["ace"]["views"]


def test_mismatched_runs_are_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a = make_run(tmp_path, "a", "calibrated", n=200)
    b = make_run(tmp_path, "b", "calibrated", n=150)
    assert main(["metrics", str(a), str(b), "--n-boot", "50"]) == 2
    assert "different examples" in capsys.readouterr().err


def test_too_many_runs(tmp_path: Path) -> None:
    assert main(["metrics", "x", "y", "z"]) == 2


def test_missing_run(tmp_path: Path) -> None:
    # A path that doesn't exist is read as a predictions file: bad input, exit 6
    # (docs/cli-design.md §5.1, approved 2026-11-22).
    assert main(["metrics", str(tmp_path / "nope")]) == 6


def test_signal_option_scores_a_recorded_signal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from dataclasses import replace

    from ask_or_act.core.runs import load_run, save_run

    run = make_run(tmp_path, "r", "calibrated")
    rec = load_run(run)
    # Give half the decisions an extra signal that is always 1.0 (maximally overconfident).
    decisions = [
        replace(d, signals={**d.signals, "verbalized": 1.0}) if i % 2 else d
        for i, d in enumerate(rec.decisions)
    ]
    run2 = save_run(tmp_path, "r2", rec.metadata, decisions, rec.examples)
    assert main(["metrics", str(run2), "--n-boot", "50", "--signal", "verbalized"]) == 0
    out = capsys.readouterr().out
    assert "confidence = verbalized" in out
    assert "100 decision(s) lack signal 'verbalized'" in out
