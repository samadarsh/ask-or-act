"""``ask-or-act demo`` (docs/cli-design.md §3.7). No network.

The demo must work from nothing in under 30 seconds, never overwrite a folder, and
show what it promises: an overconfident model (a calibration warning beyond noise), a
real act/ask split, and the same summaries ``thresholds`` and ``metrics`` print.
"""

import csv
import subprocess
import sys
import time
from pathlib import Path

import pytest

import ask_or_act as aoa
from ask_or_act import demo
from ask_or_act.cli import main


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("NO_COLOR", "FORCE_COLOR", "TERM"):
        monkeypatch.delenv(name, raising=False)


def _words(text: str) -> str:
    return " ".join(text.split())


def test_demo_writes_data_and_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    folder = tmp_path / "ask-or-act-demo"
    data, report = folder / demo.FILE_NAME, folder / demo.REPORT_NAME
    assert sorted(p.name for p in folder.iterdir()) == [demo.REPORT_NAME, demo.FILE_NAME]
    assert "saved: ask-or-act-demo/report.html" in out
    page = report.read_text(encoding="utf-8")
    assert demo.TITLE in page
    assert 'class="badge target">TARGET' in page
    assert page.count('class="mark"') == 1  # the chosen threshold, on the curve
    assert "of 500 unlabelled rows have confidence" in page
    assert "ask-or-act check ask-or-act-demo/support-tickets.csv" in out
    with data.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == demo.N_LABELLED + demo.N_NEW
    assert list(rows[0]) == ["id", "prediction", "confidence", "human_label"]  # no text
    assert sum(r["human_label"] == "" for r in rows) == demo.N_NEW


def test_the_summaries_are_what_thresholds_and_metrics_print(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["demo"]) == 0
    out = _words(capsys.readouterr().out)
    path = "ask-or-act-demo/support-tickets.csv"
    for argv in (["thresholds", path], ["metrics", path]):
        main(argv)
        printed = capsys.readouterr().out
        assert _words(printed) in out, argv


def test_the_model_is_overconfident_with_a_real_act_ask_split() -> None:
    folder = demo.new_folder(Path("d"))
    path = demo.write_data(folder)
    th = aoa.find_thresholds(path)
    assert th.threshold is not None
    assert 0.4 < th.test_coverage < 0.8  # acts on part, asks a human for the rest
    ev = aoa.evaluate(path, n_boot=200)
    text = ev.summary()
    assert "overstates accuracy by about 8 points" in text
    assert "so this is beyond noise" in text


def test_the_data_is_the_same_every_time() -> None:
    assert demo.rows() == demo.rows()
    assert demo.rows(1) != demo.rows()


def test_an_existing_folder_is_never_overwritten(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mine = tmp_path / "ask-or-act-demo"
    mine.mkdir()
    (mine / "support-tickets.csv").write_text("my own file\n")
    assert main(["demo", "-q"]) == 0
    assert main(["demo", "-q"]) == 0
    out = capsys.readouterr().out
    assert (mine / "support-tickets.csv").read_text() == "my own file\n"
    assert (tmp_path / "ask-or-act-demo-2" / "report.html").is_file()
    assert (tmp_path / "ask-or-act-demo-3" / "report.html").is_file()
    assert out.splitlines()[1] == (
        "saved: ask-or-act-demo-2/report.html "
        "(1 model, 2,000 labelled rows; opens offline, no external files)"
    )


def test_dir_option(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["demo", "--dir", "nested/try", "-q"]) == 0
    assert (tmp_path / "nested" / "try" / "report.html").is_file()
    assert "nested/try/support-tickets.csv" in capsys.readouterr().out


def test_demo_runs_in_under_30_seconds(tmp_path: Path) -> None:
    """The whole command as a user runs it: a new process, imports included."""
    start = time.perf_counter()
    r = subprocess.run(
        [sys.executable, "-m", "ask_or_act", "demo", "-q"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    elapsed = time.perf_counter() - start
    assert r.returncode == 0, r.stderr
    assert elapsed < 30, f"demo took {elapsed:.1f} s"
