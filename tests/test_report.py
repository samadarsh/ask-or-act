"""`ask-or-act report`: one static HTML file from saved runs. Fakes only, no network."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pytest

from ask_or_act.cli import main
from ask_or_act.core.runs import save_run
from ask_or_act.core.types import Example
from ask_or_act.data.sampling import Split
from ask_or_act.metrics.report import Settings
from ask_or_act.report import build as report
from ask_or_act.report import svg
from tests.helpers import fake
from tests.metrics_helpers import sample_examples

SECRET = "PLANTED-INPUT-TEXT"
FAST = Settings(n_boot=30)


def _examples(split: Split, n: int) -> list[Example]:
    return [
        Example(e.id, f"{SECRET} {e.id}", e.options, e.gold, meta=dict(e.meta))
        for e in sample_examples("banking77", split)[:n]
    ]


def _save(
    runs: Path, name: str, kind: str, split: Split, *, n: int = 200, alpha: float | None = 0.1
) -> Path:
    examples = _examples(split, n)
    decider = fake(kind)
    decisions = asyncio.run(decider.decide_batch(examples))
    contents: dict[str, Any] = {"name": name}
    if alpha is not None:
        contents["guarantee"] = {"alpha": alpha, "delta": 0.1, "registered": "2026-11-21"}
    meta: dict[str, Any] = {
        "name": name,
        "status": "completed",
        "dataset": {"name": "banking77", "split": split.value, "limit": n},
        "decider": decider.config.to_json(),
        "config": {"path": "configs/none.toml", "contents": contents},
        "git": {"commit": "abc1234def"},
    }
    return save_run(runs, f"2026-{name}", meta, decisions, examples)


@pytest.fixture
def world(tmp_path: Path) -> tuple[list[Path], Path]:
    runs = tmp_path / "runs"
    paths = [
        _save(runs, "cheap-cal", "calibrated", Split.CALIBRATION),
        _save(runs, "cheap-test", "calibrated", Split.TEST),
        _save(runs, "strong-cal", "overconfident", Split.CALIBRATION),
        _save(runs, "strong-test", "overconfident", Split.TEST),
    ]
    seen = tmp_path / "seen.json"
    first = [e.id for e in _examples(Split.TEST, 5)]
    seen.write_text(json.dumps({"datasets": {"banking77": {"n": 5, "ids": first}}}))
    return paths, seen


def test_report_has_every_section_and_never_input_text(world: tuple[list[Path], Path]) -> None:
    paths, seen = world
    html = report.build(paths, seen_path=seen, settings=FAST, raw_dir=Path("/nonexistent"))
    assert html.startswith("<!doctype html>")
    assert SECRET not in html  # ids and numbers only
    for needle in (
        "Accuracy and calibration",
        "Reliability diagrams",
        "Risk–coverage",
        "Guaranteed thresholds",
        "Cascades",
        "Sensitivity: without pilot-seen examples",
        "prefers-color-scheme: dark",
    ):
        assert needle in html
    assert html.count("<svg") >= 6
    assert "http://" not in html.replace("http://www.w3.org", "")  # no external services
    assert "https://" not in html


def test_registered_target_is_labelled_and_others_are_exploratory(
    world: tuple[list[Path], Path],
) -> None:
    paths, seen = world
    html = report.build(paths, seen_path=seen, settings=FAST)
    # One PRE-REGISTERED row per decider (α = 0.1) plus one for the cascade; the other
    # α values are EXPLORATORY.
    assert html.count('badge pre">PRE-REGISTERED') == 2 + 1 + 1  # 2 deciders, cascade, legend
    assert 'badge expl">EXPLORATORY' in html
    assert "not a clean pre-registration" in html  # the Banking77 disclosure


def test_sensitivity_drops_the_seen_ids(world: tuple[list[Path], Path]) -> None:
    paths, seen = world
    html = report.build(paths, seen_path=seen, settings=FAST)
    section = html[html.index('id="sensitivity"') :]
    rows = re.findall(r"<tr><td><code>([^<]+)</code></td><td>(\d+)</td><td>(\d+)</td>", section)
    assert ("cheap-test", "200", "195") in rows
    assert all(name.endswith("-test") for name, _, _ in rows)  # test halves only


def test_without_seen_drops_whole_mt_bench_comparisons() -> None:
    from tests.test_judge import votes

    s = votes(
        [
            ("a-vs-b", "a", "model_a", "model_a", "h1"),
            ("a-vs-b", "b", "model_b", "model_a", "h2"),
            ("c-vs-d", "c", "tie", "tie", "h1"),
        ]
    )
    assert report.without_seen(s, {"human:0"}).ids == ("human:2",)


def test_run_without_a_registered_target_gets_only_exploratory_rows(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    paths = [
        _save(runs, "c", "calibrated", Split.CALIBRATION, alpha=None),
        _save(runs, "t", "calibrated", Split.TEST, alpha=None),
    ]
    html = report.build(paths, seen_path=None, settings=FAST)
    assert 'badge pre">PRE-REGISTERED' in html  # the legend only
    assert html.count('badge pre">PRE-REGISTERED') == 1


def test_charts_are_inline_svg_with_tooltips_and_nice_axes() -> None:
    chart = svg.step_lines(
        [svg.Series("a", [0.5, 1.0], [0.1, 0.27])], title="t", x_label="x", y_label="y"
    )
    assert chart.startswith("<svg")
    assert "<title>a</title>" in chart
    assert ">50%<" in chart  # 0.27 → axis top 0.50, ticks every 10%
    assert svg.nice_top(0.27) == pytest.approx(0.5)
    assert svg.nice_top(0.04) == pytest.approx(0.05)
    assert svg.nice_top(0.9) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="small multiples"):
        svg.step_lines(
            [svg.Series(str(k), [1.0], [0.1]) for k in range(4)], title="t", x_label="", y_label=""
        )


def test_cli_writes_the_report(world: tuple[list[Path], Path], tmp_path: Path) -> None:
    paths, seen = world
    out = tmp_path / "r" / "report.html"
    argv = [
        "report",
        *map(str, paths),
        "--out",
        str(out),
        "--seen-ids",
        str(seen),
        "--n-boot",
        "30",
    ]
    assert main(argv) == 0
    assert out.read_text().startswith("<!doctype html>")
    assert (
        main(["report", str(tmp_path / "missing")]) == 6
    )  # a missing path is bad input (docs/cli-design.md §5)
