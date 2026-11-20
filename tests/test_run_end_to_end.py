"""`ask-or-act run` end to end on a tiny fake Banking77, with downloads replaced by a fake."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ask_or_act.cli import main
from ask_or_act.core.runs import load_run
from ask_or_act.data import hub, manifest
from ask_or_act.data.examples import load_examples, select
from ask_or_act.data.manifest import Manifest, ManifestError
from ask_or_act.data.sampling import SampledExample, Split, fingerprint
from ask_or_act.data.sources import BANKING77

LABELS = ["card_arrival", "top_up"]
ROWS: list[dict[str, Any]] = [{"text": f"query {i}", "label": i % 2} for i in range(8)]


def write_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: Sequence[dict[str, Any]] = ROWS
) -> Path:
    """Fake raw parquet + a manifest over it; hub.fetch returns the local file."""
    parquet = tmp_path / "test.parquet"
    meta = {"info": {"features": {"label": {"names": LABELS}}}}
    table = pa.Table.from_pylist(list(rows)).replace_schema_metadata(
        {"huggingface": json.dumps(meta)}
    )
    pq.write_table(table, parquet)
    monkeypatch.setattr(hub, "fetch", lambda source, raw_dir: [parquet])

    examples = tuple(
        SampledExample(
            f"test:{i}",
            Split.TEST if i < 6 else Split.CALIBRATION,
            LABELS[int(r["label"])],
            fingerprint(r),
        )
        for i, r in enumerate(ROWS)
    )
    path = tmp_path / "banking77.json"
    manifest.write(Manifest("banking77", "toy", BANKING77, 1, examples), path)
    return path


def write_config(
    tmp_path: Path, manifest_path: Path, *, params: str = "", budget: float = 0.0
) -> Path:
    cfg = tmp_path / "run.toml"
    cfg.write_text(f"""
name = "e2e"
[dataset]
manifest = "{manifest_path}"
split = "test"
[decider]
type = "fake"
model = "fake-calibrated"
[decider.params]
kind = "calibrated"
seed = 1
{params}
[budget]
max_usd = {budget}
[cache]
path = "{tmp_path / "cache.sqlite"}"
[output]
runs_dir = "{tmp_path / "runs"}"
""")
    return cfg


def runs(tmp_path: Path) -> list[Path]:
    d = tmp_path / "runs"
    return sorted(d.iterdir()) if d.exists() else []


# ------------------------------------------------------------------ examples


def test_load_examples_builds_verified_examples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = manifest.read(write_world(tmp_path, monkeypatch))
    examples = load_examples(m, tmp_path, "test")
    assert [ex.id for ex in examples] == [f"test:{i}" for i in range(6)]
    assert examples[1].input == "query 1"
    assert examples[1].gold == "top_up"
    assert examples[1].options == tuple(LABELS)
    assert len(load_examples(m, tmp_path, "all")) == 8


def test_load_examples_rejects_changed_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    changed = [dict(r) for r in ROWS]
    changed[2]["text"] = "edited upstream"
    m = manifest.read(write_world(tmp_path, monkeypatch, changed))
    with pytest.raises(ManifestError, match="test:2"):
        load_examples(m, tmp_path, "test")


def test_limit_is_a_deterministic_subset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m = manifest.read(write_world(tmp_path, monkeypatch))
    first = select(m, "all", limit=3)
    assert len(first) == 3
    assert first == select(m, "all", limit=3)


# ------------------------------------------------------------------ the CLI


def test_run_end_to_end_then_fully_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = write_config(tmp_path, write_world(tmp_path, monkeypatch))

    assert main(["run", str(cfg)]) == 0
    (run1,) = runs(tmp_path)
    record = load_run(run1)
    assert len(record.decisions) == 6
    assert all(d.ok and not d.cached for d in record.decisions)
    meta = record.metadata
    assert meta["status"] == "completed"
    assert meta["counts"] == {
        "examples": 6,
        "cached": 0,
        "refresh": False,
        "calls": 6,
        "errors": 0,
        "skipped": 0,
    }
    for key in (
        "git",
        "environment",
        "config",
        "dataset",
        "seeds",
        "cost",
        "started_at",
        "finished_at",
    ):
        assert key in meta
    assert isinstance(meta["dataset"], dict)
    assert meta["dataset"]["source"] == {"repo": BANKING77.repo, "revision": BANKING77.revision}
    assert isinstance(meta["seeds"], dict)
    assert meta["seeds"] == {"manifest": 1, "decider": 1}

    # Second run: everything comes from the cache, zero calls.
    capsys.readouterr()
    assert main(["run", str(cfg)]) == 0
    assert "6 cached, 0 API call(s)" in capsys.readouterr().out
    (run2,) = set(runs(tmp_path)) - {run1}  # same-second ids don't sort by start time
    rec2 = load_run(run2)
    assert rec2.metadata["counts"] == {
        "examples": 6,
        "cached": 6,
        "refresh": False,
        "calls": 0,
        "errors": 0,
        "skipped": 0,
    }
    assert [d.choice for d in rec2.decisions] == [d.choice for d in record.decisions]
    assert all(d.cached for d in rec2.decisions)


def test_dry_run_calls_nothing_and_saves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = write_config(tmp_path, write_world(tmp_path, monkeypatch))
    assert main(["run", str(cfg), "--dry-run"]) == 0
    assert "dry run: nothing was called" in capsys.readouterr().out
    assert runs(tmp_path) == []


PRICED = "input_price_per_mtok = 1000.0"  # ~$0.001 per token: makes the fake cost money


def test_refuses_over_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = write_config(tmp_path, write_world(tmp_path, monkeypatch), params=PRICED, budget=0.0)
    assert main(["run", str(cfg), "--yes"]) == 3
    assert "exceeds the budget" in capsys.readouterr().out
    assert runs(tmp_path) == []


def test_spending_needs_explicit_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = write_config(tmp_path, write_world(tmp_path, monkeypatch), params=PRICED, budget=1.0)
    assert main(["run", str(cfg)]) == 4
    assert "--yes" in capsys.readouterr().out
    assert runs(tmp_path) == []

    assert main(["run", str(cfg), "--yes"]) == 0
    (run,) = runs(tmp_path)
    cost = load_run(run).metadata["cost"]
    assert isinstance(cost, dict)
    assert isinstance(cost["charged_usd"], float)
    assert 0 < cost["charged_usd"] <= 1.0
    assert cost["charged_usd"] == pytest.approx(cost["estimated_usd"])
    assert cost["total_cost_usd"] == pytest.approx(cost["charged_usd"])

    # Re-run: nothing charged, but the true cost is still reported.
    assert main(["run", str(cfg), "--yes"]) == 0
    (rerun,) = set(runs(tmp_path)) - {run}
    cost2 = load_run(rerun).metadata["cost"]
    assert isinstance(cost2, dict)
    assert cost2["charged_usd"] == 0
    assert cost2["total_cost_usd"] == pytest.approx(cost["total_cost_usd"])
    assert cost2["saved_by_cache_usd"] == pytest.approx(cost["total_cost_usd"])


def test_bad_config_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = tmp_path / "bad.toml"
    cfg.write_text('name = "x"\n')
    assert main(["run", str(cfg)]) == 2
    assert "missing required table [dataset]" in capsys.readouterr().err
