"""Checks the committed manifests in data/samples/ with no network access.

These guard the properties the study relies on: sizes, label balance, no leakage
between calibration and test, and no raw text.
"""

from collections import Counter
from pathlib import Path

import pytest

from ask_or_act.data import manifest
from ask_or_act.data.sampling import Split
from ask_or_act.data.sources import DATASETS, DEFAULT_SEED

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"


@pytest.mark.parametrize("name", list(DATASETS))
def test_manifest_matches_its_spec(name: str) -> None:
    m = manifest.read(SAMPLES / f"{name}.json")
    spec = DATASETS[name]
    assert m.name == name
    assert m.source == spec.source
    assert m.seed == DEFAULT_SEED
    ids = [e.id for e in m.examples]
    assert len(ids) == len(set(ids))


def test_banking77() -> None:
    m = manifest.read(SAMPLES / "banking77.json")
    assert m.counts() == {"calibration": 770, "test": 770}
    per = Counter((e.label, e.split) for e in m.examples)
    assert len({lb for lb, _ in per}) == 77
    assert set(per.values()) == {10}
    assert all(e.id.startswith("test:") for e in m.examples)  # official test split only


def test_mt_bench() -> None:
    m = manifest.read(SAMPLES / "mt_bench.json")
    assert m.counts() == {"calibration": 500, "test": 500}
    question_splits: dict[str, set[Split]] = {}
    comparison_rows: Counter[str] = Counter()
    for e in m.examples:
        question = str(e.meta["comparison"]).split("-")[0]
        question_splits.setdefault(question, set()).add(e.split)
        comparison_rows[str(e.meta["comparison"])] += 1
    assert all(len(s) == 1 for s in question_splits.values()), "question in both splits"
    # Every vote on a sampled comparison was included.
    n_votes = {str(e.meta["comparison"]): int(e.meta["n_votes"]) for e in m.examples}
    assert all(comparison_rows[c] == n for c, n in n_votes.items())
    assert any(n > 1 for n in n_votes.values()), "no multi-voter comparisons to measure"


def test_aegis() -> None:
    m = manifest.read(SAMPLES / "aegis.json")
    assert m.counts() == {"calibration": 500, "test": 500}
    labels = Counter((e.label, e.split) for e in m.examples)
    assert {lb for lb, _ in labels} == {"safe", "unsafe"}
    assert labels["safe", Split.CALIBRATION] == labels["safe", Split.TEST]


def test_aegis_manifest_has_no_text_fields() -> None:
    import json

    raw = json.loads((SAMPLES / "aegis.json").read_text())
    allowed = {"id", "split", "label", "fingerprint", "meta"}
    assert all(set(e) == allowed for e in raw["examples"])
    assert all(set(e["meta"]) == {"source_file"} for e in raw["examples"])
