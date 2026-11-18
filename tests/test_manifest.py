import json
from pathlib import Path

import pytest

from ask_or_act.data import manifest
from ask_or_act.data.manifest import Manifest, ManifestError, Source
from ask_or_act.data.sampling import SampledExample, Split

SRC = Source(repo="org/ds", revision="abc123", files=("data/test.parquet",))


def sample_manifest() -> Manifest:
    return Manifest(
        name="toy",
        description="a toy",
        source=SRC,
        seed=1,
        examples=(
            SampledExample("a", Split.CALIBRATION, "x", "f1", {"n_votes": 2}),
            SampledExample("b", Split.TEST, "y", "f2"),
        ),
    )


def test_round_trip(tmp_path: Path) -> None:
    m = sample_manifest()
    path = tmp_path / "sub" / "toy.json"
    manifest.write(m, path)
    assert manifest.read(path) == m


def test_output_is_valid_json_with_one_example_per_line() -> None:
    text = manifest.dumps(sample_manifest())
    raw = json.loads(text)
    assert raw["counts"] == {"calibration": 1, "test": 1}
    example_lines = [ln for ln in text.splitlines() if ln.strip().startswith('{"id"')]
    assert len(example_lines) == 2


def test_empty_manifest_round_trips() -> None:
    m = Manifest("e", "", SRC, 1, ())
    assert manifest.loads(manifest.dumps(m)) == m


def test_tampered_counts_rejected() -> None:
    raw = json.loads(manifest.dumps(sample_manifest()))
    raw["counts"]["test"] = 99
    with pytest.raises(ManifestError, match="counts"):
        manifest.loads(json.dumps(raw))


def test_malformed_rejected() -> None:
    with pytest.raises(ManifestError, match="malformed"):
        manifest.loads('{"name": "x"}')
    raw = json.loads(manifest.dumps(sample_manifest()))
    raw["examples"][0]["split"] = "train"
    with pytest.raises(ManifestError, match="malformed"):
        manifest.loads(json.dumps(raw))


def test_verify_passes_on_matching_source() -> None:
    manifest.verify(sample_manifest(), {"a": "f1", "b": "f2", "c": "unused"})


def test_verify_reports_changed_and_missing_rows() -> None:
    with pytest.raises(ManifestError, match="2 example") as exc:
        manifest.verify(sample_manifest(), {"a": "CHANGED"})
    assert "a: fingerprint" in str(exc.value)
    assert "b: missing" in str(exc.value)
