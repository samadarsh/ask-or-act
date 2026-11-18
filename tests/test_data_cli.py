"""`ask-or-act data ...` with the network layer replaced by fakes."""

from pathlib import Path

import pytest

from ask_or_act.cli import main
from ask_or_act.data import hub, manifest
from ask_or_act.data.manifest import Manifest
from ask_or_act.data.sampling import SampledExample, Split
from ask_or_act.data.sources import BANKING77


def fake_manifest(name: str, raw_dir: Path, seed: int) -> Manifest:
    ex = (SampledExample("test:0", Split.CALIBRATION, "x", "f0"),)
    return Manifest(name, "fake", BANKING77, seed, ex)


def test_sample_writes_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hub, "build_manifest", fake_manifest)
    assert main(["data", "sample", "banking77", "--samples-dir", str(tmp_path), "--seed", "3"]) == 0
    m = manifest.read(tmp_path / "banking77.json")
    assert m.seed == 3


@pytest.mark.parametrize(("source_fp", "code"), [("f0", 0), ("changed", 1)])
def test_verify_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_fp: str, code: int
) -> None:
    manifest.write(fake_manifest("banking77", tmp_path, 1), tmp_path / "banking77.json")
    monkeypatch.setattr(hub, "current_fingerprints", lambda name, raw: {"test:0": source_fp})
    assert main(["data", "verify", "banking77", "--samples-dir", str(tmp_path)]) == code


def test_unknown_dataset_rejected() -> None:
    with pytest.raises(SystemExit):
        main(["data", "sample", "imagenet"])
