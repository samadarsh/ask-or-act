"""Network and file I/O for the benchmark datasets (needs the ``data`` extra).

Kept separate so the rest of ``ask_or_act.data`` stays importable, and testable,
without the network or optional dependencies.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ask_or_act.data.manifest import Manifest, Source
from ask_or_act.data.sampling import Candidate
from ask_or_act.data.sources import (
    DATASETS,
    Row,
    aegis_candidates,
    banking77_candidates,
    mt_bench_candidates,
)


def fetch(source: Source, raw_dir: Path) -> list[Path]:
    """Download the source's files at its pinned revision (cached after the first call)."""
    from huggingface_hub import hf_hub_download

    local_dir = raw_dir / source.repo
    return [
        Path(
            hf_hub_download(
                source.repo,
                f,
                repo_type="dataset",
                revision=source.revision,
                local_dir=local_dir,
            )
        )
        for f in source.files
    ]


def read_parquet(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rows plus the Hugging Face feature metadata embedded in the parquet schema."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    meta = table.schema.metadata or {}
    features: dict[str, Any] = (
        json.loads(meta[b"huggingface"]).get("info", {}).get("features", {})
        if b"huggingface" in meta
        else {}
    )
    rows: list[dict[str, Any]] = table.to_pylist()
    return rows, features


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    """A JSON array or JSON Lines file."""
    text = path.read_text(encoding="utf-8").lstrip()
    if text.startswith("["):
        rows: list[dict[str, Any]] = json.loads(text)
        return rows
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_candidates(name: str, paths: Sequence[Path]) -> list[Candidate]:
    if name == "banking77":
        rows, features = read_parquet(paths[0])
        return banking77_candidates(rows, features["label"]["names"])
    if name == "mt_bench":
        rows, _ = read_parquet(paths[0])
        return mt_bench_candidates(rows)
    if name == "aegis":
        by_file: dict[str, Sequence[Row]] = {p.name: read_json_rows(p) for p in paths}
        return aegis_candidates(by_file)
    raise KeyError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")


def rows_by_id(name: str, paths: Sequence[Path]) -> tuple[dict[str, Row], list[str]]:
    """Raw rows keyed by the same ids the manifests use, plus class names if any."""
    if name == "banking77":
        rows, features = read_parquet(paths[0])
        return {f"test:{i}": r for i, r in enumerate(rows)}, list(features["label"]["names"])
    if name == "mt_bench":
        rows, _ = read_parquet(paths[0])
        return {f"human:{i}": r for i, r in enumerate(rows)}, []
    if name == "aegis":
        return {str(r["id"]): r for p in paths for r in read_json_rows(p)}, []
    raise KeyError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")


def build_manifest(name: str, raw_dir: Path, seed: int) -> Manifest:
    spec = DATASETS[name]
    candidates = load_candidates(name, fetch(spec.source, raw_dir))
    examples = spec.sample(candidates, seed)
    return Manifest(name, spec.description, spec.source, seed, tuple(examples))


def current_fingerprints(name: str, raw_dir: Path) -> dict[str, str]:
    spec = DATASETS[name]
    return {c.id: c.fingerprint for c in load_candidates(name, fetch(spec.source, raw_dir))}
