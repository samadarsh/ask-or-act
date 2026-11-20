"""Turn a committed sample manifest plus the pinned raw data into ``Example`` objects.

Before building anything, the raw rows are checked against the manifest fingerprints.
A run can only use exactly the committed sample.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ask_or_act.core.types import JSON, Example
from ask_or_act.data import hub
from ask_or_act.data.manifest import Manifest, verify
from ask_or_act.data.sampling import SampledExample, fingerprint, seeded_order

MT_BENCH_OPTIONS = ("model_a", "model_b", "tie")
AEGIS_OPTIONS = ("safe", "unsafe")


def select(manifest: Manifest, split: str, limit: int | None = None) -> list[SampledExample]:
    """The manifest entries for a split ("calibration", "test" or "all").

    With ``limit``, take a seeded-random subset. This is a smoke-test subset, not a
    stratified one.
    """
    chosen = [e for e in manifest.examples if split == "all" or e.split.value == split]
    if limit is not None:
        chosen = sorted(chosen, key=lambda e: seeded_order(e.id, manifest.seed))[:limit]
        chosen.sort(key=lambda e: (e.split, e.id))
    return chosen


def _example(
    name: str, entry: SampledExample, row: Mapping[str, Any], label_names: Sequence[str]
) -> Example:
    if name == "banking77":
        return Example(entry.id, str(row["text"]), tuple(label_names), entry.label)
    if name == "mt_bench":
        meta: dict[str, JSON] = {
            "question_id": int(row["question_id"]),
            "turn": int(row["turn"]),
            "model_a": str(row["model_a"]),
            "model_b": str(row["model_b"]),
            "judge": str(row["judge"]),
            **dict(entry.meta),
        }
        payload: JSON = {
            "turn": int(row["turn"]),
            "conversation_a": row["conversation_a"],
            "conversation_b": row["conversation_b"],
        }
        return Example(entry.id, payload, MT_BENCH_OPTIONS, entry.label, meta=meta)
    if name == "aegis":
        # Unsafe text lives only in memory; runs and caches store ids and decisions only.
        return Example(entry.id, str(row["prompt"]), AEGIS_OPTIONS, entry.label)
    raise KeyError(f"no example builder for dataset {name!r}")


def load_examples(
    manifest: Manifest, raw_dir: Path, split: str = "test", limit: int | None = None
) -> list[Example]:
    entries = select(manifest, split, limit)
    rows, label_names = hub.rows_by_id(manifest.name, hub.fetch(manifest.source, raw_dir))
    wanted = {e.id for e in entries}
    present = {i: fingerprint(r) for i, r in rows.items() if i in wanted}
    verify(
        Manifest(
            manifest.name, manifest.description, manifest.source, manifest.seed, tuple(entries)
        ),
        present,
    )
    return [_example(manifest.name, e, rows[e.id], label_names) for e in entries]
