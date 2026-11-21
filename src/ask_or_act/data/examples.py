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


def question_of(entry: SampledExample) -> str:
    """MT-Bench question of a vote: ``q104`` from comparison ``q104-t2-a-vs-b``."""
    comparison = entry.meta.get("comparison")
    if not isinstance(comparison, str):
        raise ValueError(f"{entry.id}: no 'comparison' in manifest meta (MT-Bench only)")
    return comparison.split("-", 1)[0]


def _largest_remainder(sizes: Mapping[str, int], total: int) -> dict[str, int]:
    """Split ``total`` over groups in proportion to ``sizes`` (Hamilton's method):
    floor of each exact share, then one more for the largest remainders (ties by
    name, so the result is deterministic)."""
    n = sum(sizes.values())
    exact = {k: total * v / n for k, v in sizes.items()}
    out = {k: int(x) for k, x in exact.items()}
    left = total - sum(out.values())
    for k in sorted(exact, key=lambda k: (out[k] - exact[k], k))[:left]:
        out[k] += 1
    return out


def select(
    manifest: Manifest,
    split: str,
    limit: int | None = None,
    *,
    per_label: int | None = None,
    stratify: str = "random",
) -> list[SampledExample]:
    """The manifest entries for a split ("calibration", "test" or "all"), optionally a
    subset (see core/config.py ``DatasetConfig``).

    Every subset takes entries in seeded order (a hash of id and seed), within each
    label or question where stratified. So subsets are nested: the first N per label
    are always the same N, whatever larger N another config uses. Selecting by gold
    label happens here, in the harness; deciders never see labels.
    """
    chosen = [e for e in manifest.examples if split == "all" or e.split.value == split]

    def order(key: str) -> str:
        return seeded_order(key, manifest.seed)

    ranked = sorted(chosen, key=lambda e: order(e.id))
    if per_label is not None:
        taken: dict[str, int] = {}
        picked = []
        for e in ranked:
            if taken.get(e.label, 0) < per_label:
                taken[e.label] = taken.get(e.label, 0) + 1
                picked.append(e)
        chosen = picked
    elif limit is not None and stratify == "label":
        sizes: dict[str, int] = {}
        for e in ranked:
            sizes[e.label] = sizes.get(e.label, 0) + 1
        quota = _largest_remainder(sizes, min(limit, len(ranked)))
        seen: dict[str, int] = {}
        picked = []
        for e in ranked:
            seen[e.label] = seen.get(e.label, 0) + 1
            if seen[e.label] <= quota[e.label]:
                picked.append(e)
        chosen = picked
    elif limit is not None and stratify == "question":
        # Whole questions in seeded order until at least ``limit`` votes.
        by_q: dict[str, list[SampledExample]] = {}
        for e in chosen:
            by_q.setdefault(question_of(e), []).append(e)
        picked = []
        for q in sorted(by_q, key=order):
            if len(picked) >= limit:
                break
            picked.extend(by_q[q])
        chosen = picked
    elif limit is not None:
        chosen = ranked[:limit]
    return sorted(chosen, key=lambda e: (e.split, e.id))


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
    manifest: Manifest,
    raw_dir: Path,
    split: str = "test",
    limit: int | None = None,
    *,
    per_label: int | None = None,
    stratify: str = "random",
) -> list[Example]:
    entries = select(manifest, split, limit, per_label=per_label, stratify=stratify)
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
