"""Sample manifests: the committed list of example IDs for a benchmark sample.

A manifest stores IDs, splits, labels and content fingerprints. It never stores text,
which matters for the safety dataset. Rerunning an experiment means re-downloading the
pinned source revision and selecting these IDs. :func:`verify` checks that the source
rows still hash to the recorded fingerprints.
"""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ask_or_act.data.sampling import SAMPLER_VERSION, SampledExample, Split


class ManifestError(ValueError):
    """A manifest is malformed or doesn't match the source data."""


@dataclass(frozen=True)
class Source:
    """A pinned Hugging Face dataset: repo, exact commit, and the files we read."""

    repo: str
    revision: str
    files: tuple[str, ...]


@dataclass(frozen=True)
class Manifest:
    name: str
    description: str
    source: Source
    seed: int
    examples: tuple[SampledExample, ...]
    sampler_version: int = SAMPLER_VERSION

    def counts(self) -> dict[str, int]:
        c = Counter(e.split for e in self.examples)
        return {s.value: c[s] for s in Split}

    def split(self, split: Split) -> list[SampledExample]:
        return [e for e in self.examples if e.split == split]


def dumps(m: Manifest) -> str:
    """Serialize to JSON with one example per line, so git diffs stay readable."""
    header: dict[str, Any] = {
        "name": m.name,
        "description": m.description,
        "source": {
            "repo": m.source.repo,
            "revision": m.source.revision,
            "files": list(m.source.files),
        },
        "seed": m.seed,
        "sampler_version": m.sampler_version,
        "counts": m.counts(),
    }
    lines = [json.dumps(header, indent=2, ensure_ascii=False)[:-2] + ',\n  "examples": [']
    rows = [
        "    "
        + json.dumps(
            {
                "id": e.id,
                "split": e.split.value,
                "label": e.label,
                "fingerprint": e.fingerprint,
                "meta": dict(e.meta),
            },
            ensure_ascii=False,
        )
        for e in m.examples
    ]
    lines.append(",\n".join(rows))
    lines.append("  ]\n}\n")
    return "\n".join(lines)


def loads(text: str) -> Manifest:
    try:
        raw = json.loads(text)
        src = raw["source"]
        examples = tuple(
            SampledExample(
                id=e["id"],
                split=Split(e["split"]),
                label=e["label"],
                fingerprint=e["fingerprint"],
                meta=e.get("meta", {}),
            )
            for e in raw["examples"]
        )
        m = Manifest(
            name=raw["name"],
            description=raw["description"],
            source=Source(src["repo"], src["revision"], tuple(src["files"])),
            seed=int(raw["seed"]),
            examples=examples,
            sampler_version=int(raw["sampler_version"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError(f"malformed manifest: {exc}") from exc
    if raw.get("counts") != m.counts():
        raise ManifestError(f"counts header {raw.get('counts')} != actual {m.counts()}")
    ids = [e.id for e in m.examples]
    if len(ids) != len(set(ids)):
        raise ManifestError("manifest contains duplicate ids")
    return m


def write(m: Manifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(m), encoding="utf-8")


def read(path: Path) -> Manifest:
    return loads(path.read_text(encoding="utf-8"))


def verify(m: Manifest, fingerprints_by_id: Mapping[str, str]) -> None:
    """Raise if any manifest example is missing from the source or its content changed."""
    problems: Sequence[str] = [
        f"{e.id}: missing from source"
        if e.id not in fingerprints_by_id
        else f"{e.id}: fingerprint {fingerprints_by_id[e.id]} != recorded {e.fingerprint}"
        for e in m.examples
        if fingerprints_by_id.get(e.id) != e.fingerprint
    ]
    if problems:
        shown = "; ".join(problems[:5])
        raise ManifestError(f"{m.name}: {len(problems)} example(s) don't match source: {shown}")
