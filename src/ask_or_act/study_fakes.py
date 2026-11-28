"""A fake study, shaped like the real one, for testing ``study_analysis`` without the
network and without any study data.

``make_fake_study(root)`` writes run configs under ``root/configs/``, runs fake deciders
on the committed sample manifests (data/samples/, ids and labels only, no raw text),
saves completed runs under ``root/runs/``, and writes ``root/fakes.toml``, a spec in the
same format as configs/analysis/study.toml. Same root, same files: every draw is seeded.

What it imitates:

- the study's sizes:
  - Banking77: 4 per intent, and 1 per intent for self-consistency;
  - Aegis: 250 and 100;
  - MT-Bench: the whole half, and whole questions up to ≥ 50 votes;
- the confidence scales: verbalized confidence in steps of 0.01, self-consistency in
  thirds (k = 3), and a continuous "classic" baseline;
- the registered targets, through [guarantee] tables;
- the two cascades kinds;
- **synthetic** provider metadata: response ids, timestamps and a system_fingerprint.
  The calibration halves are dated 2026-12-01 and the test halves 2026-12-02. One fake
  model on MT-Bench gets a new fingerprint on its test half, so the S6 split path runs.
  None of this is Groq data.
"""

import asyncio
import hashlib
import re
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from ask_or_act.core.config import load_config
from ask_or_act.core.runs import GoldRecord, save_run
from ask_or_act.core.types import JSON, DeciderConfig, Decision, Example
from ask_or_act.data import manifest
from ask_or_act.data.sampling import Split
from ask_or_act.deciders.fake import FakeDecider

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"
SEEN = Path(__file__).resolve().parents[2] / "configs" / "pilot" / "seen_test_ids.json"
TARGETS = {"banking77": (0.10, 0.10), "aegis": (0.10, 0.10), "mt_bench": (0.25, 0.10)}

# (name, kind, fake params) per dataset. "llm" fakes get verbalized and self-consistency
# runs; "classic" gets a continuous confidence and no self-consistency.
DECIDERS: dict[str, list[tuple[str, str, dict[str, JSON]]]] = {
    "banking77": [
        ("fake-calibrated", "llm", {"kind": "calibrated", "low": 0.6, "high": 1.0}),
        ("fake-overconfident", "llm", {"kind": "overconfident", "low": 0.5, "high": 1.0}),
        ("fake-weak", "llm", {"kind": "calibrated", "low": 0.3, "high": 0.95}),
        ("fake-classic", "classic", {"kind": "calibrated", "low": 0.4, "high": 1.0}),
    ],
    "aegis": [
        ("fake-calibrated", "llm", {"kind": "calibrated", "low": 0.6, "high": 1.0}),
        ("fake-overconfident", "llm", {"kind": "overconfident", "low": 0.5, "high": 1.0}),
        ("fake-weak", "llm", {"kind": "calibrated", "low": 0.3, "high": 0.95}),
        ("fake-safeguard", "llm", {"kind": "calibrated", "low": 0.7, "high": 1.0}),
        ("fake-classic", "classic", {"kind": "calibrated", "low": 0.5, "high": 1.0}),
    ],
    "mt_bench": [
        ("fake-calibrated", "llm", {"kind": "calibrated", "low": 0.4, "high": 0.95}),
        ("fake-overconfident", "llm", {"kind": "overconfident", "low": 0.35, "high": 0.9}),
        ("fake-classic", "classic", {"kind": "calibrated", "low": 0.3, "high": 0.7}),
    ],
}
CASCADES = [
    ("banking77", ("fake-weak", "fake-calibrated")),
    ("banking77", ("fake-classic", "fake-calibrated")),
    ("aegis", ("fake-safeguard", "fake-calibrated")),
    ("aegis", ("fake-weak", "fake-calibrated")),
    ("aegis", ("fake-classic", "fake-calibrated")),
]
DAY = {"calibration": 1796083200, "test": 1796169600}  # 2026-12-01 / 12-02 00:00 UTC
COMPARISON = re.compile(r"^q(\d+)-t(\d+)-(.+)-vs-(.+)$")


def _h(*parts: object) -> int:
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def _examples(dataset: str, split: Split) -> list[Example]:
    m = manifest.read(SAMPLES / f"{dataset}.json")
    rows = m.split(split)
    options = tuple(sorted({e.label for e in m.examples}))
    out = []
    for e in rows:
        meta: dict[str, JSON] = dict(e.meta)
        if dataset == "mt_bench":  # display order isn't in the manifest: use the id's order
            q, t, a, b = COMPARISON.match(str(meta["comparison"])).groups()  # type: ignore[union-attr]
            meta |= {"question_id": int(q), "turn": int(t), "model_a": a, "model_b": b}
        out.append(Example(e.id, e.id, options, e.label, meta=meta))
    return out


def _subset(dataset: str, xs: list[Example], sc: bool) -> list[Example]:
    """Nested seeded subsets, as in the study (docs/study.md)."""
    order = sorted(xs, key=lambda e: _h("subset", e.id))
    if dataset == "banking77":
        per, seen = (1 if sc else 4), dict[str, int]()
        keep = []
        for e in order:
            if seen.get(e.gold, 0) < per:
                seen[e.gold] = seen.get(e.gold, 0) + 1
                keep.append(e)
        chosen = {e.id for e in keep}
    elif dataset == "aegis":
        chosen = {e.id for e in order[: 100 if sc else 250]}
    else:
        if not sc:
            return xs
        qs = sorted({e.meta["question_id"] for e in xs}, key=lambda q: _h("q", q))
        chosen, n = set(), 0
        for q in qs:
            ids = [e.id for e in xs if e.meta["question_id"] == q]
            chosen |= set(ids)
            n += len(ids)
            if n >= 50:
                break
    return [e for e in xs if e.id in chosen]


def _snap(d: Decision, step: float) -> Decision:
    if d.confidence is None:
        return d
    c = min(1.0, max(0.0, round(d.confidence / step) * step))
    return replace(d, confidence=c)


def _with_calls(ds: Sequence[Decision], name: str, split: str, dataset: str) -> list[Decision]:
    """Synthetic provider metadata (see the module docstring)."""
    out = []
    for i, d in enumerate(ds):
        pool = [f"fp_fake{(_h(name, 'fp', k) % 9000) + 1000}" for k in range(3)]
        fp = pool[_h(name, d.example_id) % 3]
        if (
            dataset == "mt_bench"
            and name == "fake-overconfident"
            and split == "test"
            and i % 3 == 0
        ):
            fp = "fp_fakenew1"
        created = DAY[split] + 3600 + i
        raw: dict[str, JSON] = {
            "id": f"resp-{name}-{split}-{d.example_id}",
            "created": created,
            "system_fingerprint": fp,
        }
        out.append(replace(d, raw=raw))
    return out


def _toml(
    name: str, dataset: str, split: str, params: dict[str, JSON], signal: str | None, runs: Path
) -> str:
    alpha, delta = TARGETS[dataset]
    extra = {
        "verbalized": 'confidence_signal = "verbalized"\n',
        "self_consistency": 'confidence_signal = "self_consistency"\nsamples = 3\n',
        None: "",
    }[signal]
    p = "".join(
        f"{k} = {v!r}\n" if not isinstance(v, str) else f'{k} = "{v}"\n' for k, v in params.items()
    )
    return (
        f'# FAKE run config for the analysis tests: no network, no study data.\nname = "{name}"\n\n'
        f'[dataset]\nmanifest = "{SAMPLES / (dataset + ".json")}"\nsplit = "{split}"\n\n'
        f'[decider]\ntype = "fake"\nmodel = "{name}"\nprompt_version = "none"\n\n'
        f"[decider.params]\n{p}{extra}\n"
        "[budget]\nmax_usd = 0.0\n\n[cache]\nenabled = false\n\n"
        f'[output]\nruns_dir = "{runs}"\n\n'
        f'[guarantee]\nalpha = {alpha}\ndelta = {delta}\nregistered = "fake"\n'
    )


def make_fake_study(root: Path) -> Path:
    """Build the fake study under ``root``; returns the spec path."""
    cfg_dir, runs = root / "configs", root / "runs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    spec = [
        'label = "fakes"',
        f'seen_ids = "{SEEN}"',
        'notes = { banking77 = "fake data: the registration note would go here" }',
        "",
    ]
    for dataset, deciders in DECIDERS.items():
        halves = {s: _examples(dataset, Split(s)) for s in ("calibration", "test")}
        for seed, (name, kind, params) in enumerate(deciders, start=1):
            entry = [f'[[decider]]\nname = "{name}"\ndataset = "{dataset}"\nkind = "{kind}"']
            for sc in (False, True) if kind == "llm" else (False,):
                for split in ("calibration", "test"):
                    xs = _subset(dataset, halves[split], sc)
                    signal = ("self_consistency" if sc else "verbalized") if kind == "llm" else None
                    run_name = (
                        f"fake-{name.removeprefix('fake-')}-{dataset}-{split}{'-sc' if sc else ''}"
                    )
                    path = cfg_dir / f"{run_name}.toml"
                    full = {**params, "seed": seed * 10 + (5 if sc else 0)}
                    path.write_text(_toml(run_name, dataset, split, full, signal, runs))
                    cfg = load_config(path)
                    fake = FakeDecider(DeciderConfig("fake", name, "none", full))
                    ds = asyncio.run(fake.decide_batch(xs))
                    step = {"verbalized": 0.01, "self_consistency": 1 / 3, None: 0.0}[signal]
                    if step:
                        ds = [_snap(d, step) for d in ds]
                    if kind == "llm":
                        ds = _with_calls(ds, name, split, dataset)
                    meta: dict[str, JSON] = {
                        "name": run_name,
                        "status": "completed",
                        "config": {"path": str(path), "sha256": cfg.sha256, "contents": cfg.raw},
                        "dataset": {"name": dataset, "split": split},
                        "decider": {"type": "fake", "model": name},
                    }
                    save_run(
                        runs,
                        f"20261201T000000Z-{run_name}-fake00",
                        meta,
                        ds,
                        [GoldRecord.of(e) for e in xs],
                    )
                    key = ("sc_" if sc else "") + ("cal" if split == "calibration" else "test")
                    entry.append(f'{key} = "{path}"')
            spec += ["\n".join(entry), ""]
    for dataset, stages in CASCADES:
        spec += [
            f'[[cascade]]\ndataset = "{dataset}"\nstages = {list(stages)!r}'.replace("'", '"'),
            "",
        ]
    out = root / "fakes.toml"
    out.write_text("\n".join(spec))
    return out
