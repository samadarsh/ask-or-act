"""Training data for the free baselines. Never touches the evaluation test split.

| dataset   | training data                                           | labels used          |
|-----------|---------------------------------------------------------|----------------------|
| banking77 | official train split (10,003)                           | gold intents         |
| aegis     | official train split; human *or* LLM-jury prompt labels | safe/unsafe          |
| mt_bench  | our sample's *calibration* half (no official train)     | expert votes         |

**Leakage guards:**

* Aegis training rows whose prompt text also appears in test or validation are
  dropped (the dataset repeats some prompts).
* MT-Bench training rows come from calibration questions, which never overlap the
  test questions by construction.
* Every TrainingSet records its example ids. The classic decider refuses to predict on
  any id it was trained on.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ask_or_act.core.types import JSON
from ask_or_act.data import hub
from ask_or_act.data.sources import AEGIS, AEGIS_LABELS, AEGIS_TRAIN, BANKING77_TRAIN

SAMPLES_DIR = Path("data/samples")


@dataclass(frozen=True)
class TrainingSet:
    ids: tuple[str, ...]
    inputs: tuple[JSON, ...]  # same shape as Example.input for that dataset
    labels: tuple[str, ...]
    description: str
    # Rows that must stay together in CV folds (MT-Bench: question id). None = i.i.d.
    groups: tuple[str, ...] | None = None


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def banking77(raw_dir: Path) -> TrainingSet:
    (path,) = hub.fetch(BANKING77_TRAIN, raw_dir)
    rows, features = hub.read_parquet(path)
    names = features["label"]["names"]
    return TrainingSet(
        tuple(f"train:{i}" for i in range(len(rows))),
        tuple(str(r["text"]) for r in rows),
        tuple(str(names[int(r["label"])]) for r in rows),
        "Banking77 official train split",
    )


def aegis(raw_dir: Path) -> TrainingSet:
    (train_path,) = hub.fetch(AEGIS_TRAIN, raw_dir)
    eval_prompts = {
        _hash(str(r.get("prompt", "")))
        for p in hub.fetch(AEGIS, raw_dir)
        for r in hub.read_json_rows(p)
    }
    ids, inputs, labels = [], [], []
    for r in hub.read_json_rows(train_path):
        prompt = str(r.get("prompt", ""))
        if (
            r.get("prompt_label") not in AEGIS_LABELS
            or prompt == "REDACTED"
            or _hash(prompt) in eval_prompts
        ):
            continue
        ids.append(f"train:{r['id']}")
        inputs.append(prompt)
        labels.append(str(r["prompt_label"]))
    return TrainingSet(
        tuple(ids),
        tuple(inputs),
        tuple(labels),
        "Aegis 2.0 train split (human + LLM-jury prompt labels), "
        "minus prompts that also occur in test/validation",
    )


def mt_bench(raw_dir: Path) -> TrainingSet:
    from ask_or_act.data import manifest
    from ask_or_act.data.examples import load_examples

    m = manifest.read(SAMPLES_DIR / "mt_bench.json")
    examples = load_examples(m, raw_dir, "calibration")
    return TrainingSet(
        tuple(e.id for e in examples),
        tuple(e.input for e in examples),
        tuple(e.gold for e in examples),
        "MT-Bench sample, calibration half (questions disjoint from test); "
        "calibration-half predictions are 5-fold cross-fitted by question",
        tuple(str(e.meta["question_id"]) for e in examples),
    )


def load(dataset: str, raw_dir: Path) -> TrainingSet:
    loaders: dict[str, Any] = {"banking77": banking77, "aegis": aegis, "mt_bench": mt_bench}
    if dataset not in loaders:
        raise ValueError(f"no training data for dataset {dataset!r}")
    result: TrainingSet = loaders[dataset](raw_dir)
    return result
