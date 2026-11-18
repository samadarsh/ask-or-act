"""The three benchmark datasets: pinned sources and row -> Candidate conversion.

The conversion functions are pure (rows in, candidates out) so they can be tested with
fake rows and no network. See docs/datasets.md for why each choice was made.
"""

import hashlib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ask_or_act.data.manifest import Source
from ask_or_act.data.sampling import (
    Candidate,
    SampledExample,
    fingerprint,
    grouped_split,
    stratified_split,
)

type Row = Mapping[str, Any]

# Fixed before any results existed (it's the date the plan was approved), so it can't
# have been tuned to make results look good.
DEFAULT_SEED = 20261118


# ----------------------------------------------------------------------- Banking77

BANKING77 = Source(
    repo="PolyAI/banking77",
    revision="1846984189f2bc0ac51c6fc3a600fa9e1714e0c6",
    files=("data/test-00000-of-00001.parquet",),
)


def banking77_candidates(test_rows: Sequence[Row], label_names: Sequence[str]) -> list[Candidate]:
    """Official *test* split only. The classic baseline trains on the train split."""
    return [
        Candidate(id=f"test:{i}", label=label_names[int(r["label"])], fingerprint=fingerprint(r))
        for i, r in enumerate(test_rows)
    ]


def banking77_sample(cands: Sequence[Candidate], seed: int) -> list[SampledExample]:
    return stratified_split(cands, seed=seed, per_label=20)  # 77 x 20 = 1,540


# ----------------------------------------------------------------------- MT-Bench

MT_BENCH = Source(
    repo="lmsys/mt_bench_human_judgments",
    revision="f7d2896d2cc5d80f8b55c2bbc722613555233c25",
    files=("data/human-00000-of-00001-25f4910818759289.parquet",),
)


def comparison_key(row: Row) -> str:
    """Identifies one comparison regardless of which model was shown as A or B."""
    a, b = sorted((str(row["model_a"]), str(row["model_b"])))
    return f"q{row['question_id']}-t{row['turn']}-{a}-vs-{b}"


def mt_bench_candidates(human_rows: Sequence[Row]) -> list[Candidate]:
    """One candidate per human vote.

    ``unit`` = comparison: all votes on the same comparison are sampled together, and
    ``meta.n_votes`` records how many there are, for measuring human-human agreement.
    ``group`` = question: a question never appears in both calibration and test.
    """
    votes = Counter(comparison_key(r) for r in human_rows)
    out = []
    for i, r in enumerate(human_rows):
        key = comparison_key(r)
        out.append(
            Candidate(
                id=f"human:{i}",
                label=str(r["winner"]),
                fingerprint=fingerprint(r),
                group=f"q{r['question_id']}",
                unit=key,
                meta={"comparison": key, "n_votes": votes[key], "judge": str(r["judge"])},
            )
        )
    return out


def mt_bench_sample(cands: Sequence[Candidate], seed: int) -> list[SampledExample]:
    return grouped_split(cands, seed=seed, total=1000)


# ----------------------------------------------------------------------- Aegis 2.0

AEGIS = Source(
    repo="nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
    revision="d86bb8bedff51d25ac834ab7838f1cc61acb7a2c",
    files=("test.json", "validation.json"),
)

AEGIS_LABELS = frozenset({"safe", "unsafe"})


def aegis_candidates(rows_by_file: Mapping[str, Sequence[Row]]) -> list[Candidate]:
    """Prompt-safety candidates.

    Keeps rows whose ``prompt_label`` is human-annotated, drops ``REDACTED`` prompts,
    and keeps one row per distinct prompt text (first occurrence, files in the given
    order). Duplicates would otherwise straddle calibration and test.
    Prompt text is hashed, never stored.
    """
    seen_prompts: set[str] = set()
    out = []
    for file, rows in rows_by_file.items():
        for r in rows:
            prompt = str(r.get("prompt", ""))
            if (
                r.get("prompt_label_source") != "human"
                or r.get("prompt_label") not in AEGIS_LABELS
                or prompt == "REDACTED"
            ):
                continue
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            if prompt_hash in seen_prompts:
                continue
            seen_prompts.add(prompt_hash)
            out.append(
                Candidate(
                    id=str(r["id"]),
                    label=str(r["prompt_label"]),
                    fingerprint=fingerprint(r),
                    meta={"source_file": file},
                )
            )
    return out


def aegis_sample(cands: Sequence[Candidate], seed: int) -> list[SampledExample]:
    return stratified_split(cands, seed=seed, total=1000)


# ----------------------------------------------------------------------- registry


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    description: str
    source: Source
    sample: Callable[[Sequence[Candidate], int], list[SampledExample]]


DATASETS: dict[str, DatasetSpec] = {
    "banking77": DatasetSpec(
        "banking77",
        "Routing: 77-intent banking queries. 20 per intent from the official test split.",
        BANKING77,
        banking77_sample,
    ),
    "mt_bench": DatasetSpec(
        "mt_bench",
        "Judging: MT-Bench expert pairwise votes. Split by question; all votes on a "
        "comparison kept together.",
        MT_BENCH,
        mt_bench_sample,
    ),
    "aegis": DatasetSpec(
        "aegis",
        "Safety: Aegis 2.0 human-labelled prompts (test + validation), deduplicated, "
        "stratified safe/unsafe. IDs only; no text.",
        AEGIS,
        aegis_sample,
    ),
}
