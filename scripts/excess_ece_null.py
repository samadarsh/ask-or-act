"""Null check for Q1/Q1b's excess-ECE test (docs/analysis-plan.md, Amendment 2).

Two *perfectly calibrated* fake deciders on the same n examples: each is right with
probability equal to its confidence. They differ only in confidence granularity:

- "coarse": 4 values, {0.25, 0.5, 0.75, 1.0};
- "fine": uniform on [0.3, 1.0] in steps of 0.01.

Both are calibrated, so any rejection is a false positive. For each setting this reports
how often the excess-ECE test (the Q1 test) and the raw-ECE test reject at 0.05, one test
at a time, before Holm. The excess test should reject at most about 5%.

Outcomes are drawn independently, or "shared" (one uniform per example for both
deciders, so their errors are positively correlated, as on real paired data).

    uv run python scripts/excess_ece_null.py [--reps 500] [--n-boot 2000]

Fake data only; no network. Seeded: the same arguments give the same table.
"""

import argparse
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np

from ask_or_act import study_analysis as sa
from ask_or_act.metrics.data import Scored
from ask_or_act.metrics.report import Settings

COARSE = np.array([0.25, 0.5, 0.75, 1.0])


def _scored(conf: np.ndarray, correct: np.ndarray) -> Scored:
    n = len(conf)
    ids = tuple(f"x{i}" for i in range(n))
    return Scored(
        ids=ids,
        gold=("g",) * n,
        pred=tuple("g" if c else "h" for c in correct),
        confidence=conf,
        correct=correct,
        probabilities=({},) * n,
        options=("g", "h"),
        groups=ids,
        meta=({},) * n,
        dataset="sim",
        n_errors=0,
    )


def _confidences(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "coarse":
        return rng.choice(COARSE, size=n)
    return np.round(rng.uniform(0.3, 1.0, size=n), 2)


def one(args: tuple[int, str, str, int, str, int, int]) -> tuple[bool, bool, float, float]:
    k, kind_a, kind_b, n, outcomes, rep, n_boot = args
    rng = np.random.default_rng([k, rep])  # setting index, repetition
    ca, cb = _confidences(kind_a, n, rng), _confidences(kind_b, n, rng)
    if outcomes == "shared":
        u = rng.random(n)
        za, zb = u < ca, u < cb
    else:
        za, zb = rng.random(n) < ca, rng.random(n) < cb
    settings = Settings(n_boot=n_boot)  # seed 0 and 15 bins, as in the analysis
    a, b = _scored(ca, za), _scored(cb, zb)
    ex = sa.excess_test(a, b, settings)
    raw = sa.raw_ece_gap(a, b, settings)
    return ex["p"] <= 0.05, raw["p"] <= 0.05, ex["estimate"], raw["estimate"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--reps", type=int, default=500)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)
    settings = [
        ("coarse", "fine", n, outcomes)
        for n, outcomes in product((25, 300), ("independent", "shared"))
    ] + [("fine", "fine", n, "independent") for n in (25, 300)]
    print(
        "| A vs B (both calibrated) | n | outcomes | excess-ECE rejects | raw-ECE rejects "
        "| mean excess gap | mean raw gap |"
    )
    print("|---|---|---|---|---|---|---|")
    with ProcessPoolExecutor() as pool:
        for k, (ka, kb, n, outcomes) in enumerate(settings):
            jobs = [(k, ka, kb, n, outcomes, r, args.n_boot) for r in range(args.reps)]
            res = list(pool.map(one, jobs, chunksize=8))
            ex = np.mean([r[0] for r in res])
            raw = np.mean([r[1] for r in res])
            se = lambda x: 1.96 * math.sqrt(x * (1 - x) / args.reps)  # noqa: E731
            print(
                f"| {ka} vs {kb} | {n} | {outcomes} | {ex:.1%} ± {se(ex):.1%} "
                f"| {raw:.1%} ± {se(raw):.1%} | {np.mean([r[2] for r in res]):+.4f} "
                f"| {np.mean([r[3] for r in res]):+.4f} |",
                flush=True,
            )
    print(f"\n{args.reps} repetitions each, {args.n_boot} resamples, rejection at p ≤ 0.05.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
