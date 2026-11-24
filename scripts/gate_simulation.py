"""How often does ``ask-or-act gate`` fail when it shouldn't, and catch what it should?

Every replicate draws two predictions files on the same 2,000 items (1,000 calibration
/ 1,000 test after the hashed split), a baseline and a new model, and runs the real
gate (``gate.run_gate``) on them. Items have a difficulty p_i ~ Beta(5, 1) shared by
both models, so the files are paired as real ones are. A model is correct on item i with
probability clip(p_i + shift) and states the confidence

    logit(conf_i) = logit(p_i + shift) + bias + N(0, noise²).

Scenarios:

* **null**: both files from the same model. The true differences are 0, inside H0, so
  every FAIL is false.
* **boundary**: the same, with every tolerance 0. Now the truth is *on* H0's boundary,
  the hardest case for a test's size: the Holm-adjusted family should fail at most
  ``level`` (5%) of the time.
* **accuracy at −1 point**: the new model is exactly 1 point less accurate, the
  default tolerance. Again on the boundary, for the accuracy check alone (Tango's
  lower tail, which his paper doesn't simulate).
* **power**: a 4-point accuracy drop, an overconfident new model (bias +0.8) and a
  noisier confidence (lower coverage).

Results go in docs/gate-simulation.md. Run: ``uv run python scripts/gate_simulation.py``
(about 15 minutes on 8 cores).
"""

import math
import multiprocessing as mp
import sys
import time
import zlib
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ask_or_act import gate

N_ITEMS = 2000
N_BOOT = 1000


@dataclass(frozen=True)
class Model:
    shift: float = 0.0
    bias: float = 0.0
    noise: float = 0.35


@dataclass(frozen=True)
class Scenario:
    name: str
    new: Model
    base: Model = field(default_factory=Model)
    tolerance: gate.Tolerances = field(default_factory=gate.Tolerances)
    reps: int = 400


def _logit(x: np.ndarray) -> np.ndarray:
    return np.log(x / (1 - x))


def _draw(rng: np.random.Generator, p: np.ndarray, m: Model) -> dict[str, list[object]]:
    q = np.clip(p + m.shift, 0.01, 0.999)
    correct = rng.uniform(size=len(p)) < q
    z = _logit(q) + m.bias + rng.normal(0.0, m.noise, len(p))
    conf = np.clip(1 / (1 + np.exp(-z)), 0.001, 0.999)
    return {
        "id": [f"i{k:05d}" for k in range(len(p))],
        "prediction": ["y" if c else "n" for c in correct],
        "confidence": [round(float(x), 4) for x in conf],
        "human_label": ["y"] * len(p),
    }


def one(args: tuple[Scenario, int]) -> dict[str, str]:
    sc, rep = args
    rng = np.random.default_rng([zlib.crc32(sc.name.encode()), rep])  # stable across runs
    p = np.clip(rng.beta(5.0, 1.0, N_ITEMS), 0.02, 0.995)
    base = _draw(rng, p, sc.base)
    new = _draw(rng, p, sc.new)
    policy = gate.Policy(tolerance=sc.tolerance, n_boot=N_BOOT, seed=rep)
    try:
        r = gate.run_gate(new, base, policy=policy)
    except Exception as exc:  # e.g. too few rows to certify: record, don't crash
        return {"error": type(exc).__name__}
    out = {c.name: c.status for c in r.checks}
    out["gate"] = "pass" if r.passed else "fail"
    out["strict"] = (
        "fail" if any(c.status in ("fail", "inconclusive") for c in r.checks) else "pass"
    )
    out["paired_fail"] = (
        "fail" if any(c.status == "fail" for c in r.checks if c.name in gate.PAIRED) else "pass"
    )
    return out


SCENARIOS = [
    Scenario("null (same model, default tolerances)", Model()),
    Scenario(
        "boundary (same model, all tolerances 0)",
        Model(),
        tolerance=gate.Tolerances(accuracy=0.0, calibration=0.0, coverage=0.0),
    ),
    Scenario("accuracy at −1 point (the tolerance)", Model(shift=-0.01)),
    Scenario("power: accuracy −4 points", Model(shift=-0.04), reps=200),
    Scenario("power: overconfident (bias +0.8)", Model(bias=0.8), reps=200),
    Scenario("power: noisier confidence (σ 0.35 → 1.2)", Model(noise=1.2), reps=200),
]


def ci(k: int, n: int) -> str:
    """Wilson 95% interval for a rate, in percent."""
    if n == 0:
        return "—"
    z = 1.96
    ph = k / n
    centre = (ph + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{100 * max(0.0, centre - half):.1f}–{100 * min(1.0, centre + half):.1f}"


def main() -> None:
    t0 = time.perf_counter()
    rows = []
    with mp.Pool() as pool:
        for sc in SCENARIOS:
            res = pool.map(one, [(sc, r) for r in range(sc.reps)])
            ok = [r for r in res if "error" not in r]
            errors = Counter(r["error"] for r in res if "error" in r)
            cols = {}
            for name in (
                "accuracy",
                "calibration",
                "coverage",
                "certified",
                "paired_fail",
                "gate",
                "strict",
            ):
                k = sum(r.get(name) == "fail" for r in ok)
                cols[name] = (k, len(ok))
            inc = {
                name: sum(r.get(name) == "inconclusive" for r in ok) / max(1, len(ok))
                for name in gate.PAIRED
            }
            rows.append((sc, cols, inc, errors))
            print(f"done: {sc.name} ({time.perf_counter() - t0:.0f} s)", file=sys.stderr)
    print(
        f"{N_ITEMS:,} items per file (1,000 calibration / 1,000 test), {N_BOOT:,} bootstrap "
        f"resamples per gate run, level 0.05.\n"
    )
    print(
        "| scenario | runs | accuracy | calibration | coverage | certified "
        "| any paired check | gate FAIL | strict FAIL |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for sc, cols, _, errors in rows:
        n = cols["gate"][1]
        cells = [
            f"{100 * k / max(1, m):.1f}% ({ci(k, m)})"
            for k, m in (
                cols[c]
                for c in (
                    "accuracy",
                    "calibration",
                    "coverage",
                    "certified",
                    "paired_fail",
                    "gate",
                    "strict",
                )
            )
        ]
        note = f" ({sum(errors.values())} errors)" if errors else ""
        print(f"| {sc.name} | {n}{note} | " + " | ".join(cells) + " |")
    print("\nInconclusive rates (passes unless strict):\n")
    print("| scenario | accuracy | calibration | coverage |")
    print("|---|---|---|---|")
    for sc, _, inc, _ in rows:
        print(f"| {sc.name} | " + " | ".join(f"{100 * inc[c]:.1f}%" for c in gate.PAIRED) + " |")
    print(f"\n{time.perf_counter() - t0:.0f} s in all.", file=sys.stderr)


if __name__ == "__main__":
    main()
