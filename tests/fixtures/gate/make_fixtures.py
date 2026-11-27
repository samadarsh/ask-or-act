"""Write the paired files the repo's own Action workflow gates (.github/workflows/gate.yml).

    uv run python tests/fixtures/gate/make_fixtures.py

The files are committed; this script documents how they were made, and a test checks
they still match it. No text, no real data: ids, support-ticket intents and numbers from fixed
seeds.
2,000 items share one difficulty draw (Beta(5, 1)), so the files are paired the way
real ones are (the same model as scripts/gate_simulation.py):

* base.csv: the baseline;
* candidate.csv: another run of the same model. The gate passes;
* regressed.csv: 8 points less accurate. The gate fails (exit 1).
"""

import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
N = 2000
INTENTS = ("refund", "card_lost", "card_declined", "transfer_pending", "change_pin", "top_up")


def write(path: Path, *, seed: int, shift: float = 0.0) -> None:
    items = np.random.default_rng(0)
    p = np.clip(items.beta(5.0, 1.0, N), 0.02, 0.995)
    label = items.integers(0, len(INTENTS), N)
    rng = np.random.default_rng(1000 + seed)
    q = np.clip(p + shift, 0.01, 0.999)
    right = rng.uniform(size=N) < q
    z = np.log(q / (1 - q)) + rng.normal(0.0, 0.35, N)
    conf = np.clip(1 / (1 + np.exp(-z)), 0.001, 0.999)
    wrong = (label + rng.integers(1, len(INTENTS), N)) % len(INTENTS)  # another intent
    with path.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["id", "prediction", "confidence", "human_label"])
        for i in range(N):
            pred = INTENTS[label[i]] if right[i] else INTENTS[wrong[i]]
            w.writerow([f"t{i:05d}", pred, f"{conf[i]:.4f}", INTENTS[label[i]]])


FILES = {
    "base.csv": {"seed": 1},
    "candidate.csv": {"seed": 3},
    "regressed.csv": {"seed": 2, "shift": -0.08},
}


def main(folder: Path = HERE) -> None:
    for name, kw in FILES.items():
        write(folder / name, **kw)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
