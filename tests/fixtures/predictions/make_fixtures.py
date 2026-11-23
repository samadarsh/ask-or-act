"""Write the synthetic predictions files that docs/cli-design.md's real-output sessions use.

    uv run python tests/fixtures/predictions/make_fixtures.py

The files are committed; this script documents how they were made. No text, no real
data: ids, labels and numbers from a fixed seed. The model is overconfident on purpose
(stated confidence about 8 points above its real accuracy), like many LLMs.
"""

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
INTENTS = (
    "refund",
    "card_lost",
    "card_declined",
    "transfer_pending",
    "change_pin",
    "top_up",
    "exchange_rate",
    "close_account",
)


def _overconfident(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(correct, stated confidence): P(correct) = skill; confidence ≈ skill + 0.08."""
    skill = rng.uniform(0.35, 1.0, n)
    correct = rng.uniform(size=n) < skill
    conf = np.clip(skill + 0.08 + rng.normal(0, 0.03, n), 0.3, 0.999).round(3)
    return correct, conf


def support_bot(rng: np.random.Generator) -> list[dict[str, str]]:
    """2,400 rows: 38 failed decisions, 400 unlabelled, a split column (1,000 / 1,000
    labelled)."""
    n = 2400
    correct, conf = _overconfident(rng, n)
    gold = rng.choice(INTENTS, n)
    failed = set(rng.choice(n, 38, replace=False).tolist())
    unlabelled = set(rng.choice(sorted(set(range(n)) - failed), 400, replace=False).tolist())
    labelled = [i for i in range(n) if i not in unlabelled]
    cal = set(rng.choice(labelled, 1000, replace=False).tolist())
    rows = []
    for i in range(n):
        g = str(gold[i])
        pred = g if correct[i] else str(rng.choice([x for x in INTENTS if x != g]))
        rows.append(
            {
                "id": f"t-{i:05d}",
                "prediction": "" if i in failed else pred,
                "confidence": "" if i in failed else f"{conf[i]:.3f}",
                "human_label": "" if i in unlabelled else g,
                "split": "calibration" if i in cal or (i in unlabelled and i % 2) else "test",
            }
        )
    return rows


def scores(rng: np.random.Generator) -> list[dict[str, str]]:
    """812 rows with other column names and percent confidences."""
    n = 812
    correct, conf = _overconfident(rng, n)
    gold = rng.choice(INTENTS, n)
    out = []
    for i in range(n):
        g = str(gold[i])
        pred = g if correct[i] else str(rng.choice([x for x in INTENTS if x != g]))
        out.append(
            {"id": f"s-{i:04d}", "pred": pred, "conf": str(round(conf[i] * 100)), "label": g}
        )
    return out


def moderation(rng: np.random.Generator) -> list[dict[str, object]]:
    """600 rows of a binary classifier whose score is P(unsafe), as JSONL."""
    n = 600
    unsafe = rng.uniform(size=n) < 0.3
    p = np.where(unsafe, rng.beta(6, 2, n), rng.beta(2, 6, n)).round(3)
    return [
        {
            "id": f"m-{i:04d}",
            "prediction": "unsafe" if p[i] >= 0.5 else "safe",
            "confidence": float(p[i]),
            "human_label": "unsafe" if unsafe[i] else "safe",
        }
        for i in range(n)
    ]


def tiny(rng: np.random.Generator) -> list[dict[str, str]]:
    """40 labelled rows: too few for a guarantee at α = 0.1."""
    correct, conf = _overconfident(rng, 40)
    conf = np.maximum(conf, 0.5)  # two labels: confidence in the chosen one is >= 0.5
    return [
        {
            "id": f"p-{i:02d}",
            "prediction": "yes" if correct[i] else "no",
            "confidence": f"{conf[i]:.2f}",
            "human_label": "yes",
        }
        for i in range(40)
    ]


def large(rng: np.random.Generator, small: list[dict[str, str]]) -> list[dict[str, str]]:
    """A stronger, better-calibrated model on the same rows as support-bot.csv (same ids,
    labels, split and failures), for compare and cascade examples."""
    n = len(small)
    skill = rng.uniform(0.55, 1.0, n)
    correct = rng.uniform(size=n) < skill
    conf = np.clip(skill + rng.normal(0, 0.03, n), 0.3, 0.999).round(3)
    out = []
    for i, row in enumerate(small):
        g = row["human_label"] or str(rng.choice(INTENTS))
        pred = g if correct[i] else str(rng.choice([x for x in INTENTS if x != g]))
        failed = row["prediction"] == ""
        out.append(
            {
                **row,
                "prediction": "" if failed else pred,
                "confidence": "" if failed else f"{conf[i]:.3f}",
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    rng = np.random.default_rng(20261122)
    small = support_bot(rng)
    _write_csv(HERE / "support-bot.csv", small)
    _write_csv(HERE / "scores.csv", scores(rng))
    with (HERE / "moderation.jsonl").open("w", encoding="utf-8") as f:
        for row in moderation(rng):
            f.write(json.dumps(row) + "\n")
    _write_csv(HERE / "pilot.csv", tiny(rng))
    _write_csv(HERE / "support-bot-large.csv", large(rng, small))  # last: keeps the others


if __name__ == "__main__":
    main()
