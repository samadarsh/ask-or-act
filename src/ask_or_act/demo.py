"""``ask-or-act demo``: a working result before you have prepared anything.

docs/cli-design.md §3.7. The demo writes a predictions file and an HTML report into a
new folder and prints the same summaries ``thresholds`` and ``metrics`` would.

The data is generated here, never downloaded, and holds no text: ids, intent labels
and numbers from a fixed seed. It simulates a support-ticket router whose model is
**overconfident**, as many LLMs are, so the demo shows a calibration warning and a
real act/ask split (some decisions acted on, the rest sent to a human).
"""

import csv
import itertools
from pathlib import Path

import numpy as np

SEED = 0
N_LABELLED = 2000  # tickets a human has also labelled
N_NEW = 500  # recent tickets nobody has labelled yet
FILE_NAME = "support-tickets.csv"
REPORT_NAME = "report.html"
DEFAULT_DIR = Path("ask-or-act-demo")
TITLE = "ask-or-act demo: a support-ticket router"
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


def rows(seed: int = SEED) -> list[dict[str, str]]:
    """The demo's predictions, one dict per ticket.

    Each ticket has a chance ``p`` that the model routes it correctly, drawn from
    Beta(5, 1): most tickets are easy, some are hard. The model's stated confidence is
    ``p`` pushed outward on the log-odds scale, logit(conf) = 1.3·logit(p) + 0.4 plus a
    little noise. So confidence overstates accuracy by about 8 points on average, most
    in the middle of the range, while the most confident tickets are still mostly right.
    The confidences are spread out (no pile-up at one value), as a real model's are.
    """
    rng = np.random.default_rng(seed)
    n = N_LABELLED + N_NEW
    p = np.clip(rng.beta(5.0, 1.0, n), 0.02, 0.995)
    correct = rng.uniform(size=n) < p
    z = 1.3 * np.log(p / (1 - p)) + 0.4 + rng.normal(0.0, 0.25, n)
    conf = np.clip(1 / (1 + np.exp(-z)), 0.13, 0.999)
    gold = rng.choice(INTENTS, n)
    out = []
    for i in range(n):
        label = str(gold[i])
        wrong = [x for x in INTENTS if x != label]
        pred = label if correct[i] else str(rng.choice(wrong))
        out.append(
            {
                "id": f"ticket-{i:05d}",
                "prediction": pred,
                "confidence": f"{conf[i]:.3f}",
                "human_label": label if i < N_LABELLED else "",
            }
        )
    return out


def new_folder(base: Path) -> Path:
    """Create and return ``base``, or ``base-2``, ``base-3``, … if it exists: the demo
    never writes into an existing folder."""
    for k in itertools.count(1):
        path = base if k == 1 else base.with_name(f"{base.name}-{k}")
        try:
            path.mkdir(parents=True)
        except FileExistsError:
            continue
        return path
    raise AssertionError("unreachable")  # pragma: no cover


def write_data(folder: Path, seed: int = SEED) -> Path:
    path = folder / FILE_NAME
    data = rows(seed)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(data)
    return path
