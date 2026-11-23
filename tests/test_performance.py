"""100,000-row files stay fast and small (docs/cli-design.md §8).

The limits are generous (CI machines vary); the measured numbers are in the doc. A
regression that makes ``check`` quadratic, or keeps several copies of every row, would
blow through them.
"""

import time
import tracemalloc
from pathlib import Path

import numpy as np

from ask_or_act.predictions import check

N = 100_000
MAX_SECONDS = 15.0
MAX_PEAK_MB = 500.0


def _write(path: Path) -> None:
    rng = np.random.default_rng(1)
    labels = np.array([f"intent_{k:02d}" for k in range(77)])
    gold = rng.choice(labels, N)
    skill = rng.uniform(0.3, 1.0, N)
    pred = np.where(rng.uniform(size=N) < skill, gold, rng.choice(labels, N))
    conf = np.clip(skill + 0.08, 0.0, 0.999).round(4)
    split = np.where(rng.uniform(size=N) < 0.5, "calibration", "test")
    lines = [f"r{i},{pred[i]},{conf[i]},{gold[i]},{split[i]}" for i in range(N)]
    path.write_text("id,prediction,confidence,human_label,split\n" + "\n".join(lines) + "\n")


def test_check_on_100k_rows_is_fast_and_small(tmp_path: Path) -> None:
    path = tmp_path / "big.csv"
    _write(path)
    start = time.perf_counter()
    result = check(path)
    elapsed = time.perf_counter() - start
    assert result.ok
    assert result.n_rows == N
    assert elapsed < MAX_SECONDS, f"check took {elapsed:.1f} s"

    tracemalloc.start()
    check(path)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak / 1e6 < MAX_PEAK_MB, f"peak {peak / 1e6:.0f} MB"


def test_thresholds_metrics_and_compare_on_100k_rows(tmp_path: Path) -> None:
    """Fewer bootstrap resamples than the default (time is linear in them); the doc
    records the full-default timings."""
    from ask_or_act import api

    path = tmp_path / "big.csv"
    _write(path)
    table = api.load(path).table
    start = time.perf_counter()
    th = api.find_thresholds(table)
    assert th.n_cal + (th.n_test or 0) == N
    ev = api.evaluate(table, n_boot=100)
    assert ev.n > 40_000
    cmp = api.compare(table, table, n_boot=20)
    assert cmp.n == ev.n
    elapsed = time.perf_counter() - start
    assert elapsed < 60.0, f"took {elapsed:.1f} s"
