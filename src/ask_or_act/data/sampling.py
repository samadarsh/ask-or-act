"""Deterministic stratified sampling and calibration/test splitting.

Every "random" choice here is made by sorting on ``sha256(f"{seed}:{key}")`` instead of
using :mod:`random`. That gives:

* Byte-identical results on every Python version and OS. ``random.shuffle`` and
  ``random.sample`` are only guaranteed stable within one Python version.
* Stability under insertion: adding a new row to the source doesn't reshuffle the
  others, because each row's position depends only on its own key.

The committed sample manifests depend on this exact algorithm. If you change it, bump
``SAMPLER_VERSION`` and regenerate the manifests.
"""

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

SAMPLER_VERSION = 1

type Meta = Mapping[str, str | int]


class Split(StrEnum):
    CALIBRATION = "calibration"
    TEST = "test"


class SamplingError(ValueError):
    """The requested sample can't be drawn from the given candidates."""


@dataclass(frozen=True)
class Candidate:
    """One row of a source dataset that may be sampled.

    ``group``: rows with the same group always land in the same split (prevents leakage,
    e.g. two judgments of the same MT-Bench question). ``unit``: rows with the same unit
    are selected or skipped together (e.g. all human votes on one comparison).
    Both default to the row's own id.
    """

    id: str
    label: str
    fingerprint: str
    group: str = ""
    unit: str = ""
    meta: Meta = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Frozen dataclass: object.__setattr__ is the sanctioned way to fill defaults.
        if not self.group:
            object.__setattr__(self, "group", self.id)
        if not self.unit:
            object.__setattr__(self, "unit", self.id)


@dataclass(frozen=True)
class SampledExample:
    id: str
    split: Split
    label: str
    fingerprint: str
    meta: Meta = field(default_factory=dict)


def fingerprint(row: Mapping[str, Any]) -> str:
    """Short content hash of a source row, used to detect upstream data changes.

    Keys are sorted, so the hash doesn't depend on dict order. 16 hex chars = 64 bits:
    ample for integrity checks over thousands of rows, and keeps the manifests small.
    """
    blob = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def seeded_order(key: str, seed: int) -> str:
    """Sort key giving a seed-dependent pseudo-random order over keys."""
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def proportional_quotas(counts: Mapping[str, int], total: int) -> dict[str, int]:
    """Split ``total`` across labels in proportion to ``counts``.

    Uses the largest-remainder method so the quotas sum to exactly ``total``. Ties are
    broken by label name, so the result is deterministic.
    """
    available = sum(counts.values())
    if total > available:
        raise SamplingError(f"asked for {total} examples but only {available} available")
    if available == 0:
        return {}
    exact = {label: total * n / available for label, n in counts.items()}
    quotas = {label: int(x) for label, x in exact.items()}
    leftover = total - sum(quotas.values())
    by_remainder = sorted(exact, key=lambda lb: (-(exact[lb] - quotas[lb]), lb))
    for label in by_remainder[:leftover]:
        quotas[label] += 1
    return quotas


def _check_unique_ids(candidates: Sequence[Candidate]) -> None:
    dupes = [i for i, n in Counter(c.id for c in candidates).items() if n > 1]
    if dupes:
        raise SamplingError(f"duplicate candidate ids: {sorted(dupes)[:5]}")


def _sorted_output(examples: Iterable[SampledExample]) -> list[SampledExample]:
    return sorted(examples, key=lambda e: (e.split, e.id))


def _to_example(c: Candidate, split: Split) -> SampledExample:
    return SampledExample(c.id, split, c.label, c.fingerprint, c.meta)


def stratified_split(
    candidates: Sequence[Candidate],
    *,
    seed: int,
    per_label: int | None = None,
    total: int | None = None,
) -> list[SampledExample]:
    """Sample candidates stratified by label, then split them 50/50 into calibration/test.

    Pass exactly one of:

    * ``per_label``: take this many of every label (must be even); half go to each split.
    * ``total``: take ``total`` examples with label proportions matching the population;
      each split gets ``total // 2`` (test gets the extra one if odd), and each split is
      itself stratified.

    Ignores ``group``/``unit``; use :func:`grouped_split` when rows must stay together.
    """
    if (per_label is None) == (total is None):
        raise ValueError("pass exactly one of per_label or total")
    _check_unique_ids(candidates)

    by_label: dict[str, list[Candidate]] = defaultdict(list)
    for c in candidates:
        by_label[c.label].append(c)
    for members in by_label.values():
        members.sort(key=lambda c: seeded_order(c.id, seed))

    if per_label is not None:
        if per_label <= 0 or per_label % 2:
            raise ValueError("per_label must be a positive even number")
        short = {lb: len(m) for lb, m in by_label.items() if len(m) < per_label}
        if short:
            raise SamplingError(f"labels with fewer than {per_label} candidates: {short}")
        cal_q = dict.fromkeys(by_label, per_label // 2)
        test_q = dict(cal_q)
    else:
        assert total is not None
        counts = {lb: len(m) for lb, m in by_label.items()}
        cal_q = proportional_quotas(counts, total // 2)
        test_q = proportional_quotas(counts, total - total // 2)
        over = {lb: counts[lb] for lb in counts if cal_q[lb] + test_q[lb] > counts[lb]}
        if over:
            raise SamplingError(f"not enough candidates for labels: {over}")

    out: list[SampledExample] = []
    for label, members in by_label.items():
        n_cal, n_test = cal_q[label], test_q[label]
        out += [_to_example(c, Split.CALIBRATION) for c in members[:n_cal]]
        out += [_to_example(c, Split.TEST) for c in members[n_cal : n_cal + n_test]]
    return _sorted_output(out)


def grouped_split(
    candidates: Sequence[Candidate], *, seed: int, total: int
) -> list[SampledExample]:
    """Sample about ``total`` rows, never splitting a group or a unit.

    1. Groups are put in seeded order and alternately assigned to calibration and test,
       so each split gets half the groups.
    2. Within each split, units are visited in seeded order, and each unit is taken
       whole if it fits in the remaining budget of ``total // 2`` rows (or
       ``total - total // 2`` for test). Units that don't fit are skipped, so the
       budget is filled exactly when possible.

    Every row of a unit must belong to the same group.
    """
    _check_unique_ids(candidates)

    units: dict[str, list[Candidate]] = defaultdict(list)
    for c in candidates:
        units[c.unit].append(c)
    for unit, rows in units.items():
        if len({r.group for r in rows}) > 1:
            raise SamplingError(f"unit {unit!r} spans more than one group")

    groups = sorted({c.group for c in candidates}, key=lambda g: seeded_order(g, seed))
    split_of_group = {
        g: Split.CALIBRATION if i % 2 == 0 else Split.TEST for i, g in enumerate(groups)
    }
    budgets = {Split.CALIBRATION: total // 2, Split.TEST: total - total // 2}

    out: list[SampledExample] = []
    for split, budget in budgets.items():
        split_units = [u for u, rows in units.items() if split_of_group[rows[0].group] == split]
        split_units.sort(key=lambda u: seeded_order(u, seed))
        taken = 0
        for u in split_units:
            rows = units[u]
            if taken + len(rows) <= budget:
                out += [_to_example(c, split) for c in rows]
                taken += len(rows)
            if taken == budget:
                break
        if taken < budget:
            raise SamplingError(f"{split}: could only fill {taken} of {budget} rows")
    return _sorted_output(out)
