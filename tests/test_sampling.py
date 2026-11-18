from collections import Counter

import pytest

from ask_or_act.data.sampling import (
    Candidate,
    SamplingError,
    Split,
    fingerprint,
    grouped_split,
    proportional_quotas,
    seeded_order,
    stratified_split,
)


def make(n_per_label: dict[str, int]) -> list[Candidate]:
    return [
        Candidate(id=f"{label}-{i}", label=label, fingerprint=f"fp{label}{i}")
        for label, n in n_per_label.items()
        for i in range(n)
    ]


# ------------------------------------------------------------- hashing primitives


def test_fingerprint_ignores_key_order() -> None:
    assert fingerprint({"a": 1, "b": "x"}) == fingerprint({"b": "x", "a": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})
    assert len(fingerprint({"a": 1})) == 16


def test_seeded_order_is_pinned() -> None:
    # Golden value. If this fails, the hashing changed and every committed manifest is
    # stale: bump SAMPLER_VERSION and regenerate, don't just update this number.
    assert seeded_order("test:0", 20261118)[:16] == "4ec1bd8c1fdac34d"


def test_seeded_order_depends_on_seed() -> None:
    assert seeded_order("x", 1) != seeded_order("x", 2)


# ------------------------------------------------------------- proportional_quotas


def test_quotas_sum_exactly_and_follow_proportions() -> None:
    q = proportional_quotas({"a": 600, "b": 300, "c": 100}, 500)
    assert q == {"a": 300, "b": 150, "c": 50}
    q = proportional_quotas({"a": 1, "b": 1, "c": 1}, 2)
    assert sum(q.values()) == 2
    assert q == {"a": 1, "b": 1, "c": 0}  # remainder ties broken by label name


def test_quotas_reject_more_than_available() -> None:
    with pytest.raises(SamplingError):
        proportional_quotas({"a": 3}, 4)


# ------------------------------------------------------------- stratified_split


def test_per_label_takes_equal_halves() -> None:
    out = stratified_split(make({"a": 30, "b": 25}), seed=1, per_label=20)
    counts = Counter((e.label, e.split) for e in out)
    assert counts == {
        ("a", Split.CALIBRATION): 10,
        ("a", Split.TEST): 10,
        ("b", Split.CALIBRATION): 10,
        ("b", Split.TEST): 10,
    }


def test_per_label_is_deterministic_and_seed_sensitive() -> None:
    cands = make({"a": 50, "b": 50})
    first = stratified_split(cands, seed=7, per_label=10)
    assert first == stratified_split(list(reversed(cands)), seed=7, per_label=10)
    assert first != stratified_split(cands, seed=8, per_label=10)


def test_per_label_rejects_short_labels_and_odd_counts() -> None:
    with pytest.raises(SamplingError, match="fewer than 20"):
        stratified_split(make({"a": 30, "b": 5}), seed=1, per_label=20)
    with pytest.raises(ValueError, match="even"):
        stratified_split(make({"a": 30}), seed=1, per_label=3)


def test_total_mode_stratifies_each_split() -> None:
    out = stratified_split(make({"safe": 600, "unsafe": 400}), seed=1, total=100)
    counts = Counter((e.label, e.split) for e in out)
    assert counts == {
        ("safe", Split.CALIBRATION): 30,
        ("unsafe", Split.CALIBRATION): 20,
        ("safe", Split.TEST): 30,
        ("unsafe", Split.TEST): 20,
    }


def test_exactly_one_mode_required() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        stratified_split(make({"a": 4}), seed=1)
    with pytest.raises(ValueError, match="exactly one"):
        stratified_split(make({"a": 4}), seed=1, per_label=2, total=2)


def test_duplicate_ids_rejected() -> None:
    c = Candidate(id="x", label="a", fingerprint="f")
    with pytest.raises(SamplingError, match="duplicate"):
        stratified_split([c, c], seed=1, total=2)


def test_adding_a_row_does_not_reshuffle_others() -> None:
    # Hash ordering: a new candidate can displace at most one selection per split.
    base = make({"a": 40})
    extra = [*base, Candidate(id="a-new", label="a", fingerprint="f")]
    before = {e.id for e in stratified_split(base, seed=3, per_label=10)}
    after = {e.id for e in stratified_split(extra, seed=3, per_label=10)}
    assert len(before - after) <= 2


# ------------------------------------------------------------- grouped_split


def grouped_cands() -> list[Candidate]:
    """10 groups (questions) x 3 units (comparisons) x 1-3 rows (votes)."""
    out = []
    for g in range(10):
        for u in range(3):
            for v in range(1 + (g + u) % 3):
                out.append(
                    Candidate(
                        id=f"g{g}u{u}v{v}",
                        label="model_a",
                        fingerprint="f",
                        group=f"g{g}",
                        unit=f"g{g}u{u}",
                    )
                )
    return out


def test_grouped_split_keeps_groups_and_units_whole() -> None:
    cands = grouped_cands()
    out = grouped_split(cands, seed=5, total=20)
    assert Counter(e.split for e in out) == {Split.CALIBRATION: 10, Split.TEST: 10}

    by_id = {c.id: c for c in cands}
    split_of_group: dict[str, Split] = {}
    for e in out:
        g = by_id[e.id].group
        assert split_of_group.setdefault(g, e.split) == e.split, "group straddles splits"

    chosen = {e.id for e in out}
    for unit in {c.unit for c in cands}:
        members = {c.id for c in cands if c.unit == unit}
        assert members <= chosen or not (members & chosen), f"unit {unit} partially taken"


def test_grouped_split_is_deterministic() -> None:
    cands = grouped_cands()
    assert grouped_split(cands, seed=5, total=20) == grouped_split(cands[::-1], seed=5, total=20)


def test_grouped_split_errors_when_budget_cannot_be_filled() -> None:
    with pytest.raises(SamplingError, match="could only fill"):
        grouped_split(grouped_cands(), seed=5, total=1000)


def test_unit_spanning_groups_rejected() -> None:
    cands = [
        Candidate(id="1", label="x", fingerprint="f", group="g1", unit="u"),
        Candidate(id="2", label="x", fingerprint="f", group="g2", unit="u"),
    ]
    with pytest.raises(SamplingError, match="spans"):
        grouped_split(cands, seed=1, total=2)
