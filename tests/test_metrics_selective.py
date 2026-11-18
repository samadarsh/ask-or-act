"""Risk–coverage, stepwise AURC with tie blocks, AUROC."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ask_or_act.metrics.selective import (
    aurc,
    auroc,
    excess_aurc,
    n_unique,
    optimal_aurc,
    risk_coverage,
)

rng = np.random.default_rng(11)


def test_aurc_without_ties_equals_the_textbook_sum() -> None:
    c = rng.permutation(np.linspace(0, 1, 200))
    z = rng.random(200) < c
    order = np.argsort(-c)
    errors = np.cumsum(~z[order])
    textbook = float(np.mean(errors / np.arange(1, 201)))  # (1/N) Σ_i risk(top-i)
    assert aurc(c, z) == pytest.approx(textbook)


def test_all_tied_is_a_single_block() -> None:
    z = np.array([True, False, True, True])
    rc = risk_coverage(np.full(4, 0.8), z)
    assert list(rc.coverage) == [1.0]
    assert list(rc.risk) == [0.25]
    assert aurc(np.full(4, 0.8), z) == pytest.approx(0.25)


def test_ties_make_aurc_order_independent() -> None:
    c = np.array([0.9, 0.9, 0.9, 0.5, 0.5])
    z = np.array([False, True, True, True, False])
    base = aurc(c, z)
    for _ in range(10):
        p = rng.permutation(5)
        assert aurc(c[p], z[p]) == pytest.approx(base)
    # Hand value: block 1 (3 items, 1 error): Δcov 0.6 × risk 1/3; block 2: 0.4 × 2/5.
    assert base == pytest.approx(0.6 / 3 + 0.4 * 2 / 5)


def test_stepwise_differs_from_trapezoid_on_sparse_scores() -> None:
    # The arXiv 2608.04899 pitfall: with few distinct values, a trapezoid would credit
    # operating points that no threshold can reach.
    c = np.array([0.9] * 5 + [0.1] * 5)
    z = np.array([True] * 4 + [False] + [True] + [False] * 4)
    rc = risk_coverage(c, z)
    trapezoid = float(
        np.trapezoid(np.concatenate([[rc.risk[0]], rc.risk]), np.concatenate([[0.0], rc.coverage]))
    )
    assert aurc(c, z) == pytest.approx(0.5 * 0.2 + 0.5 * 0.5)
    assert aurc(c, z) != pytest.approx(trapezoid)


def test_oracle_and_excess() -> None:
    z = rng.random(300) < 0.7
    assert excess_aurc(z.astype(float), z) == pytest.approx(0.0)
    c = rng.random(300)
    assert excess_aurc(c, z) > 0
    assert optimal_aurc(np.ones(5, dtype=bool)) == 0.0


@pytest.mark.parametrize("decimals", [None, 1])
def test_auroc_matches_sklearn_including_ties(decimals: int | None) -> None:
    c = rng.random(800)
    if decimals is not None:
        c = np.round(c, decimals)
    z = rng.random(800) < c
    assert auroc(c, z) == pytest.approx(roc_auc_score(z, c))


def test_auroc_undefined_with_one_class() -> None:
    assert auroc(np.array([0.2, 0.9]), np.array([True, True])) is None


def test_n_unique() -> None:
    assert n_unique(np.array([0.1, 0.1, 0.5])) == 2


def test_empty_input() -> None:
    assert np.isnan(aurc(np.zeros(0), np.zeros(0, dtype=bool)))
