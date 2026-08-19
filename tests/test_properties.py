"""Mathematical properties RGA must satisfy, stated as executable claims."""

from __future__ import annotations

import numpy as np
import pytest

from rgbox import gini_score, rga, rga_curves
from rgbox.exceptions import UndefinedMetricError


def test_bounded_in_unit_interval(sample):
    y, yhat = sample
    if np.ptp(y) == 0:
        pytest.skip("undefined by design")
    value = rga(y, yhat)
    assert -1e-12 <= value <= 1 + 1e-12


def test_perfect_concordance_is_one(rng):
    y = rng.normal(size=200)
    assert rga(y, y) == pytest.approx(1.0)
    assert rga(y, 3.0 * y + 7.0) == pytest.approx(1.0)
    assert rga(y, np.exp(y)) == pytest.approx(1.0)


def test_perfect_discordance_is_zero(rng):
    y = rng.normal(size=200)
    assert rga(y, -y) == pytest.approx(0.0, abs=1e-12)


def test_uninformative_score_is_one_half(rng):
    y = rng.normal(size=400)
    assert rga(y, np.ones(400)) == pytest.approx(0.5)
    # A score independent of y averages to 0.5 across replications.
    draws = [rga(y, rng.permutation(y)) for _ in range(300)]
    assert np.mean(draws) == pytest.approx(0.5, abs=0.01)


def test_invariant_under_increasing_transforms_of_the_score(sample):
    """Only the ordering of yhat is used."""
    y, yhat = sample
    if np.ptp(y) == 0:
        pytest.skip("undefined by design")
    baseline = rga(y, yhat)
    shifted = yhat - yhat.min() + 1e-9
    for transformed in (
        2.5 * yhat + 11.0,
        np.expm1(np.clip(yhat, -20, 20)),
        np.log(shifted),
        np.arctan(yhat),
    ):
        assert rga(y, transformed) == pytest.approx(baseline, abs=1e-10)


def test_invariant_under_increasing_affine_transforms_of_the_target(rng):
    y = rng.normal(size=300)
    yhat = 0.6 * y + rng.normal(size=300)
    baseline = rga(y, yhat)
    assert rga(4.0 * y + 100.0, yhat) == pytest.approx(baseline, abs=1e-9)


def test_not_invariant_under_arbitrary_monotone_transforms_of_the_target(rng):
    """RGA is Pearson-like in y and Spearman-like in yhat.

    That asymmetry is the defining property of a Gini correlation, and it is
    the reason RGA is not simply Spearman's rho.
    """
    y = rng.lognormal(0, 1.5, 400)
    yhat = rng.normal(size=400)
    assert rga(y, yhat) != pytest.approx(rga(np.log(y), yhat), abs=1e-3)


def test_is_asymmetric(rng):
    a = rng.normal(size=300)
    b = 0.5 * a + rng.normal(size=300)
    assert rga(a, b) != pytest.approx(rga(b, a), abs=1e-6)


def test_reversing_the_score_reflects_around_one_half(sample):
    y, yhat = sample
    if np.ptp(y) == 0:
        pytest.skip("undefined by design")
    assert rga(y, -yhat) == pytest.approx(1.0 - rga(y, yhat), abs=1e-12)


def test_equals_gini_correlation_identity(sample):
    """RGA == 1/2 + cov(y, R(yhat)) / (2 cov(y, R(y)))."""
    y, yhat = sample
    if np.ptp(y) == 0:
        pytest.skip("undefined by design")
    scipy_stats = pytest.importorskip("scipy.stats")
    centred = y - y.mean()
    expected = 0.5 + (
        centred @ scipy_stats.rankdata(yhat)
    ) / (2 * (centred @ scipy_stats.rankdata(y)))
    assert rga(y, yhat) == pytest.approx(expected, abs=1e-12)


def test_equals_auroc_on_binary_targets(rng):
    metrics = pytest.importorskip("sklearn.metrics")
    for n, prevalence in [(80, 0.5), (400, 0.2), (1500, 0.05), (600, 0.9)]:
        y = rng.binomial(1, prevalence, n).astype(float)
        if y.sum() in (0, n):
            continue
        scores = rng.normal(y, 1.0, n)
        assert rga(y, scores) == pytest.approx(
            metrics.roc_auc_score(y, scores), abs=1e-12
        )


def test_equals_auroc_with_heavy_score_ties(rng):
    """Coarse scores (shallow trees, rating grades) tie heavily."""
    metrics = pytest.importorskip("sklearn.metrics")
    y = rng.binomial(1, 0.3, 600).astype(float)
    grades = np.round(rng.normal(y, 1.0, 600) * 2) / 2
    assert rga(y, grades) == pytest.approx(metrics.roc_auc_score(y, grades), abs=1e-12)


def test_equals_wilcoxon_mann_whitney(rng):
    """RGA is the normalised WMW statistic on a binary target."""
    y = rng.binomial(1, 0.4, 200).astype(float)
    scores = rng.normal(y, 1.0, 200)
    positives, negatives = scores[y == 1], scores[y == 0]
    comparison = positives[:, None] - negatives[None, :]
    statistic = (np.sum(comparison > 0) + 0.5 * np.sum(comparison == 0)) / (
        positives.size * negatives.size
    )
    assert rga(y, scores) == pytest.approx(statistic, abs=1e-12)


def test_gini_score_is_the_banking_accuracy_ratio(rng):
    metrics = pytest.importorskip("sklearn.metrics")
    y = rng.binomial(1, 0.25, 800).astype(float)
    scores = rng.normal(y, 1.0, 800)
    assert gini_score(y, scores) == pytest.approx(
        2 * metrics.roc_auc_score(y, scores) - 1, abs=1e-12
    )


def test_integer_weights_replicate_the_sample(rng):
    y = rng.normal(size=150)
    yhat = rng.normal(size=150)
    weights = rng.integers(1, 6, 150)
    assert rga(y, yhat, weights=weights) == pytest.approx(
        rga(np.repeat(y, weights), np.repeat(yhat, weights)), abs=1e-12
    )


def test_uniform_weights_are_a_no_op(sample):
    y, yhat = sample
    if np.ptp(y) == 0:
        pytest.skip("undefined by design")
    assert rga(y, yhat, weights=np.full(y.size, 3.0)) == pytest.approx(
        rga(y, yhat), abs=1e-12
    )


def test_curve_areas_reproduce_the_closed_form(rng):
    y = rng.gamma(2.0, 1.0, 300)
    yhat = 0.5 * y + rng.normal(size=300)
    curves = rga_curves(y, yhat)
    assert curves.rga == pytest.approx(rga(y, yhat), abs=1e-12)
    assert curves.lorenz[0] == 0.0 and curves.lorenz[-1] == pytest.approx(1.0)
    # Lorenz is the best ordering, dual Lorenz the worst; concordance sits
    # between them everywhere.
    assert np.all(curves.concordance <= curves.dual_lorenz + 1e-12)
    assert np.all(curves.concordance >= curves.lorenz - 1e-12)


def test_curves_reject_non_positive_totals(rng):
    with pytest.raises(UndefinedMetricError):
        rga_curves(rng.normal(-10, 1, 50), rng.normal(size=50))
