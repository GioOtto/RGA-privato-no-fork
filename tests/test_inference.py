"""The inference layer: exactness, mutual agreement, and calibration."""

from __future__ import annotations

import numpy as np
import pytest

from rgbox import rga, rga_ci, rga_compare, rga_test
from rgbox.exceptions import InputError
from rgbox.inference import bootstrap_values, influence_values, jackknife_values


def naive_jackknife(y, yhat):
    return np.array([rga(np.delete(y, k), np.delete(yhat, k)) for k in range(y.size)])


def test_fast_jackknife_is_exact(sample):
    """O(n log n) delete-one values must equal the O(n^2) ones bit for bit."""
    y, yhat = sample
    if np.ptp(y) == 0:
        pytest.skip("undefined by design")
    fast = jackknife_values(y, yhat)
    slow = naive_jackknife(y, yhat)
    assert np.max(np.abs(fast - slow)) < 1e-11


def test_bootstrap_weights_reproduce_the_resampled_sample(rng):
    """A multinomial reweighting must equal the physically resampled RGA."""
    from rgbox._ranks import sorted_index
    from rgbox.inference import _multinomial_rga

    y = rng.integers(0, 4, 200).astype(float)
    yhat = rng.integers(0, 6, 200).astype(float)
    counts = rng.multinomial(200, np.full(200, 1 / 200), size=25).astype(float)
    fast = _multinomial_rga(y, counts, sorted_index(yhat), sorted_index(y))
    for row, value in zip(counts, fast):
        expected = rga(np.repeat(y, row.astype(int)), np.repeat(yhat, row.astype(int)))
        assert value == pytest.approx(expected, abs=1e-11)


def test_the_three_methods_agree(sample):
    y, yhat = sample
    if np.ptp(y) == 0:
        pytest.skip("undefined by design")
    estimates = {
        method: rga_ci(y, yhat, method=method, n_resamples=1500, random_state=0)
        for method in ("jackknife", "influence", "bootstrap")
    }
    points = [e.estimate for e in estimates.values()]
    assert max(points) - min(points) < 1e-12
    errors = [e.standard_error for e in estimates.values()]
    if np.ptp(yhat) == 0:
        # A constant score gives exactly 0.5 on every possible sample, so the
        # estimator has no sampling variability at all. Zero is the right
        # answer, not a degenerate one.
        assert max(errors) < 1e-12
        return
    # Within 25% of each other: these are different estimators of the same
    # asymptotic variance, not the same formula computed three ways.
    assert max(errors) / min(errors) < 1.25


def test_influence_standard_error_matches_delong(rng):
    """On a binary target RGA is the AUC, so DeLong is an external oracle."""

    def delong(y, scores):
        positives, negatives = scores[y == 1], scores[y == 0]
        m, n = positives.size, negatives.size
        v10 = np.array(
            [
                (np.sum(p > negatives) + 0.5 * np.sum(p == negatives)) / n
                for p in positives
            ]
        )
        v01 = np.array(
            [
                (np.sum(positives > q) + 0.5 * np.sum(positives == q)) / m
                for q in negatives
            ]
        )
        return np.sqrt(np.var(v10, ddof=1) / m + np.var(v01, ddof=1) / n)

    for n in (200, 600, 2000):
        y = rng.binomial(1, 0.3, n).astype(float)
        scores = rng.normal(y, 1.0, n)
        ours = rga_ci(y, scores, method="influence").standard_error
        assert ours == pytest.approx(delong(y, scores), rel=0.01)


def test_jackknife_is_deterministic(continuous):
    y, yhat = continuous
    first = rga_ci(y, yhat, method="jackknife")
    second = rga_ci(y, yhat, method="jackknife")
    assert first.standard_error == second.standard_error


def test_bootstrap_is_reproducible(continuous):
    y, yhat = continuous
    kwargs = {"method": "bootstrap", "n_resamples": 400, "random_state": 42}
    assert rga_ci(y, yhat, **kwargs).ci_low == rga_ci(y, yhat, **kwargs).ci_low


def test_standard_error_shrinks_like_one_over_root_n(rng):
    errors = {}
    for n in (250, 1000, 4000):
        y = rng.normal(size=n)
        yhat = 0.6 * y + rng.normal(size=n)
        errors[n] = rga_ci(y, yhat).standard_error
    assert errors[250] / errors[1000] == pytest.approx(2.0, rel=0.35)
    assert errors[1000] / errors[4000] == pytest.approx(2.0, rel=0.35)


@pytest.mark.slow
def test_confidence_intervals_are_calibrated(rng):
    """Coverage of the nominal 95% interval, over 500 replications."""
    truth_draws = []
    big = np.random.default_rng(999)
    for _ in range(30):
        y = big.normal(size=4000)
        truth_draws.append(rga(y, 0.8 * y + big.normal(size=4000)))
    truth = float(np.mean(truth_draws))

    covered = 0
    replications = 500
    for _ in range(replications):
        y = rng.normal(size=400)
        estimate = rga_ci(y, 0.8 * y + rng.normal(size=400))
        covered += estimate.ci_low <= truth <= estimate.ci_high
    assert 0.90 <= covered / replications <= 0.99


def test_interval_kinds(continuous):
    y, yhat = continuous
    for kind in ("normal", "percentile", "basic", "bca"):
        estimate = rga_ci(
            y,
            yhat,
            method="bootstrap",
            interval=kind,
            n_resamples=800,
            random_state=1,
        )
        assert estimate.interval == kind
        assert estimate.ci_low < estimate.estimate < estimate.ci_high
    with pytest.raises(InputError, match="requires method='bootstrap'"):
        rga_ci(y, yhat, method="jackknife", interval="bca")


def test_estimate_exposes_the_gini_scale(binary):
    y, yhat = binary
    estimate = rga_ci(y, yhat)
    assert estimate.gini == pytest.approx(2 * estimate.estimate - 1)
    assert estimate.gini_ci[0] == pytest.approx(2 * estimate.ci_low - 1)
    assert float(estimate) == pytest.approx(estimate.estimate)
    assert "95% CI" in str(estimate)
    assert set(estimate.to_dict()) >= {"rga", "gini", "ci_low", "ci_high"}


# --------------------------------------------------------------- comparison


def test_identical_scores_compare_as_a_dead_heat(binary):
    y, yhat = binary
    comparison = rga_compare(y, yhat, yhat)
    assert comparison.difference == pytest.approx(0.0)
    assert comparison.p_value == pytest.approx(1.0)
    assert not comparison.significant


def test_comparison_is_antisymmetric(rng):
    y = rng.binomial(1, 0.3, 800).astype(float)
    a, b = rng.normal(y * 1.2, 1, 800), rng.normal(y * 0.7, 1, 800)
    forward = rga_compare(y, a, b)
    backward = rga_compare(y, b, a)
    assert forward.difference == pytest.approx(-backward.difference)
    assert forward.p_value == pytest.approx(backward.p_value)


def test_comparison_detects_a_real_gap(rng):
    y = rng.binomial(1, 0.35, 3000).astype(float)
    strong = rng.normal(y * 1.6, 1.0, 3000)
    weak = rng.normal(y * 0.3, 1.0, 3000)
    comparison = rga_compare(y, strong, weak)
    assert comparison.difference > 0
    assert comparison.p_value < 1e-6
    assert comparison.significant


def test_paired_comparison_beats_naive_interval_overlap(rng):
    """The point of pairing: correlated models get a tighter difference SE."""
    y = rng.normal(size=1500)
    shared = 0.7 * y + rng.normal(size=1500)
    a = shared + rng.normal(0, 0.05, 1500)
    b = shared + rng.normal(0, 0.05, 1500) - 0.02 * y
    paired = rga_compare(y, a, b).standard_error
    independent = np.hypot(rga_ci(y, a).standard_error, rga_ci(y, b).standard_error)
    assert paired < 0.5 * independent


def test_comparison_methods_agree(rng):
    y = rng.binomial(1, 0.3, 1200).astype(float)
    a, b = rng.normal(y * 1.2, 1, 1200), rng.normal(y * 0.8, 1, 1200)
    errors = [
        rga_compare(y, a, b, method=m, n_resamples=1500, random_state=0).standard_error
        for m in ("jackknife", "influence", "bootstrap")
    ]
    assert max(errors) / min(errors) < 1.3


# -------------------------------------------------------------------- tests


def test_permutation_test_rejects_a_real_signal(binary):
    y, yhat = binary
    assert rga_test(y, yhat)["p_value"] < 1e-4


def test_permutation_test_is_calibrated_under_the_null(rng):
    """p-values must be roughly uniform when the score is pure noise."""
    p_values = [
        rga_test(rng.normal(size=200), rng.normal(size=200), alternative="two-sided")[
            "p_value"
        ]
        for _ in range(400)
    ]
    assert 0.02 <= np.mean(np.array(p_values) < 0.05) <= 0.10


def test_analytic_and_monte_carlo_permutation_agree(rng):
    y = rng.normal(size=300)
    yhat = 0.3 * y + rng.normal(size=300)
    analytic = rga_test(y, yhat)
    simulated = rga_test(y, yhat, n_permutations=4000, random_state=0)
    assert analytic["null_se"] == pytest.approx(simulated["null_se"], rel=0.1)
    assert analytic["p_value"] == pytest.approx(simulated["p_value"], abs=0.03)


def test_unknown_method_rejected(continuous):
    y, yhat = continuous
    with pytest.raises(InputError, match="unknown method"):
        rga_ci(y, yhat, method="magic")
    with pytest.raises(InputError, match="unknown method"):
        rga_compare(y, yhat, yhat, method="magic")


def test_influence_values_sum_to_about_zero(continuous):
    y, yhat = continuous
    assert abs(np.mean(influence_values(y, yhat))) < 0.05


def test_bootstrap_values_shape(continuous):
    y, yhat = continuous
    draws = bootstrap_values(y, yhat, n_resamples=137, random_state=0)
    assert draws.shape == (137,)
    paired_a, paired_b = bootstrap_values(
        y, yhat, n_resamples=64, random_state=0, paired_with=yhat
    )
    # Same weights for both scores, so identical inputs give identical draws.
    assert np.allclose(paired_a, paired_b)


def test_the_delete_one_family_really_costs_two_sorts(rng, monkeypatch):
    """docs/THEORY.md 2.2 claims two sorts; it used to perform six.

    `average_ranks`, `suffix_sums_strictly_greater` and `tie_group_sums` each
    re-sorted the same array, and `_leave_one_out_sums` called all three - twice
    over, once per argument. A shared `SortedIndex` brings it back to what the
    derivation says. This asserts the count rather than the timing, so it does
    not flake on a loaded machine.
    """
    import numpy as np_module

    import rgbox._ranks as ranks_module

    calls = {"n": 0}
    real = np_module.argsort

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(ranks_module.np, "argsort", counting)

    y = rng.normal(size=500)
    yhat = 0.6 * y + rng.normal(size=500)

    calls["n"] = 0
    jackknife_values(y, yhat)
    assert calls["n"] == 2, "the exact jackknife must sort each argument once"

    calls["n"] = 0
    influence_values(y, yhat)
    assert calls["n"] == 2, "the influence function must sort each argument once"

    calls["n"] = 0
    rga(y, yhat)
    assert calls["n"] == 2


def test_bootstrap_does_not_re_sort_per_block(rng, monkeypatch):
    """The sort order does not depend on the replicate, so it is hoisted."""
    import rgbox._ranks as ranks_module

    calls = {"n": 0}
    real = np.argsort

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(ranks_module.np, "argsort", counting)

    y = rng.normal(size=400)
    yhat = 0.6 * y + rng.normal(size=400)
    calls["n"] = 0
    bootstrap_values(y, yhat, n_resamples=500, random_state=0, block_size=50)
    # Ten blocks, but still one sort per argument for the whole run.
    assert calls["n"] == 2
