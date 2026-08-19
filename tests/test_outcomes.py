"""Outcome-based fairness criteria: correctness, intervals, multiplicity."""

from __future__ import annotations

import numpy as np
import pytest

from rgbox import outcome_parity
from rgbox.exceptions import InputError
from rgbox.outcomes import FOUR_FIFTHS


@pytest.fixture
def biased(rng):
    """One group is selected far more often than the others."""
    n = 3000
    groups = rng.choice(["a", "b", "c"], n, p=[0.5, 0.3, 0.2])
    x = rng.normal(size=n) + (groups == "c") * 1.0
    y = (rng.random(n) < 1 / (1 + np.exp(-x))).astype(float)
    scores = 1 / (1 + np.exp(-(x + rng.normal(0, 0.5, n))))
    return y, scores, groups


def test_rates_match_a_manual_computation(biased):
    y, scores, groups = biased
    result = outcome_parity(y, scores, groups, threshold=0.5, random_state=0)
    decisions = scores >= 0.5

    for record in result["demographic_parity"].groups:
        mask = groups == record.group
        assert record.n_eligible == int(mask.sum())
        assert record.rate == pytest.approx(decisions[mask].mean())

    for record in result["equal_opportunity"].groups:
        mask = (groups == record.group) & (y == 1)
        assert record.rate == pytest.approx(decisions[mask].mean())

    for record in result["predictive_equality"].groups:
        mask = (groups == record.group) & (y == 0)
        assert record.rate == pytest.approx(decisions[mask].mean())


def test_gap_is_max_minus_min_of_the_rates(biased):
    y, scores, groups = biased
    result = outcome_parity(y, scores, groups, threshold=0.5, random_state=0)
    criterion = result["demographic_parity"]
    rates = [g.rate for g in criterion.groups if g.included]
    assert criterion.gap == pytest.approx(max(rates) - min(rates))
    assert criterion.worst_group != criterion.best_group


def test_disparate_impact_is_the_ratio_and_flags_the_four_fifths_rule(biased):
    y, scores, groups = biased
    result = outcome_parity(y, scores, groups, threshold=0.5, random_state=0)
    rates = [g.rate for g in result["demographic_parity"].groups if g.included]
    assert result.disparate_impact == pytest.approx(min(rates) / max(rates))
    record = result.to_dict()
    assert record["four_fifths_rule_met"] == (result.disparate_impact >= FOUR_FIFTHS)
    low, high = result["demographic_parity"].ratio_ci
    assert 0 < low < result.disparate_impact < high <= 1.0


def test_equalized_odds_is_the_worse_of_the_two_error_rate_gaps(biased):
    y, scores, groups = biased
    result = outcome_parity(y, scores, groups, threshold=0.5, random_state=0)
    assert result.equalized_odds == pytest.approx(
        max(result["equal_opportunity"].gap, result["predictive_equality"].gap)
    )


def test_a_threshold_is_required_for_scores(biased):
    y, scores, groups = biased
    with pytest.raises(InputError, match="threshold"):
        outcome_parity(y, scores, groups)


def test_pre_thresholded_decisions_need_no_threshold(biased):
    y, scores, groups = biased
    with_threshold = outcome_parity(y, scores, groups, threshold=0.5, random_state=0)
    already = outcome_parity(y, (scores >= 0.5).astype(float), groups, random_state=0)
    assert already["demographic_parity"].gap == pytest.approx(
        with_threshold["demographic_parity"].gap
    )


def test_perfect_parity_is_not_significant(rng):
    n = 2400
    groups = rng.choice(["a", "b", "c"], n)
    scores = rng.random(n)  # independent of the group by construction
    y = (rng.random(n) < 0.4).astype(float)
    result = outcome_parity(y, scores, groups, threshold=0.5, random_state=0)
    assert result["demographic_parity"].gap_p_value > 0.05


def test_a_real_disparity_is_detected(biased):
    y, scores, groups = biased
    result = outcome_parity(y, scores, groups, threshold=0.5, random_state=0)
    assert result["demographic_parity"].gap_p_value < 0.01
    assert result["demographic_parity"].gap > 0.1


def test_max_t_controls_the_family_wise_error_rate(rng):
    """The correction that rga_parity needed is needed here for the same reason."""
    adjusted_hits = unadjusted_hits = 0
    trials = 150
    for _ in range(trials):
        n = 1000
        groups = rng.integers(0, 5, n)  # five groups, ten pairs, exact parity
        scores = rng.random(n)
        y = (rng.random(n) < 0.4).astype(float)
        result = outcome_parity(
            y,
            scores,
            groups,
            threshold=0.5,
            min_group_size=50,
            n_resamples=500,
            random_state=int(rng.integers(1e9)),
        )
        criterion = result["demographic_parity"]
        adjusted_hits += criterion.gap_p_value < 0.05
        unadjusted_hits += criterion.gap_p_value_unadjusted < 0.05
    assert adjusted_hits / trials < 0.12
    assert unadjusted_hits > adjusted_hits


def test_wilson_intervals_survive_a_degenerate_rate(rng):
    """A group nobody is selected in must not produce a zero-width interval."""
    n = 400
    groups = np.where(np.arange(n) < 200, "a", "b")
    scores = np.where(groups == "a", 0.9, 0.1)
    y = (rng.random(n) < 0.5).astype(float)
    result = outcome_parity(y, scores, groups, threshold=0.5, random_state=0)
    rates = {g.group: g for g in result["demographic_parity"].groups}
    assert rates["b"].rate == 0.0
    assert rates["b"].ci_high > 0.0
    assert rates["a"].rate == 1.0
    assert rates["a"].ci_low < 1.0


def test_a_group_can_be_eligible_for_one_criterion_and_not_another(rng):
    """equal_opportunity conditions on y == 1, so its group sizes differ."""
    n = 900
    groups = np.where(np.arange(n) < 800, "big", "small")
    y = np.zeros(n)
    y[:400] = 1.0  # the 'small' group has no positives at all
    scores = rng.random(n)
    result = outcome_parity(
        y, scores, groups, threshold=0.5, min_group_size=50, random_state=0
    )
    opportunity = {g.group: g for g in result["equal_opportunity"].groups}
    assert opportunity["small"].n_eligible == 0
    assert opportunity["small"].rate is None
    assert result["equal_opportunity"].gap is None
    # ...while demographic parity, which conditions on nothing, still works.
    assert result["demographic_parity"].gap is not None


def test_non_binary_target_is_refused(rng):
    y = rng.normal(size=300)
    with pytest.raises(InputError, match="binary"):
        outcome_parity(y, rng.random(300), rng.integers(0, 2, 300), threshold=0.5)


def test_missing_group_labels_are_refused(rng):
    y = (rng.random(300) < 0.5).astype(float)
    groups = np.array(["a"] * 300, dtype=object)
    groups[:40] = None
    with pytest.raises(InputError, match="missing label"):
        outcome_parity(y, rng.random(300), groups, threshold=0.5)


def test_results_serialise_and_are_deterministic(biased):
    import json

    y, scores, groups = biased
    first = outcome_parity(
        y, scores, groups, threshold=0.5, random_state=0, attribute="region"
    ).to_dict()
    second = outcome_parity(
        y, scores, groups, threshold=0.5, random_state=0, attribute="region"
    ).to_dict()
    assert first == second
    assert json.loads(json.dumps(first, default=str))["attribute"] == "region"


def test_str_names_the_threshold_and_the_criteria(biased):
    y, scores, groups = biased
    text = str(outcome_parity(y, scores, groups, threshold=0.5, random_state=0))
    assert "threshold = 0.5" in text
    assert "demographic_parity" in text
    assert "equal_opportunity" in text
    assert "four-fifths" in text


def test_interpretation_warns_against_reading_it_as_ranking_quality(biased):
    y, scores, groups = biased
    result = outcome_parity(y, scores, groups, threshold=0.5, random_state=0)
    assert "not a substitute" in result.INTERPRETATION
    assert "rga_parity" in result.to_dict()["interpretation"]
