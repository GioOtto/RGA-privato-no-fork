"""Worst-cohort search: it must find a planted weakness and resist noise."""

from __future__ import annotations

import numpy as np
import pytest

from rgbox import worst_cohort
from rgbox.exceptions import InputError

pytest.importorskip("pandas")


@pytest.fixture
def planted(rng):
    """A model that is fine everywhere except on one intersection of two bins."""
    pd = pytest.importorskip("pandas")
    n = 4000
    region = rng.choice(["north", "centre", "south"], n)
    channel = rng.choice(["branch", "online"], n)
    signal = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-signal))).astype(float)
    scores = 1 / (1 + np.exp(-(signal + rng.normal(0, 0.4, n))))
    broken = (region == "south") & (channel == "online")
    scores = scores.copy()
    scores[broken] = rng.random(int(broken.sum()))  # pure noise on that cohort
    frame = pd.DataFrame({"region": region, "channel": channel})
    return y, scores, frame, broken


def test_finds_the_planted_cohort(planted):
    y, scores, frame, _ = planted
    result = worst_cohort(
        y, scores, frame, min_size=100, n_permutations=100, random_state=0
    )
    assert result.cohorts
    worst = result.cohorts[0]
    assert set(worst.conditions) == {"region == 'south'", "channel == 'online'"}
    assert worst.rga < result.overall_rga
    assert worst.shortfall == pytest.approx(result.overall_rga - worst.rga)


def test_the_planted_cohort_is_significant(planted):
    y, scores, frame, _ = planted
    result = worst_cohort(
        y, scores, frame, min_size=100, n_permutations=200, random_state=0
    )
    assert result.p_value < 0.05


def test_pure_noise_features_do_not_produce_a_finding(rng):
    """The whole point of the permutation p-value: searching noise finds nothing."""
    pd = pytest.importorskip("pandas")
    p_values = []
    for seed in range(20):
        local = np.random.default_rng(1000 + seed)
        n = 1500
        signal = local.normal(size=n)
        y = (local.random(n) < 1 / (1 + np.exp(-signal))).astype(float)
        scores = 1 / (1 + np.exp(-(signal + local.normal(0, 0.5, n))))
        frame = pd.DataFrame(
            {"a": local.normal(size=n), "b": local.choice(list("xyz"), n)}
        )
        p_values.append(
            worst_cohort(
                y,
                scores,
                frame,
                min_size=120,
                n_permutations=100,
                ci=False,
                random_state=seed,
            ).p_value
        )
    p_values = np.array(p_values)
    assert np.mean(p_values < 0.05) <= 0.15
    assert np.median(p_values) > 0.2


def test_the_worst_cohort_is_always_below_the_others(planted):
    y, scores, frame, _ = planted
    result = worst_cohort(
        y, scores, frame, min_size=100, n_permutations=0, random_state=0
    )
    values = [cohort.rga for cohort in result.cohorts]
    assert values == sorted(values)
    assert result.p_value is None


def test_depth_one_searches_fewer_cohorts_than_depth_two(planted):
    y, scores, frame, _ = planted
    shallow = worst_cohort(
        y, scores, frame, max_depth=1, n_permutations=0, min_size=100, ci=False
    )
    deep = worst_cohort(
        y, scores, frame, max_depth=2, n_permutations=0, min_size=100, ci=False
    )
    assert shallow.n_cohorts_searched < deep.n_cohorts_searched
    # The planted weakness is an intersection, so only depth 2 can name it.
    assert len(deep.cohorts[0].conditions) == 2


def test_every_cohort_meets_the_size_floor(planted):
    y, scores, frame, _ = planted
    result = worst_cohort(y, scores, frame, min_size=300, n_permutations=0, ci=False)
    assert all(cohort.n >= 300 for cohort in result.cohorts)


def test_numeric_features_are_binned_into_quantiles(rng):
    pd = pytest.importorskip("pandas")
    n = 2000
    x = rng.normal(size=n)
    y = (rng.random(n) < 0.5).astype(float)
    frame = pd.DataFrame({"x": x})
    result = worst_cohort(
        y,
        rng.random(n),
        frame,
        n_bins=4,
        max_depth=1,
        min_size=100,
        n_permutations=0,
        ci=False,
    )
    assert result.n_cohorts_searched == 4
    assert all("x" in cohort.label for cohort in result.cohorts)


def test_intervals_are_attached_when_asked(planted):
    y, scores, frame, _ = planted
    result = worst_cohort(
        y, scores, frame, min_size=100, n_permutations=0, random_state=0, ci=True
    )
    worst = result.cohorts[0]
    assert worst.ci_low < worst.rga < worst.ci_high
    without = worst_cohort(y, scores, frame, min_size=100, n_permutations=0, ci=False)
    assert without.cohorts[0].ci_low is None


def test_serialises_and_explains_the_selection_effect(planted):
    import json

    y, scores, frame, _ = planted
    result = worst_cohort(
        y, scores, frame, min_size=100, n_permutations=50, random_state=0
    )
    record = result.to_dict()
    json.dumps(record, default=str)
    assert record["cohorts"][0]["cohort"] == result.cohorts[0].label
    assert "worst of many" in record["selection_note"]
    assert "cohorts searched" in str(result)


def test_is_deterministic(planted):
    y, scores, frame, _ = planted
    first = worst_cohort(
        y, scores, frame, min_size=100, n_permutations=50, random_state=7
    ).to_dict()
    second = worst_cohort(
        y, scores, frame, min_size=100, n_permutations=50, random_state=7
    ).to_dict()
    assert first == second


def test_rejects_a_frame_without_column_labels(rng):
    y = (rng.random(300) < 0.5).astype(float)
    with pytest.raises(InputError, match="column labels"):
        worst_cohort(y, rng.random(300), np.zeros((300, 2)))


def test_rejects_a_reckless_size_floor(planted):
    y, scores, frame, _ = planted
    with pytest.raises(InputError, match="at least 3"):
        worst_cohort(y, scores, frame, min_size=2)


def test_rejects_an_unsupported_depth(planted):
    y, scores, frame, _ = planted
    with pytest.raises(InputError, match="max_depth"):
        worst_cohort(y, scores, frame, max_depth=3)
