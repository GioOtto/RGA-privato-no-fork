"""Worst-cohort search: it must find a planted weakness and resist noise."""

from __future__ import annotations

import numpy as np
import pytest

from rgbox import worst_cohort
from rgbox.cohorts import _bin_conditions
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


def test_a_numeric_feature_with_a_nan_is_still_binned_as_numeric(rng):
    """One NaN must not turn an interval-scaled feature into a categorical one.

    as_1d_float raises the same InputError for "not numeric" and for "numeric
    but contains a NaN". Conflating the two produced one bin per distinct
    float - so no bin could clear min_size, the search reported nothing, and
    depth 2 spent O(n^2) getting there.
    """
    pd = pytest.importorskip("pandas")
    n = 800
    y = (rng.random(n) < 0.4).astype(float)
    scores = rng.random(n) * 0.5 + y * 0.3
    values = rng.normal(size=n)
    values[7] = np.nan
    frame = pd.DataFrame({"f": values})

    conditions = _bin_conditions(frame, ["f"], 4)
    labels = [name for _, name, _ in conditions]
    assert len(conditions) == 5, labels  # four quantile bins plus missing
    assert labels[-1] == "f is missing"

    result = worst_cohort(y, scores, frame, min_size=100, n_permutations=0)
    assert result.n_cohorts_searched > 0


def test_every_row_lands_in_some_bin(rng):
    """Missing is a bin, not a silent deletion - for numbers and for strings."""
    pd = pytest.importorskip("pandas")
    n = 300
    numeric = rng.normal(size=n)
    numeric[[1, 2, 3]] = np.nan
    numeric[4] = np.inf
    categorical = np.array(rng.choice(["x", "y"], n), dtype=object)
    categorical[[5, 6]] = None
    frame = pd.DataFrame({"num": numeric, "cat": categorical})

    for column in ("num", "cat"):
        covered = np.zeros(n, dtype=bool)
        for _, _, mask in _bin_conditions(frame, [column], 4):
            covered |= mask
        assert covered.all(), f"{column}: {int((~covered).sum())} rows in no bin"


def test_an_all_missing_feature_contributes_no_bin(rng):
    """Its one bin would hold every row, which is the sample, not a cohort.

    Left in, it scored a shortfall of 0 against itself and then intersected
    with every other bin to produce an exact duplicate of that bin, so `top`
    came back half-filled with copies. A constant column does the same.
    """
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"f": np.full(200, np.nan), "c": ["z"] * 200})
    assert _bin_conditions(frame, ["f", "c"], 4) == []


def test_a_useless_feature_does_not_duplicate_the_useful_ones(rng):
    pd = pytest.importorskip("pandas")
    n = 400
    y = (rng.random(n) < 0.4).astype(float)
    scores = rng.random(n) * 0.5 + y * 0.3
    frame = pd.DataFrame({"good": rng.normal(size=n), "dead": np.full(n, np.nan)})

    result = worst_cohort(y, scores, frame, min_size=50, n_permutations=0, ci=False)
    labels = [cohort.label for cohort in result.cohorts]
    assert len(labels) == len(set(labels))
    assert not any("dead" in label for label in labels)


def test_a_categorical_column_with_pandas_na_is_binned(rng):
    """`arr == level` over the whole column made pandas.NA raise.

    NA == "x" is NA rather than False, and numpy asking for its truth value
    raises "boolean value of NA is ambiguous", so one missing entry in a
    string-dtype column took the whole search down - against the documented
    promise that missing values join an "is missing" bin.
    """
    pd = pytest.importorskip("pandas")
    n = 300
    y = (rng.random(n) < 0.4).astype(float)
    scores = rng.random(n) * 0.5 + y * 0.3
    values = ["x"] * 140 + ["y"] * 120 + [pd.NA] * 40

    for dtype in ("string", "category", "object"):
        column = (
            np.array(values, dtype=object)
            if dtype == "object"
            else pd.array(values, dtype=dtype)
        )
        frame = pd.DataFrame({"seg": column})
        assert [name for _, name, _ in _bin_conditions(frame, ["seg"], 4)] == [
            "seg == 'x'",
            "seg == 'y'",
            "seg is missing",
        ], dtype

        result = worst_cohort(y, scores, frame, min_size=30, n_permutations=0, ci=False)
        assert "seg is missing" in {cohort.label for cohort in result.cohorts}, dtype


def test_cohorts_compare_and_hash(planted):
    """A frozen dataclass with an ndarray field breaks == and hash() for good."""
    y, scores, frame, _ = planted
    first = worst_cohort(y, scores, frame, min_size=100, n_permutations=0)
    second = worst_cohort(y, scores, frame, min_size=100, n_permutations=0)

    assert first.cohorts[0] == second.cohorts[0]
    assert first == second
    assert len(set(first.cohorts)) == len(first.cohorts)


def test_the_selection_note_describes_the_test_that_is_actually_run(planted):
    """The note is serialised into to_dict() and quoted in reports.

    It used to say the p-value permuted the *score* and took the *best*-found
    cohort. The module docstring argues at length that permuting the score is
    the wrong null, and the code takes the worst.
    """
    y, scores, frame, _ = planted
    note = worst_cohort(y, scores, frame, min_size=100, n_permutations=0).SELECTION_NOTE
    assert "permutations of the cohort definitions" in note
    assert "worst-found cohort" in note
    assert "score permutations" not in note
