"""Fairness: correctness of the gap, its interval, and its interpretation."""

from __future__ import annotations

import numpy as np
import pytest

from rgbox import labels_from_dummies, proxy_leakage, rga, rga_parity, rgf
from rgbox.exceptions import InputError

pytest.importorskip("pandas")


def test_returns_a_number_not_a_sentence(rng):
    """Upstream returned a formatted string, unusable in any pipeline."""
    y = rng.binomial(1, 0.4, 600).astype(float)
    scores = rng.normal(y, 1.0, 600)
    groups = rng.binomial(1, 0.5, 600)
    result = rga_parity(y, scores, groups)
    assert isinstance(float(result), float)
    assert 0.0 <= float(result) <= 1.0
    assert result.gap == float(result)
    assert isinstance(result.to_dict(), dict)


def test_per_group_values_match_a_manual_split(rng):
    y = rng.binomial(1, 0.35, 800).astype(float)
    scores = rng.normal(y, 1.0, 800)
    groups = rng.binomial(1, 0.5, 800)
    result = rga_parity(y, scores, groups, min_group_size=0)
    for record in result.groups:
        mask = groups == record.group
        assert record.rga == pytest.approx(rga(y[mask], scores[mask]))
        assert record.n == int(mask.sum())


def test_gap_is_max_minus_min(rng):
    y = rng.binomial(1, 0.4, 900).astype(float)
    scores = rng.normal(y, 1.0, 900)
    groups = rng.choice(["a", "b", "c"], 900)
    result = rga_parity(y, scores, groups, min_group_size=0)
    values = [g.rga for g in result.groups if g.included]
    assert result.gap == pytest.approx(max(values) - min(values))


def test_every_group_carries_a_confidence_interval(rng):
    y = rng.binomial(1, 0.3, 700).astype(float)
    scores = rng.normal(y, 1.0, 700)
    groups = rng.binomial(1, 0.5, 700)
    result = rga_parity(y, scores, groups)
    for record in result.groups:
        assert record.ci_low < record.rga < record.ci_high


def test_small_groups_are_reported_but_excluded_from_the_gap(rng):
    """A 12-obligor segment must not drive the headline number."""
    y = np.concatenate([rng.binomial(1, 0.3, 800), rng.binomial(1, 0.3, 12)]).astype(
        float
    )
    scores = np.concatenate([rng.normal(y[:800], 1.0), rng.normal(0, 1, 12)])
    groups = np.array(["big"] * 800 + ["tiny"] * 12)
    result = rga_parity(y, scores, groups, min_group_size=50)
    labels = {record.group: record for record in result.groups}
    assert labels["tiny"].n == 12
    assert not labels["tiny"].included
    assert "tiny" in result.excluded
    # With only one eligible group there is no gap to report.
    assert result.gap is None
    assert "not computable" in str(result)


def test_perfect_parity_gives_a_non_significant_gap(rng):
    """Identical data-generating processes: any observed gap is noise."""
    n = 4000
    y = rng.binomial(1, 0.35, n).astype(float)
    scores = rng.normal(y, 1.0, n)
    groups = rng.binomial(1, 0.5, n)  # independent of everything
    result = rga_parity(y, scores, groups)
    assert result.gap_p_value > 0.05


def test_a_real_disparity_is_detected(rng):
    """One group is ranked well, the other by noise."""
    n = 3000
    groups = rng.binomial(1, 0.5, n)
    y = rng.binomial(1, 0.4, n).astype(float)
    scores = np.where(groups == 1, rng.normal(y * 2.5, 1.0, n), rng.normal(0, 1.0, n))
    result = rga_parity(y, scores, groups, n_resamples=2000)
    assert result.gap > 0.15
    # The adjusted value is a Monte-Carlo tail probability, so it saturates at
    # 1 / (n_resamples + 1) rather than shrinking without limit. Hitting the
    # floor exactly is the strongest verdict the simulation can return.
    assert result.gap_p_value == pytest.approx(1 / 2001)
    assert result.gap_p_value_unadjusted < 1e-6


def test_two_groups_need_no_correction(rng):
    """One pair, nothing selected: adjusted and unadjusted must agree."""
    n = 2500
    y = rng.binomial(1, 0.35, n).astype(float)
    scores = rng.normal(y * 0.8, 1.0, n)
    groups = rng.binomial(1, 0.5, n)
    result = rga_parity(y, scores, groups, n_resamples=4000, random_state=0)
    assert result.gap_p_value == pytest.approx(result.gap_p_value_unadjusted, abs=0.02)
    assert "1 pair" in result.multiplicity


def test_the_correction_tightens_as_pairs_multiply(rng):
    """Same data, more levels: the adjustment must grow with the family."""
    n = 3000
    y = rng.binomial(1, 0.35, n).astype(float)
    scores = rng.normal(y * 0.8, 1.0, n)
    ratios = []
    for k in (2, 3, 5):
        groups = rng.integers(0, k, n)  # independent of y and scores
        result = rga_parity(y, scores, groups, n_resamples=4000, random_state=0)
        ratios.append(result.gap_p_value / max(result.gap_p_value_unadjusted, 1e-12))
        assert len(result.pairwise) == k * (k - 1) // 2
        # Never anti-conservative: correcting can only raise a p-value.
        assert result.gap_p_value >= result.gap_p_value_unadjusted - 1e-9
    assert ratios[0] < ratios[1] < ratios[2]


def test_max_t_controls_the_family_wise_error_rate():
    """Type I error under exact parity, five groups, at a nominal 5%.

    This is the defect the correction exists for: selecting the widest of ten
    pairs and referring it to a normal rejected 27% of the time. Twelve
    replications is far too few to pin the rate down, so the assertion is only
    that the uncorrected test is visibly broken and the corrected one is not -
    which is a gap wide enough to survive the noise.
    """
    master = np.random.default_rng(4242)
    n, trials = 1200, 12
    unadjusted = adjusted = 0
    for _ in range(trials):
        draw = np.random.default_rng(master.integers(2**32))
        y = draw.binomial(1, 0.35, n).astype(float)
        scores = draw.normal(y * 0.8, 1.0, n)  # identical for every group
        groups = draw.integers(0, 5, n)
        result = rga_parity(y, scores, groups, n_resamples=1000, random_state=0)
        unadjusted += result.gap_p_value_unadjusted < 0.05
        adjusted += result.gap_p_value < 0.05
    assert unadjusted >= 2  # the old behaviour, ~27% of the time
    assert adjusted <= 1  # the corrected one, ~5%


def test_gap_interval_is_documented_as_non_negative(rng):
    """max - min cannot be negative, so its percentile CI never covers 0."""
    n = 2000
    y = rng.binomial(1, 0.35, n).astype(float)
    scores = rng.normal(y, 1.0, n)
    groups = rng.binomial(1, 0.5, n)
    result = rga_parity(y, scores, groups, n_resamples=800, random_state=0)
    assert result.gap_ci[0] >= 0.0
    assert "does not contain 0" in result.GAP_CI_NOTE
    # The bias-corrected point estimate is smaller than the raw max - min.
    assert result.gap_bias_corrected <= result.gap + 1e-12


def test_pairwise_comparisons_cover_every_pair(rng):
    y = rng.binomial(1, 0.4, 1500).astype(float)
    scores = rng.normal(y, 1.0, 1500)
    groups = rng.choice(["a", "b", "c"], 1500)
    result = rga_parity(y, scores, groups)
    assert len(result.pairwise) == 3
    for record in result.pairwise:
        assert record["ci_low"] < record["difference"] < record["ci_high"]


def test_single_class_subgroup_is_flagged_not_crashed(rng):
    """RGA is undefined where a subgroup has no variation in the target."""
    y = np.concatenate([rng.binomial(1, 0.4, 500), np.ones(200)]).astype(float)
    scores = rng.normal(size=700)
    groups = np.array(["mixed"] * 500 + ["all_default"] * 200)
    result = rga_parity(y, scores, groups, min_group_size=0)
    flagged = {record.group: record for record in result.groups}
    assert flagged["all_default"].rga is None
    assert not flagged["all_default"].included
    assert flagged["mixed"].rga is not None


def test_group_length_must_match(rng):
    with pytest.raises(InputError, match="entries but y has"):
        rga_parity(rng.normal(size=100), rng.normal(size=100), rng.integers(0, 2, 50))


def test_rgf_matches_the_r_code_definition(rng):
    """RGF = RGA(full scores, scores without the protected attribute)."""
    pd = pytest.importorskip("pandas")
    n = 800
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "gender": rng.binomial(1, 0.5, n).astype(float),
        }
    )

    def blind(X):
        return X["x"]

    def reliant(X):
        return X["x"] + 5.0 * X["gender"]

    assert rgf(frame, frame, blind, "gender")["rgf"] == pytest.approx(1.0, abs=1e-9)
    assert rgf(frame, frame, reliant, "gender")["rgf"] < 0.9
    assert rgf(frame, frame, reliant, "gender")["rge"] > 0.1


def test_proxy_leakage_finds_the_proxy(rng):
    pd = pytest.importorskip("pandas")
    n = 1000
    protected = rng.binomial(1, 0.5, n).astype(float)
    frame = pd.DataFrame(
        {
            "gender": protected,
            "proxy": protected * 3 + rng.normal(0, 0.3, n),
            "unrelated": rng.normal(size=n),
        }
    )
    result = proxy_leakage(frame, frame, lambda X: X["unrelated"], "gender")
    ranked = [row["variable"] for row in result["proxies"]]
    assert ranked[0] == "proxy"
    assert result["proxies"][0]["leakage"] > 0.8
    unrelated = next(r for r in result["proxies"] if r["variable"] == "unrelated")
    assert unrelated["leakage"] < 0.2


def test_rgf_accepts_a_one_hot_encoded_attribute(rng):
    """A multi-level attribute is removed as a unit, not one dummy at a time."""
    pd = pytest.importorskip("pandas")
    n = 1200
    region = rng.choice(["north", "centre", "south"], n, p=[0.45, 0.30, 0.25])
    frame = pd.DataFrame({"x": rng.normal(size=n)})
    frame = pd.concat(
        [frame, pd.get_dummies(region, prefix="r", drop_first=True).astype(float)],
        axis=1,
    )
    dummies = [c for c in frame.columns if c.startswith("r_")]

    def reliant(X):
        return X["x"] + 4.0 * X[dummies[0]] - 3.0 * X[dummies[1]]

    whole = rgf(frame, frame, reliant, dummies)
    assert whole["attribute"] == dummies
    # Not a general law: group RGE is not monotone, so a group *can* score
    # below its members (see the counterexample in explainability.py's module
    # docstring). Here the two levels push the score in opposite directions, so
    # removing both costs more than removing either - and this asserts that
    # this fixture behaves that way, not that every fixture must.
    per_level = [rgf(frame, frame, reliant, d)["rge"] for d in dummies]
    assert whole["rge"] >= max(per_level) - 1e-12
    assert rgf(frame, frame, lambda X: X["x"], dummies)["rgf"] == pytest.approx(
        1.0, abs=1e-9
    )


def test_a_tuple_is_one_multiindex_column_not_a_group():
    """Regression: a tuple label was expanded into one column per element.

    A ``MultiIndex`` column *is* a tuple, so unwrapping it turned
    ``("demo", "gender")`` into the two non-existent columns ``"demo"`` and
    ``"gender"`` and raised. A list stays the only way to spell a group.
    """
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(11)
    n = 400
    frame = pd.DataFrame(
        {
            ("demo", "gender"): rng.integers(0, 2, n).astype(float),
            ("fin", "income"): rng.normal(size=n),
        }
    )
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)

    def model(X):
        return X[("fin", "income")] + 2.0 * X[("demo", "gender")]

    result = rgf(frame, frame, model, ("demo", "gender"))
    assert result["attribute"] == ("demo", "gender")
    assert result["rge"] > 0.0


def test_proxy_leakage_accepts_a_one_hot_group(rng):
    """A candidate is a proxy if it reconstructs *any* level of the attribute."""
    pd = pytest.importorskip("pandas")
    n = 1200
    region = rng.choice(["north", "centre", "south"], n, p=[0.45, 0.30, 0.25])
    frame = pd.DataFrame(
        {
            "unrelated": rng.normal(size=n),
            # Carries the "south" level only, so it is invisible to any single
            # other dummy but is a genuine proxy for the attribute.
            "postcode": (region == "south").astype(float) + rng.normal(0, 0.05, n),
        }
    )
    frame = pd.concat(
        [frame, pd.get_dummies(region, prefix="r", drop_first=True).astype(float)],
        axis=1,
    )
    dummies = [c for c in frame.columns if c.startswith("r_")]

    result = proxy_leakage(frame, frame, lambda X: X["unrelated"], dummies)
    # The dummies are the attribute; they must not be scored as its proxies.
    assert {row["variable"] for row in result["proxies"]} == {"unrelated", "postcode"}
    top = result["proxies"][0]
    assert top["variable"] == "postcode"
    assert top["leakage"] > 0.8
    assert top["level"] in dummies
    unrelated = next(r for r in result["proxies"] if r["variable"] == "unrelated")
    assert unrelated["leakage"] < 0.2


def test_labels_from_dummies_inverts_a_drop_first_encoding(rng):
    pd = pytest.importorskip("pandas")
    n = 500
    region = rng.choice(["north", "centre", "south"], n, p=[0.45, 0.30, 0.25])
    dummies = pd.get_dummies(region, prefix="r", drop_first=True).astype(float)
    labels = labels_from_dummies(dummies, list(dummies.columns), reference="r_centre")

    # get_dummies drops the first level alphabetically, i.e. "centre".
    rebuilt = np.where(
        labels == "r_centre", "centre", [str(v).removeprefix("r_") for v in labels]
    )
    assert list(rebuilt) == list(region)


def test_labels_from_dummies_rejects_overlapping_rows():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"a": [1.0, 0.0, 1.0], "b": [1.0, 1.0, 0.0]})
    with pytest.raises(InputError, match="more than one"):
        labels_from_dummies(frame, ["a", "b"])


def test_labels_from_dummies_rejects_non_binary_columns():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [1.0, 0.0, 0.0]})
    with pytest.raises(InputError, match="0/1 encoding"):
        labels_from_dummies(frame, ["a", "b"])
