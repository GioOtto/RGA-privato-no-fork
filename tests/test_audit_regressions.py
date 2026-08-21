"""Regression tests for the defects found auditing 44a736c.

Every test here fails on the commit before the fix. They are collected in one
file, rather than scattered into the per-module suites, because they share a
theme worth keeping visible: each one is a case where the library returned a
*plausible* answer instead of raising - a p-value of 1 on the largest disparity
representable, an RGA reported as its own complement, a robustness curve made
of one repeated number, an importance computed by a method nobody named. The
existing suites cover the happy paths thoroughly; what they had in common was
trusting that a wrong argument would announce itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from rgbox import (
    InputError,
    UndefinedMetricError,
    contamination_curve,
    outcome_parity,
    rga,
    rga_curves,
    rgbox_report,
    rge,
    rge_shapley,
    rgr_curve,
    worst_cohort,
)
from rgbox._validation import as_group_labels
from rgbox.inference import bootstrap_values, rga_test
from rgbox.predictors import predict_scores

pd = pytest.importorskip("pandas")


# --------------------------------------------------------------------------
# outcome_parity: the null must not be the observed variance
# --------------------------------------------------------------------------


@pytest.fixture
def split_sample():
    """Two groups of 100; group 'a' never selected, group 'b' always."""
    y = np.tile(np.r_[np.zeros(50), np.ones(50)], 2)
    groups = np.r_[["a"] * 100, ["b"] * 100]
    decisions = np.r_[np.zeros(100), np.ones(100)]
    return y, decisions, groups


def test_a_zero_versus_one_gap_is_significant(split_sample):
    """The headline defect: p = 1.0 on a 100-point selection-rate gap.

    Both plug-in standard errors are exactly zero at rates of 0 and 1, so the
    statistic was 0/0 -> 0 and the p-value 2 * (1 - Phi(0)) = 1. The max-T
    draws degenerated identically, being scaled by the same zero errors.
    """
    y, decisions, groups = split_sample
    criterion = outcome_parity(y, decisions, groups, min_group_size=50, random_state=0)[
        "demographic_parity"
    ]
    assert criterion.gap == pytest.approx(1.0)
    assert criterion.gap_p_value_unadjusted < 1e-40
    # The adjusted value cannot go below the resampling floor, and should be there.
    assert criterion.gap_p_value == pytest.approx(1 / 2001)
    # Pooled, not plug-in: sqrt(0.5 * 0.5 / 100) per group.
    pair = criterion.pairwise[0]
    assert pair["standard_error"] == 0.0
    assert pair["null_standard_error"] == pytest.approx(np.hypot(0.05, 0.05))


def test_a_degenerate_gap_gets_a_usable_interval(split_sample):
    """Wald collapsed to (1.0, 1.0); Agresti-Caffo does not."""
    y, decisions, groups = split_sample
    criterion = outcome_parity(y, decisions, groups, min_group_size=50, random_state=0)[
        "demographic_parity"
    ]
    low, high = criterion.gap_ci
    assert low < high
    assert 0.9 < low < 1.0
    assert high <= 1.0


def test_exact_parity_is_still_not_significant(rng):
    """The fix must not buy its power by rejecting the null.

    Three groups drawn from one common rate, so every gap is noise. The
    family-wise p-value has to stay near nominal; an uncorrected one would not.
    """
    hits = 0
    trials = 200
    for trial in range(trials):
        y = (rng.random(450) < 0.5).astype(float)
        groups = rng.choice(["a", "b", "c"], 450)
        decisions = (rng.random(450) < 0.3).astype(float)
        result = outcome_parity(
            y,
            decisions,
            groups,
            min_group_size=50,
            n_resamples=500,
            random_state=trial,
        )
        hits += result["demographic_parity"].gap_p_value < 0.05
    assert hits / trials < 0.12


def test_everybody_selected_is_not_a_disparity(rng):
    """Pooled rate 1 means every group agrees, so p = 1 is the right answer."""
    y = (rng.random(200) < 0.5).astype(float)
    groups = np.r_[["a"] * 100, ["b"] * 100]
    criterion = outcome_parity(
        y, np.ones(200), groups, min_group_size=50, random_state=0
    )["demographic_parity"]
    assert criterion.gap == 0.0
    assert criterion.gap_p_value_unadjusted == pytest.approx(1.0)


def test_a_nan_threshold_is_rejected(rng):
    """`values >= nan` is False everywhere, which read as perfect parity."""
    y = (rng.random(200) < 0.5).astype(float)
    groups = np.r_[["a"] * 100, ["b"] * 100]
    with pytest.raises(InputError, match="finite"):
        outcome_parity(y, rng.random(200), groups, threshold=float("nan"))


def test_zero_resamples_is_rejected(rng):
    """(1 + 0) / (0 + 1) made every gap, however extreme, come back p = 1.0."""
    y = (rng.random(200) < 0.5).astype(float)
    groups = np.r_[["a"] * 100, ["b"] * 100]
    with pytest.raises(InputError, match="n_resamples"):
        outcome_parity(y, (rng.random(200) < 0.5).astype(float), groups, n_resamples=0)


def test_partial_equalized_odds_is_none_not_a_number(rng):
    """One computable component is not equalised odds, and must not be reported as it."""
    n = 900
    groups = np.where(np.arange(n) < 800, "big", "small")
    y = np.zeros(n)
    y[:400] = 1.0  # 'small' has no positives, so the TPR gap does not exist
    result = outcome_parity(
        y,
        (np.arange(n) % 3 == 0).astype(float),
        groups,
        min_group_size=50,
        random_state=0,
    )
    assert result["equal_opportunity"].gap is None
    assert result["predictive_equality"].gap is not None
    assert result.equalized_odds is None
    assert result.to_dict()["equalized_odds"] is None
    assert any("equalised odds is None" in note for note in result.notes)


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------


def test_block_size_zero_raises_instead_of_hanging(binary):
    """`take = min(0, remaining)` is 0 forever, so `done` never advanced."""
    y, yhat = binary
    with pytest.raises(InputError, match="block_size"):
        bootstrap_values(y, yhat, n_resamples=10, block_size=0)


@pytest.mark.parametrize("bad", [0, -1, 2.5, "50"])
def test_bad_block_size_is_rejected(binary, bad):
    y, yhat = binary
    with pytest.raises(InputError):
        bootstrap_values(y, yhat, n_resamples=10, block_size=bad)


def test_unknown_alternative_is_rejected(binary):
    """ "grater" ran a two-sided test and returned a different p-value."""
    y, yhat = binary
    with pytest.raises(InputError, match="alternative"):
        rga_test(y, yhat, alternative="grater")


def test_alternatives_still_disagree_as_they_should(rng):
    """Guard against "validate it" turning into "collapse them all into one".

    Deliberately a weak signal: on a well-separated sample every one-sided and
    two-sided p-value underflows to 0.0 and the test would pass vacuously.
    """
    y = rng.binomial(1, 0.4, 300).astype(float)
    yhat = rng.normal(y * 0.15, 1.0, 300)
    one_sided = rga_test(y, yhat, alternative="greater")["p_value"]
    two_sided = rga_test(y, yhat, alternative="two-sided")["p_value"]
    less = rga_test(y, yhat, alternative="less")["p_value"]
    assert 0.0 < one_sided < 1.0
    assert two_sided == pytest.approx(2.0 * min(one_sided, less))
    assert less == pytest.approx(1.0 - one_sided)


@pytest.mark.parametrize("alternative", ["greater", "less", "two-sided"])
def test_a_constant_score_gives_p_one_everywhere(binary, alternative):
    """A point-mass null: nothing is more extreme, so the exact p-value is 1.

    The analytic branch reported 0.5 one-sided; the Monte-Carlo branch raised
    ZeroDivisionError building its own statistic on the same input.
    """
    y, _ = binary
    constant = np.full(y.size, 3.0)
    analytic = rga_test(y, constant, alternative=alternative)
    monte_carlo = rga_test(
        y, constant, alternative=alternative, n_permutations=50, random_state=0
    )
    assert analytic["p_value"] == pytest.approx(1.0)
    assert monte_carlo["p_value"] == pytest.approx(1.0)
    assert analytic["statistic"] == 0.0
    assert monte_carlo["statistic"] == 0.0


@pytest.mark.parametrize("bad", [0, -3])
def test_bad_permutation_counts_are_rejected(binary, bad):
    y, yhat = binary
    with pytest.raises(InputError, match="n_permutations"):
        rga_test(y, yhat, n_permutations=bad)


# --------------------------------------------------------------------------
# core
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1e-8, 1e-3, 1.0, 7.0, 1e8])
def test_rga_is_invariant_to_the_scale_of_the_weights(binary, scale):
    """Same relative weights, same answer - `1e-8` used to be "degenerate"."""
    y, yhat = binary
    reference = rga(y, yhat, weights=np.ones(y.size))
    assert rga(y, yhat, weights=np.full(y.size, scale)) == pytest.approx(reference)


def test_weight_scale_invariance_holds_for_uneven_weights(binary, rng):
    y, yhat = binary
    weights = rng.random(y.size) + 0.1
    reference = rga(y, yhat, weights=weights)
    assert rga(y, yhat, weights=weights * 1e-9) == pytest.approx(reference)
    assert rga(y, yhat, weights=weights * 1e9) == pytest.approx(reference)


def test_rga_curves_matches_rga_on_a_constant_target():
    """`sum(y) > 0` does not imply dispersion; this used to be ZeroDivisionError."""
    y, yhat = np.ones(20), np.arange(20.0)
    with pytest.raises(UndefinedMetricError):
        rga_curves(y, yhat)
    with pytest.raises(UndefinedMetricError):
        rga(y, yhat)


def test_rga_curves_still_works_on_a_dispersed_target(binary):
    """The new guard must not fire on ordinary input."""
    y, yhat = binary
    assert rga_curves(y, yhat).rga == pytest.approx(rga(y, yhat))


# --------------------------------------------------------------------------
# pos_label
# --------------------------------------------------------------------------


@pytest.fixture
def margin_model(rng):
    svm = pytest.importorskip("sklearn.svm")
    X = rng.normal(size=(300, 3))
    labels = np.where(X[:, 0] + 0.3 * rng.normal(size=300) > 0, "good", "bad")
    return svm.LinearSVC().fit(X, labels), X, labels


def test_pos_label_flips_a_decision_function(margin_model):
    """The margin points at classes_[1]; asking for classes_[0] means negate it.

    Ignoring pos_label here reported RGA as 1 - RGA, with no error: 0.96 as
    0.04 on this fixture.
    """
    model, X, labels = margin_model
    negative, positive = model.classes_
    y_positive = (labels == positive).astype(float)
    y_negative = (labels == negative).astype(float)

    default = rga(y_positive, predict_scores(model, X))
    flipped = rga(y_negative, predict_scores(model, X, pos_label=negative))
    kept = rga(y_positive, predict_scores(model, X, pos_label=positive))

    assert default > 0.9  # the fixture is a well-separated problem
    assert flipped == pytest.approx(default)
    assert kept == pytest.approx(default)
    assert predict_scores(model, X, pos_label=negative) == pytest.approx(
        -predict_scores(model, X)
    )


def test_a_bogus_pos_label_is_rejected_on_the_margin_branch(margin_model):
    """The predict_proba branch rejected this; decision_function accepted it."""
    model, X, _ = margin_model
    with pytest.raises(InputError, match="not one of the model's classes"):
        predict_scores(model, X, pos_label="not-a-class")


def test_make_rga_scorer_honours_pos_label(rng):
    """`pos_label=0` and `pos_label=1` returned scorers that behaved identically."""
    linear_model = pytest.importorskip("sklearn.linear_model")
    selection = pytest.importorskip("sklearn.model_selection")
    from rgbox.sklearn_api import make_rga_scorer

    X = rng.normal(size=(300, 3))
    labels = np.where(X[:, 0] > 0, "yes", "no")
    model = linear_model.LogisticRegression(max_iter=500)

    # Binary RGA is symmetric under relabelling, so honouring pos_label means
    # both orientations agree - and both are computable at all, which they were
    # not for string labels before, since the metric saw the raw labels.
    as_yes = selection.cross_val_score(
        model, X, labels, scoring=make_rga_scorer(pos_label="yes"), cv=3
    )
    as_no = selection.cross_val_score(
        model, X, labels, scoring=make_rga_scorer(pos_label="no"), cv=3
    )
    assert as_yes == pytest.approx(as_no)
    assert np.all(as_yes > 0.9)


def test_pos_label_with_a_regressor_scorer_is_rejected():
    from rgbox.sklearn_api import make_rga_scorer

    with pytest.raises(InputError, match="needs_proba=False"):
        make_rga_scorer(needs_proba=False, pos_label=1)


# --------------------------------------------------------------------------
# explainability
# --------------------------------------------------------------------------


@pytest.fixture
def frame(rng):
    data = pd.DataFrame(
        {"x1": rng.normal(size=300), "x2": rng.normal(size=300)},
    )
    return data, (lambda X: np.asarray(X["x1"]) * 1.5 + np.asarray(X["x2"]) * 0.5)


@pytest.mark.parametrize("bad", ["meam", "retrian", "MEAN", "permutation", ""])
def test_unknown_removal_methods_are_rejected(frame, bad):
    """Anything unrecognised silently computed mean substitution."""
    data, model = frame
    with pytest.raises(InputError, match="unknown removal method"):
        rge(data, data, model, ["x1"], method=bad)


def test_a_retrain_typo_no_longer_returns_a_mean_substitution(frame):
    """ "retrian" returned a number instead of demanding the refit callable."""
    data, model = frame
    with pytest.raises(InputError, match="unknown removal method"):
        rge(data, data, model, ["x1"], method="retrian")
    # while the correctly spelled one still asks for what it needs
    with pytest.raises(InputError, match="refit"):
        rge(data, data, model, ["x1"], method="retrain")


@pytest.mark.parametrize("bad", [0, -1])
def test_bad_shapley_permutation_counts_are_rejected(frame, bad):
    """0 raised ZeroDivisionError; -1 returned -0.0 for every predictor."""
    data, model = frame
    with pytest.raises(InputError, match="n_permutations"):
        rge_shapley(data, data, model, ["x1", "x2"], n_permutations=bad)


def test_shapley_handles_multiindex_columns_in_the_sampled_path(rng):
    """A list of equal-length tuples became a 2-D object array when permuted."""
    data = pd.DataFrame(rng.normal(size=(200, 2)))
    data.columns = pd.MultiIndex.from_tuples([("num", "x1"), ("num", "x2")])
    columns = [("num", "x1"), ("num", "x2")]
    result = rge_shapley(
        data,
        data,
        lambda X: np.asarray(X.iloc[:, 0]),
        columns,
        n_permutations=5,
        random_state=0,
    )
    assert set(result.values) == set(columns)
    assert all(isinstance(key, tuple) for key in result.values)


# --------------------------------------------------------------------------
# cohorts
# --------------------------------------------------------------------------


@pytest.fixture
def cohort_data(rng):
    n = 600
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "region": rng.choice(["N", "S", "C"], size=n),
        }
    )
    y = (rng.random(n) < 0.5).astype(float)
    return y, rng.random(n), frame


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"top": -1}, "top"),
        ({"top": 0}, "top"),
        ({"n_permutations": -1}, "n_permutations"),
        ({"n_bins": 0}, "n_bins"),
        ({"n_bins": 1}, "n_bins"),
        ({"n_bins": -1}, "n_bins"),
    ],
)
def test_worst_cohort_rejects_degenerate_counts(cohort_data, kwargs, match):
    """`top=-1` returned all-but-the-best; `n_bins<2` searched nothing quietly."""
    y, yhat, frame = cohort_data
    with pytest.raises(InputError, match=match):
        worst_cohort(y, yhat, frame, ci=False, **kwargs)


def test_worst_cohort_still_allows_skipping_the_p_value(cohort_data):
    """n_permutations=0 is documented as meaningful and must stay allowed."""
    y, yhat, frame = cohort_data
    assert worst_cohort(y, yhat, frame, n_permutations=0, ci=False).p_value is None


def test_the_n_permutations_docstring_matches_the_code():
    """It said "permutations of the score" - the null the module argues against."""
    doc = worst_cohort.__doc__
    assert "Permutations of the score" not in doc
    assert "cohort definitions" in doc


# --------------------------------------------------------------------------
# robustness
# --------------------------------------------------------------------------


def test_shuffle_has_no_robustness_curve(frame):
    """Six grid points, one experiment: every value was bit-for-bit identical."""
    data, model = frame
    with pytest.raises(InputError, match="no magnitude to sweep"):
        rgr_curve(data, model, ["x1"], kind="shuffle", random_state=0)


def test_shuffle_is_still_available_as_a_single_benchmark(frame):
    """Rejecting the curve must not remove the perturbation."""
    from rgbox import rgr

    data, model = frame
    results = rgr(data, model, ["x1"], kind="shuffle", random_state=0)
    assert results[0].kind == "shuffle"
    assert 0.0 <= results[0].rgr <= 1.0


@pytest.mark.parametrize("bad", [0, -2])
def test_bad_repeat_counts_are_rejected(frame, bad):
    from rgbox import rgr

    data, model = frame
    with pytest.raises(InputError, match="n_repeats"):
        rgr(data, model, ["x1"], kind="gaussian", n_repeats=bad)


def test_an_unknown_perturbation_kind_is_rejected(frame):
    from rgbox import perturb

    data, _ = frame
    with pytest.raises(InputError, match="unknown perturbation kind"):
        perturb(data, "x1", 0.1, kind="tailswapp")


# --------------------------------------------------------------------------
# accuracy
# --------------------------------------------------------------------------


def test_zero_repeats_is_rejected_by_contamination_curve(binary):
    """It left every contaminated row's mean over an empty list: NaN."""
    y, yhat = binary
    with pytest.raises(InputError, match="n_repeats"):
        contamination_curve(y, yhat, n_repeats=0)


@pytest.mark.parametrize("bad", [2.0, -0.5])
def test_out_of_range_fractions_are_rejected(binary, bad):
    """These reached rng.choice and surfaced as raw NumPy ValueErrors."""
    y, yhat = binary
    with pytest.raises(InputError, match="fractions"):
        contamination_curve(y, yhat, fractions=(0.0, bad), n_repeats=2)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def test_two_dimensional_group_labels_are_rejected():
    """A 2x2 block became four labels in row-major order, silently."""
    with pytest.raises(InputError, match="one-dimensional"):
        as_group_labels(np.array([["a", "b"], ["c", "d"]]), "groups", 4)


@pytest.mark.parametrize(
    "shape_ok",
    [np.array(["a", "b", "c", "d"]), np.array([["a"], ["b"], ["c"], ["d"]])],
)
def test_column_shaped_group_labels_are_still_accepted(shape_ok):
    assert list(as_group_labels(shape_ok, "groups", 4)) == ["a", "b", "c", "d"]


# --------------------------------------------------------------------------
# the report artefact
# --------------------------------------------------------------------------


@pytest.fixture
def report_inputs(rng):
    n = 400
    frame = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    y = (rng.random(n) < 1 / (1 + np.exp(-frame.x1.to_numpy()))).astype(int)
    return y, frame, (lambda X: np.asarray(X["x1"]))


def test_html_is_escaped(report_inputs):
    """Model names and labels come from the caller's data, not from this package."""
    y, frame, model = report_inputs
    payload = "<img src=x onerror=alert(1)>"
    report = rgbox_report(
        y=y,
        X_test=frame,
        model=model,
        X_train=frame,
        variables=["x1"],
        model_name=payload,
        random_state=0,
    )
    html = report.to_html()
    assert payload not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    # the surrounding structure this package generates is still real markup
    assert "<h1>" in html and "<table>" in html


def test_quotes_in_labels_are_escaped_too(report_inputs):
    y, frame, model = report_inputs
    report = rgbox_report(
        y=y,
        X_test=frame,
        model=model,
        X_train=frame,
        variables=["x1"],
        model_name='a "quoted" name',
        random_state=0,
    )
    assert "&quot;quoted&quot;" in report.to_html()


@pytest.mark.parametrize(("level", "label"), [(0.90, "90% CI"), (0.99, "99% CI")])
def test_the_report_states_the_level_it_used(report_inputs, level, label):
    """A 90% report was captioned "95% CI"; the numbers were right all along."""
    y, frame, model = report_inputs
    report = rgbox_report(
        y=y,
        X_test=frame,
        model=model,
        X_train=frame,
        variables=["x1"],
        level=level,
        random_state=0,
    )
    markdown = report.to_markdown()
    assert label in markdown
    assert "95% CI" not in markdown
    assert report.accuracy["rga"]["level"] == pytest.approx(level)


def test_the_default_level_still_says_95(report_inputs):
    y, frame, model = report_inputs
    report = rgbox_report(
        y=y, X_test=frame, model=model, X_train=frame, variables=["x1"], random_state=0
    )
    assert "95% CI" in report.to_markdown()


def test_one_failing_section_does_not_take_the_report_down(rng):
    """A string column among `variables` reached perturb() and killed the run."""
    n = 400
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "region": rng.choice(["N", "S"], size=n),
        }
    )
    y = (rng.random(n) < 0.5).astype(float)
    report = rgbox_report(
        y=y,
        X_test=frame,
        model=(lambda X: np.asarray(X["x1"])),
        X_train=frame,
        variables=["x1", "region"],
        random_state=0,
    )
    # accuracy is the report and is never skipped
    assert report.accuracy["rga"]["rga"] > 0.0
    assert any("robustness skipped" in warning for warning in report.warnings)
    assert report.to_markdown()
    assert report.to_html()


def test_a_bug_in_the_model_is_not_swallowed_as_a_skipped_section(rng):
    """Only RGBoxError is a "section that does not apply"."""
    n = 400
    frame = pd.DataFrame({"x1": rng.normal(size=n)})
    y = (rng.random(n) < 0.5).astype(float)

    calls = {"n": 0}

    def flaky(X):
        calls["n"] += 1
        if calls["n"] > 1:  # the baseline scoring succeeds, the perturbed one does not
            raise RuntimeError("a genuine bug inside the model")
        return np.asarray(X["x1"])

    with pytest.raises(RuntimeError, match="a genuine bug"):
        rgbox_report(
            y=y,
            X_test=frame,
            model=flaky,
            X_train=frame,
            variables=["x1"],
            random_state=0,
        )
