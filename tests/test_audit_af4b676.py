"""Regression tests for the defects found re-auditing af4b676 (1.0.2).

Two of these are defects the 1.0.2 fixes introduced or left behind, which is
the reason this file exists separately from `test_audit_3b20cdf.py`: a fix that
drops a validation call inherits responsibility for what that call was
checking.
"""

from __future__ import annotations

import numpy as np
import pytest

from rgbox import InputError, rga_ovr, rga_parity, rgr
from rgbox.predictors import resolve_columns

pd = pytest.importorskip("pandas")


# --------------------------------------------------------------------------
# rga_ovr: dropping as_1d_float dropped its missing-value check with it
# --------------------------------------------------------------------------


def test_rga_ovr_rejects_a_missing_class_label():
    """A None row is 0 in every indicator: a negative for every class at once.

    No exception, and a presentable RGA for a sample in which one row belongs
    to no class. Measured before the fix: rga = 1.0000 on a target of
    ["a", "b", None, "c"], with n reported as 4 and only 3 rows classified.
    """
    y = np.array(["a", "b", None, "c"], dtype=object)
    proba = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4], [0.1, 0.1, 0.8]]
    )
    with pytest.raises(InputError, match="missing label"):
        rga_ovr(y, proba, classes=["a", "b", "c"])


def test_rga_ovr_rejects_a_nan_in_a_numeric_target():
    y = np.array([0.0, 1.0, np.nan, 1.0])
    proba = np.array([[0.8, 0.2], [0.2, 0.8], [0.5, 0.5], [0.1, 0.9]])
    with pytest.raises(InputError, match="missing label"):
        rga_ovr(y, proba, classes=[0.0, 1.0])


def test_rga_ovr_rejects_classes_that_omit_an_observed_level():
    """`classes=["a","b","d"]` on a target holding "c" scored happily.

    The "c" rows became a negative for every listed class, quietly changing
    each one-vs-rest problem, and "d" was reported as merely absent.
    """
    y = np.array(["a", "b", "c", "a"], dtype=object)
    proba = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4], [0.7, 0.2, 0.1]]
    )
    with pytest.raises(InputError, match="not named in 'classes'"):
        rga_ovr(y, proba, classes=["a", "b", "d"])


def test_rga_ovr_rejects_duplicated_classes():
    y = np.array(["a", "b", "c", "a"], dtype=object)
    proba = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4], [0.7, 0.2, 0.1]]
    )
    with pytest.raises(InputError, match="repeats"):
        rga_ovr(y, proba, classes=["a", "a", "b"])


def test_rga_ovr_requires_classes_for_heterogeneous_labels():
    """`sorted(key=repr)` invented a column order and inverted the result.

    `repr("a")` starts with a quote, so `[1, "a"]` sorts to `["a", 1]` and the
    two columns of `proba` swap. Measured before the fix on a *perfect*
    classifier over that target: rga = 0.0000, i.e. reported as perfectly
    inverted, with no error anywhere. No ordering rule can recover the order
    `proba`'s columns were built in, so the caller has to say.
    """
    y = np.array([1, "a", 1, "a"], dtype=object)
    proba = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    with pytest.raises(InputError, match="no common natural ordering"):
        rga_ovr(y, proba)
    # Told the order, it gets the right answer.
    assert rga_ovr(y, proba, classes=[1, "a"])["rga"] == pytest.approx(1.0)


def test_rga_ovr_rejects_an_unhashable_class_name():
    y = np.array(["a", "b", "c", "a"], dtype=object)
    proba = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4], [0.7, 0.2, 0.1]]
    )
    with pytest.raises(InputError, match="unhashable entry"):
        rga_ovr(y, proba, classes=["a", "b", ["c"]])


def test_rga_ovr_default_classes_keep_numeric_order():
    """`sorted(key=repr)` would put 10.0 before 2.0 and transpose two columns."""
    y = np.array([2.0, 10.0, 2.0, 10.0])
    proba = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    result = rga_ovr(y, proba)
    assert [row["class"] for row in result["per_class"]] == [2.0, 10.0]
    assert result["rga"] == pytest.approx(1.0)


def test_rga_ovr_still_accepts_a_clean_string_target():
    """The 1.0.2 feature must survive the validation added around it."""
    y = np.array(["cat", "dog", "bird", "cat", "dog", "bird"], dtype=object)
    proba = np.array(
        [
            [0.1, 0.7, 0.2],
            [0.1, 0.2, 0.7],
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
            [0.6, 0.3, 0.1],
        ]
    )
    result = rga_ovr(y, proba, classes=["bird", "cat", "dog"])
    assert result["n"] == 6
    assert all(row["rga"] is not None for row in result["per_class"])


# --------------------------------------------------------------------------
# Typed errors on the two remaining untyped paths
# --------------------------------------------------------------------------


def test_duplicate_columns_of_mixed_type_raise_a_typed_error():
    """`sorted({"x", 1})` raised TypeError from inside the error path itself.

    pandas allows a frame to mix label types, so ["x", 1, "x", 1] is a legal
    duplicate list; the InputError meant to name the duplicates was replaced by
    a TypeError naming neither them nor the argument.
    """
    frame = pd.DataFrame({"x": [1.0, 2.0], 1: [3.0, 4.0]})
    with pytest.raises(InputError, match="duplicated column"):
        resolve_columns(["x", 1, "x", 1], frame, "variables")


def test_unhashable_group_labels_raise_a_typed_error():
    """A list-valued object column is 1-D and non-missing, then died on hash()."""
    labels = pd.Series([["A"], ["B"], ["A"], ["B"]], dtype=object)
    with pytest.raises(InputError, match="unhashable label"):
        rga_parity([0.0, 1.0, 0.0, 1.0], [0.1, 0.9, 0.2, 0.8], labels, min_group_size=0)


def test_ordinary_string_labels_are_unaffected(rng):
    """The hashability guard must not reject the normal object column."""
    n = 200
    y = (rng.random(n) < 0.5).astype(float)
    scores = rng.random(n) * 0.4 + y * 0.3
    labels = np.array(["north", "south"] * (n // 2), dtype=object)
    result = rga_parity(y, scores, labels, min_group_size=10)
    assert result.gap is not None


# --------------------------------------------------------------------------
# The multi-draw RGR interval must describe the mean it is printed beside
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ci_method", ["jackknife", "influence"])
def test_the_rgr_interval_shrinks_as_draws_are_added(rng, ci_method):
    """`mean(SE_j**2)` did not fall with m at all.

    SE_j is the sampling variance of a *single* draw and carries the whole
    sample-by-perturbation interaction; averaging the SE_j removes the
    interaction from the estimand but not from the estimate. Measured against
    the empirical variance of the reported figure at n=300, the true variance
    fell ~50x from m=1 to m=64 while mean(SE_j**2) was flat to three digits
    (1.39e-04 -> 1.41e-04).
    """
    n = 400
    frame = pd.DataFrame({"x": rng.normal(size=n)})
    baseline = frame["x"].to_numpy()

    errors = [
        rgr(
            frame,
            lambda X: np.asarray(X["x"]),
            ["x"],
            yhat=baseline,
            kind="gaussian",
            magnitude=0.8,
            n_repeats=m,
            ci=True,
            ci_method=ci_method,
            random_state=4,
        )[0].estimate.standard_error
        for m in (1, 4, 16)
    ]
    assert errors[0] > errors[1] > errors[2], errors


def test_the_bootstrap_pooling_shares_weights_across_draws(rng):
    """Averaging replicates is only a bootstrap of the mean under shared weights.

    Passing `random_state` straight through advanced a live Generator per draw,
    so each draw was reweighted differently and the element-wise average of the
    replicate vectors was a bootstrap distribution of nothing.
    """
    from rgbox.robustness import _draw_values

    n = 200
    baseline = rng.normal(size=n)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    # Same seed, different scores: the multinomial weights must be identical,
    # so the two replicate vectors must move together rather than independently.
    rep_a = _draw_values(baseline, a, "bootstrap", 200, 99)
    rep_b = _draw_values(baseline, b, "bootstrap", 200, 99)
    again = _draw_values(baseline, a, "bootstrap", 200, 99)
    assert np.array_equal(rep_a, again)
    assert not np.array_equal(rep_a, rep_b)


def test_the_pooled_variance_double_counts_the_interaction_by_exactly_one_term(rng):
    """Pins the *known* residual bias of `T1 + B/m`, so it cannot be forgotten.

    Under the orthogonal decomposition g(S,P) = mu + a(S) + b(P) + c(S,P) the
    quantity wanted is var_a + (var_b + var_c)/m, while T1 -> var_a + var_c/m
    and B/m -> (var_b + var_c)/m. The sum therefore carries var_c/m twice. This
    is a property of the estimator, not of RGA, so it is checked on the
    decomposition directly.
    """
    var_a, var_b, var_c, m = 1.0, 2.0, 3.0, 8

    truth = var_a + (var_b + var_c) / m
    t1_limit = var_a + var_c / m
    b_over_m_limit = (var_b + var_c) / m
    assert t1_limit + b_over_m_limit == pytest.approx(truth + var_c / m)

    # And the same thing by simulation, so the algebra is not just asserted.
    draws = np.array(
        [
            rng.normal(0, np.sqrt(var_a))
            + rng.normal(0, np.sqrt(var_b), m).mean()
            + rng.normal(0, np.sqrt(var_c), m).mean()
            for _ in range(60_000)
        ]
    )
    assert np.var(draws, ddof=1) == pytest.approx(truth, rel=0.03)


def test_the_rgr_interval_is_still_centred_on_the_reported_point(rng):
    n = 300
    frame = pd.DataFrame({"x": rng.normal(size=n)})
    baseline = frame["x"].to_numpy()
    result = rgr(
        frame,
        lambda X: np.asarray(X["x"]),
        ["x"],
        yhat=baseline,
        kind="gaussian",
        magnitude=0.5,
        n_repeats=6,
        ci=True,
        random_state=3,
    )[0]
    midpoint = (result.estimate.ci_low + result.estimate.ci_high) / 2
    assert midpoint == pytest.approx(result.rgr)
    assert result.estimate.estimate == pytest.approx(result.rgr)
