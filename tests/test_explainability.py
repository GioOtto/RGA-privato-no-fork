"""RGE: scale, removal strategies, collinearity, Shapley."""

from __future__ import annotations

import numpy as np
import pytest

from rgbox import rge, rge_group, rge_shapley
from rgbox.exceptions import InputError

pytest.importorskip("pandas")
pytest.importorskip("sklearn")


def test_removing_every_predictor_gives_exactly_one_half(fitted_logit):
    """The documented [0, 1] range is not attainable; the real cap is 0.5.

    With no predictor left the reduced score is constant, all its ranks tie,
    RGA is exactly 0.5 and RGE is exactly 0.5 - whatever the model, whatever
    the data.
    """
    context = fitted_logit
    result = rge_group(
        context["X_train"],
        context["X_test"],
        context["model"],
        list(context["X_train"].columns),
        yhat=context["yhat"],
    )
    assert result.rge == pytest.approx(0.5, abs=1e-9)


def test_normalization_restores_the_unit_scale(fitted_logit):
    context = fitted_logit
    result = rge_group(
        context["X_train"],
        context["X_test"],
        context["model"],
        list(context["X_train"].columns),
        yhat=context["yhat"],
        normalize=True,
    )
    assert result.rge == pytest.approx(1.0, abs=1e-9)


def test_an_irrelevant_predictor_scores_near_zero(fitted_logit):
    context = fitted_logit
    results = {
        item.variables[0]: item.rge
        for item in rge(
            context["X_train"],
            context["X_test"],
            context["model"],
            ["dti", "noise"],
            yhat=context["yhat"],
        )
    }
    assert results["noise"] < 0.02
    assert results["dti"] > results["noise"]


def test_results_are_sorted_descending(fitted_logit):
    context = fitted_logit
    results = rge(
        context["X_train"],
        context["X_test"],
        context["model"],
        ["noise", "dti", "age"],
        yhat=context["yhat"],
    )
    assert [item.rge for item in results] == sorted(
        [item.rge for item in results], reverse=True
    )


def test_group_rge_is_not_monotone(fitted_logit):
    """Removing *more* information can yield a *lower* RGE.

    Two near-collinear predictors are fitted with small opposing coefficients
    on very large values, so their difference carries real weight in the score.
    Deleting one alone destroys that balance and wrecks the ranking; deleting
    both cancels the damage and leaves the genuine signal (``dti``) in charge.

    Measured here: ``RGE({income}) = 0.64``, ``RGE({income_copy}) = 0.25``,
    ``RGE({income, income_copy}) = 0.07``. The group scores *below both of its
    members*, so individual RGE values are not additive contributions and must
    not be presented as if they were.
    """
    context = fitted_logit
    single = {
        name: rge_group(
            context["X_train"],
            context["X_test"],
            context["model"],
            [name],
            yhat=context["yhat"],
        ).rge
        for name in ("income", "income_copy")
    }
    together = rge_group(
        context["X_train"],
        context["X_test"],
        context["model"],
        ["income", "income_copy"],
        yhat=context["yhat"],
    ).rge
    assert together < min(single.values())


def test_rge_can_exceed_one_half(fitted_logit):
    """The 0.5 ceiling applies to the *all-predictors* group, not in general.

    When removing a predictor inverts the ranking rather than merely flattening
    it, ``RGA(full, reduced)`` drops below 0.5 and RGE rises above it. Here
    removing ``income`` alone scores 0.64. So RGE genuinely ranges over
    ``[0, 1]``; what is pinned to exactly 0.5 is the grand coalition, because a
    constant reduced score has RGA exactly 0.5.
    """
    context = fitted_logit
    value = rge_group(
        context["X_train"],
        context["X_test"],
        context["model"],
        ["income"],
        yhat=context["yhat"],
    ).rge
    assert value > 0.5


def test_collinear_predictors_mask_each_other(fitted_logit):
    """The classic marginal-importance pathology, demonstrated end to end.

    ``income`` and ``income_copy`` are the same information twice. Judged
    jointly they look almost irrelevant (0.07, only ~5x the pure-noise
    feature), while judged individually one of them looks like the single most
    important predictor in the model (0.64). Neither reading is usable; this is
    what :func:`rge_shapley` is for.
    """
    context = fitted_logit
    individual = {
        item.variables[0]: item.rge
        for item in rge(
            context["X_train"],
            context["X_test"],
            context["model"],
            ["income", "income_copy", "noise"],
            yhat=context["yhat"],
        )
    }
    joint = rge_group(
        context["X_train"],
        context["X_test"],
        context["model"],
        ["income", "income_copy"],
        yhat=context["yhat"],
    ).rge
    assert individual["income"] > 0.5  # looks dominant alone
    assert joint < 0.15  # looks negligible together
    assert joint < individual["income_copy"]  # and below either member


def test_shapley_is_efficient(fitted_logit):
    """Values must sum to the grand coalition's worth.

    Over *all* the model's predictors that worth is exactly 1 (normalised),
    because removing everything leaves a constant score.
    """
    context = fitted_logit
    result = rge_shapley(
        context["X_train"],
        context["X_test"],
        context["model"],
        list(context["X_train"].columns),
        yhat=context["yhat"],
        normalize=True,
    )
    assert sum(result.values.values()) == pytest.approx(result.total, abs=1e-9)
    assert result.total == pytest.approx(1.0, abs=1e-9)
    assert result.exact


def test_shapley_efficiency_holds_on_a_subset_too(fitted_logit):
    """Efficiency is relative to whatever coalition you asked about."""
    context = fitted_logit
    columns = ["income", "dti", "age", "noise"]
    result = rge_shapley(
        context["X_train"],
        context["X_test"],
        context["model"],
        columns,
        yhat=context["yhat"],
        normalize=True,
    )
    assert sum(result.values.values()) == pytest.approx(result.total, abs=1e-9)


def test_shapley_gives_symmetric_players_equal_credit(rng):
    """Two exactly identical predictors must receive identical Shapley values."""
    pd = pytest.importorskip("pandas")
    n = 400
    base = rng.normal(size=n)
    frame = pd.DataFrame({"a": base, "b": base, "c": rng.normal(size=n)})

    def model(X):
        return X["a"] + X["b"] + 0.2 * X["c"]

    result = rge_shapley(frame, frame, model, ["a", "b", "c"])
    assert result.values["a"] == pytest.approx(result.values["b"], abs=1e-9)


def test_shapley_sampling_approximates_the_exact_values(fitted_logit):
    context = fitted_logit
    columns = ["income", "dti", "age", "noise"]
    exact = rge_shapley(
        context["X_train"],
        context["X_test"],
        context["model"],
        columns,
        yhat=context["yhat"],
    )
    sampled = rge_shapley(
        context["X_train"],
        context["X_test"],
        context["model"],
        columns,
        yhat=context["yhat"],
        n_permutations=400,
        random_state=0,
    )
    assert not sampled.exact
    for column in columns:
        assert sampled.values[column] == pytest.approx(exact.values[column], abs=0.05)


@pytest.mark.parametrize("method", ["mean", "median", "mode", "permute"])
def test_removal_methods_all_run_and_rank_signal_above_noise(fitted_logit, method):
    context = fitted_logit
    results = {
        item.variables[0]: item.rge
        for item in rge(
            context["X_train"],
            context["X_test"],
            context["model"],
            ["dti", "noise"],
            yhat=context["yhat"],
            method=method,
            random_state=0,
        )
    }
    assert results["dti"] > results["noise"]


def test_retrain_method_matches_the_papers_r_code(rng):
    """The R scripts define RGE against a *refitted* reduced model."""
    pd = pytest.importorskip("pandas")
    from sklearn.linear_model import LinearRegression

    n = 500
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.normal(size=n),
        }
    )
    target = 3 * frame["x1"] + 0.05 * frame["x3"] + rng.normal(0, 0.5, n)
    full = LinearRegression().fit(frame, target)

    def refit(kept):
        return LinearRegression().fit(frame[kept], target)

    results = {
        item.variables[0]: item.rge
        for item in rge(
            frame,
            frame,
            full,
            ["x1", "x3"],
            method="retrain",
            refit=refit,
        )
    }
    assert results["x1"] > results["x3"]
    assert results["x1"] > 0.3


def test_retrain_without_refit_is_a_clear_error(fitted_logit):
    context = fitted_logit
    with pytest.raises(InputError, match="needs a `refit` callable"):
        rge(
            context["X_train"],
            context["X_test"],
            context["model"],
            ["dti"],
            yhat=context["yhat"],
            method="retrain",
        )


def test_string_columns_are_handled_not_crashed(credit_frame, rng):
    """Upstream raised TypeError: mean of a string dtype."""
    frame, _target = credit_frame

    def model(X):
        return np.log(np.asarray(X["income"], dtype=float)) + (
            np.asarray(X["region"]) == "north"
        ).astype(float)

    results = rge(frame, frame, model, ["region", "income"])
    assert all(np.isfinite(item.rge) for item in results)
    assert len(results) == 2


def test_yhat_is_actually_used(fitted_logit, rng):
    """Passing a different yhat must change the answer."""
    context = fitted_logit
    real = rge(
        context["X_train"],
        context["X_test"],
        context["model"],
        ["dti"],
        yhat=context["yhat"],
    )[0].rge
    junk = rge(
        context["X_train"],
        context["X_test"],
        context["model"],
        ["dti"],
        yhat=rng.normal(size=len(context["X_test"])),
    )[0].rge
    assert real != pytest.approx(junk, abs=1e-6)


def test_confidence_interval_brackets_the_estimate(fitted_logit):
    context = fitted_logit
    result = rge(
        context["X_train"],
        context["X_test"],
        context["model"],
        ["income"],
        yhat=context["yhat"],
        ci=True,
    )[0]
    low, high = result.rge_ci
    assert low < result.rge < high


def test_unknown_column_is_reported_with_the_available_ones(fitted_logit):
    context = fitted_logit
    with pytest.raises(InputError, match="not in the data"):
        rge(
            context["X_train"],
            context["X_test"],
            context["model"],
            ["nope"],
            yhat=context["yhat"],
        )


def test_numpy_input_gives_an_actionable_message(fitted_logit):
    """Upstream said '<name> is not in the variables', which misleads."""
    context = fitted_logit
    with pytest.raises(InputError, match="no column labels"):
        rge(
            context["X_train"].to_numpy(),
            context["X_test"].to_numpy(),
            context["model"],
            ["income"],
            yhat=context["yhat"],
        )


def test_inputs_are_not_mutated(fitted_logit):
    context = fitted_logit
    before = context["X_test"].copy()
    rge(
        context["X_train"],
        context["X_test"],
        context["model"],
        ["income", "dti"],
        yhat=context["yhat"],
        group=True,
    )
    assert context["X_test"].equals(before)


def test_group_permutation_is_shared_across_the_columns(rng):
    """A one-hot group must survive removal as a valid one-hot.

    Permuting each dummy independently hands the model rows that belong to two
    levels at once, so the reduced score measures the model's behaviour on data
    that cannot exist. This pins both halves of the claim: the rate that
    behaviour produces, and that the shared permutation drives it to zero.
    """
    pd = pytest.importorskip("pandas")
    n = 1500
    region = rng.choice(["north", "centre", "south"], n, p=[0.45, 0.30, 0.25])
    frame = pd.DataFrame({"income": rng.normal(size=n)})
    frame = pd.concat(
        [frame, pd.get_dummies(region, prefix="r", drop_first=True).astype(float)],
        axis=1,
    )
    dummies = [c for c in frame.columns if c.startswith("r_")]
    assert len(dummies) == 2, "drop_first leaves the reference level implicit"

    # The rate quoted in the docs and CHANGELOG: p_north * p_south = .45 * .25.
    independent = frame.copy()
    for column in dummies:
        independent[column] = np.asarray(independent[column])[rng.permutation(n)]
    impossible = float((independent[dummies].sum(axis=1) > 1).mean())
    assert impossible == pytest.approx(0.1125, abs=0.02)

    seen: list[float] = []

    def spy(X):
        seen.append(float((X[dummies].sum(axis=1) > 1).sum()))
        return X["income"].to_numpy()

    rge(frame, frame, spy, dummies, group=True, method="permute", random_state=0)
    assert seen[-1] == 0.0


def test_permute_accepts_a_generator_as_random_state(rng):
    """The per-column seed was ``int(random_state) + offset``, so a Generator
    raised ``TypeError: int() argument must be...`` before reaching the model.
    """
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=200),
            "b": rng.normal(size=200),
        }
    )
    results = rge(
        frame,
        frame,
        lambda X: X["a"] + X["b"],
        ["a", "b"],
        group=True,
        method="permute",
        random_state=np.random.default_rng(3),
    )
    assert len(results) == 1


def test_group_permutation_keeps_the_rows_intact(rng):
    """Every removed column must still come from the same original row."""
    pd = pytest.importorskip("pandas")
    n = 300
    frame = pd.DataFrame(
        {
            "a": np.arange(n, dtype=float),
            "b": np.arange(n, dtype=float) * -1.0,  # b == -a, row by row
            "c": rng.normal(size=n),
        }
    )

    seen: list[bool] = []

    def spy(X):
        seen.append(bool(np.allclose(X["b"], -X["a"])))
        return X["c"].to_numpy()

    rge(frame, frame, spy, ["a", "b"], group=True, method="permute", random_state=7)
    assert seen[-1], "the shared permutation must preserve the a/b pairing"
