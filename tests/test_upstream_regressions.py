"""One test per upstream defect, so none of them can quietly come back.

Each test names the original behaviour in its docstring. This file doubles as
the executable version of ``MIGRATION.md``.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from rgbox import predict_scores, rga, rga_parity, rge, rge_group
from rgbox.exceptions import (
    InputError,
    ModelAdapterError,
    UndefinedMetricError,
)

pytest.importorskip("pandas")


def test_core_imports_without_xgboost_or_catboost():
    """Upstream: `import safeaipackage.core` raised ModuleNotFoundError.

    ``core.py`` imported ``util.utils``, which imported CatBoost and XGBoost at
    module scope, so twenty lines of NumPy required two gradient-boosting
    libraries. ``setup.py`` meanwhile declared ``install_requires=[]``.
    """
    script = (
        "import sys; "
        "sys.modules['xgboost'] = None; sys.modules['catboost'] = None; "
        "sys.modules['sklearn'] = None; sys.modules['pandas'] = None; "
        "import rgbox.core, rgbox.inference, rgbox.accuracy; "
        "import numpy as np; "
        "print(round(rgbox.core.rga([0,0,1,1],[0.1,0.4,0.35,0.8]), 4))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "0.75"


def test_fairness_actually_uses_yhat(rng):
    """Upstream: `compute_rga_parity` accepted `yhat`, then re-predicted.

    Passing a vector of random numbers instead of the real scores returned a
    bit-identical result (0.01880059620717922 either way).
    """
    y = rng.binomial(1, 0.4, 800).astype(float)
    groups = rng.binomial(1, 0.5, 800)
    real = rga_parity(y, rng.normal(y, 1.0, 800), groups).gap
    junk = rga_parity(y, rng.normal(size=800), groups).gap
    assert real != pytest.approx(junk, abs=1e-6)


def test_fairness_returns_a_number(rng):
    """Upstream returned "The RGA-based imparity between the protected gorups
    is 0.0188." - a string, with a typo, unusable in a threshold check."""
    y = rng.binomial(1, 0.4, 600).astype(float)
    result = rga_parity(y, rng.normal(y, 1, 600), rng.binomial(1, 0.5, 600))
    assert float(result) >= 0.0
    assert result.gap < 1.0


def test_fairness_tolerates_a_level_missing_from_the_test_split(rng):
    """Upstream: levels came from `xtrain` but rows were filtered from `xtest`,
    so a train-only level produced an empty slice and
    `ValueError: Found array with 0 sample(s)` from inside the estimator."""
    y = rng.binomial(1, 0.4, 500).astype(float)
    scores = rng.normal(y, 1.0, 500)
    groups = np.array(["present"] * 500)          # the other level never appears
    result = rga_parity(y, scores, groups, min_group_size=0)
    assert result.gap is None                      # reported, not crashed
    assert len(result.groups) == 1


def test_constant_target_raises_instead_of_returning_nan(rng):
    """Upstream: silent `nan`, which then propagates into fairness gaps."""
    with pytest.raises(UndefinedMetricError):
        rga(np.ones(30), rng.normal(size=30))


def test_multiclass_model_is_not_silently_scored_as_class_one(rng):
    """Upstream: `[x[1] for x in model.predict_proba(X)]` discarded every class
    but the second, without a word."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=list("abc"))
    y = rng.integers(0, 3, 300)
    model = LogisticRegression(max_iter=500).fit(X, y)
    with pytest.raises(ModelAdapterError, match="3 classes"):
        predict_scores(model, X)
    # ...but an explicit one-vs-rest reading is available.
    scores = predict_scores(model, X, pos_label=2)
    assert scores.shape == (300,)


def test_unsupported_model_raises_a_useful_error(rng):
    """Upstream: `find_yhat` had no else branch, so `yhat` stayed unbound and
    the caller saw `UnboundLocalError` (or `AttributeError:
    __sklearn_tags__` on newer scikit-learn)."""

    class NotAModel:
        pass

    with pytest.raises(ModelAdapterError, match="exposes none of"):
        predict_scores(NotAModel(), [[1.0]])


def test_plain_callables_are_accepted(rng):
    """Upstream restricted models to an enumerated list of classes."""
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"x": rng.normal(size=100)})
    assert predict_scores(lambda X: X["x"] * 2, frame).shape == (100,)


def test_string_column_does_not_crash_variable_removal(rng):
    """Upstream: `manipulate_testdata` only special-cased pandas
    CategoricalDtype, so an object/string column reached `.mean()` and raised
    `TypeError: Could not perform reduction 'mean' with string dtype`."""
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({
        "num": rng.normal(size=200),
        "txt": rng.choice(["a", "b", "c"], 200),
    })

    def model(X):
        return np.asarray(X["num"]) + (np.asarray(X["txt"]) == "a")

    results = rge(frame, frame, model, ["txt", "num"])
    assert all(np.isfinite(item.rge) for item in results)


def test_numpy_arrays_get_an_honest_error(rng):
    """Upstream reported `'a' is not in the variables` when handed a NumPy
    array, which points at the wrong problem: the array has no column names."""
    pytest.importorskip("sklearn")
    with pytest.raises(InputError, match="no column labels"):
        rge(rng.normal(size=(50, 3)), rng.normal(size=(50, 3)),
            lambda X: X[:, 0], ["a"])


def test_removing_all_predictors_is_exactly_one_half(fitted_logit):
    """Upstream README: "When RGE is equal to 1, it shows a variable with a
    high contribution". The grand coalition is pinned to 0.5, so 1 is not on
    the reachable scale."""
    context = fitted_logit
    value = rge_group(
        context["X_train"], context["X_test"], context["model"],
        list(context["X_train"].columns), yhat=context["yhat"],
    ).rge
    assert value == pytest.approx(0.5, abs=1e-9)


def test_rgr_default_magnitude_is_not_a_neutral_choice(fitted_logit):
    """Upstream fixed `perturbation_percentage=0.05` with no justification;
    the answer moves by more than 0.3 across the legal range."""
    from rgbox import rgr

    context = fitted_logit
    values = [
        rgr(context["X_test"], context["model"], ["income"],
            yhat=context["yhat"], magnitude=m)[0].rgr
        for m in (0.01, 0.5)
    ]
    assert values[0] - values[1] > 0.15


def test_binary_column_tailswap_is_a_near_no_op(rng):
    """Upstream applied the tail swap to any column. On a 0/1 indicator it
    exchanges arbitrary tied values and leaves the mean untouched, so the
    'perturbation' carries no information about robustness."""
    pd = pytest.importorskip("pandas")
    from rgbox import perturb

    frame = pd.DataFrame({"flag": rng.binomial(1, 0.5, 400).astype(float)})
    swapped = perturb(frame, "flag", 0.1)
    assert swapped["flag"].mean() == pytest.approx(frame["flag"].mean())


def test_check_nan_raises_the_typeerror_it_documented():
    """Upstream's docstring promised `TypeError` for non-DataFrame input but
    the code never raised it."""
    from safeaipackage.util.utils import check_nan

    with pytest.raises(TypeError, match="not a"):
        check_nan([1, 2, 3])
