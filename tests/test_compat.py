"""The legacy `safeaipackage` API must keep working, call-site unchanged."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")

import pandas as pd

from safeaipackage.check_accuracy import accuracy_table, compute_gini, compute_rga
from safeaipackage.check_explainability import compute_rge_values
from safeaipackage.check_fairness import ImparityScore, compute_rga_parity
from safeaipackage.check_robustness import compute_rgr_values, perturb
from safeaipackage.core import rga as legacy_rga
from safeaipackage.util.utils import (
    check_nan,
    convert_to_dataframe,
    find_yhat,
    manipulate_testdata,
    validate_variables,
)


def test_legacy_rga_matches_the_new_one(sample):
    from rgbox import rga

    y, yhat = sample
    if np.ptp(y) == 0:
        pytest.skip("undefined by design")
    assert legacy_rga(y, yhat) == pytest.approx(rga(y, yhat))


def test_import_emits_a_fork_notice():
    import importlib

    import safeaipackage

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(safeaipackage)
    assert any("rgbox fork" in str(w.message) for w in caught)


def test_compute_rge_values_shape_and_dtype(fitted_logit):
    context = fitted_logit
    frame = compute_rge_values(
        context["X_train"], context["X_test"], context["yhat"],
        context["model"], ["income", "dti", "noise"],
    )
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["RGE"]
    assert set(frame.index) == {"income", "dti", "noise"}
    assert frame["RGE"].is_monotonic_decreasing


def test_compute_rge_values_group_mode(fitted_logit):
    context = fitted_logit
    variables = ["income", "dti"]
    frame = compute_rge_values(
        context["X_train"], context["X_test"], context["yhat"],
        context["model"], variables, group=True,
    )
    assert list(frame.index) == [str(variables)]
    assert frame.shape == (1, 1)


def test_compute_rgr_values_shape(fitted_logit):
    context = fitted_logit
    frame = compute_rgr_values(
        context["X_test"], context["yhat"], context["model"],
        ["income", "dti"], perturbation_percentage=0.1,
    )
    assert list(frame.columns) == ["RGR"]
    assert frame["RGR"].is_monotonic_decreasing


def test_perturb_keeps_the_upstream_signature(rng):
    frame = pd.DataFrame({"x": rng.normal(size=100)})
    swapped = perturb(frame, "x", 0.1)
    assert isinstance(swapped, pd.DataFrame)
    assert np.allclose(np.sort(frame["x"]), np.sort(swapped["x"]))


def test_imparity_is_a_float_that_prints_like_the_old_string(fitted_logit):
    context = fitted_logit
    result = compute_rga_parity(
        context["X_train"], context["X_test"], context["y_test"],
        context["yhat"], context["model"], "gender",
    )
    assert isinstance(result, ImparityScore)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0
    assert result < 1.0                      # usable in a threshold check
    assert "RGA-based imparity" in str(result)
    assert "gorups" not in str(result)       # upstream typo, corrected
    # And the full analysis is one attribute away.
    assert result.result.groups
    assert result.result.gap == pytest.approx(float(result))


def test_imparity_missing_protected_variable(fitted_logit):
    context = fitted_logit
    with pytest.raises(ValueError, match="is not in the variables"):
        compute_rga_parity(
            context["X_train"], context["X_test"], context["y_test"],
            context["yhat"], context["model"], "not_a_column",
        )


def test_util_helpers_preserve_upstream_behaviour(fitted_logit, rng):
    context = fitted_logit
    frames = convert_to_dataframe([1, 2, 3], np.array([4, 5, 6]))
    assert len(frames) == 2 and all(isinstance(f, pd.DataFrame) for f in frames)

    validate_variables(["income"], context["X_train"])
    with pytest.raises(ValueError, match="not in the variables"):
        validate_variables(["nope"], context["X_train"])
    with pytest.raises(ValueError, match="must be a list"):
        validate_variables("income", context["X_train"])

    check_nan(context["X_train"])
    with pytest.raises(ValueError, match="missing values"):
        check_nan(pd.DataFrame({"a": [1.0, np.nan]}))

    replaced = manipulate_testdata(
        context["X_train"], context["X_test"], context["model"], "income"
    )
    assert replaced["income"].nunique() == 1
    assert replaced["income"].iloc[0] == pytest.approx(
        context["X_train"]["income"].mean()
    )

    scores = find_yhat(context["model"], context["X_test"])
    assert np.allclose(scores, context["yhat"])


def test_check_accuracy_module_is_back(binary):
    y, yhat = binary
    assert compute_rga(y, yhat) == pytest.approx(compute_gini(y, yhat) / 2 + 0.5)
    table = accuracy_table(y, yhat)
    assert isinstance(table, pd.DataFrame)
    assert table.shape[0] == 1
    assert {"rga", "gini", "rga_ci_low", "rga_ci_high"} <= set(table.columns)


def test_top_level_reexports():
    import safeaipackage

    for name in ("rga", "compute_rge_values", "compute_rgr_values",
                 "compute_rga_parity", "perturb"):
        assert hasattr(safeaipackage, name)
