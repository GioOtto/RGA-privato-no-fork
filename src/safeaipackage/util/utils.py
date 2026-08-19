"""Legacy ``safeaipackage.util.utils``, without the mandatory boosting libraries.

Upstream imported CatBoost, XGBoost and scikit-learn at module scope here, and
every other module imported this one, so the whole package was unusable unless
all three were installed. Those imports are gone: models are duck-typed by
:mod:`rgbox.predictors`, which supports the same estimators plus anything else
that exposes ``predict``/``predict_proba``/``decision_function`` - or is simply
a callable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rgbox.exceptions import InputError
from rgbox.explainability import replace_column
from rgbox.predictors import predict_scores

__all__ = [
    "manipulate_testdata",
    "convert_to_dataframe",
    "validate_variables",
    "check_nan",
    "find_yhat",
]


def manipulate_testdata(
    xtrain: pd.DataFrame,
    xtest: pd.DataFrame,
    model: Any,
    variable: str,
) -> pd.DataFrame:
    """Replace ``variable`` in ``xtest`` by its training mean (or mode).

    Numeric columns get the training mean, everything else the training mode.
    Upstream only recognised ``pandas.CategoricalDtype`` as non-numeric, so a
    plain string column fell through to ``.mean()`` and raised
    ``TypeError: Could not perform reduction 'mean' with string dtype``.
    """
    return replace_column(xtest, variable, X_train=xtrain, method="mean")


def convert_to_dataframe(*args: Any) -> list[pd.DataFrame]:
    """Convert inputs to DataFrames with a fresh RangeIndex."""
    return [pd.DataFrame(arg).reset_index(drop=True) for arg in args]


def validate_variables(variables: list, xtrain: pd.DataFrame) -> None:
    """Check that ``variables`` is a list of columns present in ``xtrain``."""
    if not isinstance(variables, list):
        raise InputError("Variables input must be a list")
    columns = set(getattr(xtrain, "columns", []))
    for name in variables:
        if name not in columns:
            raise InputError(f"{name} is not in the variables")


def check_nan(*dataframes: pd.DataFrame) -> None:
    """Raise if any frame contains missing values."""
    for position, frame in enumerate(dataframes, start=1):
        if not hasattr(frame, "isna"):
            raise TypeError(
                f"argument {position} is a {type(frame).__name__}, not a "
                "pandas object. (Upstream documented this TypeError but never "
                "raised it.)"
            )
        if frame.isna().to_numpy().sum() > 0:
            raise InputError(f"DataFrame {position} contains missing values.")


def find_yhat(model: Any, xtest: pd.DataFrame) -> np.ndarray:
    """Score ``xtest`` with ``model``.

    Deviations from upstream, both of which were latent bugs:

    * a model that is neither a scikit-learn classifier nor regressor left
      ``yhat`` unbound and surfaced ``UnboundLocalError`` (or, on scikit-learn
      >= 1.6, ``AttributeError: __sklearn_tags__``); it now raises
      :class:`rgbox.ModelAdapterError` with an actionable message, and plain
      callables are accepted;
    * a multiclass classifier returned ``P(class == classes_[1])`` while
      silently discarding every other class; it now raises, and directs you to
      ``pos_label=`` or :func:`rgbox.rga_ovr`.
    """
    return predict_scores(model, xtest)
