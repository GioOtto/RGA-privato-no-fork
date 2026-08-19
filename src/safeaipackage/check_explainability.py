"""Legacy ``safeaipackage.check_explainability``."""

from __future__ import annotations

from typing import Any

import pandas as pd

from rgbox.explainability import rge as _rge

__all__ = ["compute_rge_values"]


def compute_rge_values(
    xtrain: pd.DataFrame,
    xtest: pd.DataFrame,
    yhat: list,
    model: Any,
    variables: list,
    group: bool = False,
) -> pd.DataFrame:
    """RGE for the given variables, individually or as one group.

    Same signature, same return shape (a one-column ``"RGE"`` DataFrame indexed
    by variable name, or by ``str(variables)`` when ``group=True``) and the same
    values as upstream.

    Two things worth knowing, unchanged from upstream but undocumented there:
    the attainable range is ``[0, 0.5]`` rather than ``[0, 1]`` - removing every
    predictor yields exactly 0.5, not 1 - and group RGE is not monotone in the
    group. :func:`rgbox.rge` exposes ``normalize=True`` for the first and
    :func:`rgbox.rge_shapley` for the second.
    """
    results = _rge(xtrain, xtest, model, list(variables), yhat=yhat, group=group)
    if group:
        return pd.DataFrame(
            [results[0].rge], index=[str(list(variables))], columns=["RGE"]
        )
    return pd.DataFrame(
        [item.rge for item in results],
        index=[item.variables[0] for item in results],
        columns=["RGE"],
    )
