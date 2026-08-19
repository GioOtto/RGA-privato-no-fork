"""Legacy ``safeaipackage.check_fairness``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rgbox.exceptions import InputError
from rgbox.fairness import ParityResult
from rgbox.fairness import rga_parity as _rga_parity
from rgbox.predictors import predict_scores

__all__ = ["compute_rga_parity", "ImparityScore"]


class ImparityScore(float):
    """The parity gap, usable as a number *and* printable as the old sentence.

    Upstream returned a formatted string, so the value could not be compared to
    a threshold, collected in a loop or plotted without re-parsing it. This is a
    ``float`` subclass: arithmetic and comparisons work, ``str()`` reproduces
    the legacy sentence (with the "gorups" typo corrected), and ``.result``
    exposes the full :class:`rgbox.ParityResult` with per-group RGA values,
    confidence intervals and pairwise tests.
    """

    result: ParityResult

    def __new__(cls, value: float, result: ParityResult) -> ImparityScore:
        obj = super().__new__(cls, value)
        obj.result = result
        return obj

    def __str__(self) -> str:
        return (
            "The RGA-based imparity between the protected groups is "
            f"{float(self)}."
        )

    def __repr__(self) -> str:
        return f"ImparityScore({float(self)!r})"


def compute_rga_parity(
    xtrain: pd.DataFrame,
    xtest: pd.DataFrame,
    ytest: list,
    yhat: list,
    model: Any,
    protectedvariable: str,
) -> ImparityScore:
    """RGA-based imparity across the levels of ``protectedvariable``.

    Three upstream defects are fixed here, so numbers can differ:

    1. **``yhat`` is now used.** Upstream converted it, checked it for NaN, and
       then re-predicted with ``find_yhat``, so passing a vector of random
       numbers gave a bit-identical result.
    2. **Group levels come from the test set.** Upstream enumerated the levels
       of ``xtrain`` and filtered ``xtest`` with them, so a level present in
       training but absent from the test split produced an empty slice and a
       ``ValueError: Found array with 0 sample(s)`` from inside the estimator.
    3. **A number is returned, not a sentence.**

    ``float(result)`` is the gap; ``result.result`` is the full analysis. Prefer
    :func:`rgbox.rga_parity`, which takes arrays directly and needs no model.
    """
    if protectedvariable not in getattr(xtrain, "columns", []):
        raise InputError(f"{protectedvariable} is not in the variables")
    if protectedvariable not in getattr(xtest, "columns", []):
        raise InputError(f"{protectedvariable} is not present in xtest")

    scores = (
        np.asarray(yhat, dtype=float).ravel()
        if yhat is not None
        else predict_scores(model, xtest)
    )
    result = _rga_parity(
        ytest,
        scores,
        xtest[protectedvariable],
        min_group_size=0,          # upstream applied no minimum
        attribute=protectedvariable,
    )
    if result.gap is None:
        raise InputError(
            f"cannot compare RGA across the levels of {protectedvariable!r}: "
            "fewer than two levels yielded a defined RGA on the test set. "
            "Check subgroup sizes and whether each subgroup contains both "
            "outcome classes."
        )
    return ImparityScore(result.gap, result)
