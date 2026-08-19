"""``safeaipackage.check_accuracy`` - restored.

This module existed once: compiled ``check_accuracy.cpython-38.pyc`` and
``check_accuracy.cpython-311.pyc`` are still committed in the upstream
repository, but the source was deleted, leaving the "A" of S.A.F.E. as the only
principle in the Rank Graduation Box without a module. (``check_privacy``
suffered the same fate; privacy is out of scope for this fork, since the
published Rank Graduation Box does not define a privacy measure.)

The functions here are thin wrappers over :mod:`rgbox.accuracy`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from rgbox.accuracy import accuracy_report as _accuracy_report
from rgbox.core import gini_score as _gini
from rgbox.core import rga as _rga
from rgbox.inference import rga_ci as _rga_ci
from rgbox.inference import rga_compare as _rga_compare

__all__ = [
    "compute_rga",
    "compute_rga_ci",
    "compute_gini",
    "compare_rga",
    "accuracy_table",
]


def compute_rga(y: list, yhat: list) -> float:
    """RGA of ``yhat`` against ``y``."""
    return _rga(y, yhat)


def compute_gini(y: list, yhat: list) -> float:
    """``2 * RGA - 1`` - the Gini coefficient of scorecard validation."""
    return _gini(y, yhat)


def compute_rga_ci(
    y: list,
    yhat: list,
    method: str = "jackknife",
    level: float = 0.95,
    **kwargs: Any,
):
    """RGA with a standard error and a confidence interval."""
    return _rga_ci(y, yhat, method=method, level=level, **kwargs)


def compare_rga(y: list, yhat_a: list, yhat_b: list, **kwargs: Any):
    """Paired test of ``RGA(A) == RGA(B)`` on the same evaluation sample."""
    return _rga_compare(y, yhat_a, yhat_b, **kwargs)


def accuracy_table(y: list, yhat: list, **kwargs: Any) -> pd.DataFrame:
    """One-row DataFrame with RGA, Gini and the conventional metrics."""
    report = _accuracy_report(y, yhat, **kwargs)
    record = report.to_dict()
    flat = {
        "n": record["n"],
        "rga": record["rga"]["rga"],
        "rga_se": record["rga"]["standard_error"],
        "rga_ci_low": record["rga"]["ci_low"],
        "rga_ci_high": record["rga"]["ci_high"],
        "gini": record["gini"],
    }
    flat.update(record["reference_metrics"])
    flat["p_value_vs_random"] = record.get("significance", {}).get("p_value")
    return pd.DataFrame([flat])
