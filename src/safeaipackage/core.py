"""Legacy ``safeaipackage.core``.

The upstream implementation lived here and pulled in pandas, CatBoost and
XGBoost transitively (via ``util.utils``) just to compute a rank statistic on
two arrays. This one has no such imports and returns the identical value - the
test suite asserts agreement with the upstream algorithm to ~1e-16 on
continuous, binary, count, negative and heavily tied data.
"""

from __future__ import annotations

from typing import Any

from rgbox.core import rga as _rga

__all__ = ["rga"]


def rga(y: Any, yhat: Any) -> float:
    """RANK GRADUATION ACCURACY (RGA) MEASURE.

    Parameters
    ----------
    y : list
        A list of actual values.
    yhat : list
        A list of predicted values.

    Returns
    -------
    float
        The RGA value.

    Notes
    -----
    Deviation from upstream: a constant ``y`` raises
    :class:`rgbox.UndefinedMetricError` instead of returning ``nan``. The
    measure genuinely has no value there (its denominator is zero), and a
    silent ``nan`` propagating into a fairness gap or a model-comparison table
    is worse than a stack trace. Prefer :func:`rgbox.rga`, which additionally
    accepts sample weights.
    """
    return _rga(y, yhat)
