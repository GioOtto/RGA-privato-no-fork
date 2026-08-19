"""Rank Graduation Accuracy: the one measure everything else is built from.

Definition
----------
Order the sample by the score :math:`\\hat y`. Let :math:`y^{*}` be the target
re-ordered accordingly, with tied scores replaced by their conditional mean of
:math:`y`. With :math:`y_{(\\cdot)}` the ascending sort of :math:`y`, the
original discrete definition is

.. math::

    \\mathrm{RGA} = \\frac{\\sum_i i\\,y^{*}_i - \\sum_i i\\,y_{(n+1-i)}}
                          {\\sum_i i\\,y_{(i)} - \\sum_i i\\,y_{(n+1-i)}}

i.e. the area between the dual Lorenz curve and the concordance curve, divided
by the area between the Lorenz and dual Lorenz curves.

Closed form
-----------
Expanding those sums, the :math:`(n+1)\\sum_i y_i` terms cancel and the whole
thing collapses to a ratio of two covariances-with-ranks:

.. math::

    \\mathrm{RGA} = \\frac{1}{2}
        + \\frac{\\operatorname{cov}\\big(y,\\ R(\\hat y)\\big)}
                {2\\,\\operatorname{cov}\\big(y,\\ R(y)\\big)}

with :math:`R` the average-rank function. Equivalently
:math:`\\mathrm{RGA} = (1 + \\gamma)/2` where :math:`\\gamma` is the
Schechtman-Yitzhaki *Gini correlation* between :math:`y` and :math:`\\hat y`.

This identity is not a numerical approximation: it is an algebraic rewriting of
the same estimator, and the test suite asserts agreement with the upstream
implementation to machine precision on tied, binary, count and continuous data.
It buys three things:

* **speed** - one sort per argument and two dot products, no pandas groupby /
  merge round-trip;
* **conditioning** - the original form sums terms of order :math:`n^2\\bar y`
  and then subtracts them, so it loses precision when ``y`` is far from zero;
  centring first avoids that cancellation;
* **inference** - a Gini correlation is a smooth functional of the empirical
  distribution, which is what makes the influence function and the exact
  leave-one-out recursion in :mod:`rgbox.inference` possible.

Properties (all covered by ``tests/test_properties.py``)
--------------------------------------------------------
* ``RGA in [0, 1]``; ``1`` iff the score orders the sample exactly as ``y``,
  ``0`` under exact reversal, ``0.5`` when the score is uninformative.
* Binary ``y``: ``RGA == AUROC == `` the normalised Wilcoxon-Mann-Whitney
  statistic, ties included. Hence ``gini_score == 2 * AUROC - 1``, the
  "Gini coefficient" of credit-risk practice, but defined for any ordered
  target rather than only binary ones.
* Invariant under any strictly increasing transform of ``yhat`` (it only sees
  ranks) and under increasing affine transforms of ``y``. It is *not* invariant
  under arbitrary monotone transforms of ``y``: RGA is Pearson-like in ``y``
  and Spearman-like in ``yhat``, the defining asymmetry of Gini correlations.
* Asymmetric: ``rga(a, b) != rga(b, a)`` in general.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._ranks import average_ranks, weighted_average_ranks
from ._validation import as_score_pair, as_weights
from .exceptions import UndefinedMetricError

__all__ = ["rga", "rga_score", "gini_score", "rga_curves", "RGACurves"]

_DENOM_ATOL = 1e-12


def _denominator_guard(denominator: float, y: np.ndarray) -> None:
    """RGA's denominator is (a multiple of) the Gini mean difference of ``y``."""
    scale = float(np.max(np.abs(y - y.mean()))) * y.size**2
    tol = max(_DENOM_ATOL, scale * 1e-15)
    if abs(denominator) <= tol:
        raise UndefinedMetricError(
            "RGA is undefined because the target has (numerically) zero "
            "dispersion: its Lorenz and dual Lorenz curves coincide, so the "
            "denominator is 0. This typically happens on a constant target or "
            "on a subgroup that turned out to be single-class - check subgroup "
            "sizes before comparing RGA across segments."
        )


def rga(y: Any, yhat: Any, *, weights: Any = None) -> float:
    """Rank Graduation Accuracy of ``yhat`` as a ranking of ``y``.

    Parameters
    ----------
    y :
        Observed target. Any ordered numeric values: binary, ordinal, count or
        continuous. Must not be constant.
    yhat :
        Model scores. Only their ordering matters, so predicted probabilities,
        log-odds, raw margins or decision-function values are interchangeable.
    weights :
        Optional non-negative sample weights (survey weights, stratified
        sampling, reject-inference reweighting). Integer weights give exactly
        the RGA of the corresponding replicated sample.

    Returns
    -------
    float
        A value in ``[0, 1]``. ``0.5`` means the score carries no ranking
        information about ``y``.

    Raises
    ------
    rgbox.InputError
        Mismatched lengths, NaN/infinite values, non-numeric dtype.
    rgbox.UndefinedMetricError
        ``y`` is constant, so the measure has a zero denominator.

    Examples
    --------
    >>> import numpy as np
    >>> from rgbox import rga
    >>> y = np.array([0, 0, 1, 1])
    >>> float(rga(y, [0.1, 0.4, 0.35, 0.8]))
    0.75
    """
    y_arr, yhat_arr = as_score_pair(y, yhat)
    w = as_weights(weights, y_arr.size)

    if w is None:
        centred = y_arr - y_arr.mean()
        denominator = float(centred @ average_ranks(y_arr))
        _denominator_guard(denominator, y_arr)
        numerator = float(centred @ average_ranks(yhat_arr))
    else:
        mean = float(w @ y_arr) / float(w.sum())
        centred = w * (y_arr - mean)
        denominator = float(centred @ weighted_average_ranks(y_arr, w))
        _denominator_guard(denominator, y_arr)
        numerator = float(centred @ weighted_average_ranks(yhat_arr, w))

    return 0.5 + numerator / (2.0 * denominator)


#: Scikit-learn-flavoured alias (``y_true, y_score`` ordering is the same).
rga_score = rga


def gini_score(y: Any, yhat: Any, *, weights: Any = None) -> float:
    """``2 * RGA - 1``: the Gini coefficient of credit-risk practice.

    For a binary target this is exactly the quantity a scorecard validation
    report calls "Gini" or "Accuracy Ratio" (``2 * AUROC - 1``, equivalently
    Somers' D of ``y`` on ``yhat``). Unlike AUROC it keeps its meaning when the
    target is a loss amount, an LGD, an exposure or a rating notch, which is
    the practical reason to reach for the rank graduation family at all.

    Ranges over ``[-1, 1]``; ``0`` under an uninformative score.
    """
    return 2.0 * rga(y, yhat, weights=weights) - 1.0


@dataclass(frozen=True)
class RGACurves:
    """The three curves whose areas define RGA, ready to plot.

    Attributes
    ----------
    fraction :
        ``k / n`` for ``k = 0..n``; the x axis of all three curves.
    lorenz :
        Cumulative share of ``y`` sorted ascending - the best attainable
        ordering.
    dual_lorenz :
        Cumulative share of ``y`` sorted descending - the worst attainable
        ordering.
    concordance :
        Cumulative share of ``y`` reordered by the model's score, with ties
        replaced by their conditional mean.
    rga :
        ``(area(dual) - area(concordance)) / (area(dual) - area(lorenz))``,
        identically equal to :func:`rga`.
    """

    fraction: np.ndarray
    lorenz: np.ndarray
    dual_lorenz: np.ndarray
    concordance: np.ndarray
    rga: float


def rga_curves(y: Any, yhat: Any) -> RGACurves:
    """Build the Lorenz / dual Lorenz / concordance curves behind RGA.

    Useful for the "why is this model worse" page of a validation report: the
    vertical gap between the concordance curve and the Lorenz curve localises
    *where in the score distribution* the ranking breaks down, which a single
    scalar cannot.

    Requires ``sum(y) > 0`` - the classical Lorenz normalisation divides by the
    total. RGA itself has no such restriction; use :func:`rga` for targets that
    can be negative or sum to zero.
    """
    y_arr, yhat_arr = as_score_pair(y, yhat)
    total = float(y_arr.sum())
    if not total > 0:
        raise UndefinedMetricError(
            "Lorenz curves require a strictly positive total; got "
            f"sum(y) = {total!r}. The RGA value itself is still well defined "
            "for such targets - call rga() instead of rga_curves()."
        )

    n = y_arr.size
    order = np.argsort(yhat_arr, kind="stable")
    # Conditional mean of y within each group of tied scores.
    scores_sorted = yhat_arr[order]
    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    np.not_equal(scores_sorted[1:], scores_sorted[:-1], out=is_new[1:])
    gid = np.cumsum(is_new) - 1
    group_mean = np.bincount(gid, weights=y_arr[order]) / np.bincount(gid)
    y_star = group_mean[gid]

    ascending = np.sort(y_arr)
    curves = {
        "lorenz": ascending,
        "dual_lorenz": ascending[::-1],
        "concordance": y_star,
    }
    built = {
        name: np.concatenate(([0.0], np.cumsum(values) / total))
        for name, values in curves.items()
    }
    # Rectangle rule: the endpoint corrections of the trapezoid rule are shared
    # by all three curves and cancel in the ratio, so this is exact.
    areas = {name: float(values[1:].sum()) for name, values in built.items()}
    value = (areas["dual_lorenz"] - areas["concordance"]) / (
        areas["dual_lorenz"] - areas["lorenz"]
    )
    return RGACurves(
        fraction=np.arange(n + 1, dtype=np.float64) / n,
        lorenz=built["lorenz"],
        dual_lorenz=built["dual_lorenz"],
        concordance=built["concordance"],
        rga=value,
    )
