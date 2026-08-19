"""scikit-learn integration.

Upstream had none: RGA could not be used as a ``cross_val_score`` metric, in a
``GridSearchCV``, or in any of the model-selection machinery a team already
runs. That is a large part of why a new metric never gets adopted - it has to
be reachable from the tools people already use.

scikit-learn is imported lazily, inside the functions, so it stays an optional
dependency of the package.
"""

from __future__ import annotations

from typing import Any

from .core import gini_score, rga

__all__ = ["rga_scorer", "gini_scorer", "make_rga_scorer"]


def make_rga_scorer(
    *,
    needs_proba: bool = True,
    pos_label: Any = None,
    gini: bool = False,
    **scorer_kwargs: Any,
):
    """Build a scikit-learn scorer for RGA (or Gini).

    Parameters
    ----------
    needs_proba :
        ``True`` for classifiers (uses ``predict_proba``); set ``False`` for
        regressors, where ``predict`` is the score.
    pos_label :
        Positive class for binary classification.
    gini :
        Score on the ``2 * RGA - 1`` scale instead.

    Examples
    --------
    >>> from sklearn.model_selection import cross_val_score      # doctest: +SKIP
    >>> from rgbox.sklearn_api import make_rga_scorer            # doctest: +SKIP
    >>> cross_val_score(model, X, y, scoring=make_rga_scorer())  # doctest: +SKIP

    Notes
    -----
    RGA is a "greater is better" score in ``[0, 1]``, so it plugs into
    ``GridSearchCV`` without a sign flip. Because it reads only ranks, it is
    invariant to any monotone recalibration of the model output - handy when
    comparing a raw margin against a Platt-scaled probability.
    """
    from sklearn.metrics import make_scorer

    metric = gini_score if gini else rga
    kwargs: dict[str, Any] = dict(greater_is_better=True, **scorer_kwargs)
    if needs_proba:
        # scikit-learn >= 1.4 prefers `response_method`; fall back for older.
        try:
            return make_scorer(metric, response_method="predict_proba", **kwargs)
        except TypeError:  # pragma: no cover - legacy scikit-learn
            return make_scorer(metric, needs_proba=True, **kwargs)
    return make_scorer(metric, **kwargs)


def rga_scorer(estimator: Any, X: Any, y: Any) -> float:
    """Callable scorer usable directly as ``scoring=rga_scorer``.

    Works for classifiers and regressors alike: it picks ``predict_proba``,
    ``decision_function`` or ``predict`` in that order, exactly like the rest
    of the package.
    """
    from .predictors import predict_scores

    return rga(y, predict_scores(estimator, X))


def gini_scorer(estimator: Any, X: Any, y: Any) -> float:
    """As :func:`rga_scorer`, on the ``2 * RGA - 1`` scale."""
    from .predictors import predict_scores

    return gini_score(y, predict_scores(estimator, X))
