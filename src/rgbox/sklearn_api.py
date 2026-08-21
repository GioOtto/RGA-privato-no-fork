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

import numpy as np

from .core import gini_score, rga
from .exceptions import InputError

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
        Positive class for binary classification. Supplying it returns a
        **callable scorer** built on :func:`rgbox.predictors.predict_scores`
        rather than one built by ``sklearn.metrics.make_scorer``: both the
        score column and ``y_true`` have to be re-expressed relative to the
        chosen class, and only the former is something ``make_scorer`` can do.

        This argument was previously accepted and then dropped on the floor -
        it reached neither ``make_scorer`` nor the metric, so
        ``make_rga_scorer(pos_label=0)`` and ``make_rga_scorer(pos_label=1)``
        returned scorers that behaved identically. Either the argument means
        something or it should not be in the signature.
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
    metric = gini_score if gini else rga

    if pos_label is not None:
        if not needs_proba:
            raise InputError(
                "pos_label is meaningless with needs_proba=False: that path "
                "scores a regressor's `predict` output, which has no classes."
            )
        if scorer_kwargs:
            raise InputError(
                f"make_rga_scorer(pos_label=...) builds a callable scorer "
                f"directly and cannot forward make_scorer options "
                f"{sorted(scorer_kwargs)!r}."
            )

        def scorer(estimator: Any, X: Any, y: Any) -> float:
            from .predictors import predict_scores

            scores = predict_scores(estimator, X, pos_label=pos_label)
            # The score is now P(class == pos_label), so the target has to be
            # the indicator of the same class. Re-expressing only one of the
            # two is what turns RGA into 1 - RGA.
            return metric(np.asarray(y) == pos_label, scores)

        scorer.__name__ = f"{'gini' if gini else 'rga'}_scorer_{pos_label!r}"
        return scorer

    import inspect

    from sklearn.metrics import make_scorer

    kwargs: dict[str, Any] = dict(greater_is_better=True, **scorer_kwargs)
    if needs_proba:
        # scikit-learn >= 1.4 prefers `response_method`; older versions want
        # `needs_proba=True`. The capability is detected from the signature,
        # not from a TypeError at construction: `make_scorer` forwards every
        # unrecognised keyword to the *metric*, so on scikit-learn 1.0 -
        # the floor this package declares - the old code built a scorer
        # happily and then failed at scoring time with "rga() got an
        # unexpected keyword argument 'response_method'". Inside
        # `cross_val_score` that surfaced as a column of NaN scores rather
        # than an error, and the `except TypeError` fallback could never run.
        parameters = inspect.signature(make_scorer).parameters
        if "response_method" in parameters:
            return make_scorer(metric, response_method="predict_proba", **kwargs)
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
