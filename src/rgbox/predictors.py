"""Turning "a model" into "a function that returns one score per row".

Upstream hard-imported CatBoost, XGBoost and scikit-learn at module scope, in
``util/utils.py``, which every other module imports. The practical consequence
is that ``rga(y, yhat)`` - twenty lines of NumPy on two arrays - could not run
without two gradient-boosting libraries installed. It also meant support was
restricted to an enumerated list of classes: a Keras model, a statsmodels fit,
an ONNX session, a pickled scorecard or a plain lambda were all rejected.

Here nothing is imported at module scope. Models are duck-typed, and the
fallback accepts any callable, so an internal scoring engine wrapped in a
one-line lambda works exactly as well as an ``XGBClassifier``.

Score conventions
-----------------
RGA only reads the ordering of the score, so any strictly increasing transform
is equivalent: probabilities, log-odds, margins and decision-function values
all give identical results. What matters is *direction* - the score must
increase with the target. For binary classifiers we take the probability of
the positive class; for a model whose score runs the other way, negate it or
pass ``greater_is_better=False``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

import numpy as np

from .exceptions import InputError, ModelAdapterError

__all__ = ["ScoreFunction", "as_score_function", "predict_scores"]

ScoreFunction = Callable[[Any], np.ndarray]


def _looks_like_classifier(model: Any) -> bool:
    if getattr(model, "_estimator_type", None) == "classifier":
        return True
    # scikit-learn >= 1.6 moved this onto __sklearn_tags__.
    tags = getattr(model, "__sklearn_tags__", None)
    if callable(tags):
        try:
            return getattr(tags(), "estimator_type", None) == "classifier"
        except Exception:  # pragma: no cover - defensive against tag churn
            pass
    return hasattr(model, "predict_proba") and hasattr(model, "classes_")


def _class_index(model: Any, pos_label: Any) -> int:
    """Position of ``pos_label`` in the model's ``classes_``."""
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise ModelAdapterError(
            "pos_label was given but the model exposes no `classes_` attribute."
        )
    matches = np.flatnonzero(np.asarray(classes) == pos_label)
    if matches.size != 1:
        raise InputError(
            f"pos_label={pos_label!r} is not one of the model's classes "
            f"{list(classes)!r}."
        )
    return int(matches[0])


def _positive_column(model: Any, proba: np.ndarray, pos_label: Any) -> np.ndarray:
    """Select the column of ``predict_proba`` that the score should follow."""
    if proba.ndim != 2:
        raise ModelAdapterError(
            f"predict_proba returned an array of shape {proba.shape}; expected "
            "(n_samples, n_classes)."
        )
    n_classes = proba.shape[1]

    if n_classes == 1:
        raise ModelAdapterError(
            "the model was fitted on a single class; RGA is undefined."
        )

    if pos_label is None:
        if n_classes > 2:
            raise ModelAdapterError(
                f"this model predicts {n_classes} classes. A single RGA value "
                "needs a single ordered score, and silently taking column 1 - "
                "as the upstream implementation did - reports "
                "P(class == classes_[1]) while ignoring every other class. "
                "Choose one of: pass `pos_label=<class>` for a one-vs-rest "
                "reading, pass an explicit score function, or use "
                "rgbox.accuracy.rga_ovr for the one-vs-rest average."
            )
        return proba[:, 1]

    return proba[:, _class_index(model, pos_label)]


def as_score_function(
    model: Any,
    *,
    pos_label: Any = None,
    greater_is_better: bool = True,
) -> ScoreFunction:
    """Wrap ``model`` into a plain ``X -> scores`` callable.

    Accepted, in priority order:

    1. any plain callable (including a bare function or ``lambda``) that is not
       an estimator - used directly;
    2. a classifier exposing ``predict_proba`` - probability of the positive
       class, resolved through ``classes_`` when ``pos_label`` is given;
    3. anything exposing ``decision_function`` - the margin;
    4. anything exposing ``predict`` - the prediction.

    Raises
    ------
    rgbox.ModelAdapterError
        If none of the above applies, or the model is multiclass and no
        ``pos_label`` was supplied. Upstream left ``yhat`` unbound in this case
        and surfaced an ``UnboundLocalError`` from inside the library.
    """
    sign = 1.0 if greater_is_better else -1.0

    is_estimator = any(
        hasattr(model, attr)
        for attr in ("predict", "predict_proba", "decision_function")
    )
    if callable(model) and not is_estimator:

        def from_callable(X: Any) -> np.ndarray:
            return sign * np.asarray(model(X), dtype=np.float64).ravel()

        return from_callable

    if hasattr(model, "predict_proba") and _looks_like_classifier(model):

        def from_proba(X: Any) -> np.ndarray:
            proba = np.asarray(model.predict_proba(X), dtype=np.float64)
            return sign * _positive_column(model, proba, pos_label).ravel()

        return from_proba

    if hasattr(model, "decision_function"):
        # A binary decision_function is a single margin oriented towards
        # `classes_[1]` by scikit-learn's convention, so asking for
        # `classes_[0]` means the same margin with the opposite sign. This used
        # to ignore pos_label entirely on this branch: the argument was
        # accepted, the orientation was never changed, and RGA came back as
        # 1 - RGA with no error and no warning - 0.964 reported as 0.036 on the
        # LinearSVC in the test suite. A pos_label naming a class the model was
        # never fitted on was accepted here too, while the predict_proba branch
        # rejected it.
        margin_sign = sign
        if pos_label is not None:
            index = _class_index(model, pos_label)
            n_classes = len(np.asarray(model.classes_))
            if n_classes != 2:
                raise ModelAdapterError(
                    f"pos_label with decision_function needs a binary model; "
                    f"this one has {n_classes} classes. Pass an explicit score "
                    "function, or use rgbox.accuracy.rga_ovr."
                )
            if index == 0:
                margin_sign = -margin_sign

        def from_margin(X: Any) -> np.ndarray:
            scores = np.asarray(model.decision_function(X), dtype=np.float64)
            if scores.ndim > 1 and scores.shape[1] > 1:
                raise ModelAdapterError(
                    "decision_function returned one column per class; supply "
                    "pos_label or an explicit score function."
                )
            return margin_sign * scores.ravel()

        return from_margin

    if hasattr(model, "predict"):

        def from_predict(X: Any) -> np.ndarray:
            return sign * np.asarray(model.predict(X), dtype=np.float64).ravel()

        return from_predict

    raise ModelAdapterError(
        f"cannot score with an object of type {type(model).__name__!r}: it "
        "exposes none of predict_proba / decision_function / predict and is "
        "not callable. Pass a function `X -> scores` instead."
    )


def predict_scores(
    model: Any,
    X: Any,
    *,
    pos_label: Any = None,
    greater_is_better: bool = True,
) -> np.ndarray:
    """One-shot convenience wrapper around :func:`as_score_function`."""
    scores = as_score_function(
        model, pos_label=pos_label, greater_is_better=greater_is_better
    )(X)
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n_rows = len(X) if hasattr(X, "__len__") else scores.size
    if scores.size != n_rows:
        raise ModelAdapterError(
            f"the model returned {scores.size} scores for {n_rows} rows."
        )
    return scores


def _is_label(value: Any, available_set: set) -> bool:
    """True when ``value`` is itself one of the frame's column labels."""
    try:
        return value in available_set
    except TypeError:  # unhashable, so it cannot be a label
        return False


def resolve_columns(
    columns: Sequence[Any] | Any, frame: Any, argument: str
) -> list[Any]:
    """Normalise a column selection and check it against the frame.

    A tuple is ambiguous: it is both a sequence of labels and a legal label in
    its own right, because that is how pandas spells a ``MultiIndex`` column.
    An exact match against the frame's labels wins - ``("demo", "gender")`` on
    a MultiIndex frame is *that one column*, not the two columns ``"demo"`` and
    ``"gender"``. Pass a list to select several columns; a list is never a
    label, so it is never ambiguous.

    A ``set`` is **rejected**. It reads like a natural way to spell "these
    columns", but it has no order, and iteration order for strings varies with
    ``PYTHONHASHSEED`` - so the same call in two processes returned RGE and
    Shapley values in different orders. This package promises reproducible
    re-runs, which a set cannot deliver.
    """
    if isinstance(columns, (set, frozenset)):
        raise InputError(
            f"{argument!r} was given as a set. Sets have no order, and Python "
            "varies it between processes, so the results would not be "
            "reproducible. Pass a list instead: "
            f"{sorted(columns, key=repr)!r}."
        )

    available = getattr(frame, "columns", None)
    # Column labels are hashable by construction, so this cannot raise.
    available_set = set(available) if available is not None else set()

    # A list is always a group; it can never be a label. A tuple is a group
    # only when it is not itself one of the frame's columns. Anything else - a
    # string, a number, any scalar - is a lone label to be wrapped.
    is_group = isinstance(columns, list) or (
        isinstance(columns, tuple) and not _is_label(columns, available_set)
    )
    if not is_group:
        columns = [columns]
    columns = list(columns)
    if not columns:
        raise InputError(f"{argument!r} is empty; nothing to evaluate.")
    if available is None:
        raise InputError(
            f"{argument!r} was given by name, but the data has no column "
            "labels. Pass a pandas DataFrame (a bare NumPy array carries no "
            "names, and upstream reported this as "
            "'<name> is not in the variables')."
        )
    missing = [column for column in columns if column not in available_set]
    if missing:
        raise InputError(
            f"{argument}: column(s) {missing!r} are not in the data. "
            f"Available: {list(available)[:20]!r}"
            f"{' ...' if len(list(available)) > 20 else ''}"
        )
    duplicates = {c for c in columns if columns.count(c) > 1}
    if duplicates:
        # sorted(key=repr), for the same reason as the set branch above: pandas
        # allows a frame to mix label types, so `["x", 1, "x", 1]` is a legal
        # duplicate list, and plain `sorted` on {"x", 1} raises TypeError
        # ("'<' not supported between instances of 'str' and 'int'") from
        # inside the error path - replacing this InputError with a TypeError
        # that names neither the argument nor the duplicates.
        raise InputError(
            f"{argument}: duplicated column(s) {sorted(duplicates, key=repr)!r}."
        )
    return columns
