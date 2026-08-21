"""Input coercion and checking.

One entry point (:func:`as_score_pair`) is used by every metric so that error
messages, NaN policy and dtype handling are identical everywhere.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .exceptions import InputError, InsufficientDataError

__all__ = [
    "as_1d_float",
    "as_score_pair",
    "as_weights",
    "check_level",
    "check_count",
    "check_finite_scalar",
    "as_group_labels",
    "distinct_levels",
    "is_missing",
]

#: Type names of the missing-value singletons of pandas. Matched by name so
#: that pandas stays an optional dependency.
_MISSING_TYPE_NAMES = frozenset({"NAType", "NaTType"})


def is_missing(value: Any) -> bool:
    """True for ``None``, ``float('nan')``, ``pandas.NA`` and ``pandas.NaT``.

    Duck-typed on purpose: importing pandas here would make it a hard
    dependency of the numpy-only core.
    """
    if value is None:
        return True
    if type(value).__name__ in _MISSING_TYPE_NAMES:
        return True
    try:
        return bool(value != value)  # NaN is the only value unequal to itself
    except (TypeError, ValueError):  # pragma: no cover - exotic __ne__
        return False


def _ravel_columnish(value: Any, name: str) -> np.ndarray:
    """Accept lists, Series, 1-column DataFrames and (n,) / (n,1) arrays."""
    # Duck-typed pandas support: no pandas import, so pandas stays optional.
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    arr = np.asarray(value)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    elif arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0, :]
    if arr.ndim != 1:
        raise InputError(
            f"{name!r} must be one-dimensional (or an (n, 1) column); "
            f"got shape {arr.shape}."
        )
    return arr


def _count_missing(arr: np.ndarray) -> int:
    """Count ``None`` / NaN / ``pandas.NA`` / ``NaT``, vectorised where possible.

    ``as_group_labels`` runs this on every parity, segment and outcome call, so
    the object-dtype fallback - materialise a Python list and ask each element
    whether it is missing - must not be the path a plain float or string column
    takes. It costs two orders of magnitude more than the numpy predicate and
    more than the metric it is guarding.
    """
    kind = arr.dtype.kind
    if kind in "fc":
        return int(np.isnan(arr).sum())
    if kind in "iub":
        return 0  # integers and booleans have no missing state
    if kind in "US":
        # A fixed-width string array stores the *text* "nan"; it holds no
        # missing value, which is what the object path below would conclude.
        return 0
    if kind in "Mm":
        return int(np.isnat(arr).sum())
    # Object dtype: the elements are arbitrary Python, so ask each one.
    return sum(1 for value in arr.tolist() if is_missing(value))


def as_1d_float(value: Any, name: str, *, allow_nan: bool = False) -> np.ndarray:
    """Coerce to a contiguous 1-D float64 array, rejecting NaN by default."""
    arr = _ravel_columnish(value, name)
    if arr.dtype == object or arr.dtype.kind in "US":
        # Booleans and numeric strings are worth converting; free text is not.
        try:
            arr = arr.astype(np.float64)
        except (TypeError, ValueError) as exc:
            # A pandas nullable column (Int64, boolean, string) arrives here as
            # an object array holding pandas.NA, and the conversion fails for
            # that reason rather than because the values are categorical.
            # Reporting "encode categories to numbers first" would send the
            # reader after the wrong problem.
            n_missing = _count_missing(arr)
            if n_missing:
                raise InputError(
                    f"{name!r} contains {n_missing} missing value(s) "
                    "(None/NaN/pandas.NA). Handle missing values explicitly "
                    "before scoring - drop the rows, impute them, or encode "
                    "'missing' as its own level."
                ) from exc
            raise InputError(
                f"{name!r} has non-numeric dtype {arr.dtype!r} and cannot be "
                "converted to float. RGA is defined on ordered values; encode "
                "categories to numbers first."
            ) from exc
    else:
        arr = arr.astype(np.float64, copy=False)
    if not allow_nan and not np.all(np.isfinite(arr)):
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        raise InputError(
            f"{name!r} contains {n_nan} NaN and {n_inf} infinite value(s). "
            "Handle missing values explicitly before scoring."
        )
    return np.ascontiguousarray(arr)


def as_score_pair(
    y: Any, yhat: Any, *, y_name: str = "y", yhat_name: str = "yhat", min_size: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce and cross-validate a (target, score) pair."""
    y_arr = as_1d_float(y, y_name)
    yhat_arr = as_1d_float(yhat, yhat_name)
    if y_arr.size != yhat_arr.size:
        raise InputError(
            f"{y_name!r} and {yhat_name!r} must have the same length; got "
            f"{y_arr.size} and {yhat_arr.size}."
        )
    if y_arr.size < min_size:
        raise InsufficientDataError(
            f"at least {min_size} observations are required; got {y_arr.size}."
        )
    return y_arr, yhat_arr


def as_weights(weights: Any, n: int) -> np.ndarray | None:
    """Coerce optional sample weights; ``None`` means uniform.

    The weights are **rescaled to mean 1** before being returned. RGA is a
    ratio whose numerator and denominator both scale as the square of the
    weight scale, so multiplying every weight by a positive constant cannot
    change the answer - but the zero-denominator guard in :mod:`rgbox.core`
    measures that denominator against a tolerance derived from ``y`` alone,
    which does *not* scale. Without this normalisation ``weights = [1, 1, 1,
    1]`` was computed happily while ``weights = [1e-8] * 4`` - the same
    relative weights, and just as legal under the checks below - was rejected
    as numerically degenerate. Normalising first makes the guard see the same
    denominator for both, which is the invariance the measure actually has.
    """
    if weights is None:
        return None
    w = as_1d_float(weights, "weights")
    if w.size != n:
        raise InputError(f"'weights' must have length {n}; got {w.size}.")
    if np.any(w < 0):
        raise InputError("'weights' must be non-negative.")
    total = w.sum()
    if total <= 0:
        raise InputError("'weights' must have a positive sum.")
    if np.count_nonzero(w) < 2:
        raise InsufficientDataError(
            "at least two observations must carry non-zero weight."
        )
    return w * (w.size / total)


def check_level(level: float) -> float:
    if not (0.0 < level < 1.0):
        raise InputError(f"'level' must lie strictly in (0, 1); got {level!r}.")
    return float(level)


def check_count(value: Any, name: str, *, minimum: int = 1) -> int:
    """Coerce a resampling / iteration count and reject the degenerate ones.

    Every counter in this package - ``n_resamples``, ``n_permutations``,
    ``n_repeats``, ``block_size``, ``n_bins``, ``top`` - feeds a loop, an array
    allocation or a divisor, and each one had its own failure mode when given 0
    or a negative number: an infinite loop in ``bootstrap_values``, a
    ``ZeroDivisionError`` in ``rge_shapley``, a silently empty search in
    ``worst_cohort``, and - worst of the family - a p-value of exactly 1.0
    returned with no warning by ``outcome_parity(n_resamples=0)``, because the
    empty max-T sample makes ``(1 + 0) / (0 + 1)`` the answer for every gap.
    A wrong count must fail here, loudly, rather than downstream in NumPy or
    not at all.

    ``minimum=0`` is passed by the callers where "do none of this" is a
    documented, meaningful request - skipping the permutation p-value of
    :func:`rgbox.worst_cohort`, for instance.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise InputError(
            f"{name!r} must be an integer; got {value!r} of type "
            f"{type(value).__name__}."
        )
    count = int(value)
    if count < minimum:
        raise InputError(f"{name!r} must be at least {minimum}; got {count}.")
    return count


def check_finite_scalar(value: Any, name: str) -> float:
    """Coerce a scalar that must be a real, finite number.

    NaN is the dangerous one, and silently: ``values >= float('nan')`` is
    ``False`` for every row, so a NaN threshold does not raise - it turns the
    whole sample into "nobody was selected" and reports perfect parity.
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{name!r} must be a number; got {value!r}.") from exc
    if not math.isfinite(number):
        raise InputError(
            f"{name!r} must be finite; got {value!r}. A NaN compares False "
            "against everything, so it would silently produce an empty "
            "selection rather than an error."
        )
    return number


def as_group_labels(values: Any, name: str, expected: int) -> np.ndarray:
    """Coerce a per-row grouping variable and reject missing labels.

    Missing labels are rejected rather than dropped. Dropping them silently
    removes rows from a fairness or monitoring table whose totals then no
    longer add up to the sample, and NaN in particular cannot even be counted:
    ``float('nan') != float('nan')``, so every missing row becomes its own
    "level" of size zero. Encode missing as an explicit level if it should be
    reported, or filter the rows if it should not.

    Shape is checked with the same policy as :func:`as_1d_float`: ``(n,)``,
    ``(n, 1)`` and ``(1, n)`` are accepted, anything else is rejected. This
    used to ``ravel()`` whatever it was handed, so a two-dimensional block
    passed by mistake - a 2x2 frame alongside four observations, say - became
    four group labels in row-major order with no complaint, while the numeric
    arguments of the very same call rejected it. One policy, both arguments.
    """
    arr = _ravel_columnish(values, name)
    if arr.size != expected:
        raise InputError(f"{name!r} has {arr.size} entries but y has {expected}.")

    n_missing = _count_missing(arr)
    if n_missing:
        raise InputError(
            f"{name!r} contains {n_missing} missing label(s) "
            "(None/NaN/pandas.NA). Every row must belong to a named level: a "
            "missing label cannot be compared with anything, and silently "
            f"dropping those rows would leave the {name!r} table adding up to "
            "fewer rows than the sample. Either filter them out yourself, or "
            "map them to an explicit level such as 'missing'."
        )
    return arr


def distinct_levels(labels: np.ndarray) -> list[tuple[Any, np.ndarray]]:
    """Levels in order of first appearance, each with its boolean row mask.

    ``labels`` must already have passed :func:`as_group_labels`, so it carries
    no missing values and equality-based masking is well defined.
    """
    seen = list(dict.fromkeys(labels.tolist()))
    return [(level, labels == level) for level in seen]
