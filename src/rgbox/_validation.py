"""Input coercion and checking.

One entry point (:func:`as_score_pair`) is used by every metric so that error
messages, NaN policy and dtype handling are identical everywhere.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .exceptions import InputError, InsufficientDataError

__all__ = ["as_1d_float", "as_score_pair", "as_weights", "check_level"]


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


def as_1d_float(value: Any, name: str, *, allow_nan: bool = False) -> np.ndarray:
    """Coerce to a contiguous 1-D float64 array, rejecting NaN by default."""
    arr = _ravel_columnish(value, name)
    if arr.dtype == object or arr.dtype.kind in "US":
        # Booleans and numeric strings are worth converting; free text is not.
        try:
            arr = arr.astype(np.float64)
        except (TypeError, ValueError) as exc:
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
    """Coerce optional sample weights; ``None`` means uniform."""
    if weights is None:
        return None
    w = as_1d_float(weights, "weights")
    if w.size != n:
        raise InputError(
            f"'weights' must have length {n}; got {w.size}."
        )
    if np.any(w < 0):
        raise InputError("'weights' must be non-negative.")
    total = w.sum()
    if total <= 0:
        raise InputError("'weights' must have a positive sum.")
    if np.count_nonzero(w) < 2:
        raise InsufficientDataError(
            "at least two observations must carry non-zero weight."
        )
    return w


def check_level(level: float) -> float:
    if not (0.0 < level < 1.0):
        raise InputError(f"'level' must lie strictly in (0, 1); got {level!r}.")
    return float(level)
