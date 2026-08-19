"""Low-level rank primitives.

Everything in the Rank Graduation Box reduces to *average ranks* and
*tie-group aggregates*. Keeping those primitives in one place means the rest of
the library is a handful of dot products, and it lets the weighted variants
(needed for sample weights and for the exact bootstrap) share one code path
with the unweighted ones.

All functions are pure NumPy, ``O(n log n)``, and free of pandas.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "average_ranks",
    "weighted_average_ranks",
    "tie_group_ids",
    "suffix_sums_strictly_greater",
    "tie_group_sums",
]


def tie_group_ids(values_sorted: np.ndarray) -> np.ndarray:
    """Dense 0-based group id per element of an already-sorted array."""
    n = values_sorted.size
    if n == 0:
        return np.zeros(0, dtype=np.intp)
    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    np.not_equal(values_sorted[1:], values_sorted[:-1], out=is_new[1:])
    return np.cumsum(is_new) - 1


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks in ``1..n`` with ties resolved by the group mean.

    Equivalent to ``scipy.stats.rankdata(values, method="average")`` but without
    the SciPy dependency and slightly faster, since we already need the sort
    order and the tie groups elsewhere.
    """
    values = np.asarray(values)
    n = values.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    gid = tie_group_ids(sorted_values)
    counts = np.bincount(gid)
    # Cumulative count *before* each group; ranks are 1-based.
    start = np.concatenate(([0], np.cumsum(counts)[:-1]))
    group_rank = start + (counts + 1.0) / 2.0
    out = np.empty(n, dtype=np.float64)
    out[order] = group_rank[gid]
    return out


def weighted_average_ranks(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Average ranks in a sample where observation ``i`` appears ``w_i`` times.

    For a tie group carrying total weight ``W`` and preceded by cumulative
    weight ``C``, every member of the group receives rank ``C + (W + 1) / 2``.
    With integer weights this reproduces exactly the average ranks of the
    physically replicated sample, which is what makes the multinomial bootstrap
    in :mod:`rgbox.inference` exact rather than approximate.
    """
    values = np.asarray(values)
    weights = np.asarray(weights, dtype=np.float64)
    n = values.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    gid = tie_group_ids(sorted_values)
    group_weight = np.bincount(gid, weights=sorted_weights)
    before = np.concatenate(([0.0], np.cumsum(group_weight)[:-1]))
    group_rank = before + (group_weight + 1.0) / 2.0
    out = np.empty(n, dtype=np.float64)
    out[order] = group_rank[gid]
    return out


def suffix_sums_strictly_greater(
    values: np.ndarray, payload: np.ndarray
) -> np.ndarray:
    """``out[i] = sum of payload[j] over all j with values[j] > values[i]``.

    Ties are excluded, which is what the leave-one-out rank update needs.
    """
    values = np.asarray(values)
    payload = np.asarray(payload, dtype=np.float64)
    n = values.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_payload = payload[order]
    gid = tie_group_ids(sorted_values)
    n_groups = int(gid[-1]) + 1
    counts = np.bincount(gid, minlength=n_groups)
    # Index of the first element belonging to the *next* tie group.
    next_group_start = np.cumsum(counts)
    # Suffix sums with a trailing zero so `next_group_start == n` reads as 0.
    suffix = np.concatenate((np.cumsum(sorted_payload[::-1])[::-1], [0.0]))
    out = np.empty(n, dtype=np.float64)
    out[order] = suffix[next_group_start[gid]]
    return out


def tie_group_sums(values: np.ndarray, payload: np.ndarray) -> np.ndarray:
    """``out[i] = sum of payload[j] over all j with values[j] == values[i]``."""
    values = np.asarray(values)
    payload = np.asarray(payload, dtype=np.float64)
    n = values.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    gid = tie_group_ids(values[order])
    sums = np.bincount(gid, weights=payload[order])
    out = np.empty(n, dtype=np.float64)
    out[order] = sums[gid]
    return out
