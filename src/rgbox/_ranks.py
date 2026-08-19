"""Low-level rank primitives.

Everything in the Rank Graduation Box reduces to *average ranks* and
*tie-group aggregates*. Keeping those primitives in one place means the rest of
the library is a handful of dot products, and it lets the weighted variants
(needed for sample weights and for the exact bootstrap) share one code path
with the unweighted ones.

All functions are pure NumPy, ``O(n log n)``, and free of pandas.

One sort, several aggregates
----------------------------
Every aggregate here needs the same three things: the sort order, the sorted
values, and the tie-group id of each sorted position. :func:`sorted_index`
computes them once and :class:`SortedIndex` carries them, so a caller that
needs average ranks *and* suffix sums *and* tie-group sums over the same array
pays for one ``argsort`` rather than three. That matters: the exact jackknife
needs all three per argument, and computing them independently made
``jackknife_values`` cost six sorts where the derivation in ``docs/THEORY.md``
says two.

The public one-shot functions (:func:`average_ranks` and friends) are thin
wrappers that build a throwaway index, so single-aggregate call sites read the
same as before.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SortedIndex",
    "sorted_index",
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


@dataclass(frozen=True)
class SortedIndex:
    """One ``argsort`` of an array, plus the tie structure every aggregate needs.

    Attributes
    ----------
    order :
        ``argsort(values, kind="stable")``.
    sorted_values :
        ``values[order]``.
    gid :
        Dense 0-based tie-group id of each *sorted* position.
    n_groups :
        Number of distinct values.
    """

    order: np.ndarray
    sorted_values: np.ndarray
    gid: np.ndarray
    n_groups: int

    @property
    def size(self) -> int:
        return int(self.order.size)

    def scatter(self, per_element_sorted: np.ndarray) -> np.ndarray:
        """Undo the sort: map a value-per-sorted-position back to input order."""
        out = np.empty(self.size, dtype=np.float64)
        out[self.order] = per_element_sorted
        return out


def sorted_index(values: np.ndarray) -> SortedIndex:
    """Sort ``values`` once and describe its tie structure."""
    values = np.asarray(values)
    n = values.size
    if n == 0:
        empty = np.zeros(0, dtype=np.intp)
        return SortedIndex(empty, values[:0], empty, 0)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    gid = tie_group_ids(sorted_values)
    return SortedIndex(order, sorted_values, gid, int(gid[-1]) + 1)


def average_ranks_from(index: SortedIndex) -> np.ndarray:
    """Average ranks in ``1..n`` from a prepared :class:`SortedIndex`."""
    if index.size == 0:
        return np.zeros(0, dtype=np.float64)
    counts = np.bincount(index.gid, minlength=index.n_groups)
    # Cumulative count *before* each group; ranks are 1-based.
    start = np.concatenate(([0], np.cumsum(counts)[:-1]))
    group_rank = start + (counts + 1.0) / 2.0
    return index.scatter(group_rank[index.gid])


def suffix_sums_strictly_greater_from(
    index: SortedIndex, payload: np.ndarray
) -> np.ndarray:
    """``out[i] = sum of payload[j] over all j with values[j] > values[i]``."""
    if index.size == 0:
        return np.zeros(0, dtype=np.float64)
    payload = np.asarray(payload, dtype=np.float64)
    sorted_payload = payload[index.order]
    counts = np.bincount(index.gid, minlength=index.n_groups)
    # Index of the first element belonging to the *next* tie group.
    next_group_start = np.cumsum(counts)
    # Suffix sums with a trailing zero so `next_group_start == n` reads as 0.
    suffix = np.concatenate((np.cumsum(sorted_payload[::-1])[::-1], [0.0]))
    return index.scatter(suffix[next_group_start[index.gid]])


def tie_group_sums_from(index: SortedIndex, payload: np.ndarray) -> np.ndarray:
    """``out[i] = sum of payload[j] over all j with values[j] == values[i]``."""
    if index.size == 0:
        return np.zeros(0, dtype=np.float64)
    payload = np.asarray(payload, dtype=np.float64)
    sums = np.bincount(
        index.gid, weights=payload[index.order], minlength=index.n_groups
    )
    return index.scatter(sums[index.gid])


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks in ``1..n`` with ties resolved by the group mean.

    Equivalent to ``scipy.stats.rankdata(values, method="average")`` but without
    the SciPy dependency and slightly faster, since we already need the sort
    order and the tie groups elsewhere.
    """
    return average_ranks_from(sorted_index(values))


def weighted_average_ranks(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Average ranks in a sample where observation ``i`` appears ``w_i`` times.

    For a tie group carrying total weight ``W`` and preceded by cumulative
    weight ``C``, every member of the group receives rank ``C + (W + 1) / 2``.
    With integer weights this reproduces exactly the average ranks of the
    physically replicated sample, which is what makes the multinomial bootstrap
    in :mod:`rgbox.inference` exact rather than approximate.
    """
    weights = np.asarray(weights, dtype=np.float64)
    index = sorted_index(values)
    if index.size == 0:
        return np.zeros(0, dtype=np.float64)
    sorted_weights = weights[index.order]
    group_weight = np.bincount(
        index.gid, weights=sorted_weights, minlength=index.n_groups
    )
    before = np.concatenate(([0.0], np.cumsum(group_weight)[:-1]))
    group_rank = before + (group_weight + 1.0) / 2.0
    return index.scatter(group_rank[index.gid])


def suffix_sums_strictly_greater(values: np.ndarray, payload: np.ndarray) -> np.ndarray:
    """``out[i] = sum of payload[j] over all j with values[j] > values[i]``.

    Ties are excluded, which is what the leave-one-out rank update needs.
    """
    return suffix_sums_strictly_greater_from(sorted_index(values), payload)


def tie_group_sums(values: np.ndarray, payload: np.ndarray) -> np.ndarray:
    """``out[i] = sum of payload[j] over all j with values[j] == values[i]``."""
    return tie_group_sums_from(sorted_index(values), payload)
