"""Finding the cohort where the model is worst, instead of being told where to look.

:func:`rgbox.rga_by_segment` answers "how does the model do on *these* slices",
which presumes you already know which slices to ask about. The failure mode of
model monitoring is the slice nobody thought to cut: a rating band crossed with
a channel, a vintage crossed with a region. This module searches instead of
asking - the "error analysis" of the Responsible AI Toolbox, but rank-based and
in pure NumPy.

The method
----------
Bin every candidate feature into quantile bins, enumerate every cohort
definable by one bin condition (depth 1) or by an intersection of two (depth
2), compute RGA inside each cohort that is big enough, and rank ascending. The
enumeration is exhaustive within the depth limit, so there is no search
heuristic to tune and the result is reproducible.

The catch, and the correction
-----------------------------
Searching thousands of cohorts for the worst one is a *selection* problem, and
a large one: the minimum of many noisy estimates is far below their common
mean even when every cohort is identical in truth. This is the same effect that
made the unadjusted ``max - min`` p-value in :func:`rgbox.rga_parity` fire on
27.3% of perfectly fair five-level attributes - only much worse here, because
the family is thousands of cohorts rather than ten pairs.

So the same discipline applies, adapted to the fact that cohorts *overlap*
(unlike the disjoint groups of a parity table, whose independence made the
max-T draws exact). Overlapping cohorts share rows, so their estimates are
positively correlated, and simulating them as independent would over-state how
extreme a minimum to expect.

The null is therefore obtained by permutation, and **which** thing is permuted
is the whole point. The hypothesis to reject is *homogeneity* - "the model
ranks equally well everywhere" - not "the model has no ranking information".
Permuting the score against the target would test the second: it drives every
cohort's RGA to 0.5, so an observed worst cohort of 0.44 stops looking
unusual and the test loses all its power against exactly the case it exists
for. Instead the **cohort definitions are permuted against the (target, score)
pairs**: row *i*'s features are reassigned to another row. That keeps every
cohort's size, every overlap and the model's overall RGA exactly as observed,
and destroys only the association between belonging to a cohort and being
ranked badly. ``p_value`` is the share of permutations whose worst-found
cohort was at least as bad as the observed one - a family-wise, search-aware
p-value.

Read ``worst_cohort(...).p_value`` before acting on ``worst_cohort(...).cohorts[0]``.
A worst cohort always exists; a *significant* one does not.

Range restriction: the one result that is real but not a defect
---------------------------------------------------------------
Slicing on a feature the model **uses** lowers RGA inside every slice, by
construction and with no bug involved. Conditioning on a narrow band of a
predictor removes the between-band variation the score was exploiting, so
inside the band there is less left to rank - the same reason a within-decile
AUC is always below the overall one. Measured on a clean logistic model whose
only predictor is ``x``: the overall RGA is 0.72, while the middle quartile of
``x`` crossed with a level of an unrelated attribute scores 0.45, and the
search reports ``p = 0.005``.

That p-value is not wrong. Ranking quality genuinely is not homogeneous across
those cohorts; it just has a benign explanation. Two practical consequences:

* **prefer slicing on variables the model does not use** - portfolio, channel,
  vintage, region - where a shortfall has no mechanical explanation and points
  at something worth fixing;
* when you do slice on a model feature, compare a cohort against its *siblings*
  from the same feature rather than against the overall RGA. A band that scores
  far below the other bands of the same variable is a finding; a band that
  scores below the overall figure along with all its siblings is arithmetic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._validation import as_1d_float, as_score_pair, check_level, is_missing
from .core import rga
from .exceptions import InputError, UndefinedMetricError
from .inference import rga_ci
from .predictors import resolve_columns

__all__ = ["Cohort", "CohortSearch", "worst_cohort"]


@dataclass(frozen=True)
class Cohort:
    """One cohort: how it is defined, how big it is, how the model does on it."""

    conditions: tuple[str, ...]
    # compare=False: a frozen dataclass derives __eq__ and __hash__ from its
    # fields, and an ndarray field poisons both - `a == b` raises on the
    # ambiguous array truth value and hash() raises on unhashability, which
    # also takes out any CohortSearch holding a list of these. The mask is a
    # derived view of `conditions` against the same frame, so leaving it out
    # of the comparison loses nothing.
    mask: np.ndarray = field(repr=False, compare=False)
    n: int
    rga: float
    shortfall: float
    standard_error: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None

    @property
    def label(self) -> str:
        return " AND ".join(self.conditions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cohort": self.label,
            "conditions": list(self.conditions),
            "n": self.n,
            "rga": self.rga,
            "gini": 2 * self.rga - 1,
            "shortfall_vs_overall": self.shortfall,
            "standard_error": self.standard_error,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }


@dataclass(frozen=True)
class CohortSearch:
    """The ranked cohorts, the overall figure they are measured against, and a p-value."""

    overall_rga: float
    n: int
    cohorts: list[Cohort]
    n_cohorts_searched: int
    max_depth: int
    min_size: int
    p_value: float | None
    n_permutations: int
    level: float

    #: Why the p-value is not optional reading.
    SELECTION_NOTE = (
        "The worst of many cohorts is low partly because it is the worst of "
        "many. p_value is the share of permutations of the cohort definitions "
        "whose worst-found cohort was at least as bad, so it accounts for the "
        "whole search; a single cohort's own confidence interval does not. "
        "The definitions are permuted rather than the score: permuting the "
        "score would test 'no ranking signal at all', which is not the "
        "hypothesis at issue."
    )

    def __float__(self) -> float:
        if not self.cohorts:
            raise UndefinedMetricError("no cohort met the size floor.")
        return float(self.cohorts[0].rga)

    def __str__(self) -> str:
        if not self.cohorts:
            return (
                f"No cohort of at least {self.min_size} rows was found "
                f"(searched {self.n_cohorts_searched})."
            )
        worst = self.cohorts[0]
        tail = "" if self.p_value is None else f", p = {self.p_value:.4g}"
        return (
            f"Worst cohort: {worst.label} - RGA {worst.rga:.4f} on {worst.n} rows, "
            f"{worst.shortfall:.4f} below the overall {self.overall_rga:.4f}"
            f"{tail} ({self.n_cohorts_searched} cohorts searched)"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_rga": self.overall_rga,
            "n": self.n,
            "n_cohorts_searched": self.n_cohorts_searched,
            "max_depth": self.max_depth,
            "min_size": self.min_size,
            "p_value": self.p_value,
            "n_permutations": self.n_permutations,
            "level": self.level,
            "selection_note": self.SELECTION_NOTE,
            "cohorts": [cohort.to_dict() for cohort in self.cohorts],
        }


def _missing_bin(column: Any, present: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """A bin for the rows the feature says nothing about, if there are any.

    Missing gets its own bin rather than being dropped. "the rows where this
    field was never filled in" is exactly the kind of slice this search exists
    to surface, and dropping those rows would make them unsearchable without
    saying so anywhere in the result.
    """
    absent = ~present
    if not absent.any():
        return []
    return [(f"{column} is missing", absent)]


def _bin_conditions(
    X: Any, columns: Sequence[Any], n_bins: int
) -> list[tuple[str, np.ndarray]]:
    """One boolean mask per (feature, bin), with a human-readable condition.

    A bin that holds every row is dropped: it is the whole sample under
    another name, not a cohort. It has shortfall 0 by construction, and at
    depth 2 it pairs with every other bin to produce an exact duplicate of
    that bin - so ``top=10`` comes back half-filled with copies. A feature
    that is entirely missing, or constant, is what produces one.
    """
    out: list[tuple[str, np.ndarray]] = []
    for column in columns:
        raw = X[column]
        try:
            # allow_nan, and the reason matters: as_1d_float raises the same
            # InputError for "this column is not numeric" and for "this
            # numeric column contains a NaN". Left to the default, one missing
            # value in an interval-scaled feature would be caught below as
            # categorical, and the level-per-value branch would then emit one
            # bin per distinct float - n bins instead of n_bins, a depth-2
            # search that is O(n^2) in mask intersections, and, since no
            # single-value bin can clear min_size, an empty result reported
            # without a word. Measured at n=4000: 35 seconds to search nothing.
            values = as_1d_float(raw, str(column), allow_nan=True)
        except InputError:
            # Genuinely categorical / string column: every level is a bin.
            arr = np.asarray(raw.to_numpy() if hasattr(raw, "to_numpy") else raw)
            values = arr.tolist()
            present = np.fromiter(
                (not is_missing(value) for value in values),
                dtype=bool,
                count=arr.size,
            )
            # Grouped by value in one pass rather than one `arr == level`
            # sweep per level. Broadcasting the comparison over the whole
            # column is what made a `pandas.NA` blow up here - NA == "x" is
            # NA, not False, and numpy asking for its truth value raises
            # "boolean value of NA is ambiguous", so a string-dtype column
            # with a single missing entry took the search down. Restricting
            # the comparison to `present` rows keeps NA out of it entirely.
            levels: dict[Any, list[int]] = {}
            for index in np.flatnonzero(present).tolist():
                levels.setdefault(values[index], []).append(index)
            for level, indices in levels.items():
                mask = np.zeros(arr.size, dtype=bool)
                mask[indices] = True
                out.append((f"{column} == {level!r}", mask))
            out.extend(_missing_bin(column, present))
            continue

        # Non-finite values carry no position on the scale, so they cannot sit
        # inside any interval; they are collected into the missing bin instead.
        present = np.isfinite(values)
        if not present.any():
            out.extend(_missing_bin(column, present))
            continue
        observed = values[present]

        distinct = np.unique(observed)
        if distinct.size <= n_bins:
            for level in distinct:
                out.append((f"{column} == {level:g}", present & (values == level)))
            out.extend(_missing_bin(column, present))
            continue
        # Quantile edges, deduplicated: a heavily tied column can collapse
        # several quantiles onto the same edge, which would make empty bins.
        edges = np.unique(np.quantile(observed, np.linspace(0, 1, n_bins + 1)))
        for i in range(edges.size - 1):
            low, high = edges[i], edges[i + 1]
            last = i == edges.size - 2
            mask = present & (values >= low)
            mask &= values <= high if last else values < high
            operator = "<=" if last else "<"
            out.append((f"{low:g} <= {column} {operator} {high:g}", mask))
        out.extend(_missing_bin(column, present))
    return [(name, mask) for name, mask in out if not mask.all()]


def _search(
    y: np.ndarray,
    yhat: np.ndarray,
    conditions: list[tuple[str, np.ndarray]],
    min_size: int,
    max_depth: int,
) -> list[tuple[tuple[str, ...], np.ndarray, int, float]]:
    """Every cohort within the depth limit that is large enough to score."""
    found: list[tuple[tuple[str, ...], np.ndarray, int, float]] = []

    def consider(names: tuple[str, ...], mask: np.ndarray) -> None:
        size = int(mask.sum())
        if size < min_size:
            return
        try:
            value = rga(y[mask], yhat[mask])
        except (UndefinedMetricError, InputError):
            return  # single-class or degenerate cohort: nothing to report
        found.append((names, mask, size, value))

    for i, (name_i, mask_i) in enumerate(conditions):
        consider((name_i,), mask_i)
        if max_depth < 2:
            continue
        for name_j, mask_j in conditions[i + 1 :]:
            # Two bins of the same feature never intersect; skip the work.
            combined = mask_i & mask_j
            if int(combined.sum()) < min_size:
                continue
            consider((name_i, name_j), combined)
    return found


def worst_cohort(
    y: Any,
    yhat: Any,
    X: Any,
    features: Sequence[Any] | None = None,
    *,
    n_bins: int = 4,
    max_depth: int = 2,
    min_size: int = 100,
    top: int = 10,
    n_permutations: int = 200,
    level: float = 0.95,
    ci: bool = True,
    random_state: Any = None,
) -> CohortSearch:
    """Search for the cohort on which the model ranks worst.

    Parameters
    ----------
    y, yhat :
        Target and scores on the evaluation sample.
    X :
        Frame of candidate features to slice on. It does not have to be the
        model's design matrix, and usually should not be: slicing on a
        predictor the model *uses* depresses RGA inside every slice by range
        restriction alone, so the cleanest cuts are on variables the model
        never saw - portfolio, channel, vintage, region. See the module
        docstring.
    features :
        Which columns to slice on. Defaults to all of them.
    n_bins :
        Quantile bins per numeric feature. A feature with at most ``n_bins``
        distinct values is used as-is, one bin per value; a string or
        categorical feature is always one bin per level. Rows whose value is
        missing (NaN, infinite, ``None``, ``pandas.NA``) join a ``"<column> is
        missing"`` bin instead of being dropped, so a cohort defined by the
        absence of a field is searched like any other.
    max_depth :
        1 for single conditions, 2 (default) to also search every intersection
        of two. Cost is ``O((n_bins * n_features)^max_depth)`` cohorts.
    min_size :
        Cohorts below this many rows are not scored at all. Keep it well above
        the 3-row floor of RGA itself: a 12-row cohort will happily produce an
        RGA of 0.2 out of pure noise, and it is the whole point of this
        function that it will find it.
    top :
        How many of the worst cohorts to return.
    n_permutations :
        Permutations of the score used to calibrate ``p_value``. Set to 0 to
        skip it - which is only sensible if you intend to treat the result as
        exploratory and never quote a number from it.
    ci :
        Attach a per-cohort confidence interval. These are *not* corrected for
        the search; ``p_value`` is what accounts for it.

    Returns
    -------
    CohortSearch
        ``.cohorts`` is sorted worst-first, ``.p_value`` says whether the worst
        one is worse than searching noise would produce.

    Notes
    -----
    The per-cohort interval and the family-wise p-value answer different
    questions and can disagree: a cohort's own interval can sit far below the
    overall RGA while ``p_value`` is 0.4, because the search examined thousands
    of cohorts and one of them was bound to look bad. Act on the p-value.

    A small p-value says ranking quality is not homogeneous across cohorts. It
    does not say the model is broken: if the cohorts are cut on the model's own
    predictors, range restriction produces genuine heterogeneity with a benign
    cause. The module docstring has the measured example and what to do about
    it.
    """
    y_arr, yhat_arr = as_score_pair(y, yhat, min_size=3)
    level = check_level(level)
    if max_depth not in (1, 2):
        raise InputError(f"'max_depth' must be 1 or 2; got {max_depth!r}.")
    if min_size < 3:
        raise InputError(
            f"'min_size' must be at least 3 (RGA's own floor); got {min_size!r}. "
            "Values below ~50 make the search report noise."
        )
    if getattr(X, "columns", None) is None:
        raise InputError(
            "'X' must be a frame with column labels: the cohorts are described "
            "by feature name, and a bare NumPy array carries none."
        )
    if len(X) != y_arr.size:
        raise InputError(f"'X' has {len(X)} rows but y has {y_arr.size}.")

    columns = (
        resolve_columns(features, X, "features")
        if features is not None
        else list(X.columns)
    )
    conditions = _bin_conditions(X, columns, n_bins)
    overall = rga(y_arr, yhat_arr)

    found = _search(y_arr, yhat_arr, conditions, min_size, max_depth)
    found.sort(key=lambda item: item[3])

    cohorts: list[Cohort] = []
    for names, mask, size, value in found[:top]:
        estimate = None
        if ci:
            try:
                estimate = rga_ci(y_arr[mask], yhat_arr[mask], level=level)
            except (UndefinedMetricError, InputError):
                estimate = None
        cohorts.append(
            Cohort(
                conditions=names,
                mask=mask,
                n=size,
                rga=value,
                shortfall=overall - value,
                standard_error=None if estimate is None else estimate.standard_error,
                ci_low=None if estimate is None else estimate.ci_low,
                ci_high=None if estimate is None else estimate.ci_high,
            )
        )

    p_value = None
    if n_permutations and found:
        # Permute the cohort *definitions*, not the score: reassigning row i's
        # features to another row keeps every cohort size, every overlap and
        # the overall RGA exactly as observed, and removes only the link
        # between membership and being ranked badly. That is the null of
        # homogeneity, which is the one worth rejecting - permuting the score
        # instead would test "the model has no signal at all", drive every
        # cohort to 0.5, and leave the test with no power against a genuinely
        # weak cohort.
        rng = np.random.default_rng(random_state)
        observed = found[0][3]
        at_least_as_bad = 0
        for _ in range(n_permutations):
            shuffle = rng.permutation(y_arr.size)
            permuted = [(name, mask[shuffle]) for name, mask in conditions]
            null_found = _search(y_arr, yhat_arr, permuted, min_size, max_depth)
            if null_found and min(item[3] for item in null_found) <= observed:
                at_least_as_bad += 1
        p_value = (1 + at_least_as_bad) / (n_permutations + 1)

    return CohortSearch(
        overall_rga=overall,
        n=y_arr.size,
        cohorts=cohorts,
        n_cohorts_searched=len(found),
        max_depth=max_depth,
        min_size=min_size,
        p_value=p_value,
        n_permutations=n_permutations,
        level=level,
    )
