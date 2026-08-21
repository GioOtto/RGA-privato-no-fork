"""Fairness diagnostics built on RGA - and an honest account of what they mean.

What RGA parity is, and is not
------------------------------
``max_g RGA_g - min_g RGA_g`` measures whether the model *ranks equally well*
inside each protected group. That is **AUC parity** (also called predictive
parity of discrimination, or "equal discriminatory power"). It is a legitimate
model-quality question - a scorecard that separates good from bad obligors in
one segment but not another is broken for that segment - and it is exactly what
a model-risk function should ask about a rating model.

It is *not* demographic parity, nor equalised odds, nor equal opportunity, and
it does not follow from or imply any of them:

* a model can rank perfectly within both groups (gap 0) while assigning one
  group systematically worse scores - discriminatory, yet "fair" by this
  measure;
* a model can be calibrated and equal-opportunity-fair while ranking better in
  the larger group simply because it had more data - "unfair" by this measure
  without treating anyone differently.

Report it alongside outcome-based criteria, never instead of them.

Why the gap needs an interval
-----------------------------
The gap is a difference of two noisy estimates, and protected subgroups are
usually small. Two hundred obligors in the minority segment give an RGA
standard error around 0.04, so an *observed* gap of 0.08 between a segment of
200 and one of 12000 is roughly one standard error of pure noise. Reporting the
gap as a bare number - as upstream did, formatted into an f-string - invites
acting on nothing. Every estimate here carries a confidence interval, and
subgroups below ``min_group_size`` are reported but excluded from the headline
gap.

Why the gap's p-value needs a multiplicity correction
-----------------------------------------------------
``max - min`` picks the widest of ``k(k-1)/2`` pairwise comparisons, so its z
statistic is a maximum, not a fixed contrast, and referring it to a standard
normal over-rejects badly. Measured type I error under *exact* parity, against
a nominal 5%:

===============  ======  ==================
groups           pairs   unadjusted p < .05
===============  ======  ==================
2                1       4.3%
3                3       13.3%
5                10      27.3%
===============  ======  ==================

A module whose whole argument is "do not act on noise" cannot ship a headline
test that fires on a quarter of perfectly fair five-level attributes - and the
multi-level case is exactly the one this fork advertises as an improvement over
upstream. ``gap_p_value`` is therefore a **max-T family-wise** p-value: because
the groups are disjoint samples, their RGA estimates are independent under H0,
so the joint null of all pairwise statistics is simulated directly from the
per-group standard errors, with no re-estimation. It uses the real correlation
between pairs sharing a group, so it is far less conservative than Bonferroni.
``gap_p_value_unadjusted`` keeps the raw value for comparison; every entry in
``pairwise`` carries both.

Two consequences are worth stating plainly. With two groups there is a single
pair and nothing to correct for, so the adjusted and unadjusted values agree up
to simulation noise rather than coinciding exactly. And because the adjusted
value is a Monte-Carlo tail probability it cannot fall below
``1 / (n_resamples + 1)`` - about 5e-4 at the default 2000 draws - so it
saturates on strong disparities where the unadjusted value keeps going to
1e-15. That floor is a resolution limit, not a weaker verdict: read a saturated
``gap_p_value`` as "smaller than the simulation can measure", and raise
``n_resamples`` if a specific figure is needed.

Three additional bugs in the upstream implementation are fixed here:

* it derived the group levels from ``xtrain`` but filtered ``xtest``, so a
  level present in training and absent from the test split produced an empty
  slice and a ``ValueError`` from deep inside the estimator;
* it accepted a ``yhat`` argument, validated it, and then never used it -
  re-predicting from the model instead. Passing random noise as ``yhat``
  returned a bit-identical result;
* it returned a formatted string ("The RGA-based imparity between the protected
  gorups is 0.0188."), unusable in a loop, a threshold check or a plot.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._validation import (
    as_1d_float,
    as_group_labels,
    as_score_pair,
    check_level,
    distinct_levels,
)
from .core import rga
from .exceptions import InputError, InsufficientDataError, UndefinedMetricError
from .inference import (
    RGAEstimate,
    _bootstrap_values,
    _normal_cdf,
    _normal_quantile,
    rga_ci,
)
from .predictors import resolve_columns

__all__ = [
    "GroupRGA",
    "ParityResult",
    "labels_from_dummies",
    "rga_parity",
    "rgf",
    "proxy_leakage",
]


@dataclass(frozen=True)
class GroupRGA:
    """RGA restricted to one level of the protected attribute."""

    group: Any
    n: int
    rga: float | None
    standard_error: float | None
    ci_low: float | None
    ci_high: float | None
    included: bool
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "n": self.n,
            "rga": self.rga,
            "gini": None if self.rga is None else 2 * self.rga - 1,
            "standard_error": self.standard_error,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "included": self.included,
            "note": self.note,
        }


@dataclass(frozen=True)
class ParityResult:
    """Per-group RGA, the headline gap, and every pairwise comparison."""

    groups: list[GroupRGA]
    gap: float | None
    gap_ci: tuple[float, float] | None
    gap_p_value: float | None
    best_group: Any
    worst_group: Any
    level: float
    method: str
    attribute: Any = None
    pairwise: list[dict[str, Any]] = field(default_factory=list, repr=False)
    excluded: list[Any] = field(default_factory=list)
    gap_excess_over_noise: float | None = None
    gap_noise_floor: float | None = None
    gap_p_value_unadjusted: float | None = None
    multiplicity: str = ""

    #: ``max - min`` is non-negative by construction, so its percentile
    #: interval never contains 0 - not even under perfect parity, where the
    #: sampling noise of each group's RGA still produces a positive maximum
    #: gap. Read ``gap_ci`` as "how large could the worst-case spread be", and
    #: use ``gap_p_value`` (or the signed intervals in ``pairwise``) to *test*
    #: parity.
    #:
    #: ``gap_noise_floor`` is what the gap would average under *exact* parity
    #: given these group sizes, and ``gap_excess_over_noise`` is the observed
    #: gap minus that floor. It can be negative, which means the observed
    #: spread is no larger than sampling noise alone would produce.
    #:
    #: ``gap_p_value`` carries the *same* selection effect and is corrected for
    #: it: it is a max-T family-wise p-value, not the raw normal tail of the
    #: widest pair. ``gap_p_value_unadjusted`` keeps the raw value, which is
    #: only valid when there are exactly two groups. Being simulated, the
    #: adjusted value bottoms out at ``1 / (n_resamples + 1)``; ``multiplicity``
    #: records the draw count so that floor can be read off the result.
    GAP_CI_NOTE = (
        "gap_ci is a percentile interval for the non-negative statistic "
        "max(RGA) - min(RGA); it does not contain 0 even under exact parity. "
        "Test parity with gap_p_value, which is corrected by max-T for having "
        "selected the widest of the pairs; gap_p_value_unadjusted is not. "
        "gap_excess_over_noise is the gap minus gap_noise_floor, the spread "
        "that exact parity would produce by itself at these group sizes; it "
        "averages 0 under exact parity and may be negative."
    )

    def __float__(self) -> float:
        if self.gap is None:
            raise UndefinedMetricError("no gap could be computed.")
        return float(self.gap)

    def __str__(self) -> str:
        if self.gap is None:
            return "RGA parity: not computable (no two eligible groups)."
        pct = round(self.level * 100)
        interval = (
            f" ({pct}% CI {self.gap_ci[0]:.4f}-{self.gap_ci[1]:.4f})"
            if self.gap_ci
            else ""
        )
        tail = (
            f", p = {self.gap_p_value:.4g} (max-T adjusted)"
            if self.gap_p_value is not None
            else ""
        )
        excluded = (
            f"; excluded as too small: {self.excluded!r}" if self.excluded else ""
        )
        return (
            f"RGA parity gap = {self.gap:.4f}{interval}{tail} "
            f"[worst: {self.worst_group!r}, best: {self.best_group!r}]{excluded}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "attribute": self.attribute,
            "gap": self.gap,
            "gap_ci_low": None if self.gap_ci is None else self.gap_ci[0],
            "gap_ci_high": None if self.gap_ci is None else self.gap_ci[1],
            "gap_ci_note": self.GAP_CI_NOTE,
            "gap_excess_over_noise": self.gap_excess_over_noise,
            "gap_noise_floor": self.gap_noise_floor,
            "gap_p_value": self.gap_p_value,
            "gap_p_value_unadjusted": self.gap_p_value_unadjusted,
            "multiplicity": self.multiplicity,
            "best_group": self.best_group,
            "worst_group": self.worst_group,
            "level": self.level,
            "method": self.method,
            "groups": [g.to_dict() for g in self.groups],
            "pairwise": list(self.pairwise),
            "excluded_small_groups": list(self.excluded),
        }


def _null_group_draws(
    standard_errors: np.ndarray,
    rng: np.random.Generator,
    n_draws: int,
) -> np.ndarray:
    """Simulate the joint null ``H0: all RGAs equal``, one column per group.

    Groups are disjoint samples, so under H0 the per-group estimates are
    *independent* normals centred on the common value, with the standard errors
    already computed. Centring them at zero loses nothing, because everything
    downstream is a difference. One ``(n_draws, k)`` matrix therefore carries
    the whole joint null and serves both the max-T p-value and the noise floor
    of the gap - with no re-estimation of RGA and no resampling of the data.
    """
    return rng.normal(size=(n_draws, standard_errors.size)) * standard_errors


def _max_t_null(standard_errors: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Null distribution of ``max`` over pairs of ``|z|``, for the max-T test.

    The headline gap is ``max - min`` over groups, so its z statistic is the
    largest of the ``k(k-1)/2`` pairwise ones and is *selected*, not fixed in
    advance. Referring it to a standard normal - which is what an unadjusted
    p-value does - rejects far too often: measured type I error under exact
    parity was 13.3% at three groups and 27.3% at five, against a nominal 5%.

    Given the joint null draws of :func:`_null_group_draws`, form every pair and
    take the maximum. Because this uses the real correlation structure between
    pairs that share a group, it is markedly less conservative than Bonferroni
    while still controlling the family-wise error rate.
    """
    k = standard_errors.size
    pairs_i, pairs_j = np.triu_indices(k, k=1)
    denominator = np.hypot(standard_errors[pairs_i], standard_errors[pairs_j])
    # A degenerate pair contributes no evidence; keep it finite and harmless.
    denominator = np.where(denominator > 0, denominator, np.inf)
    statistics = (draws[:, pairs_i] - draws[:, pairs_j]) / denominator
    return np.max(np.abs(statistics), axis=1)


def _null_spread(draws: np.ndarray) -> np.ndarray:
    """``max - min`` across groups under the joint null - the gap's noise floor.

    ``max - min`` is non-negative by construction, so under *exact* parity its
    expectation is strictly positive and grows with the number of groups and
    with the per-group standard errors: measured at 0.034 for two groups of
    300, 0.084 for five groups of 200. Subtracting this expectation from the
    observed gap is what :attr:`ParityResult.gap_excess_over_noise` reports.
    """
    return draws.max(axis=1) - draws.min(axis=1)


def _stratified_gap_draws(
    y: np.ndarray,
    yhat: np.ndarray,
    masks: list[np.ndarray],
    n_resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap distribution of ``max - min``, resampling inside each group.

    Each group is resampled independently - which is what "stratified" means
    here, and what keeps the very different group sizes honest - via the exact
    vectorised multinomial bootstrap of :mod:`rgbox.inference`. Replicates in
    which any group degenerates (a resample that turns out single-class) are
    dropped as a whole, since a gap needs every group to have a value.
    """
    per_group = []
    for mask in masks:
        replicates = _bootstrap_values(
            y[mask], yhat[mask], n_resamples=n_resamples, random_state=rng
        )
        per_group.append(np.asarray(replicates, dtype=np.float64))
    stacked = np.vstack(per_group)  # (k, n_resamples)
    usable = np.all(np.isfinite(stacked), axis=0)
    stacked = stacked[:, usable]
    if stacked.shape[1] == 0:
        return np.zeros(0, dtype=np.float64)
    return stacked.max(axis=0) - stacked.min(axis=0)


def labels_from_dummies(
    X: Any,
    columns: Sequence[Any] | Any,
    *,
    reference: Any = "reference",
) -> np.ndarray:
    """Rebuild one categorical label per row from a one-hot encoding.

    :func:`rga_parity` needs one label per row, but a model's design matrix
    usually carries the protected attribute already split into dummies. This
    inverts that split, so the same ``protected=[...]`` argument that
    :func:`rgf` uses to remove the attribute as a unit can also drive the
    per-group comparison.

    Rows with every dummy at 0 get ``reference``: that is the level a
    ``drop_first=True`` encoding omits, and dropping those rows instead would
    silently exclude the largest group from the parity gap.

    Raises
    ------
    rgbox.InputError
        If a column is not 0/1, or a row has more than one dummy set - which
        means the columns are not one attribute's encoding, and any label read
        off them would be arbitrary.
    """
    columns = resolve_columns(columns, X, "columns")
    block = np.column_stack(
        [as_1d_float(X[column], f"X[{column!r}]") for column in columns]
    )
    if not np.all(np.isin(block, (0.0, 1.0))):
        raise InputError(
            f"columns {columns!r} are not a 0/1 encoding, so they cannot be "
            "inverted into one label per row."
        )
    per_row = block.sum(axis=1)
    overlapping = int(np.count_nonzero(per_row > 1))
    if overlapping:
        raise InputError(
            f"{overlapping} row(s) have more than one of {columns!r} set to 1. "
            "These columns are not one attribute's one-hot encoding; pass the "
            "dummies of a single attribute."
        )

    names = np.empty(len(columns), dtype=object)
    for position, column in enumerate(columns):
        names[position] = column

    labels = np.empty(block.shape[0], dtype=object)
    labels[:] = reference
    encoded = per_row == 1
    labels[encoded] = names[block[encoded].argmax(axis=1)]
    return labels


def rga_parity(
    y: Any,
    yhat: Any,
    groups: Any,
    *,
    min_group_size: int = 50,
    level: float = 0.95,
    method: str = "jackknife",
    n_resamples: int = 2000,
    random_state: Any = None,
    attribute: Any = None,
) -> ParityResult:
    """Compare ranking quality across the levels of a protected attribute.

    Parameters
    ----------
    y, yhat :
        Target and scores, on the evaluation sample.
    groups :
        Protected attribute, one label per row. Any number of levels is
        allowed; upstream documented "binary" but silently iterated over every
        unique value and took ``max - min``. **Missing labels are rejected**,
        not dropped - see :func:`rgbox._validation.as_group_labels`.
    min_group_size :
        Levels smaller than this are still reported, with their own interval,
        but do not enter the headline gap. Set to 0 to include everything.

    Returns
    -------
    ParityResult
        ``float(result)`` is the gap, ``str(result)`` is a one-line summary and
        ``result.to_dict()`` serialises the whole thing. The gap's confidence
        interval comes from a stratified bootstrap that resamples inside each
        group, which handles both the ``max - min`` selection effect and the
        very different group sizes.
    """
    y_arr, yhat_arr = as_score_pair(y, yhat)
    group_values = as_group_labels(groups, "groups", y_arr.size)
    level = check_level(level)

    records: list[GroupRGA] = []
    eligible: dict[Any, RGAEstimate] = {}
    eligible_masks: dict[Any, np.ndarray] = {}
    excluded: list[Any] = []

    for label, mask in distinct_levels(group_values):
        size = int(mask.sum())
        if size < 3:
            records.append(
                GroupRGA(
                    label,
                    size,
                    None,
                    None,
                    None,
                    None,
                    False,
                    "fewer than 3 observations",
                )
            )
            excluded.append(label)
            continue
        try:
            estimate = rga_ci(
                y_arr[mask],
                yhat_arr[mask],
                method=method,
                level=level,
                n_resamples=n_resamples,
                random_state=random_state,
            )
        except (UndefinedMetricError, InsufficientDataError) as exc:
            records.append(
                GroupRGA(
                    label, size, None, None, None, None, False, str(exc).split(".")[0]
                )
            )
            excluded.append(label)
            continue
        included = size >= min_group_size
        records.append(
            GroupRGA(
                label,
                size,
                estimate.estimate,
                estimate.standard_error,
                estimate.ci_low,
                estimate.ci_high,
                included,
                "" if included else f"n < min_group_size ({min_group_size})",
            )
        )
        if included:
            eligible[label] = estimate
            eligible_masks[label] = mask
        else:
            excluded.append(label)

    records.sort(key=lambda record: (record.rga is None, record.rga))

    if len(eligible) < 2:
        return ParityResult(
            groups=records,
            gap=None,
            gap_ci=None,
            gap_p_value=None,
            best_group=None,
            worst_group=None,
            level=level,
            method=method,
            attribute=attribute,
            excluded=excluded,
        )

    labels = list(eligible)
    point = {label: eligible[label].estimate for label in labels}
    # Ordering, rather than max() and min(), is what keeps the two ends
    # distinct - the same collapse fixed in :func:`rgbox.outcome_parity`. Both
    # return the *first* extremal element, so groups that all share one RGA
    # put best on top of worst, {best, worst} becomes a one-element set, and
    # the widest-pair lookup below raises StopIteration. A model that ranks
    # perfectly inside every group does it (RGA 1.0 throughout), and so does
    # any attribute whose groups happen to tie.
    by_rga = sorted(labels, key=lambda label: point[label])
    worst, best = by_rga[0], by_rga[-1]
    gap = point[best] - point[worst]

    # Pairwise comparisons: disjoint samples, so variances simply add.
    z_crit = _normal_quantile(1.0 - (1.0 - level) / 2.0)
    rng = np.random.default_rng(random_state)

    # One simulation of the joint null serves two purposes: the family-wise
    # correction for having picked the widest pair out of many, and the noise
    # floor of the gap.
    group_ses = np.array(
        [eligible[label].standard_error for label in labels], dtype=np.float64
    )
    null_draws = _null_group_draws(group_ses, rng, n_resamples)
    max_t_draws = _max_t_null(group_ses, null_draws)
    gap_noise_floor = float(np.mean(_null_spread(null_draws)))

    def adjusted(statistic: float) -> float:
        return float(
            (1 + np.count_nonzero(max_t_draws >= abs(statistic)))
            / (max_t_draws.size + 1)
        )

    pairwise: list[dict[str, Any]] = []
    for i, first in enumerate(labels):
        for second in labels[i + 1 :]:
            difference = point[first] - point[second]
            se = float(
                np.hypot(
                    eligible[first].standard_error, eligible[second].standard_error
                )
            )
            statistic = difference / se if se > 0 else 0.0
            pairwise.append(
                {
                    "group_a": first,
                    "group_b": second,
                    "difference": difference,
                    "standard_error": se,
                    "ci_low": difference - z_crit * se,
                    "ci_high": difference + z_crit * se,
                    "p_value": 2.0 * (1.0 - _normal_cdf(abs(statistic))),
                    "p_value_adjusted": adjusted(statistic),
                }
            )

    # Stratified bootstrap for the max-min gap.
    draws = _stratified_gap_draws(
        y_arr, yhat_arr, [eligible_masks[label] for label in labels], n_resamples, rng
    )
    if draws.size >= max(50, n_resamples // 2):
        alpha = 1.0 - level
        gap_ci = tuple(float(v) for v in np.quantile(draws, [alpha / 2, 1 - alpha / 2]))
    else:
        gap_ci = None

    # How much of the observed spread is more than exact parity would produce
    # by itself. Subtracting the *bootstrap* mean instead - which is what this
    # did before - corrects nothing: the bootstrap resamples inside groups
    # around the observed per-group values, so its world already contains the
    # observed spread and its mean sits on top of the gap rather than above a
    # true zero. Measured under exact parity, that removed only 24-37% of the
    # bias and exceeded the raw gap more often than not when a real gap existed.
    gap_excess_over_noise = gap - gap_noise_floor

    # The headline p-value is for the widest pair, which is the gap.
    widest = next(
        record
        for record in pairwise
        if {record["group_a"], record["group_b"]} == {best, worst}
    )

    return ParityResult(
        groups=records,
        gap=gap,
        gap_ci=gap_ci,
        gap_p_value=widest["p_value_adjusted"],
        best_group=best,
        worst_group=worst,
        level=level,
        method=f"{method} + stratified bootstrap",
        attribute=attribute,
        pairwise=pairwise,
        excluded=excluded,
        gap_excess_over_noise=gap_excess_over_noise,
        gap_noise_floor=gap_noise_floor,
        gap_p_value_unadjusted=widest["p_value"],
        multiplicity=(f"max-T over {len(pairwise)} pair(s), {n_resamples} draws"),
    )


def rgf(
    X_train: Any,
    X_test: Any,
    model: Any,
    protected: Any,
    *,
    yhat: Any = None,
    refit: Any = None,
    method: str = "mean",
    normalize: bool = False,
    pos_label: Any = None,
    greater_is_better: bool = True,
    random_state: Any = None,
) -> dict[str, Any]:
    """Rank Graduation Fairness *as defined in the paper's own R code*.

    The R scripts shipped with this repository define
    ``RGF = RGA(yhat_full, yhat_without_the_protected_variable)`` - structurally
    an RGE computed on the protected attribute, answering "how much of the
    model's ranking is driven by the sensitive attribute?". The Python package
    implemented something entirely different under the fairness heading (the
    ``max - min`` RGA gap across groups). Both are provided here, under names
    that say which is which.

    Returns a dict with ``rgf`` (the RGA between full and reduced scores; 1
    means the protected attribute is irrelevant to the ranking) and ``rge``
    (``1 - rgf``; 0 means irrelevant).

    ``protected`` may be a single column or a **list** of columns. Pass the
    list for a one-hot encoded categorical attribute: all of its dummies are
    removed together, as one attribute. Removing a single dummy would measure
    "how much does the model use *this level*", which is a different question
    and, for a drop-first encoding, not even well posed for the reference
    level. With ``method="permute"`` the group shares one permutation, so the
    encoding stays a valid one-hot.

    A tuple is read as a single ``MultiIndex`` label, not as a group - see
    :func:`rgbox.predictors.resolve_columns`. Use a list for a group.

    The sibling fairness functions take the same argument in the same form:
    :func:`proxy_leakage` scores each candidate against the group's strongest
    level, and :func:`rgbox.rgbox_report` rebuilds one categorical label per
    row with :func:`labels_from_dummies` before measuring parity.
    """
    from .explainability import rge as _rge

    columns = resolve_columns(protected, X_test, "protected")
    results = _rge(
        X_train,
        X_test,
        model,
        columns,
        yhat=yhat,
        method=method,
        group=True,
        normalize=normalize,
        refit=refit,
        pos_label=pos_label,
        greater_is_better=greater_is_better,
        random_state=random_state,
    )
    result = results[0]
    return {
        "attribute": protected,
        "rgf": result.rga_reduced,
        "rge": result.rge,
        "normalized": normalize,
        "method": method,
        "n": result.n,
        "interpretation": (
            "RGF near 1 means the ranking barely uses the protected attribute; "
            "RGF near 0.5 means the attribute drives it."
        ),
    }


def proxy_leakage(
    X: Any,
    protected: Any,
    candidates: Sequence[Any] | None = None,
    *,
    yhat: Any = None,
    ordered: bool = False,
) -> dict[str, Any]:
    """Rank the predictors by how strongly each proxies for ``protected``.

    Dropping a protected attribute from the feature list does not remove its
    influence if a correlated predictor carries it - the standard proxy
    problem, and the usual reason a "we don't use gender" claim fails review.
    This fits nothing new: it scores each candidate predictor by the RGA between
    that predictor and the protected attribute, i.e. how well the predictor
    alone ranks the protected attribute. Values far from 0.5 in either
    direction indicate a proxy.

    Parameters
    ----------
    X :
        The evaluation frame. Everything is measured on these columns; no
        model is fitted and none is needed.
    protected :
        Column label, or a **list** of one-hot dummies, exactly as in
        :func:`rgf`. RGA needs an ordered target and a multi-level attribute
        has no order, so a group is scored one level at a time and each
        candidate is reported against its **worst** level: a predictor that
        reconstructs any single level is a proxy for the attribute. ``level``
        names the level that produced the reported figure. Direction does not
        matter - ``leakage`` is ``|2·RGA - 1|``, so under a drop-first encoding
        a predictor that identifies the omitted reference level scores just as
        high, from the other side, on the dummies that are present.

        A **single column holding more than two levels** is the case that
        needs a decision from the caller, and ``ordered`` is where it is made.
        A one-hot group was already decomposed level by level, but one integer
        column was passed to RGA whole, which reads its codes as a scale:
        relabelling ``{north: 0, centre: 1, south: 2}`` to
        ``{centre: 0, north: 1, south: 2}`` is the same attribute and changed
        the reported leakage from 0.623 to 0.596. See ``ordered``.
    candidates :
        Which columns to score. Defaults to every column of ``X`` except the
        protected attribute's own dummies.
    ordered :
        Whether a single multi-level column is genuinely ordinal. ``False``
        (the default) decomposes it one-vs-rest, exactly as a one-hot group is
        decomposed, so the answer no longer depends on how the levels happen to
        be numbered. Pass ``True`` for an attribute whose codes really are a
        scale - an age band, a rating grade - to score it in one pass against
        that order. Binary and one-hot inputs are unaffected either way.
    yhat :
        Optional model scores on ``X``. When given, the result carries an extra
        ``model_leakage`` entry measuring how well the *model's own output*
        ranks the protected attribute - the question a reviewer asks
        immediately after seeing which features proxy it.

    Notes
    -----
    Until 1.0.1 the signature was
    ``proxy_leakage(X_train, X_test, model, protected, ...)`` plus ``method``,
    ``pos_label``, ``greater_is_better`` and ``random_state``, and **none of
    those seven arguments was ever read**: passing a different model, a
    different training frame or random noise as ``yhat`` returned a
    bit-identical result. That is the same defect this fork documents as
    upstream's headline bug, so the unused parameters are gone and ``yhat`` now
    does something.
    """
    if hasattr(protected, "columns") or hasattr(protected, "dtypes"):
        raise InputError(
            "proxy_leakage's signature changed in 1.0.1: it is now "
            "proxy_leakage(X, protected, candidates=None, *, yhat=None). The "
            "old (X_train, X_test, model, protected) form accepted a training "
            "frame and a model and then used neither. Pass the evaluation "
            "frame and the protected attribute only."
        )
    protected_columns = resolve_columns(protected, X, "protected")
    protected_values = {
        column: as_1d_float(X[column], f"X[{column!r}]") for column in protected_columns
    }
    ordinal_note = ""
    if len(protected_columns) == 1 and not ordered:
        # One column, more than two levels: RGA would read the level codes as a
        # scale, and a nominal attribute has none. Split it one-vs-rest so the
        # result is invariant to how the levels are numbered - which is exactly
        # what already happened to a one-hot encoded group.
        (only_column,) = protected_columns
        levels = np.unique(protected_values[only_column])
        if levels.size > 2:
            protected_values = {
                f"{only_column} == {level:g}": (
                    protected_values[only_column] == level
                ).astype(np.float64)
                for level in levels
            }
            ordinal_note = (
                f"{only_column!r} has {levels.size} levels and was scored "
                "one-vs-rest, so the reported leakage does not depend on how "
                "the levels are numbered. Pass ordered=True to treat the codes "
                "as a genuine scale instead."
            )
    excluded = set(protected_columns)
    columns = (
        resolve_columns(candidates, X, "candidates")
        if candidates is not None
        else [c for c in X.columns if c not in excluded]
    )

    def worst_level(values: np.ndarray) -> tuple[Any, float, float] | None:
        """The protected level this vector reconstructs best, and how well."""
        worst: tuple[Any, float, float] | None = None
        for level, target in protected_values.items():
            try:
                score = rga(target, values)
            except UndefinedMetricError:
                continue  # this level is constant here; the others may not be
            leakage = abs(2 * score - 1)
            if worst is None or leakage > worst[2]:
                worst = (level, score, leakage)
        return worst

    rows = []
    for column in columns:
        try:
            values = as_1d_float(X[column], str(column))
        except InputError:
            rows.append(
                {
                    "variable": column,
                    "rga": None,
                    "level": None,
                    "note": "non-numeric column, skipped",
                }
            )
            continue
        worst = worst_level(values)
        if worst is None:
            rows.append(
                {
                    "variable": column,
                    "rga": None,
                    "level": None,
                    "note": "protected attribute is constant here",
                }
            )
            continue
        rows.append(
            {
                "variable": column,
                "rga": worst[1],
                "level": worst[0],
                "leakage": worst[2],
                "note": "",
            }
        )
    rows.sort(key=lambda row: (row.get("leakage") is None, -(row.get("leakage") or 0)))

    result: dict[str, Any] = {
        "protected": protected,
        "proxies": rows,
        "ordered": ordered,
        "ordinal_note": ordinal_note,
    }
    if yhat is not None:
        scores = as_1d_float(yhat, "yhat")
        if scores.size != len(X):
            raise InputError(
                f"'yhat' has {scores.size} entries but X has {len(X)} rows."
            )
        found = worst_level(scores)
        result["model_leakage"] = (
            {
                "rga": None,
                "level": None,
                "leakage": None,
                "note": "protected attribute is constant here",
            }
            if found is None
            else {
                "rga": found[1],
                "level": found[0],
                "leakage": found[2],
                "note": (
                    "how well the model's own score ranks the protected "
                    "attribute; 0 means it carries none of it"
                ),
            }
        )
    return result
