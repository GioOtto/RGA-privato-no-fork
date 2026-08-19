"""Outcome-based fairness criteria, with intervals and a multiplicity correction.

Why this module exists
----------------------
:mod:`rgbox.fairness` says, of RGA parity: *"Report it alongside outcome-based
criteria, never instead of them."* It then offered no way to do so. This module
closes that gap.

RGA parity asks whether the model *ranks* equally well inside each group. The
criteria here ask what the model actually *does* to people, which is a
different question and the one regulation is usually written about:

======================  ===============================================
criterion               statistic
======================  ===============================================
demographic parity      ``P(D = 1 | g)`` - the selection rate
disparate impact        ``min_g P(D=1|g) / max_g P(D=1|g)`` - the ratio
equal opportunity       ``P(D = 1 | Y = 1, g)`` - the true positive rate
predictive equality     ``P(D = 1 | Y = 0, g)`` - the false positive rate
equalised odds          the larger of the TPR and FPR gaps
======================  ===============================================

Every one is a difference (or a ratio) of within-group proportions, so all of
it is closed-form: no resampling, no SciPy, no pandas.

Thresholds, and why you must pass one
-------------------------------------
Everything above is defined on a **decision**, not on a score, so it needs a
threshold - and the rest of this package is deliberately threshold-free. That
tension is not resolved by picking a default: a fairness number computed at a
cut-off nobody chose is worse than no number at all, because it looks
authoritative. So there is no default. Pass ``decisions`` already at 0/1, or
pass scores together with an explicit ``threshold``. Whichever you pass is
recorded in the result and printed in the report.

What this adds over what already exists
---------------------------------------
Fairlearn has computed these since 0.4 and has had bootstrap intervals since
0.11. Two things here are not in it:

* **the intervals are analytic** - Wilson score intervals per group, and a
  two-proportion normal interval for each gap - so they cost nothing and are
  deterministic, which matters for a quarterly report that must be diffable;
* **the headline p-value is corrected for multiplicity by max-T**, exactly as
  in :func:`rgbox.rga_parity`. ``max - min`` over ``k`` groups selects the
  widest of ``k(k-1)/2`` pairs, and referring that to a normal over-rejects
  badly - measured at 27.3% against a nominal 5% for five groups on the RGA
  side. The same selection effect applies here, with the same fix: group
  proportions are computed on disjoint rows, so under the null of exact parity
  they are independent normals with known standard errors, and the joint null
  of every pairwise statistic is simulable directly from them.

As in :func:`rgbox.rga_parity`, ``gap_noise_floor`` records what ``max - min``
would average under exact parity at these group sizes, and
``gap_excess_over_noise`` is the observed gap net of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._validation import (
    as_1d_float,
    as_group_labels,
    check_level,
    distinct_levels,
)
from .exceptions import InputError
from .fairness import _max_t_null, _null_group_draws, _null_spread
from .inference import _normal_cdf, _normal_quantile

__all__ = [
    "GroupRate",
    "CriterionResult",
    "OutcomeParityResult",
    "outcome_parity",
]

#: The "four-fifths rule" of US employment-discrimination practice: a selection
#: rate ratio below 0.8 is conventionally treated as evidence of adverse
#: impact. It is a rule of thumb from one jurisdiction, not a law of nature.
FOUR_FIFTHS = 0.8


def _wilson_interval(successes: float, n: int, z: float) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Preferred over the Wald interval because subgroup counts are small and
    rates are often near 0 or 1, exactly where Wald degenerates to a
    zero-width interval around an impossible point estimate.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = (z / denominator) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class GroupRate:
    """One group's rate for one criterion."""

    group: Any
    n: int
    n_eligible: int
    n_selected: int
    rate: float | None
    standard_error: float | None
    ci_low: float | None
    ci_high: float | None
    included: bool
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "n": self.n,
            "n_eligible": self.n_eligible,
            "n_selected": self.n_selected,
            "rate": self.rate,
            "standard_error": self.standard_error,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "included": self.included,
            "note": self.note,
        }


@dataclass(frozen=True)
class CriterionResult:
    """Per-group rates for one criterion, plus the gap and its uncertainty."""

    name: str
    description: str
    conditioned_on: str
    groups: list[GroupRate]
    gap: float | None
    gap_ci: tuple[float, float] | None
    gap_p_value: float | None
    gap_p_value_unadjusted: float | None
    gap_noise_floor: float | None
    gap_excess_over_noise: float | None
    best_group: Any
    worst_group: Any
    ratio: float | None = None
    ratio_ci: tuple[float, float] | None = None
    pairwise: list[dict[str, Any]] = field(default_factory=list, repr=False)
    multiplicity: str = ""

    def __str__(self) -> str:
        if self.gap is None:
            return f"{self.name}: not computable (fewer than two eligible groups)."
        tail = "" if self.gap_p_value is None else f", p = {self.gap_p_value:.4g}"
        interval = (
            ""
            if self.gap_ci is None
            else f" (CI {self.gap_ci[0]:+.4f}..{self.gap_ci[1]:+.4f})"
        )
        ratio = "" if self.ratio is None else f", ratio = {self.ratio:.4f}"
        return (
            f"{self.name}: gap = {self.gap:.4f}{interval}{tail}{ratio} "
            f"[worst: {self.worst_group!r}, best: {self.best_group!r}]"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "conditioned_on": self.conditioned_on,
            "gap": self.gap,
            "gap_ci_low": None if self.gap_ci is None else self.gap_ci[0],
            "gap_ci_high": None if self.gap_ci is None else self.gap_ci[1],
            "gap_p_value": self.gap_p_value,
            "gap_p_value_unadjusted": self.gap_p_value_unadjusted,
            "gap_noise_floor": self.gap_noise_floor,
            "gap_excess_over_noise": self.gap_excess_over_noise,
            "best_group": self.best_group,
            "worst_group": self.worst_group,
            "ratio": self.ratio,
            "ratio_ci_low": None if self.ratio_ci is None else self.ratio_ci[0],
            "ratio_ci_high": None if self.ratio_ci is None else self.ratio_ci[1],
            "multiplicity": self.multiplicity,
            "groups": [g.to_dict() for g in self.groups],
            "pairwise": list(self.pairwise),
        }


@dataclass(frozen=True)
class OutcomeParityResult:
    """Every outcome-based criterion, on one sample, at one threshold."""

    criteria: dict[str, CriterionResult]
    n: int
    level: float
    threshold: float | None
    attribute: Any = None
    equalized_odds: float | None = None
    disparate_impact: float | None = None
    notes: list[str] = field(default_factory=list)

    #: Printed in the generated report, because the distinction is the single
    #: most common misreading of a fairness table.
    INTERPRETATION = (
        "These are outcome-based criteria: they describe what the model does "
        "at a chosen threshold, not how well it ranks. They are the companion "
        "to rga_parity (which is AUC parity), not a substitute, and they "
        "cannot all hold at once except in degenerate cases - demographic "
        "parity and equalised odds are mutually incompatible whenever "
        "prevalence differs between groups."
    )

    def __getitem__(self, name: str) -> CriterionResult:
        return self.criteria[name]

    def __str__(self) -> str:
        head = (
            f"Outcome parity (n = {self.n}"
            + ("" if self.threshold is None else f", threshold = {self.threshold:g}")
            + ")"
        )
        lines = [head]
        lines.extend(f"  {criterion}" for criterion in self.criteria.values())
        if self.disparate_impact is not None:
            verdict = (
                "below the four-fifths rule"
                if self.disparate_impact < FOUR_FIFTHS
                else "above the four-fifths rule"
            )
            lines.append(
                f"  disparate impact ratio = {self.disparate_impact:.4f} ({verdict})"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "level": self.level,
            "threshold": self.threshold,
            "attribute": self.attribute,
            "equalized_odds": self.equalized_odds,
            "disparate_impact": self.disparate_impact,
            "four_fifths_rule_met": (
                None
                if self.disparate_impact is None
                else bool(self.disparate_impact >= FOUR_FIFTHS)
            ),
            "interpretation": self.INTERPRETATION,
            "notes": list(self.notes),
            "criteria": {
                name: criterion.to_dict() for name, criterion in self.criteria.items()
            },
        }


def _as_decisions(decisions: Any, threshold: float | None, n: int) -> np.ndarray:
    values = as_1d_float(decisions, "decisions")
    if values.size != n:
        raise InputError(f"'decisions' has {values.size} entries but y_true has {n}.")
    if threshold is not None:
        return (values >= threshold).astype(np.float64)
    unique = np.unique(values)
    if not np.all(np.isin(unique, (0.0, 1.0))):
        raise InputError(
            "'decisions' must be 0/1. It holds "
            f"{unique.size} distinct values, so it looks like a score rather "
            "than a decision: pass `threshold=` to say where the cut-off is. "
            "There is deliberately no default threshold - every criterion "
            "here is defined on a decision, and choosing the cut-off silently "
            "would make the numbers look authoritative without anyone having "
            "picked them."
        )
    return values


def _criterion(
    name: str,
    description: str,
    conditioned_on: str,
    labels: np.ndarray,
    decisions: np.ndarray,
    eligible: np.ndarray,
    *,
    min_group_size: int,
    level: float,
    z_crit: float,
    rng: np.random.Generator,
    n_resamples: int,
    want_ratio: bool,
) -> CriterionResult:
    """Per-group selection rates over the rows flagged by ``eligible``."""
    records: list[GroupRate] = []
    kept: dict[Any, tuple[float, float, int]] = {}  # label -> (rate, se, n)

    for label, mask in distinct_levels(labels):
        size = int(mask.sum())
        rows = mask & eligible
        n_eligible = int(rows.sum())
        n_selected = int(decisions[rows].sum())
        if n_eligible == 0:
            records.append(
                GroupRate(
                    label,
                    size,
                    0,
                    0,
                    None,
                    None,
                    None,
                    None,
                    False,
                    f"no rows with {conditioned_on}" if conditioned_on else "no rows",
                )
            )
            continue
        rate = n_selected / n_eligible
        standard_error = math.sqrt(rate * (1.0 - rate) / n_eligible)
        low, high = _wilson_interval(n_selected, n_eligible, z_crit)
        included = n_eligible >= min_group_size
        records.append(
            GroupRate(
                label,
                size,
                n_eligible,
                n_selected,
                rate,
                standard_error,
                low,
                high,
                included,
                "" if included else f"eligible n < min_group_size ({min_group_size})",
            )
        )
        if included:
            kept[label] = (rate, standard_error, n_eligible)

    records.sort(key=lambda record: (record.rate is None, record.rate))

    if len(kept) < 2:
        return CriterionResult(
            name=name,
            description=description,
            conditioned_on=conditioned_on,
            groups=records,
            gap=None,
            gap_ci=None,
            gap_p_value=None,
            gap_p_value_unadjusted=None,
            gap_noise_floor=None,
            gap_excess_over_noise=None,
            best_group=None,
            worst_group=None,
        )

    names = list(kept)
    rates = {label: kept[label][0] for label in names}
    best = max(names, key=lambda label: rates[label])
    worst = min(names, key=lambda label: rates[label])
    gap = rates[best] - rates[worst]

    standard_errors = np.array([kept[label][1] for label in names], dtype=np.float64)
    null_draws = _null_group_draws(standard_errors, rng, n_resamples)
    max_t_draws = _max_t_null(standard_errors, null_draws)
    noise_floor = float(np.mean(_null_spread(null_draws)))

    def adjusted(statistic: float) -> float:
        return float(
            (1 + np.count_nonzero(max_t_draws >= abs(statistic)))
            / (max_t_draws.size + 1)
        )

    pairwise: list[dict[str, Any]] = []
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            difference = rates[first] - rates[second]
            se = float(np.hypot(kept[first][1], kept[second][1]))
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

    widest = next(
        record
        for record in pairwise
        if {record["group_a"], record["group_b"]} == {best, worst}
    )
    gap_se = widest["standard_error"]

    ratio = ratio_ci = None
    if want_ratio:
        low_rate, high_rate = rates[worst], rates[best]
        if high_rate > 0 and low_rate > 0:
            ratio = low_rate / high_rate
            # Delta method on the log ratio: the interval stays inside (0, inf)
            # and is not forced to be symmetric around a bounded quantity.
            n_low, n_high = kept[worst][2], kept[best][2]
            log_se = math.sqrt(
                (1 - low_rate) / (n_low * low_rate)
                + (1 - high_rate) / (n_high * high_rate)
            )
            ratio_ci = (
                float(math.exp(math.log(ratio) - z_crit * log_se)),
                float(min(1.0, math.exp(math.log(ratio) + z_crit * log_se))),
            )
        elif high_rate > 0:
            ratio = 0.0

    return CriterionResult(
        name=name,
        description=description,
        conditioned_on=conditioned_on,
        groups=records,
        gap=gap,
        gap_ci=(gap - z_crit * gap_se, gap + z_crit * gap_se),
        gap_p_value=widest["p_value_adjusted"],
        gap_p_value_unadjusted=widest["p_value"],
        gap_noise_floor=noise_floor,
        gap_excess_over_noise=gap - noise_floor,
        best_group=best,
        worst_group=worst,
        ratio=ratio,
        ratio_ci=ratio_ci,
        pairwise=pairwise,
        multiplicity=f"max-T over {len(pairwise)} pair(s), {n_resamples} draws",
    )


def outcome_parity(
    y_true: Any,
    decisions: Any,
    groups: Any,
    *,
    threshold: float | None = None,
    positive_label: Any = None,
    min_group_size: int = 50,
    level: float = 0.95,
    n_resamples: int = 2000,
    random_state: Any = None,
    attribute: Any = None,
) -> OutcomeParityResult:
    """Demographic parity, equal opportunity, equalised odds, disparate impact.

    The companion to :func:`rgbox.rga_parity`, which measures ranking quality
    per group and is *not* any of these.

    Parameters
    ----------
    y_true :
        Observed binary outcome. Two distinct values are required; the larger
        is the positive one unless ``positive_label`` says otherwise.
    decisions :
        The model's **decision**, 0/1 - or its scores, if ``threshold`` is
        given, in which case the decision is ``scores >= threshold``.
    groups :
        Protected attribute, one label per row. Missing labels are rejected.
    threshold :
        Cut-off applied to ``decisions``. There is **no default**: see the
        module docstring.
    positive_label :
        Which value of ``y_true`` counts as the positive outcome. Defaults to
        the larger of the two.
    min_group_size :
        Groups with fewer than this many *eligible* rows for a criterion are
        reported but excluded from that criterion's gap. Note that a group can
        be included for demographic parity and excluded for equal opportunity,
        because the second conditions on ``y_true == positive``.
    level, n_resamples, random_state :
        Confidence level, and the draw count and seed for the max-T
        simulation. The estimates themselves are closed-form; only the
        multiplicity correction is simulated.

    Returns
    -------
    OutcomeParityResult
        ``result["demographic_parity"]`` and friends are
        :class:`CriterionResult` objects; ``result.to_dict()`` serialises
        everything.

    Examples
    --------
    >>> import numpy as np
    >>> from rgbox import outcome_parity
    >>> rng = np.random.default_rng(0)
    >>> y = (rng.random(400) < 0.4).astype(float)
    >>> scores = rng.random(400)
    >>> groups = np.where(np.arange(400) < 200, "a", "b")
    >>> result = outcome_parity(y, scores, groups, threshold=0.5)
    >>> sorted(result.criteria)
    ['demographic_parity', 'equal_opportunity', 'predictive_equality']
    """
    y_arr = as_1d_float(y_true, "y_true")
    n = y_arr.size
    decision_values = _as_decisions(decisions, threshold, n)
    labels = as_group_labels(groups, "groups", n)
    level = check_level(level)
    z_crit = _normal_quantile(1.0 - (1.0 - level) / 2.0)
    rng = np.random.default_rng(random_state)

    unique = np.unique(y_arr)
    if unique.size != 2:
        raise InputError(
            f"'y_true' must be binary for outcome-based criteria; it has "
            f"{unique.size} distinct value(s). RGA parity "
            "(rgbox.rga_parity) is defined for ordinal and continuous "
            "targets; these criteria are not."
        )
    positive = unique[1] if positive_label is None else float(positive_label)
    if positive not in set(unique.tolist()):
        raise InputError(
            f"positive_label={positive_label!r} is not one of the values of "
            f"'y_true' ({unique.tolist()!r})."
        )
    is_positive = y_arr == positive

    everything = np.ones(n, dtype=bool)
    shared = {
        "min_group_size": min_group_size,
        "level": level,
        "z_crit": z_crit,
        "rng": rng,
        "n_resamples": n_resamples,
    }
    criteria = {
        "demographic_parity": _criterion(
            "demographic_parity",
            "P(decision = 1 | group): equal selection rates across groups.",
            "",
            labels,
            decision_values,
            everything,
            want_ratio=True,
            **shared,
        ),
        "equal_opportunity": _criterion(
            "equal_opportunity",
            "P(decision = 1 | outcome = 1, group): equal true positive rates.",
            "y_true == positive",
            labels,
            decision_values,
            is_positive,
            want_ratio=False,
            **shared,
        ),
        "predictive_equality": _criterion(
            "predictive_equality",
            "P(decision = 1 | outcome = 0, group): equal false positive rates.",
            "y_true == negative",
            labels,
            decision_values,
            ~is_positive,
            want_ratio=False,
            **shared,
        ),
    }

    tpr_gap = criteria["equal_opportunity"].gap
    fpr_gap = criteria["predictive_equality"].gap
    equalized = (
        None
        if tpr_gap is None and fpr_gap is None
        else max(gap for gap in (tpr_gap, fpr_gap) if gap is not None)
    )

    notes: list[str] = []
    if threshold is None:
        notes.append("decisions were supplied already thresholded.")
    for name, criterion in criteria.items():
        if criterion.gap is None:
            notes.append(
                f"{name}: not computable - fewer than two groups had "
                f"{min_group_size} eligible rows."
            )
    if tpr_gap is None or fpr_gap is None:
        notes.append(
            "equalised odds needs both the TPR and the FPR gap; it is reported "
            "from whichever was computable."
        )

    return OutcomeParityResult(
        criteria=criteria,
        n=n,
        level=level,
        threshold=threshold,
        attribute=attribute,
        equalized_odds=equalized,
        disparate_impact=criteria["demographic_parity"].ratio,
        notes=notes,
    )
