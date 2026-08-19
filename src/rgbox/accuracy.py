"""Accuracy reporting - the module upstream deleted.

Compiled ``.pyc`` files for ``check_accuracy`` and ``check_privacy`` are still
committed in the original repository, but the sources were removed, leaving the
"A" of S.A.F.E. with no module of its own. This restores it and makes it do
something a validation report actually needs: put RGA next to the metrics the
reader already trusts, on the same sample, with intervals.

The one number to lead with is ``gini`` (``2 * RGA - 1``). On a binary target
it is bit-identical to the ``2 * AUROC - 1`` that every scorecard validation
already quotes; on a continuous target - loss given default, exposure, a
recovery amount - it keeps working, which is the whole point of the rank
graduation family.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._ranks import average_ranks
from ._validation import (
    as_1d_float,
    as_group_labels,
    as_score_pair,
    check_level,
    distinct_levels,
)
from .core import rga
from .exceptions import InputError, InsufficientDataError, UndefinedMetricError
from .inference import RGAComparison, RGAEstimate, rga_ci, rga_compare, rga_test

__all__ = [
    "AccuracyReport",
    "accuracy_report",
    "rga_ovr",
    "compare_models",
    "rga_by_segment",
    "contamination_curve",
]


def _spearman(y: np.ndarray, yhat: np.ndarray) -> float:
    a, b = average_ranks(y), average_ranks(yhat)
    a = a - a.mean()
    b = b - b.mean()
    denominator = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / denominator if denominator > 0 else float("nan")


def _pearson(y: np.ndarray, yhat: np.ndarray) -> float:
    a, b = y - y.mean(), yhat - yhat.mean()
    denominator = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / denominator if denominator > 0 else float("nan")


def _auroc(y: np.ndarray, yhat: np.ndarray) -> float | None:
    """AUROC by the Mann-Whitney U statistic, for a two-valued target.

    Computed *independently* of :func:`rga` - a rank-sum over the positives
    rather than a ratio of Gini covariances - specifically so that the two
    agreeing is evidence rather than a tautology. Earlier versions reported
    ``auroc`` by copying the RGA estimate, which could not disagree and so
    reassured nobody. Ties contribute 1/2 via the average ranks, matching
    ``sklearn.metrics.roc_auc_score``.
    """
    levels = np.unique(y)
    if levels.size != 2:
        return None
    positive = y == levels[1]
    n_pos = int(positive.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = average_ranks(yhat)
    u = float(ranks[positive].sum()) - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def _kendall(y: np.ndarray, yhat: np.ndarray) -> float | None:
    try:  # optional: SciPy is not a dependency of this package
        from scipy.stats import kendalltau
    except ImportError:
        return None
    return float(kendalltau(y, yhat).statistic)


@dataclass(frozen=True)
class AccuracyReport:
    """RGA alongside the conventional metrics, on one sample."""

    n: int
    rga: RGAEstimate
    gini: float
    gini_ci: tuple[float, float]
    is_binary: bool
    reference: dict[str, float | None] = field(default_factory=dict)
    significance: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"Accuracy report (n = {self.n})",
            f"  {self.rga}",
            f"  Gini (2*RGA-1)      {self.gini:+.4f} "
            f"[{self.gini_ci[0]:+.4f}, {self.gini_ci[1]:+.4f}]",
        ]
        for name, value in self.reference.items():
            if value is not None:
                lines.append(f"  {name:<20}{value:+.4f}")
        p_value = self.significance.get("p_value")
        if p_value is not None:
            lines.append(
                f"  H0: RGA = 0.5       p = {p_value:.3g} "
                f"({self.significance.get('method', '')})"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "rga": self.rga.to_dict(),
            "gini": self.gini,
            "gini_ci_low": self.gini_ci[0],
            "gini_ci_high": self.gini_ci[1],
            "is_binary": self.is_binary,
            "reference_metrics": dict(self.reference),
            "significance": dict(self.significance),
        }


def accuracy_report(
    y: Any,
    yhat: Any,
    *,
    method: str = "jackknife",
    level: float = 0.95,
    n_resamples: int = 2000,
    random_state: Any = None,
) -> AccuracyReport:
    """RGA with an interval, plus the reference metrics for the same sample.

    ``reference`` always includes Spearman, Pearson, RMSE, MAE and, when SciPy
    is installed, Kendall's tau. On a binary target it also includes AUROC,
    computed independently as a Mann-Whitney rank sum; it agrees with RGA to
    machine precision, which is a useful reassurance for a reader meeting RGA
    for the first time - and, because the two are computed by different routes,
    a live check on the implementation rather than a restatement of it.
    """
    y_arr, yhat_arr = as_score_pair(y, yhat, min_size=3)
    level = check_level(level)
    estimate = rga_ci(
        y_arr,
        yhat_arr,
        method=method,
        level=level,
        n_resamples=n_resamples,
        random_state=random_state,
    )
    unique = np.unique(y_arr)
    is_binary = unique.size == 2

    residual = y_arr - yhat_arr
    reference: dict[str, float | None] = {
        "spearman": _spearman(y_arr, yhat_arr),
        "pearson": _pearson(y_arr, yhat_arr),
        "kendall_tau": _kendall(y_arr, yhat_arr),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
    }
    if is_binary:
        # Independently computed (rank-sum, not the RGA covariance ratio), so
        # agreement is a genuine cross-check rather than a restatement.
        reference["auroc"] = _auroc(y_arr, yhat_arr)

    try:
        significance = rga_test(y_arr, yhat_arr, alternative="greater")
    except UndefinedMetricError:  # pragma: no cover - guarded upstream
        significance = {}

    return AccuracyReport(
        n=y_arr.size,
        rga=estimate,
        gini=estimate.gini,
        gini_ci=estimate.gini_ci,
        is_binary=is_binary,
        reference=reference,
        significance=significance,
    )


def rga_ovr(
    y: Any,
    proba: Any,
    *,
    classes: Sequence[Any] | None = None,
    average: str = "macro",
) -> dict[str, Any]:
    """One-vs-rest RGA for a multiclass target.

    Upstream's ``find_yhat`` took column 1 of ``predict_proba`` regardless of
    how many classes there were, so a three-class model was silently scored as
    ``P(class == classes_[1])`` with the other classes discarded. Here every
    class gets its own binary RGA (class vs rest, using that class's predicted
    probability), and the summary is their average - ``"macro"`` for an
    unweighted mean, ``"weighted"`` for one weighted by class prevalence, in
    the spirit of Hand & Till.
    """
    y_arr = as_1d_float(y, "y")
    proba_arr = np.asarray(proba, dtype=np.float64)
    if proba_arr.ndim != 2:
        raise InputError(
            f"'proba' must be (n_samples, n_classes); got shape {proba_arr.shape}."
        )
    if proba_arr.shape[0] != y_arr.size:
        raise InputError(
            f"'proba' has {proba_arr.shape[0]} rows but y has {y_arr.size}."
        )
    if classes is None:
        classes = np.unique(y_arr).tolist()
    if len(classes) != proba_arr.shape[1]:
        raise InputError(
            f"'classes' has {len(classes)} entries but 'proba' has "
            f"{proba_arr.shape[1]} columns."
        )

    per_class = []
    for index, label in enumerate(classes):
        indicator = (y_arr == label).astype(np.float64)
        support = int(indicator.sum())
        if support == 0 or support == y_arr.size:
            per_class.append(
                {
                    "class": label,
                    "n_positive": support,
                    "rga": None,
                    "note": "class absent or exhaustive",
                }
            )
            continue
        per_class.append(
            {
                "class": label,
                "n_positive": support,
                "rga": rga(indicator, proba_arr[:, index]),
                "note": "",
            }
        )

    usable = [row for row in per_class if row["rga"] is not None]
    if not usable:
        raise UndefinedMetricError("no class has both positive and negative cases.")
    if average == "weighted":
        weights = np.array([row["n_positive"] for row in usable], dtype=np.float64)
        summary = float(np.average([row["rga"] for row in usable], weights=weights))
    elif average == "macro":
        summary = float(np.mean([row["rga"] for row in usable]))
    else:
        raise InputError(f"unknown average {average!r}; use 'macro' or 'weighted'.")

    return {
        "rga": summary,
        "gini": 2 * summary - 1,
        "average": average,
        "per_class": per_class,
        "n": y_arr.size,
    }


def compare_models(
    y: Any,
    scores: dict[Any, Any],
    *,
    baseline: Any = None,
    method: str = "jackknife",
    level: float = 0.95,
    n_resamples: int = 2000,
    random_state: Any = None,
) -> dict[str, Any]:
    """Rank several candidate models and test each against a baseline.

    ``scores`` maps a model name to its score vector on the *same* evaluation
    sample. Every comparison is paired, so the strong correlation between two
    models scored on the same rows is used rather than discarded.
    """
    if len(scores) < 1:
        raise InputError("'scores' must contain at least one model.")
    estimates = {
        name: rga_ci(
            y,
            values,
            method=method,
            level=level,
            n_resamples=n_resamples,
            random_state=random_state,
        )
        for name, values in scores.items()
    }
    ordered = sorted(estimates.items(), key=lambda kv: kv[1].estimate, reverse=True)
    if baseline is None:
        baseline = ordered[-1][0]
    if baseline not in scores:
        raise InputError(f"baseline {baseline!r} is not among the models.")

    comparisons: list[RGAComparison] = [
        rga_compare(
            y,
            scores[name],
            scores[baseline],
            method=method,
            level=level,
            n_resamples=n_resamples,
            random_state=random_state,
        )
        for name, _ in ordered
        if name != baseline
    ]
    return {
        "baseline": baseline,
        "ranking": [
            {"model": name, **estimate.to_dict()} for name, estimate in ordered
        ],
        "vs_baseline": [
            {"model": name, **comparison.to_dict()}
            for (name, _), comparison in zip(
                [pair for pair in ordered if pair[0] != baseline], comparisons
            )
        ],
        "level": level,
        "method": method,
    }


def rga_by_segment(
    y: Any,
    yhat: Any,
    segments: Any,
    *,
    min_size: int = 50,
    level: float = 0.95,
    method: str = "jackknife",
    n_resamples: int = 2000,
    random_state: Any = None,
) -> list[dict[str, Any]]:
    """RGA within each segment (portfolio, vintage, region, rating band).

    Same machinery as :func:`rgbox.fairness.rga_parity` but framed as
    performance monitoring rather than fairness. Segments smaller than
    ``min_size`` are reported with a flag rather than silently dropped or
    silently trusted, and a segment too small to estimate at all (fewer than 3
    rows) is reported with a note rather than raising - a two-obligor vintage
    or rating band is a normal thing to find in a portfolio cut, and it must
    not take the whole report down with it.

    A *missing* segment label is a different matter and is rejected: it is not
    a small segment, it is a row that belongs to no segment, and letting it
    through made the table's row counts disagree with the sample.
    """
    y_arr, yhat_arr = as_score_pair(y, yhat)
    labels = as_group_labels(segments, "segments", y_arr.size)
    rows = []
    for label, mask in distinct_levels(labels):
        size = int(mask.sum())
        record: dict[str, Any] = {
            "segment": label,
            "n": size,
            "reliable": size >= min_size,
        }
        try:
            estimate = rga_ci(
                y_arr[mask],
                yhat_arr[mask],
                method=method,
                level=level,
                n_resamples=n_resamples,
                random_state=random_state,
            )
            record.update(estimate.to_dict())
        except (UndefinedMetricError, InsufficientDataError, InputError) as exc:
            # InsufficientDataError is a *sibling* of InputError under
            # RGBoxError, not a subclass, so it has to be named: a segment with
            # fewer than 3 rows used to escape this handler and abort the call.
            record.update({"rga": None, "note": str(exc).split(".")[0]})
        rows.append(record)
    rows.sort(key=lambda row: (row.get("rga") is None, row.get("rga") or 0.0))
    return rows


def contamination_curve(
    y: Any,
    yhat: Any,
    *,
    fractions: Sequence[float] = (0.0, 0.01, 0.02, 0.05, 0.10),
    magnitude: float = 50.0,
    n_repeats: int = 20,
    random_state: Any = None,
) -> dict[str, Any]:
    """Quantify the claim that RGA is more outlier-robust than RMSE.

    The rank graduation papers assert robustness to outlying observations but
    do not measure it. This replaces a fraction of the *targets* with values
    ``magnitude`` standard deviations away and tracks how far RGA and RMSE move
    from their clean values, averaged over ``n_repeats`` draws. RGA reads ranks
    and so shifts by a bounded amount; RMSE reads values and diverges.

    Returns the two sensitivity curves, each expressed as a relative change
    from the uncontaminated sample.
    """
    y_arr, yhat_arr = as_score_pair(y, yhat, min_size=10)
    rng = np.random.default_rng(random_state)
    spread = float(np.std(y_arr))
    clean_rga = rga(y_arr, yhat_arr)
    clean_rmse = float(np.sqrt(np.mean((y_arr - yhat_arr) ** 2)))

    rows = []
    for fraction in fractions:
        count = round(fraction * y_arr.size)
        rga_draws, rmse_draws = [], []
        for _ in range(n_repeats if count else 1):
            contaminated = y_arr.copy()
            if count:
                picked = rng.choice(y_arr.size, size=count, replace=False)
                sign = rng.choice([-1.0, 1.0], size=count)
                contaminated[picked] = contaminated[picked] + sign * magnitude * spread
            try:
                rga_draws.append(rga(contaminated, yhat_arr))
            except UndefinedMetricError:
                continue
            rmse_draws.append(float(np.sqrt(np.mean((contaminated - yhat_arr) ** 2))))
        rows.append(
            {
                "fraction": fraction,
                "n_contaminated": count,
                "rga": float(np.mean(rga_draws)),
                "rga_relative_change": abs(float(np.mean(rga_draws)) - clean_rga)
                / max(abs(clean_rga), 1e-12),
                "rmse": float(np.mean(rmse_draws)),
                "rmse_relative_change": abs(float(np.mean(rmse_draws)) - clean_rmse)
                / max(abs(clean_rmse), 1e-12),
            }
        )
    return {
        "clean_rga": clean_rga,
        "clean_rmse": clean_rmse,
        "magnitude_in_sd": magnitude,
        "n_repeats": n_repeats,
        "curve": rows,
    }
