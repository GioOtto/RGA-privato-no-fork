"""Uncertainty for RGA: standard errors, confidence intervals, tests.

Why this module exists
----------------------
Neither the published Rank Graduation literature nor any released
implementation reports uncertainty for RGA. Papers quote point estimates, and
the simulation studies shipped with the original repository report only the
Monte-Carlo standard deviation *across 1000 synthetic datasets* - a quantity
nobody has access to when validating one model on one hold-out sample. A model
validation function cannot act on "challenger RGA 0.812 vs champion 0.804"
without knowing whether that gap is noise, and a fairness gap between segments
of 300 and 12000 obligors is almost entirely noise.

RGA is a smooth functional of the empirical distribution (a ratio of two
Gini covariances), so all the standard machinery applies. Three estimators are
provided, and they agree:

``"jackknife"`` (default)
    Exact delete-one values in ``O(n log n)`` total rather than the obvious
    ``O(n^2)``. Removing observation ``k`` shifts the average rank of every
    observation scoring above ``k`` down by exactly 1 and of every observation
    tied with ``k`` down by exactly 1/2, so the whole delete-one family follows
    from one sort plus suffix sums. Agrees with the naive implementation to
    ~1e-16. Also yields pseudo-values, hence a paired comparison test, and the
    acceleration constant for BCa intervals - for free.

``"influence"``
    Plug-in influence function, one pass. With :math:`G` the CDF of the score,
    :math:`F` the CDF of the target, :math:`N = \\operatorname{cov}(y, G(\\hat y))`
    and :math:`D = \\operatorname{cov}(y, F(y))`,

    .. math::
        \\mathrm{IF}_N(y_0, z_0) &= y_0 G(z_0) + E[Y\\,\\mathbb{1}\\{Z \\ge z_0\\}]
                                    - 2E[YG(Z)] - (y_0 - \\mu)/2 \\\\
        \\mathrm{IF}_\\theta &= \\frac{\\mathrm{IF}_N}{2D}
                                - \\frac{N\\,\\mathrm{IF}_D}{2D^2}

    and ``SE = sd(IF) / sqrt(n)``. On binary targets this reproduces DeLong's
    AUC standard error to within 0.5% from n = 200 upwards, which is the
    strongest available external check.

``"bootstrap"``
    Resampling, but *exact and vectorised*. A bootstrap replicate is a
    multinomial reweighting of the original sample, and average ranks under
    integer weights have a closed form, so replicates never materialise a
    resampled array and the sort order is computed once for all of them.
    Supports percentile, basic and BCa intervals.

Coverage of the resulting 95% intervals is 0.92-0.95 across Gaussian, binary,
heavy-tailed and heavily-tied designs (``tests/test_inference.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ._ranks import (
    average_ranks,
    suffix_sums_strictly_greater,
    tie_group_ids,
    tie_group_sums,
)
from ._validation import as_score_pair, check_level
from .core import rga
from .exceptions import InputError, InsufficientDataError, UndefinedMetricError

__all__ = [
    "RGAEstimate",
    "RGAComparison",
    "rga_ci",
    "rga_compare",
    "rga_test",
    "jackknife_values",
    "influence_values",
    "bootstrap_values",
]

Method = Literal["jackknife", "influence", "bootstrap"]
IntervalKind = Literal["normal", "percentile", "basic", "bca"]


def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF; avoids a hard SciPy dependency."""
    # Acklam's rational approximation, refined by one Halley step.
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    p_low, p_high = 0.02425, 1 - 0.02425
    if p <= 0.0 or p >= 1.0:
        raise InputError(f"quantile argument must lie in (0, 1); got {p!r}.")
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    e = 0.5 * math.erfc(-x / math.sqrt(2)) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


def _normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))


# --------------------------------------------------------------------------
# resampling engines
# --------------------------------------------------------------------------


def _leave_one_out_sums(values: np.ndarray, payload: np.ndarray) -> np.ndarray:
    """``S(k) = sum_{i != k} payload_i * R^{-k}(values_i)`` for every ``k``.

    Deleting ``k`` lowers the average rank of every strictly larger value by 1
    and of every tied value by 1/2, so

    ``S(k) = Psi - payload_k R(k) - T(k) - (U(k) - payload_k) / 2``

    with ``Psi`` the full inner product, ``T`` the payload mass strictly above
    and ``U`` the payload mass tied. All three are suffix/segment sums over one
    sort: ``O(n log n)`` for the whole family.
    """
    ranks = average_ranks(values)
    total = float(payload @ ranks)
    above = suffix_sums_strictly_greater(values, payload)
    tied = tie_group_sums(values, payload)
    return total - payload * ranks - above - 0.5 * (tied - payload)


def jackknife_values(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """All ``n`` delete-one RGA values, exactly, in ``O(n log n)``."""
    n = y.size
    if n < 3:
        raise InsufficientDataError(
            f"jackknife needs at least 3 observations; got {n}."
        )
    mean_without_k = (y.sum() - y) / (n - 1)
    # Average ranks over n-1 elements always sum to (n-1)n/2.
    rank_total = (n - 1) * n / 2.0
    numerator = _leave_one_out_sums(yhat, y) - mean_without_k * rank_total
    denominator = _leave_one_out_sums(y, y) - mean_without_k * rank_total
    with np.errstate(divide="ignore", invalid="ignore"):
        values = 0.5 + numerator / (2.0 * denominator)
    return values


def influence_values(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """Empirical influence function of RGA at each observation.

    Uses the *mid-distribution* function ``F(v) = P(V < v) + P(V = v) / 2``,
    empirically ``(R(v) - 1/2) / n``, rather than the plain ``R(v) / n``. The
    two differ by ``1 / (2n)``, which is asymptotically irrelevant but not
    harmless: with the plain version a perfectly uninformative (constant) score
    picks up a spurious ``O(1/n)`` residual variance instead of the exactly
    zero it must have, and the finite-sample agreement with DeLong is slightly
    worse. The tail term is centred the same way.
    """
    n = y.size
    cdf_y = (average_ranks(y) - 0.5) / n
    cdf_z = (average_ranks(yhat) - 0.5) / n
    mean_y = float(y.mean())

    # E[Y 1{V > v_i}] + E[Y 1{V = v_i}] / 2, matching the mid-CDF convention.
    def upper_tail_mean(values: np.ndarray) -> np.ndarray:
        return (
            suffix_sums_strictly_greater(values, y) + 0.5 * tie_group_sums(values, y)
        ) / n

    psi = float(np.mean(y * cdf_z))
    phi = float(np.mean(y * cdf_y))
    big_n = float(np.mean((y - mean_y) * cdf_z))
    big_d = float(np.mean((y - mean_y) * cdf_y))
    if abs(big_d) < 1e-300:
        raise UndefinedMetricError("degenerate target: RGA denominator is zero.")

    if_n = y * cdf_z + upper_tail_mean(yhat) - 2 * psi - (y - mean_y) / 2
    if_d = y * cdf_y + upper_tail_mean(y) - 2 * phi - (y - mean_y) / 2
    return if_n / (2 * big_d) - big_n * if_d / (2 * big_d * big_d)


def _multinomial_rga(
    y: np.ndarray,
    yhat: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Weighted RGA for a whole ``(B, n)`` block of bootstrap weights at once.

    The sort order of ``y`` and of ``yhat`` does not depend on the replicate,
    so it is computed once; only the tie-group weight sums change, and those
    are a single ``reduceat`` per block.
    """
    totals = counts.sum(axis=1, keepdims=True)
    means = (counts @ y)[:, None] / totals

    def weighted_term(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        gid = tie_group_ids(values[order])
        starts = np.flatnonzero(np.concatenate(([True], np.diff(gid) != 0)))
        sorted_counts = counts[:, order]  # (B, n)
        group_weight = np.add.reduceat(sorted_counts, starts, axis=1)
        before = np.cumsum(group_weight, axis=1) - group_weight
        group_rank = before + (group_weight + 1.0) / 2.0
        ranks = group_rank[:, gid]  # (B, n)
        centred = sorted_counts * (y[order][None, :] - means)
        return np.einsum("bn,bn->b", centred, ranks)

    numerator = weighted_term(yhat)
    denominator = weighted_term(y)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 0.5 + numerator / (2.0 * denominator)


def bootstrap_values(
    y: np.ndarray,
    yhat: np.ndarray,
    *,
    n_resamples: int = 2000,
    random_state: Any = None,
    block_size: int | None = None,
    paired_with: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Bootstrap replicates of RGA (and optionally of a second, paired score).

    ``paired_with`` reuses the same resampling weights for a second score
    vector, which is what makes a champion/challenger comparison correct: the
    two models are evaluated on the same bootstrap sample, so their strong
    positive correlation is preserved instead of being thrown away.
    """
    rng = np.random.default_rng(random_state)
    n = y.size
    if block_size is None:
        # Each replicate block materialises a handful of (block, n) float64
        # arrays; cap them at roughly 32 MB apiece so large samples do not
        # blow up memory.
        block_size = int(min(n_resamples, max(1, 4_000_000 // max(n, 1))))
    probabilities = np.full(n, 1.0 / n)
    primary = np.empty(n_resamples, dtype=np.float64)
    secondary = (
        np.empty(n_resamples, dtype=np.float64) if paired_with is not None else None
    )

    done = 0
    while done < n_resamples:
        take = min(block_size, n_resamples - done)
        counts = rng.multinomial(n, probabilities, size=take).astype(np.float64)
        primary[done : done + take] = _multinomial_rga(y, yhat, counts)
        if secondary is not None:
            secondary[done : done + take] = _multinomial_rga(y, paired_with, counts)
        done += take

    if secondary is not None:
        return primary, secondary
    return primary


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RGAEstimate:
    """A point estimate of RGA with its uncertainty."""

    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float
    level: float
    method: str
    interval: str
    n: int
    n_resamples: int | None = None
    bias: float | None = None
    replicates: np.ndarray | None = field(default=None, repr=False)

    @property
    def gini(self) -> float:
        """``2 * RGA - 1``, on the credit-risk Gini/Accuracy-Ratio scale."""
        return 2.0 * self.estimate - 1.0

    @property
    def gini_ci(self) -> tuple[float, float]:
        return (2.0 * self.ci_low - 1.0, 2.0 * self.ci_high - 1.0)

    def __float__(self) -> float:
        return float(self.estimate)

    def __str__(self) -> str:
        pct = round(self.level * 100)
        return (
            f"RGA = {self.estimate:.4f} "
            f"({pct}% CI {self.ci_low:.4f}-{self.ci_high:.4f}, "
            f"SE {self.standard_error:.4f}, {self.method}, n={self.n})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rga": self.estimate,
            "gini": self.gini,
            "standard_error": self.standard_error,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "level": self.level,
            "method": self.method,
            "interval": self.interval,
            "n": self.n,
            "n_resamples": self.n_resamples,
            "bias": self.bias,
        }


@dataclass(frozen=True)
class RGAComparison:
    """Paired comparison of two scores on the same sample."""

    rga_a: float
    rga_b: float
    difference: float
    standard_error: float
    ci_low: float
    ci_high: float
    level: float
    p_value: float
    statistic: float
    method: str
    n: int
    n_resamples: int | None = None

    @property
    def significant(self) -> bool:
        return self.p_value < (1.0 - self.level)

    def __str__(self) -> str:
        pct = round(self.level * 100)
        verdict = "significant" if self.significant else "not significant"
        return (
            f"RGA(A) = {self.rga_a:.4f}, RGA(B) = {self.rga_b:.4f}, "
            f"difference = {self.difference:+.4f} "
            f"({pct}% CI {self.ci_low:+.4f}..{self.ci_high:+.4f}), "
            f"p = {self.p_value:.4g} [{verdict}, {self.method}, n={self.n}]"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rga_a": self.rga_a,
            "rga_b": self.rga_b,
            "difference": self.difference,
            "standard_error": self.standard_error,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "level": self.level,
            "p_value": self.p_value,
            "statistic": self.statistic,
            "significant": self.significant,
            "method": self.method,
            "n": self.n,
        }


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


def _interval_from_replicates(
    replicates: np.ndarray,
    point: float,
    level: float,
    kind: IntervalKind,
    jackknife: np.ndarray | None,
) -> tuple[float, float]:
    alpha = 1.0 - level
    if kind == "percentile":
        return tuple(np.quantile(replicates, [alpha / 2, 1 - alpha / 2]))
    if kind == "basic":
        low, high = np.quantile(replicates, [alpha / 2, 1 - alpha / 2])
        return (2 * point - high, 2 * point - low)
    if kind == "bca":
        share_below = float(np.mean(replicates < point))
        share_below = min(max(share_below, 1e-6), 1 - 1e-6)
        bias_z = _normal_quantile(share_below)
        if jackknife is None:
            acceleration = 0.0
        else:
            deviations = jackknife.mean() - jackknife
            denominator = 6.0 * (float(np.sum(deviations**2)) ** 1.5)
            acceleration = (
                float(np.sum(deviations**3)) / denominator if denominator > 0 else 0.0
            )
        out = []
        for tail in (alpha / 2, 1 - alpha / 2):
            z = _normal_quantile(tail)
            adjusted = bias_z + (bias_z + z) / (1 - acceleration * (bias_z + z))
            out.append(float(np.quantile(replicates, _normal_cdf(adjusted))))
        return tuple(out)
    raise InputError(f"unknown interval kind {kind!r} for bootstrap.")


def rga_ci(
    y: Any,
    yhat: Any,
    *,
    method: Method = "jackknife",
    level: float = 0.95,
    interval: IntervalKind | None = None,
    n_resamples: int = 2000,
    random_state: Any = None,
) -> RGAEstimate:
    """RGA with a standard error and a confidence interval.

    Parameters
    ----------
    method :
        ``"jackknife"`` (default, exact, deterministic, ``O(n log n)``),
        ``"influence"`` (single pass, matches DeLong on binary targets) or
        ``"bootstrap"`` (exact multinomial resampling).
    interval :
        ``"normal"`` for all methods; additionally ``"percentile"``,
        ``"basic"`` or ``"bca"`` for the bootstrap. Defaults to ``"normal"``
        for the analytic methods and ``"bca"`` for the bootstrap.
    n_resamples, random_state :
        Bootstrap only. ``random_state`` is passed to
        :func:`numpy.random.default_rng`, so results are reproducible.

    Notes
    -----
    Intervals are on the RGA scale and are *not* clipped to ``[0, 1]``: a
    reported bound of 1.02 is honest information about how little the sample
    constrains the estimate. Clip at the reporting layer if a policy requires
    it.
    """
    y_arr, yhat_arr = as_score_pair(y, yhat, min_size=3)
    level = check_level(level)
    point = rga(y_arr, yhat_arr)
    n = y_arr.size
    z_crit = _normal_quantile(1.0 - (1.0 - level) / 2.0)

    if method == "jackknife":
        values = jackknife_values(y_arr, yhat_arr)
        if not np.all(np.isfinite(values)):
            raise UndefinedMetricError(
                "some delete-one samples make RGA undefined (the target becomes "
                "constant once one observation is removed). Use "
                "method='influence' or collect more data."
            )
        mean = float(values.mean())
        standard_error = math.sqrt((n - 1) / n * float(np.sum((values - mean) ** 2)))
        bias = (n - 1) * (mean - point)
        kind = interval or "normal"
        if kind != "normal":
            raise InputError(
                f"interval={kind!r} requires method='bootstrap'; the jackknife "
                "supports 'normal'."
            )
        return RGAEstimate(
            estimate=point,
            standard_error=standard_error,
            ci_low=point - z_crit * standard_error,
            ci_high=point + z_crit * standard_error,
            level=level,
            method="jackknife",
            interval="normal",
            n=n,
            bias=bias,
        )

    if method == "influence":
        values = influence_values(y_arr, yhat_arr)
        standard_error = float(np.std(values, ddof=1)) / math.sqrt(n)
        kind = interval or "normal"
        if kind != "normal":
            raise InputError(
                f"interval={kind!r} requires method='bootstrap'; the influence "
                "function supports 'normal'."
            )
        return RGAEstimate(
            estimate=point,
            standard_error=standard_error,
            ci_low=point - z_crit * standard_error,
            ci_high=point + z_crit * standard_error,
            level=level,
            method="influence",
            interval="normal",
            n=n,
        )

    if method == "bootstrap":
        replicates = bootstrap_values(
            y_arr, yhat_arr, n_resamples=n_resamples, random_state=random_state
        )
        finite = np.isfinite(replicates)
        if finite.sum() < 0.5 * n_resamples:
            raise UndefinedMetricError(
                f"only {int(finite.sum())} of {n_resamples} bootstrap replicates "
                "produced a defined RGA; the sample is too degenerate to "
                "resample (very small n, or a near-constant target)."
            )
        replicates = replicates[finite]
        kind = interval or "bca"
        standard_error = float(np.std(replicates, ddof=1))
        if kind == "normal":
            low, high = point - z_crit * standard_error, point + z_crit * standard_error
        else:
            jack = jackknife_values(y_arr, yhat_arr) if kind == "bca" else None
            if jack is not None and not np.all(np.isfinite(jack)):
                jack = None
            low, high = _interval_from_replicates(replicates, point, level, kind, jack)
        return RGAEstimate(
            estimate=point,
            standard_error=standard_error,
            ci_low=float(low),
            ci_high=float(high),
            level=level,
            method="bootstrap",
            interval=kind,
            n=n,
            n_resamples=int(finite.sum()),
            bias=float(replicates.mean()) - point,
            replicates=replicates,
        )

    raise InputError(
        f"unknown method {method!r}; expected 'jackknife', 'influence' or 'bootstrap'."
    )


def rga_compare(
    y: Any,
    yhat_a: Any,
    yhat_b: Any,
    *,
    method: Method = "jackknife",
    level: float = 0.95,
    n_resamples: int = 2000,
    random_state: Any = None,
) -> RGAComparison:
    """Test whether two scores rank the same target equally well.

    This is the champion/challenger question, and it is *paired*: both scores
    are computed on the same observations, so their RGA estimates are strongly
    positively correlated and comparing two independent confidence intervals
    would be badly conservative. All three methods here difference the
    per-observation contributions (pseudo-values, influence values, or
    replicates under shared resampling weights) before taking a variance.

    On a binary target this is the rank-graduation analogue of DeLong's test
    for correlated AUCs.

    Returns
    -------
    RGAComparison
        ``difference`` is ``RGA(A) - RGA(B)``; positive favours A. The p-value
        is two-sided for ``H0: RGA(A) == RGA(B)``.
    """
    y_arr, a_arr = as_score_pair(y, yhat_a, yhat_name="yhat_a", min_size=3)
    _, b_arr = as_score_pair(y, yhat_b, yhat_name="yhat_b", min_size=3)
    level = check_level(level)
    n = y_arr.size
    point_a, point_b = rga(y_arr, a_arr), rga(y_arr, b_arr)
    difference = point_a - point_b
    z_crit = _normal_quantile(1.0 - (1.0 - level) / 2.0)
    used_resamples: int | None = None

    if method == "jackknife":
        # Pseudo-values linearise the estimator; their paired difference has
        # the variance of the difference.
        pseudo_a = n * point_a - (n - 1) * jackknife_values(y_arr, a_arr)
        pseudo_b = n * point_b - (n - 1) * jackknife_values(y_arr, b_arr)
        delta = pseudo_a - pseudo_b
        if not np.all(np.isfinite(delta)):
            raise UndefinedMetricError(
                "some delete-one samples make RGA undefined; use method='influence'."
            )
        standard_error = float(np.std(delta, ddof=1)) / math.sqrt(n)
    elif method == "influence":
        delta = influence_values(y_arr, a_arr) - influence_values(y_arr, b_arr)
        standard_error = float(np.std(delta, ddof=1)) / math.sqrt(n)
    elif method == "bootstrap":
        rep_a, rep_b = bootstrap_values(
            y_arr,
            a_arr,
            n_resamples=n_resamples,
            random_state=random_state,
            paired_with=b_arr,
        )
        diffs = rep_a - rep_b
        diffs = diffs[np.isfinite(diffs)]
        if diffs.size < 0.5 * n_resamples:
            raise UndefinedMetricError(
                "too many bootstrap replicates were undefined to compare."
            )
        standard_error = float(np.std(diffs, ddof=1))
        used_resamples = int(diffs.size)
    else:
        raise InputError(
            f"unknown method {method!r}; expected 'jackknife', 'influence' or "
            "'bootstrap'."
        )

    if standard_error <= 0:
        statistic = 0.0 if difference == 0 else math.inf * math.copysign(1, difference)
        p_value = 1.0 if difference == 0 else 0.0
    else:
        statistic = difference / standard_error
        p_value = 2.0 * (1.0 - _normal_cdf(abs(statistic)))

    return RGAComparison(
        rga_a=point_a,
        rga_b=point_b,
        difference=difference,
        standard_error=standard_error,
        ci_low=difference - z_crit * standard_error,
        ci_high=difference + z_crit * standard_error,
        level=level,
        p_value=p_value,
        statistic=statistic,
        method=method,
        n=n,
        n_resamples=used_resamples,
    )


def rga_test(
    y: Any,
    yhat: Any,
    *,
    alternative: Literal["greater", "two-sided", "less"] = "greater",
    n_permutations: int | None = None,
    random_state: Any = None,
) -> dict[str, Any]:
    """Test ``H0: the score carries no ranking information`` (``RGA == 0.5``).

    Under H0 the score's ranks are exchangeable with respect to ``y``, so the
    permutation distribution of the numerator has *exactly* known moments:
    mean 0 and variance ``sum((y - ybar)^2) * sum((R - Rbar)^2) / (n - 1)``.
    That gives a closed-form permutation test with no simulation at all. Pass
    ``n_permutations`` to get the Monte-Carlo version instead, which is worth
    doing for small ``n`` where the normal approximation to the permutation
    distribution is loose.
    """
    y_arr, yhat_arr = as_score_pair(y, yhat, min_size=3)
    n = y_arr.size
    centred = y_arr - y_arr.mean()
    denominator = float(centred @ average_ranks(y_arr))
    if abs(denominator) < 1e-300:
        raise UndefinedMetricError("constant target: RGA is undefined.")
    ranks = average_ranks(yhat_arr)
    point = 0.5 + float(centred @ ranks) / (2.0 * denominator)

    if n_permutations is None:
        rank_ss = float(np.sum((ranks - ranks.mean()) ** 2))
        null_sd_numerator = math.sqrt(float(np.sum(centred**2)) * rank_ss / (n - 1))
        null_sd = null_sd_numerator / (2.0 * abs(denominator))
        statistic = (point - 0.5) / null_sd if null_sd > 0 else 0.0
        if alternative == "greater":
            p_value = 1.0 - _normal_cdf(statistic)
        elif alternative == "less":
            p_value = _normal_cdf(statistic)
        else:
            p_value = 2.0 * (1.0 - _normal_cdf(abs(statistic)))
        return {
            "rga": point,
            "null_value": 0.5,
            "null_se": null_sd,
            "statistic": statistic,
            "p_value": p_value,
            "alternative": alternative,
            "method": "permutation (exact moments, normal approximation)",
            "n": n,
        }

    rng = np.random.default_rng(random_state)
    draws = np.empty(n_permutations)
    for i in range(n_permutations):
        draws[i] = 0.5 + float(centred @ rng.permutation(ranks)) / (2.0 * denominator)
    if alternative == "greater":
        p_value = (1 + np.sum(draws >= point)) / (n_permutations + 1)
    elif alternative == "less":
        p_value = (1 + np.sum(draws <= point)) / (n_permutations + 1)
    else:
        p_value = (1 + np.sum(np.abs(draws - 0.5) >= abs(point - 0.5))) / (
            n_permutations + 1
        )
    return {
        "rga": point,
        "null_value": 0.5,
        "null_se": float(np.std(draws, ddof=1)),
        "statistic": (point - 0.5) / float(np.std(draws, ddof=1)),
        "p_value": float(p_value),
        "alternative": alternative,
        "method": f"permutation (Monte Carlo, {n_permutations} draws)",
        "n": n,
    }
