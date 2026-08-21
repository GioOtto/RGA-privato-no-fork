"""Rank Graduation Robustness: does the ranking survive a shock to an input?

``RGR = RGA(yhat, yhat_perturbed)``: 1 means the perturbation left the ordering
of predictions untouched, 0.5 means it destroyed it.

The scale runs to 0, not to 0.5
-------------------------------
RGR is an RGA, so it lives in ``[0, 1]``, and **0.5 is the middle of the scale,
not the bottom**. Read it as:

=========  ==========================================================
RGR        meaning
=========  ==========================================================
1          the perturbation did not change the ordering at all
0.5        the perturbation destroyed the ordering (no information)
< 0.5      the perturbation *inverted* the ordering
0          the perturbation reversed it exactly
=========  ==========================================================

Values below 0.5 are not an anomaly, and ``"tailswap"`` reaches them by
construction: swapping the lowest ``m`` fraction with the highest ``m``
fraction at ``m = 0.5`` exchanges *every* value with its mirror, which is a
verbatim reversal of the column. Measured on a 10-row column, the rank
correlation with the original is ``+0.02`` at ``m = 0.10``, ``-0.58`` at
``0.25`` and ``-1.00`` at ``0.50``. So for a model monotone in the perturbed
feature, the top of the default grid gives ``RGR = 0`` and drags AURGR well
below 0.5 - measured 0.217 for a score equal to the perturbed column. That is
the measure behaving correctly on a perturbation that runs from "mild tail
shock" to "sign flip"; it is not a floor violation.

The hyperparameter problem
--------------------------
A single RGR number is only interpretable together with the perturbation that
produced it, and the strength of that perturbation is a free parameter. On one
random forest and one feature, upstream's default sweep gives:

===================  =====
``magnitude``        RGR
===================  =====
0.01                 0.966
0.05 *(default)*     0.860
0.10                 0.781
0.20                 0.652
0.50                 0.568
===================  =====

Nothing justifies 0.05 over 0.10, and two teams using different defaults cannot
compare results. :func:`rgr_curve` therefore evaluates RGR over a grid and
reports **AURGR**, the normalised area under the robustness curve - a single
number with no free parameter, on the same ``[0, 1]`` scale as RGR itself where
higher is more robust. Report AURGR; use a point RGR only when a specific
stress scenario is mandated.

Perturbation kinds
------------------
``"tailswap"``
    The upstream scheme: pair the lowest and highest ``magnitude`` fraction of
    values and exchange them. Deterministic, distribution-preserving, and a
    genuine tail shock - up to ``magnitude = 0.5``, where it becomes an exact
    reversal of the column (see above). It is also *meaningless on discrete
    columns* - on a 0/1 indicator it swaps some zeros with some ones chosen
    arbitrarily among ties and leaves the mean unchanged, and on a string
    column upstream sorted lexicographically and produced a permutation with no
    interpretation at all. Non-numeric columns are now rejected instead of
    silently mishandled.

``"gaussian"``
    ``x + N(0, (magnitude * sd(x))^2)`` - the scheme used in the more recent
    rank-graduation literature. Stochastic, so pass ``n_repeats`` to average
    over draws and get an interval.

``"shuffle"``
    Permute the column: the strongest possible shock that keeps the marginal
    distribution. Useful as the "total loss of this input" reference point.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ._validation import (
    as_1d_float,
    check_count,
    check_finite_scalar,
    check_level,
)
from .core import rga
from .exceptions import InputError, UndefinedMetricError
from .inference import (
    RGAEstimate,
    _bootstrap_values,
    _influence_values,
    _jackknife_values,
    _normal_quantile,
)
from .predictors import as_score_function, resolve_columns

__all__ = [
    "RGRResult",
    "RGRCurve",
    "perturb",
    "rgr",
    "rgr_curve",
]

PerturbationKind = Literal["tailswap", "gaussian", "shuffle"]


def _require_numeric(column: Any, name: Any, kind: str) -> np.ndarray:
    dtype = getattr(column, "dtype", None)
    if getattr(dtype, "kind", None) not in ("i", "u", "f", "b"):
        raise InputError(
            f"perturbation kind {kind!r} needs a numeric column, but "
            f"{name!r} has dtype {dtype!r}. Ordering a categorical or string "
            "column by its labels and swapping its 'tails' produces a "
            "permutation with no meaning - encode the column numerically, or "
            "use kind='shuffle' to model a total loss of that input."
        )
    return np.asarray(column, dtype=np.float64)


def perturb(
    data: Any,
    variable: Any,
    magnitude: float = 0.05,
    *,
    kind: PerturbationKind = "tailswap",
    random_state: Any = None,
) -> Any:
    """Return a copy of ``data`` with ``variable`` perturbed.

    ``magnitude`` means the tail fraction for ``"tailswap"`` (in ``[0, 0.5]``)
    and the noise-to-standard-deviation ratio for ``"gaussian"`` (any positive
    value). It is ignored by ``"shuffle"``.
    """
    if kind not in ("tailswap", "gaussian", "shuffle"):
        raise InputError(
            f"unknown perturbation kind {kind!r}; expected 'tailswap', "
            "'gaussian' or 'shuffle'."
        )
    magnitude = check_finite_scalar(magnitude, "magnitude")
    if kind == "tailswap" and not 0.0 <= magnitude <= 0.5:
        raise InputError(
            f"for kind='tailswap', magnitude must lie in [0, 0.5]; got {magnitude!r}."
        )
    if kind == "gaussian" and magnitude < 0:
        raise InputError("for kind='gaussian', magnitude must be non-negative.")

    out = data.copy()
    column = out[variable]

    if kind == "shuffle":
        rng = np.random.default_rng(random_state)
        values = np.asarray(column)
        out[variable] = values[rng.permutation(values.size)]
        return out

    values = _require_numeric(column, variable, kind)

    if kind == "gaussian":
        rng = np.random.default_rng(random_state)
        scale = float(np.std(values))
        if scale == 0.0:
            return out  # a constant column cannot be meaningfully jittered
        out[variable] = values + rng.normal(0.0, magnitude * scale, values.size)
        return out

    if kind == "tailswap":
        n = values.size
        order = np.argsort(values, kind="stable")
        low_count = int(np.ceil(magnitude * n))
        high_start = int(np.ceil((1.0 - magnitude) * n))
        n_swap = min(low_count, n - high_start)
        if n_swap <= 0:
            return out
        lower = order[:n_swap]
        upper = order[-n_swap:][::-1]
        swapped = values.copy()
        swapped[lower] = values[upper]
        swapped[upper] = values[lower]
        out[variable] = swapped
        return out

    raise InputError(
        f"unknown perturbation kind {kind!r}; expected 'tailswap', 'gaussian' "
        "or 'shuffle'."
    )


@dataclass(frozen=True)
class RGRResult:
    """RGR for one predictor or group at one perturbation magnitude."""

    variables: tuple[Any, ...]
    rgr: float
    magnitude: float
    kind: str
    n: int
    n_repeats: int = 1
    spread: float | None = None
    estimate: RGAEstimate | None = None

    @property
    def label(self) -> str:
        return (
            self.variables[0]
            if len(self.variables) == 1
            else "{" + ", ".join(str(v) for v in self.variables) + "}"
        )

    def to_dict(self) -> dict[str, Any]:
        record = {
            "variables": list(self.variables),
            "label": self.label,
            "rgr": self.rgr,
            "magnitude": self.magnitude,
            "kind": self.kind,
            "n": self.n,
            "n_repeats": self.n_repeats,
            "perturbation_spread": self.spread,
        }
        if self.estimate is not None:
            record["ci_low"] = self.estimate.ci_low
            record["ci_high"] = self.estimate.ci_high
            record["standard_error"] = self.estimate.standard_error
        return record


@dataclass(frozen=True)
class RGRCurve:
    """RGR as a function of perturbation magnitude, plus its summary area."""

    variables: tuple[Any, ...]
    magnitudes: np.ndarray
    values: np.ndarray
    aurgr: float
    kind: str
    n: int
    per_magnitude: list[RGRResult] = field(default_factory=list, repr=False)

    @property
    def label(self) -> str:
        return (
            self.variables[0]
            if len(self.variables) == 1
            else "{" + ", ".join(str(v) for v in self.variables) + "}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "variables": list(self.variables),
            "label": self.label,
            "magnitudes": self.magnitudes.tolist(),
            "rgr": self.values.tolist(),
            "aurgr": self.aurgr,
            "kind": self.kind,
            "n": self.n,
        }


def _perturbed_scores(
    score_fn,
    X_test: Any,
    columns: Sequence[Any],
    magnitude: float,
    kind: PerturbationKind,
    rng: Any,
) -> np.ndarray:
    """Perturb every column of one group, then score.

    ``rng`` is a live :class:`numpy.random.Generator`, threaded through rather
    than re-seeded per column: ``perturb`` passes it to ``default_rng``, which
    returns a Generator unchanged, so the stream simply advances. The previous
    scheme derived a per-column seed as ``int(random_state) + offset``, which
    raised ``TypeError`` for the ``Generator`` that ``random_state`` is
    documented to accept.

    ``kind="shuffle"`` on a **group** draws one permutation and applies it to
    every column, exactly as :func:`rgbox.rge` does for ``method="permute"``.
    Advancing the generator per column instead gave each column an independent
    order, which fabricates rows that cannot exist: two independently shuffled
    dummies of one attribute are both 1 at rate ``p_i * p_j`` - measured at 12%
    of rows on a 45/30/25 attribute - so a group RGR was partly a measurement
    of how the model responds to an obligor who lives in two regions at once.
    Explainability had this right and robustness did not, for the same
    encoding and the same reason.

    ``"gaussian"`` and ``"tailswap"`` stay per column. Both are within-column
    shocks whose joint meaning is not defined by a shared row order - a shared
    permutation would not preserve a tail swap's structure - and neither
    fabricates a categorical level that does not exist, because neither moves a
    value out of the column it came from. What they do *not* do is preserve the
    correlation between the group's columns: a jointly perturbed group is
    perturbed one margin at a time. Read a group AURGR under those kinds as
    "the model's sensitivity to independent shocks to each of these inputs",
    not as a shock to their joint distribution.
    """
    if kind == "shuffle" and len(columns) > 1:
        order = np.random.default_rng(rng).permutation(len(X_test))
        perturbed = X_test.copy()
        for column in columns:
            perturbed[column] = np.asarray(perturbed[column])[order]
        return np.asarray(score_fn(perturbed), dtype=np.float64).ravel()

    perturbed = X_test
    for column in columns:
        perturbed = perturb(perturbed, column, magnitude, kind=kind, random_state=rng)
    return np.asarray(score_fn(perturbed), dtype=np.float64).ravel()


def _draw_values(
    baseline: np.ndarray,
    scores: np.ndarray,
    ci_method: str,
    n_resamples: int,
    ci_seed: int,
) -> np.ndarray:
    """The linearised values of one draw's RGA: whatever the SE is a spread of.

    Jackknife delete-one values, influence values, or bootstrap replicates.
    :func:`_pool_across_draws` averages these *across draws* before taking a
    spread, which is what makes the result an interval for the mean.

    The bootstrap branch is handed one fixed ``ci_seed`` for every draw on
    purpose. Bootstrap weights depend only on ``n``, ``n_resamples`` and the
    generator, never on the scores, so a shared seed means every draw is
    reweighted by the *same* multinomial counts and replicate ``b`` of the
    average is the average of replicate ``b`` - the paired-resampling trick
    :func:`rgbox.bootstrap_values` already uses for champion/challenger. Left
    to advance a live ``Generator`` per draw, as passing ``random_state``
    straight through did, the weights differ per draw and averaging replicates
    across them is not a bootstrap distribution of anything.
    """
    if ci_method == "jackknife":
        values = _jackknife_values(baseline, scores)
        if not np.all(np.isfinite(values)):
            raise UndefinedMetricError(
                "some delete-one samples make RGA undefined for a perturbed "
                "draw (the baseline becomes constant once one observation is "
                "removed). Use ci_method='influence'."
            )
        return values
    if ci_method == "influence":
        return _influence_values(baseline, scores)
    if ci_method == "bootstrap":
        return np.asarray(
            _bootstrap_values(
                baseline,
                scores,
                n_resamples=n_resamples,
                random_state=ci_seed,
            ),
            dtype=np.float64,
        )
    raise InputError(
        f"unknown ci_method {ci_method!r}; expected 'jackknife', 'influence' "
        "or 'bootstrap'."
    )


def _pool_across_draws(
    per_draw_values: list[np.ndarray],
    draws: list[float],
    level: float,
    ci_method: str,
    n: int,
) -> RGAEstimate:
    """Interval for the *mean* of ``m`` perturbation draws.

    The estimand and its two error terms
    ------------------------------------
    ``rgr`` reports ``theta_bar = mean_j RGA(yhat, yhat_Pj)``. Its target is
    the perturbation-expected RGR, and it carries two errors: the evaluation
    sample is finite, and only ``m`` perturbations were drawn. This adds

    ``T1 + B / m``

    with ``T1`` the sampling variance of ``theta_bar`` **given the realised
    draws** and ``B`` the spread across draws given the observed sample. ``T1``
    is obtained by linearising the statistic that is actually reported: the
    influence function of an average is the average of the influence functions,
    and likewise for jackknife delete-one values and for bootstrap replicates
    under shared weights. So the per-draw value vectors are averaged
    element-wise *first*, and the spread is taken of that average.

    This sum is **not** the law of total variance, and it is not exact. Under
    the orthogonal decomposition ``g(S, P) = mu + a(S) + b(P) + c(S, P)`` the
    target is

    ``Var(theta_bar) = var_a + (var_b + var_c) / m``,

    while ``T1 -> var_a + var_c / m`` and ``B / m -> (var_b + var_c) / m``, so
    the sum carries ``var_c / m`` twice. It is therefore biased upward by one
    interaction term over ``m``. Separating ``var_b`` from ``var_c`` needs a
    second, independent evaluation sample - or a nested resample over both - so
    it is not done here. Verified on a synthetic orthogonal design with
    ``var_a = 1, var_b = 2, var_c = 3, m = 8``: truth 1.625, ``T1 + B/m``
    2.000, over by exactly ``var_c / m = 0.375``.

    Why the mean of the per-draw variances was wrong
    ------------------------------------------------
    This computed ``W = mean_j SE_j**2`` for the first term. ``SE_j`` is the
    sampling variance of a *single* draw, and by the ANOVA decomposition
    ``g(S, P) = mu + a(S) + b(P) + c(S, P)`` it equals ``Var(a) +
    E_P Var_S(c)`` - the sample effect **plus the whole sample-by-perturbation
    interaction**. Averaging over draws does not remove the interaction from
    the estimate, though it does remove it from the estimand, so ``W`` does not
    fall as ``m`` grows. Measured on a Gaussian design at ``n = 300`` with
    1500-4000 replications per point, against the empirical variance of the
    reported estimate:

    ====  ==========  ==========  ==============
    m     true Var    ``W``       ``var(IF_bar)/n``
    ====  ==========  ==========  ==============
    1     1.08e-04    1.39e-04    1.39e-04
    4     2.47e-05    1.40e-04    6.44e-05
    16    6.72e-06    1.40e-04    4.55e-05
    64    2.17e-06    1.41e-04    4.08e-05
    ====  ==========  ==========  ==============

    The true variance falls ~50x from ``m = 1`` to ``m = 64``; ``W`` is flat to
    three digits. The two agree exactly at ``m = 1``, as they must, so the
    default path does not move.

    What is still not settled
    -------------------------
    The averaged linearisation is the right one for ``theta_bar`` and it does
    shrink with ``m``, but the table above shows it plateauing around 4e-5
    where the empirical sampling floor is nearer 1e-6 - a gap far larger than
    the ``var_c / m`` double count above accounts for.

    So: **in the Gaussian design studied so far these intervals came out
    conservative, by a factor that has not been characterised, and their
    coverage has not been validated.** That is an observation about one design,
    not a proof, and it must not be read as a guarantee: do not treat a
    multi-draw RGR interval as a calibrated confidence interval, and do not
    treat it as an upper bound either. See ``docs/THEORY.md``'s open questions.

    The single-draw interval (``n_repeats=1``, the default) is an ordinary RGA
    interval, whose coverage *is* simulation-checked in
    :mod:`rgbox.inference`, and is not affected by any of this.
    """
    m = len(per_draw_values)
    point = float(np.mean(draws))
    stacked = np.vstack(per_draw_values)
    if ci_method == "bootstrap":
        # A replicate is unusable for the average if it degenerated in any draw.
        usable = np.all(np.isfinite(stacked), axis=0)
        stacked = stacked[:, usable]
        if stacked.shape[1] < 2:
            raise UndefinedMetricError(
                "too few bootstrap replicates were defined across every "
                "perturbation draw to form an interval."
            )
    averaged = stacked.mean(axis=0)

    if ci_method == "jackknife":
        centred = averaged - averaged.mean()
        within = (n - 1) / n * float(np.sum(centred**2))
    elif ci_method == "influence":
        within = float(np.var(averaged, ddof=1)) / n
    else:
        within = float(np.var(averaged, ddof=1))

    between = float(np.var(draws, ddof=1)) if m > 1 else 0.0
    standard_error = math.sqrt(within + between / m)
    z_crit = _normal_quantile(1.0 - (1.0 - level) / 2.0)
    return RGAEstimate(
        estimate=point,
        standard_error=standard_error,
        ci_low=point - z_crit * standard_error,
        ci_high=point + z_crit * standard_error,
        level=level,
        method=(ci_method if m == 1 else f"{ci_method} + {m} draws (total variance)"),
        interval="normal",
        n=n,
        n_resamples=(int(stacked.shape[1]) if ci_method == "bootstrap" else None),
    )


def rgr(
    X_test: Any,
    model: Any,
    variables: Sequence[Any] | Any,
    *,
    yhat: Any = None,
    magnitude: float = 0.05,
    kind: PerturbationKind = "tailswap",
    group: bool = False,
    n_repeats: int = 1,
    ci: bool = False,
    ci_method: str = "jackknife",
    level: float = 0.95,
    pos_label: Any = None,
    greater_is_better: bool = True,
    random_state: Any = None,
    n_resamples: int = 2000,
) -> list[RGRResult]:
    """Rank Graduation Robustness at a single perturbation magnitude.

    ``n_repeats`` averages over independent draws for the stochastic kinds and
    reports their standard deviation in ``spread``; it is ignored (and forced
    to 1) for the deterministic ``"tailswap"``.

    With ``ci=True`` and ``n_repeats > 1`` the interval covers **both** sources
    of noise, by the law of total variance: the within-draw sampling variance
    plus the between-draw variance divided by ``m``. It is an interval for the
    expected RGR over the perturbation distribution, which is what ``rgr``
    reports. Reporting the interval of a single draw next to a mean over ``m``
    of them - as this did before 1.0.1 - stated an uncertainty for a quantity
    that is not the one in ``rgr``; pooling by Rubin's rules - as 1.0.1 did -
    stated one for an estimand with an irreducible between-draw component,
    which a perturbation the caller chooses does not have.

    ``group=True`` with ``kind="shuffle"`` applies **one** permutation to the
    whole group, so a one-hot encoded attribute stays a valid one-hot; the
    other kinds perturb each column independently. See ``_perturbed_scores``.

    Every draw uses the same seed sequence regardless of which variable is
    being perturbed, so RGR values are comparable across variables (common
    random numbers) rather than differing by noise realisation.

    Consider :func:`rgr_curve` instead: it removes the dependence on
    ``magnitude`` entirely.
    """
    columns = resolve_columns(variables, X_test, "variables")
    score_fn = as_score_function(
        model, pos_label=pos_label, greater_is_better=greater_is_better
    )
    baseline = (
        np.asarray(score_fn(X_test), dtype=np.float64).ravel()
        if yhat is None
        else as_1d_float(yhat, "yhat")
    )
    if baseline.size != len(X_test):
        raise InputError(
            f"'yhat' has {baseline.size} entries but X_test has {len(X_test)} rows."
        )
    n_repeats = check_count(n_repeats, "n_repeats")
    if kind == "tailswap":
        n_repeats = 1

    # One seed per repeat, drawn once and reused for every variable, so the
    # draws are independent across repeats but shared across variables. One
    # further seed drives the CI resampling and is *shared across draws* - see
    # _draw_values for why the bootstrap needs identical weights per draw.
    master = np.random.default_rng(random_state)
    seeds = master.integers(0, 2**63 - 1, size=n_repeats)
    ci_seed = int(master.integers(0, 2**63 - 1))
    level = check_level(level)

    groups: list[list[Any]] = [columns] if group else [[c] for c in columns]
    results: list[RGRResult] = []
    for chunk in groups:
        draws: list[float] = []
        per_draw_values: list[np.ndarray] = []
        for repeat in range(n_repeats):
            scores = _perturbed_scores(
                score_fn,
                X_test,
                chunk,
                magnitude,
                kind,
                np.random.default_rng(seeds[repeat]),
            )
            draws.append(rga(baseline, scores))
            if ci:
                per_draw_values.append(
                    _draw_values(baseline, scores, ci_method, n_resamples, ci_seed)
                )
        value = float(np.mean(draws))
        spread = float(np.std(draws, ddof=1)) if n_repeats > 1 else None
        estimate = (
            _pool_across_draws(per_draw_values, draws, level, ci_method, baseline.size)
            if ci
            else None
        )
        results.append(
            RGRResult(
                variables=tuple(chunk),
                rgr=value,
                magnitude=magnitude,
                kind=kind,
                n=baseline.size,
                n_repeats=n_repeats,
                spread=spread,
                estimate=estimate,
            )
        )

    if not group:
        results.sort(key=lambda item: item.rgr)
    return results


def rgr_curve(
    X_test: Any,
    model: Any,
    variables: Sequence[Any] | Any,
    *,
    yhat: Any = None,
    grid: Sequence[float] | None = None,
    kind: PerturbationKind = "tailswap",
    group: bool = False,
    n_repeats: int = 1,
    pos_label: Any = None,
    greater_is_better: bool = True,
    random_state: Any = None,
) -> list[RGRCurve]:
    """RGR over a grid of magnitudes, summarised by AURGR.

    AURGR is the trapezoidal area under ``magnitude -> RGR``, anchored at
    ``RGR(0) = 1`` and divided by the width of the grid, so it lies on the same
    ``[0, 1]`` scale as RGR itself and needs no hyperparameter to interpret. A
    model whose ranking is unaffected by any shock scores 1; one whose ranking
    is destroyed everywhere on the grid scores 0.5.

    **AURGR can fall below 0.5**, and routinely does. The default
    ``"tailswap"`` grid spans the perturbation's whole legal domain, and at
    ``magnitude = 0.5`` a tail swap is an exact reversal of the column, so a
    model monotone in that feature scores ``RGR = 0`` there and AURGR lands
    around 0.2. That is the average over the whole domain, which is what makes
    it free of a hyperparameter; it is not a robustness score that bottomed
    out. Compare AURGR across variables and across model versions, not against
    an absolute 0.5 threshold. See the module docstring for the scale.

    Default grids: ``0.01..0.5`` for ``"tailswap"`` (its domain), and
    ``0.05..1.0`` noise-to-sigma ratios for ``"gaussian"``.

    ``kind="shuffle"`` is rejected here, deliberately. A shuffle ignores
    ``magnitude`` - there is no dial to sweep - so a curve over it is a curve
    over a parameter that does not exist. It was accepted, fell back to the
    gaussian grid, and evaluated the same experiment at six points: with a
    ``random_state`` set, ``rgr`` re-derives its seeds from it on every call,
    so all six were bit-for-bit the same number and AURGR was a weighted
    average of one value dressed up as an area. Use
    ``rgr(..., kind="shuffle")`` for the single "this input is gone entirely"
    benchmark it actually is.
    """
    if kind == "shuffle":
        raise InputError(
            "kind='shuffle' has no magnitude to sweep, so it has no robustness "
            "curve and no AURGR. Call rgr(X_test, model, variables, "
            "kind='shuffle') for the single total-information-loss benchmark, "
            "or use kind='gaussian' / 'tailswap' for a curve."
        )
    if grid is None:
        grid = (
            [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
            if kind == "tailswap"
            else [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
        )
    magnitudes = np.asarray(sorted(float(g) for g in grid), dtype=np.float64)
    if magnitudes.size < 2:
        raise InputError("'grid' needs at least two magnitudes to form a curve.")
    if magnitudes[0] <= 0:
        raise InputError("'grid' magnitudes must be strictly positive.")

    columns = resolve_columns(variables, X_test, "variables")
    groups: list[list[Any]] = [columns] if group else [[c] for c in columns]

    per_group: dict[tuple[Any, ...], list[RGRResult]] = {
        tuple(chunk): [] for chunk in groups
    }
    for magnitude in magnitudes:
        step = rgr(
            X_test,
            model,
            columns,
            yhat=yhat,
            magnitude=float(magnitude),
            kind=kind,
            group=group,
            n_repeats=n_repeats,
            pos_label=pos_label,
            greater_is_better=greater_is_better,
            random_state=random_state,
        )
        for item in step:
            per_group[item.variables].append(item)

    curves: list[RGRCurve] = []
    for chunk, items in per_group.items():
        items.sort(key=lambda item: item.magnitude)
        values = np.array([item.rgr for item in items], dtype=np.float64)
        # Anchor at (0, 1): an unperturbed input cannot change the ranking.
        xs = np.concatenate(([0.0], magnitudes))
        ys = np.concatenate(([1.0], values))
        area = (
            float(np.trapezoid(ys, xs))
            if hasattr(np, "trapezoid")
            else float(np.trapz(ys, xs))
        )
        curves.append(
            RGRCurve(
                variables=chunk,
                magnitudes=magnitudes,
                values=values,
                aurgr=area / float(magnitudes[-1]),
                kind=kind,
                n=items[0].n,
                per_magnitude=items,
            )
        )
    curves.sort(key=lambda curve: curve.aurgr)
    return curves
