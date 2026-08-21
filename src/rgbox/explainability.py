"""Rank Graduation Explainability: how much does a predictor carry the ranking?

``RGE(x) = 1 - RGA(yhat_full, yhat_without_x)``: reorder the full model's
predictions by the reduced model's predictions and see how badly the ranking
degrades.

Four things about RGE that the documentation elsewhere does not say, all
verified in ``tests/test_explainability.py``:

**Removing everything gives exactly 0.5, not 1.**
    With no predictor left the reduced score is constant, all its ranks tie,
    RGA is *exactly* 0.5 and RGE is *exactly* 0.5 - whatever the model,
    whatever the data. So the grand total of "all the ranking information in
    this model" is 0.5, not the 1 the documentation implies, and raw RGE values
    are not comparable across datasets on the stated scale. ``normalize=True``
    rescales to ``2 * RGE``, which makes the grand total exactly 1.

    Individual RGE values are *not* capped at 0.5, though: when removing a
    predictor **inverts** the ranking rather than flattening it,
    ``RGA(full, reduced)`` falls below 0.5 and RGE rises above it. The test
    suite contains a fitted logistic regression where a single predictor scores
    0.64.

**Removal is not the same as retraining.**
    Substituting the mean holds the other coefficients fixed at values fitted
    in the presence of ``x``, and pushes rows off the data manifold when
    features are correlated - the standard Hooker-Mundler critique of
    permutation-style importance. The R code accompanying the original paper
    *retrains* (``lm(Y ~ . - X1)``); the Python package substitutes the mean.
    Those are different estimands. Both are available here, via
    ``method="mean"`` and ``method="retrain"``, and they can disagree
    substantially.

**Group RGE is not monotone.**
    Adding a feature to a group can *lower* the group's RGE. With two nearly
    collinear predictors carrying opposite coefficients, removing one of them
    destroys the balance between them and wrecks the ranking (high RGE), while
    removing both cancels the damage and leaves the genuine signal in charge
    (low RGE). Measured on the fitted model in the test suite:

    ===========================  ======
    coalition                    RGE
    ===========================  ======
    ``{income}``                 0.638
    ``{income_copy}``            0.246
    ``{income, income_copy}``    0.072
    ===========================  ======

    The group scores *below both of its members*. So individual RGE values are
    not additive contributions and must not be charted as if they were. When
    you need contributions that add up, use :func:`rge_shapley`, which is
    efficient by construction.

    The corollary matters when reading a one-hot encoded attribute: the group's
    RGE is not bounded below by its levels' RGEs either, so a group scoring
    less than one of its dummies is a real possibility, not a bug.

**Permuting a group permutes it as a block.**
    ``method="permute"`` with two or more columns draws *one* permutation and
    applies it to all of them, so each removed column still comes from the same
    original row. The alternative - a fresh shuffle per column - fabricates
    rows that cannot exist, and the resulting score describes the model's
    behaviour on impossible data. Two independently shuffled dummies both land
    on 1 at rate ``p_i * p_j``: on the drop-first encoded 45/30/25 attribute in
    the test suite that is 11.25% of rows, one in nine. Correlated columns are
    affected for the same reason, less visibly. Changed in 1.0.1; group
    ``permute`` numbers before and after are not comparable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Literal

import numpy as np

from ._validation import as_1d_float, check_count
from .core import rga
from .exceptions import InputError
from .inference import RGAEstimate, rga_ci
from .predictors import as_score_function, predict_scores, resolve_columns

__all__ = [
    "RGEResult",
    "rge",
    "rge_group",
    "rge_shapley",
    "replace_column",
]

RemovalMethod = Literal["mean", "median", "mode", "permute", "retrain"]

#: The removal methods this module implements. ``RemovalMethod`` above is a
#: type hint, and a type hint is not a runtime check: "retrain", "permute",
#: "mode" and "median" each had an explicit branch, and *every other string*
#: fell through to the mean-substitution default. ``method="meam"`` returned
#: the mean-substitution RGE, and ``method="retrian"`` returned it instead of
#: demanding the ``refit`` callable that "retrain" requires - a plausible
#: number from a method nobody asked for, which is the worst failure mode a
#: validation library has.
REMOVAL_METHODS = frozenset({"mean", "median", "mode", "permute", "retrain"})


def _check_method(method: Any) -> str:
    if method not in REMOVAL_METHODS:
        raise InputError(
            f"unknown removal method {method!r}; expected one of "
            f"{sorted(REMOVAL_METHODS)!r}."
        )
    return method


def _is_numeric(values: Any) -> bool:
    dtype = getattr(values, "dtype", None)
    kind = getattr(dtype, "kind", None)
    return kind in ("i", "u", "f", "b")


def _replacement_value(train_column: Any, method: RemovalMethod) -> Any:
    """Constant that stands in for a removed predictor."""
    numeric = _is_numeric(train_column)
    if method == "mode" or not numeric:
        # Upstream only checked for pandas CategoricalDtype, so plain string /
        # object columns fell through to .mean() and raised TypeError.
        if hasattr(train_column, "mode"):
            modes = train_column.mode(dropna=True)
            if len(modes) == 0:
                raise InputError("cannot take the mode of an all-missing column.")
            return modes.iloc[0]
        values, counts = np.unique(np.asarray(train_column), return_counts=True)
        return values[int(np.argmax(counts))]
    if method == "median":
        return float(np.nanmedian(np.asarray(train_column, dtype=np.float64)))
    return float(np.nanmean(np.asarray(train_column, dtype=np.float64)))


def replace_column(
    X_test: Any,
    column: Any,
    *,
    X_train: Any = None,
    method: RemovalMethod = "mean",
    random_state: Any = None,
) -> Any:
    """Return a copy of ``X_test`` with ``column``'s information removed.

    ``method="permute"`` shuffles the column instead of flattening it, keeping
    its marginal distribution (so the model still sees in-range values) while
    destroying its association with everything else.

    This removes **one** column. Removing several as a unit is
    :func:`rge`'s job: calling this in a loop would draw an independent
    permutation per column and break any structure they share. Pass the same
    ``random_state`` twice and you get the same permutation, which is the
    property that keeps :func:`rge_shapley` self-consistent.
    """
    _check_method(method)
    out = X_test.copy()
    source = X_train if X_train is not None else X_test
    if method == "permute":
        rng = np.random.default_rng(random_state)
        values = np.asarray(out[column])
        out[column] = values[rng.permutation(values.size)]
        return out
    out[column] = _replacement_value(source[column], method)
    return out


@dataclass(frozen=True)
class RGEResult:
    """RGE for one predictor or one group of predictors."""

    variables: tuple[Any, ...]
    rge: float
    rga_reduced: float
    normalized: bool
    method: str
    n: int
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
            "rge": self.rge,
            "rga_reduced": self.rga_reduced,
            "normalized": self.normalized,
            "method": self.method,
            "n": self.n,
        }
        if self.estimate is not None:
            record["ci_low"] = self.rge_ci[0]
            record["ci_high"] = self.rge_ci[1]
            record["standard_error"] = self.estimate.standard_error * (
                2.0 if self.normalized else 1.0
            )
        return record

    @property
    def rge_ci(self) -> tuple[float, float]:
        """CI for RGE, obtained by mapping the CI for the underlying RGA.

        ``RGE = 1 - RGA`` is decreasing, so the bounds swap.
        """
        if self.estimate is None:
            raise InputError("no interval was requested; pass ci=True to rge().")
        scale = 2.0 if self.normalized else 1.0
        return (
            scale * (1.0 - self.estimate.ci_high),
            scale * (1.0 - self.estimate.ci_low),
        )


def _reduced_scores(
    *,
    score_fn,
    X_train,
    X_test,
    columns: Sequence[Any],
    method: RemovalMethod,
    refit: Callable[[Any], Any] | None,
    random_state: Any,
    pos_label: Any,
    greater_is_better: bool,
) -> np.ndarray:
    if method == "retrain":
        if refit is None:
            raise InputError(
                "method='retrain' needs a `refit` callable: refit(kept_columns) "
                "should fit a fresh model on the training data restricted to "
                "those columns and return it. This mirrors the R code that "
                "accompanies the original paper (lm(Y ~ . - X1))."
            )
        kept = [c for c in X_train.columns if c not in set(columns)]
        if not kept:
            raise InputError(
                "method='retrain' cannot drop every predictor: there would be "
                "no model left to fit."
            )
        reduced_model = refit(kept)
        return predict_scores(
            reduced_model,
            X_test[kept],
            pos_label=pos_label,
            greater_is_better=greater_is_better,
        )

    if method == "permute" and len(columns) > 1:
        # One shared permutation for the whole group, not one per column.
        # Columns removed together are removed as a unit: independent shuffles
        # of a one-hot encoded attribute hand the model rows that belong to two
        # levels at once (rate p_i*p_j per pair of dummies), and for
        # collinear copies they destroy the joint structure the group exists to
        # hold fixed.
        order = np.random.default_rng(random_state).permutation(len(X_test))
        reduced = X_test.copy()
        for column in columns:
            reduced[column] = np.asarray(reduced[column])[order]
        return score_fn(reduced)

    reduced = X_test
    for column in columns:
        reduced = replace_column(
            reduced,
            column,
            X_train=X_train,
            method=method,
            random_state=random_state,
        )
    return score_fn(reduced)


def rge(
    X_train: Any,
    X_test: Any,
    model: Any,
    variables: Sequence[Any] | Any,
    *,
    yhat: Any = None,
    method: RemovalMethod = "mean",
    group: bool = False,
    normalize: bool = False,
    ci: bool = False,
    ci_method: str = "jackknife",
    level: float = 0.95,
    refit: Callable[[Any], Any] | None = None,
    pos_label: Any = None,
    greater_is_better: bool = True,
    random_state: Any = None,
    n_resamples: int = 2000,
) -> list[RGEResult]:
    """Rank Graduation Explainability for one or more predictors.

    Parameters
    ----------
    X_train, X_test :
        Training frame (source of replacement values) and evaluation frame.
    model :
        Anything :func:`rgbox.predictors.as_score_function` accepts, including
        a plain ``X -> scores`` callable.
    variables :
        Column name or list of column names.
    yhat :
        Full-model scores on ``X_test``. Optional - computed from ``model`` if
        omitted. When supplied it is *used*, unlike upstream's
        ``compute_rga_parity``, where the argument was accepted and then
        silently ignored.
    method :
        How a predictor is removed: ``"mean"`` (upstream behaviour, but with
        proper handling of string and categorical columns), ``"median"``,
        ``"mode"``, ``"permute"``, or ``"retrain"`` (needs ``refit``).

        ``"permute"`` on a group of two or more columns applies **one shared
        permutation** to all of them, so the rows are reordered as a block and
        every removed column still comes from the same original row. Shuffling
        each column on its own instead would fabricate rows that never existed:
        two independently shuffled dummies are both 1 at rate ``p_i * p_j``,
        11.25% of rows on the attribute used in the test suite. It changed in
        1.0.1 and moves group ``rge``
        numbers for any multi-column group - single columns are unaffected, and
        so is every other ``method``.
    group :
        ``True`` removes all ``variables`` simultaneously and returns a single
        result; ``False`` returns one result per variable, sorted descending.
        Group RGE is **not** monotone in the group - see the module docstring.
    normalize :
        Rescale to ``2 * RGE`` so the attainable range really is ``[0, 1]``.
    ci :
        Attach a confidence interval for the RGE, reflecting sampling
        variability of the evaluation set (the model itself is held fixed).

    Returns
    -------
    list[RGEResult]
        One element when ``group=True``, otherwise one per variable, sorted by
        RGE descending.
    """
    _check_method(method)
    columns = resolve_columns(variables, X_train, "variables")
    resolve_columns(columns, X_test, "variables")
    score_fn = as_score_function(
        model, pos_label=pos_label, greater_is_better=greater_is_better
    )

    if yhat is None:
        full = np.asarray(score_fn(X_test), dtype=np.float64).ravel()
    else:
        full = as_1d_float(yhat, "yhat")
        if full.size != len(X_test):
            raise InputError(
                f"'yhat' has {full.size} entries but X_test has {len(X_test)} rows."
            )

    groups: list[list[Any]] = [columns] if group else [[c] for c in columns]
    results: list[RGEResult] = []
    for chunk in groups:
        reduced = np.asarray(
            _reduced_scores(
                score_fn=score_fn,
                X_train=X_train,
                X_test=X_test,
                columns=chunk,
                method=method,
                refit=refit,
                random_state=random_state,
                pos_label=pos_label,
                greater_is_better=greater_is_better,
            ),
            dtype=np.float64,
        ).ravel()

        scale = 2.0 if normalize else 1.0
        estimate = None
        if ci:
            estimate = rga_ci(
                full,
                reduced,
                method=ci_method,
                level=level,
                random_state=random_state,
                n_resamples=n_resamples,
            )
            reduced_rga = estimate.estimate
        else:
            reduced_rga = rga(full, reduced)

        results.append(
            RGEResult(
                variables=tuple(chunk),
                rge=scale * (1.0 - reduced_rga),
                rga_reduced=reduced_rga,
                normalized=normalize,
                method=method,
                n=full.size,
                estimate=estimate,
            )
        )

    if not group:
        results.sort(key=lambda item: item.rge, reverse=True)
    return results


def rge_group(*args: Any, **kwargs: Any) -> RGEResult:
    """:func:`rge` with ``group=True``, returning the single result."""
    kwargs["group"] = True
    return rge(*args, **kwargs)[0]


@dataclass(frozen=True)
class ShapleyResult:
    """Shapley decomposition of the total ranking information."""

    values: dict[Any, float]
    total: float
    exact: bool
    n_permutations: int | None
    normalized: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": dict(self.values),
            "total": self.total,
            "exact": self.exact,
            "n_permutations": self.n_permutations,
            "normalized": self.normalized,
        }

    def ranked(self) -> list[tuple[Any, float]]:
        return sorted(self.values.items(), key=lambda kv: kv[1], reverse=True)


def rge_shapley(
    X_train: Any,
    X_test: Any,
    model: Any,
    variables: Sequence[Any] | Any,
    *,
    yhat: Any = None,
    method: RemovalMethod = "mean",
    normalize: bool = False,
    n_permutations: int | None = None,
    max_exact: int = 12,
    refit: Callable[[Any], Any] | None = None,
    pos_label: Any = None,
    greater_is_better: bool = True,
    random_state: Any = None,
) -> ShapleyResult:
    """Shapley values of the game ``v(S) = RGE(S)``, so importances add up.

    Plain RGE is a *marginal* importance and suffers the usual pathology under
    collinearity: two duplicated predictors that jointly drive the target can
    each score near zero, indistinguishable from noise, because removing either
    one alone leaves the other to do its job. Treating ``v(S) = 1 - RGA(yhat,
    yhat_without_S)`` as a cooperative game and taking Shapley values fixes
    this - the two share the credit, and by efficiency the values sum to
    ``v(all features)``, which is the total ranking information the model
    carries (0.5 unnormalised, 1.0 with ``normalize=True``).

    Cost: exact evaluation needs ``2**d`` reduced scorings, done when
    ``d <= max_exact``. Above that, or when ``n_permutations`` is given, the
    standard permutation-sampling estimator is used, costing
    ``n_permutations * d`` scorings.
    """
    _check_method(method)
    if n_permutations is not None:
        # The estimator divides by this at the end. Zero raised
        # ZeroDivisionError from inside the loop's aftermath; a negative value
        # was worse, because `range(-1)` is empty and every Shapley value came
        # back as a perfectly presentable -0.0.
        n_permutations = check_count(n_permutations, "n_permutations")
    max_exact = check_count(max_exact, "max_exact", minimum=0)
    columns = resolve_columns(variables, X_train, "variables")
    resolve_columns(columns, X_test, "variables")
    score_fn = as_score_function(
        model, pos_label=pos_label, greater_is_better=greater_is_better
    )
    full = (
        np.asarray(score_fn(X_test), dtype=np.float64).ravel()
        if yhat is None
        else as_1d_float(yhat, "yhat")
    )
    scale = 2.0 if normalize else 1.0
    cache: dict[frozenset, float] = {}

    def value(subset: frozenset) -> float:
        """v(S): information lost when the predictors in S are removed."""
        if not subset:
            return 0.0
        if subset not in cache:
            reduced = np.asarray(
                _reduced_scores(
                    score_fn=score_fn,
                    X_train=X_train,
                    X_test=X_test,
                    columns=sorted(subset, key=lambda c: columns.index(c)),
                    method=method,
                    refit=refit,
                    random_state=random_state,
                    pos_label=pos_label,
                    greater_is_better=greater_is_better,
                ),
                dtype=np.float64,
            ).ravel()
            cache[subset] = scale * (1.0 - rga(full, reduced))
        return cache[subset]

    d = len(columns)
    use_exact = n_permutations is None and d <= max_exact
    shapley = dict.fromkeys(columns, 0.0)

    if use_exact:
        from math import factorial

        others = {c: [x for x in columns if x != c] for c in columns}
        for column in columns:
            rest = others[column]
            for size in range(len(rest) + 1):
                weight = factorial(size) * factorial(d - size - 1) / factorial(d)
                for subset in combinations(rest, size):
                    frozen = frozenset(subset)
                    shapley[column] += weight * (
                        value(frozen | {column}) - value(frozen)
                    )
        return ShapleyResult(
            values=shapley,
            total=value(frozenset(columns)),
            exact=True,
            n_permutations=None,
            normalized=normalize,
        )

    draws = n_permutations if n_permutations is not None else 200
    rng = np.random.default_rng(random_state)
    for _ in range(draws):
        # Permute *positions*, not the labels themselves. Building an object
        # array out of the labels turns a list of equal-length tuples - which
        # is exactly how pandas spells MultiIndex columns, and which
        # resolve_columns explicitly supports - into a 2-D array, so shuffling
        # it yielded rows: unhashable ndarrays where the set operations below
        # need labels, and a TypeError from deep inside the loop.
        order = [columns[position] for position in rng.permutation(d)]
        running = frozenset()
        previous = 0.0
        for column in order:
            running = running | {column}
            current = value(running)
            shapley[column] += current - previous
            previous = current
    for column in columns:
        shapley[column] /= draws
    return ShapleyResult(
        values=shapley,
        total=value(frozenset(columns)),
        exact=False,
        n_permutations=draws,
        normalized=normalize,
    )
