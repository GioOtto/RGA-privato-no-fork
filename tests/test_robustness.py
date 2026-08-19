"""RGR: perturbation semantics, and the curve that removes the hyperparameter."""

from __future__ import annotations

import numpy as np
import pytest

from rgbox import perturb, rgr, rgr_curve
from rgbox.exceptions import InputError

pytest.importorskip("pandas")
pytest.importorskip("sklearn")


def test_tailswap_preserves_the_multiset_of_values(rng):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"x": rng.normal(size=200)})
    swapped = perturb(frame, "x", 0.1)
    assert np.allclose(np.sort(frame["x"]), np.sort(swapped["x"]))
    assert not np.allclose(frame["x"], swapped["x"])


def test_zero_magnitude_is_a_no_op(rng):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"x": rng.normal(size=100)})
    assert np.allclose(perturb(frame, "x", 0.0)["x"], frame["x"])


def test_tailswap_magnitude_is_bounded(rng):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"x": rng.normal(size=50)})
    for bad in (-0.01, 0.51, 1.0):
        with pytest.raises(InputError, match=r"\[0, 0.5\]"):
            perturb(frame, "x", bad)


def test_tailswap_rejects_non_numeric_columns(rng):
    """Upstream sorted strings lexicographically and returned a no-op shuffle."""
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"grade": rng.choice(["A", "B", "C"], 100)})
    with pytest.raises(InputError, match="numeric column"):
        perturb(frame, "grade", 0.1)
    # ...but "total loss of this input" is still expressible.
    shuffled = perturb(frame, "grade", kind="shuffle", random_state=0)
    assert sorted(shuffled["grade"]) == sorted(frame["grade"])


def test_gaussian_perturbation_scales_with_the_column(rng):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"x": rng.normal(0, 10, 4000)})
    jittered = perturb(frame, "x", 0.5, kind="gaussian", random_state=0)
    noise = jittered["x"] - frame["x"]
    assert np.std(noise) == pytest.approx(0.5 * np.std(frame["x"]), rel=0.1)


def test_perturbation_does_not_mutate_the_input(rng):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"x": rng.normal(size=100), "y": rng.normal(size=100)})
    before = frame.copy()
    perturb(frame, "x", 0.2)
    assert frame.equals(before)


def test_rgr_is_one_when_the_model_ignores_the_perturbed_column(rng):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {"used": rng.normal(size=300), "ignored": rng.normal(size=300)}
    )
    result = rgr(frame, lambda X: X["used"], ["ignored"])[0]
    assert result.rgr == pytest.approx(1.0, abs=1e-9)


def test_rgr_decreases_as_the_shock_grows(fitted_logit):
    context = fitted_logit
    values = [
        rgr(
            context["X_test"],
            context["model"],
            ["income"],
            yhat=context["yhat"],
            magnitude=m,
        )[0].rgr
        for m in (0.01, 0.05, 0.10, 0.25, 0.50)
    ]
    assert values == sorted(values, reverse=True)
    # And the spread across magnitudes is large - which is the whole problem
    # with quoting a single RGR at an arbitrary default.
    assert values[0] - values[-1] > 0.1


def test_aurgr_summarises_the_curve_without_a_hyperparameter(fitted_logit):
    context = fitted_logit
    curves = rgr_curve(
        context["X_test"],
        context["model"],
        ["income", "dti", "noise"],
        yhat=context["yhat"],
    )
    assert [c.aurgr for c in curves] == sorted(c.aurgr for c in curves)
    for curve in curves:
        assert 0.0 <= curve.aurgr <= 1.0
        assert curve.values.size == curve.magnitudes.size
        # AURGR must sit between the best and worst point values on the curve.
        assert curve.values.min() <= curve.aurgr <= 1.0


def test_aurgr_is_one_for_an_unused_column(rng):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {"used": rng.normal(size=300), "ignored": rng.normal(size=300)}
    )
    curve = rgr_curve(frame, lambda X: X["used"], ["ignored"])[0]
    assert curve.aurgr == pytest.approx(1.0, abs=1e-9)


def test_gaussian_repeats_report_a_spread(fitted_logit):
    context = fitted_logit
    result = rgr(
        context["X_test"],
        context["model"],
        ["income"],
        yhat=context["yhat"],
        kind="gaussian",
        magnitude=0.5,
        n_repeats=10,
        random_state=0,
    )[0]
    assert result.n_repeats == 10
    assert result.spread is not None and result.spread > 0


def test_a_generator_is_accepted_as_random_state(fitted_logit):
    """``random_state`` is documented to take a Generator; it used to crash.

    The old code derived a per-column seed as ``int(random_state) + offset``,
    which raises ``TypeError`` for a Generator. ``explainability.py`` had been
    fixed for this and ``robustness.py`` had not.
    """
    context = fitted_logit
    for state in (np.random.default_rng(7), np.random.PCG64(7), 7, None):
        result = rgr(
            context["X_test"],
            context["model"],
            ["income", "dti"],
            yhat=context["yhat"],
            kind="gaussian",
            magnitude=0.4,
            n_repeats=3,
            random_state=state,
        )
        assert len(result) == 2
        assert all(np.isfinite(item.rgr) for item in result)

    curve = rgr_curve(
        context["X_test"],
        context["model"],
        ["income"],
        yhat=context["yhat"],
        kind="gaussian",
        grid=[0.2, 0.4],
        random_state=np.random.default_rng(7),
    )[0]
    assert np.isfinite(curve.aurgr)


def test_repeats_are_seeded_independently_but_shared_across_variables(fitted_logit):
    """Common random numbers: same shocks per repeat, so variables compare."""
    context = fitted_logit
    kwargs = {
        "yhat": context["yhat"],
        "kind": "gaussian",
        "magnitude": 0.5,
        "n_repeats": 4,
        "random_state": 11,
    }
    together = rgr(context["X_test"], context["model"], ["income", "dti"], **kwargs)
    apart = [
        rgr(context["X_test"], context["model"], [name], **kwargs)[0]
        for name in ("income", "dti")
    ]
    # Scoring a variable alongside another must not change its own draws.
    for joint, alone in zip(together, apart):
        assert joint.rgr == pytest.approx(alone.rgr)
    # And the repeats must genuinely differ, or the spread would be zero.
    assert together[0].spread > 0


def test_pooled_interval_is_centred_on_the_reported_point(fitted_logit):
    """The CI used to describe the last draw, the point estimate all of them."""
    context = fitted_logit
    result = rgr(
        context["X_test"],
        context["model"],
        ["income"],
        yhat=context["yhat"],
        kind="gaussian",
        magnitude=0.5,
        n_repeats=6,
        random_state=3,
        ci=True,
    )[0]
    assert result.estimate is not None
    assert result.estimate.estimate == pytest.approx(result.rgr)
    midpoint = (result.estimate.ci_low + result.estimate.ci_high) / 2
    assert midpoint == pytest.approx(result.rgr)
    assert "Rubin" in result.estimate.method


def test_pooled_interval_is_wider_than_a_single_draw(fitted_logit):
    """Rubin's rules add the between-draw variance; ignoring it understates."""
    context = fitted_logit
    kwargs = {
        "yhat": context["yhat"],
        "kind": "gaussian",
        "magnitude": 0.6,
        "random_state": 5,
        "ci": True,
    }
    one = rgr(context["X_test"], context["model"], ["income"], n_repeats=1, **kwargs)[
        0
    ].estimate
    many = rgr(context["X_test"], context["model"], ["income"], n_repeats=8, **kwargs)[
        0
    ].estimate
    assert many.standard_error > one.standard_error
    # m == 1 must reduce exactly to the unpooled interval, so the default path
    # is untouched by the pooling.
    assert "Rubin" not in one.method


def test_tailswap_ignores_repeats_because_it_is_deterministic(fitted_logit):
    context = fitted_logit
    result = rgr(
        context["X_test"],
        context["model"],
        ["income"],
        yhat=context["yhat"],
        n_repeats=5,
    )[0]
    assert result.n_repeats == 1
    assert result.spread is None


def test_group_perturbation_is_at_least_as_damaging(fitted_logit):
    context = fitted_logit
    single = min(
        item.rgr
        for item in rgr(
            context["X_test"],
            context["model"],
            ["income", "dti"],
            yhat=context["yhat"],
            magnitude=0.2,
        )
    )
    joint = rgr(
        context["X_test"],
        context["model"],
        ["income", "dti"],
        yhat=context["yhat"],
        magnitude=0.2,
        group=True,
    )[0].rgr
    assert joint <= single + 1e-9


def test_curve_needs_at_least_two_magnitudes(fitted_logit):
    context = fitted_logit
    with pytest.raises(InputError, match="at least two magnitudes"):
        rgr_curve(
            context["X_test"],
            context["model"],
            ["income"],
            yhat=context["yhat"],
            grid=[0.1],
        )


def test_results_serialise(fitted_logit):
    context = fitted_logit
    curve = rgr_curve(
        context["X_test"],
        context["model"],
        ["income"],
        yhat=context["yhat"],
    )[0]
    record = curve.to_dict()
    assert set(record) >= {"label", "magnitudes", "rgr", "aurgr", "kind"}
    assert isinstance(record["magnitudes"], list)
