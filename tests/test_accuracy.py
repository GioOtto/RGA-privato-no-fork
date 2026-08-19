"""Accuracy reporting, multiclass, segments and the robustness claim."""

from __future__ import annotations

import numpy as np
import pytest

from rgbox import (
    accuracy_report,
    compare_models,
    contamination_curve,
    rga,
    rga_by_segment,
    rga_ovr,
)
from rgbox.exceptions import InputError, UndefinedMetricError


def test_report_puts_rga_next_to_the_usual_metrics(binary):
    y, yhat = binary
    report = accuracy_report(y, yhat)
    assert report.is_binary
    # Not a tautology any more: ``auroc`` used to be a copy of the RGA
    # estimate, so this line could not fail. It is now an independent
    # Mann-Whitney rank sum, and the agreement is the theorem being checked.
    assert report.reference["auroc"] == pytest.approx(report.rga.estimate)
    assert report.gini == pytest.approx(2 * report.rga.estimate - 1)
    assert report.significance["p_value"] < 1e-6
    assert "RGA" in str(report)
    assert set(report.to_dict()) >= {"n", "rga", "gini", "reference_metrics"}


def test_report_works_on_a_continuous_target(rng):
    y = rng.lognormal(0, 1, 400)
    yhat = 0.6 * y + rng.normal(0, 0.5, 400)
    report = accuracy_report(y, yhat)
    assert not report.is_binary
    assert "auroc" not in report.reference
    assert report.rga.estimate > 0.7


def test_auroc_is_computed_independently_of_rga():
    """Hand-computed rank sum, on a case with ties and non-0/1 labels.

    Positives are ranked 2nd, 3.5th and 3.5th of five (the tie splits ranks 3
    and 4), so U = (2 + 3.5 + 3.5) - 3*4/2 = 3 and AUROC = 3 / (3*2) = 0.5.
    """
    y = np.array([7.0, 7.0, 9.0, 9.0, 9.0])          # labels are not 0/1
    yhat = np.array([0.1, 0.9, 0.4, 0.6, 0.6])       # a tie among the positives
    report = accuracy_report(y, yhat)
    assert report.reference["auroc"] == pytest.approx(0.5)


def test_auroc_matches_sklearn_where_available(rng):
    metrics = pytest.importorskip("sklearn.metrics")
    y = rng.binomial(1, 0.3, 300).astype(float)
    yhat = np.round(rng.normal(y, 1.0, 300), 1)      # rounding forces ties
    report = accuracy_report(y, yhat)
    assert report.reference["auroc"] == pytest.approx(
        metrics.roc_auc_score(y, yhat), abs=1e-12
    )


def test_kendall_is_optional(binary):
    y, yhat = binary
    report = accuracy_report(y, yhat)
    # None when SciPy is absent, a number when present - never a crash.
    assert report.reference["kendall_tau"] is None or -1 <= report.reference[
        "kendall_tau"
    ] <= 1


# ---------------------------------------------------------------- multiclass

def test_one_vs_rest_uses_every_class(rng):
    """Upstream scored a 3-class model as P(class == classes_[1]) alone."""
    n = 900
    y = rng.integers(0, 3, n).astype(float)
    proba = rng.dirichlet(np.ones(3), n)
    # Make each column genuinely informative about its own class.
    for label in range(3):
        proba[y == label, label] += 0.5
    proba = proba / proba.sum(axis=1, keepdims=True)

    result = rga_ovr(y, proba, classes=[0.0, 1.0, 2.0])
    assert len(result["per_class"]) == 3
    assert all(row["rga"] > 0.5 for row in result["per_class"])
    assert result["rga"] == pytest.approx(
        np.mean([row["rga"] for row in result["per_class"]])
    )
    assert result["gini"] == pytest.approx(2 * result["rga"] - 1)


def test_one_vs_rest_binary_reduces_to_plain_rga(rng):
    y = rng.binomial(1, 0.4, 400).astype(float)
    p1 = 1 / (1 + np.exp(-rng.normal(y, 1.0, 400)))
    proba = np.column_stack([1 - p1, p1])
    result = rga_ovr(y, proba, classes=[0.0, 1.0])
    assert result["per_class"][1]["rga"] == pytest.approx(rga(y, p1))


def test_weighted_average_respects_prevalence(rng):
    n = 600
    y = np.concatenate([np.zeros(500), np.ones(50), np.full(50, 2.0)])
    proba = rng.dirichlet(np.ones(3), n)
    macro = rga_ovr(y, proba, classes=[0.0, 1.0, 2.0], average="macro")["rga"]
    weighted = rga_ovr(y, proba, classes=[0.0, 1.0, 2.0], average="weighted")["rga"]
    assert macro != pytest.approx(weighted, abs=1e-9)


def test_one_vs_rest_input_checks(rng):
    y = rng.integers(0, 3, 100).astype(float)
    with pytest.raises(InputError, match="n_classes"):
        rga_ovr(y, rng.normal(size=100))
    with pytest.raises(InputError, match="'classes' has"):
        rga_ovr(y, rng.dirichlet(np.ones(3), 100), classes=[0, 1])
    with pytest.raises(InputError, match="unknown average"):
        rga_ovr(y, rng.dirichlet(np.ones(3), 100), classes=[0, 1, 2], average="micro")


# ---------------------------------------------------------------- comparison

def test_compare_models_ranks_and_tests(rng):
    n = 2500
    y = rng.binomial(1, 0.3, n).astype(float)
    scores = {
        "strong": rng.normal(y * 2.0, 1.0, n),
        "weak": rng.normal(y * 0.4, 1.0, n),
        "noise": rng.normal(0, 1.0, n),
    }
    result = compare_models(y, scores, baseline="noise")
    ranking = [row["model"] for row in result["ranking"]]
    assert ranking[0] == "strong"
    assert ranking[-1] == "noise"
    strong = next(r for r in result["vs_baseline"] if r["model"] == "strong")
    assert strong["difference"] > 0 and strong["p_value"] < 1e-6


def test_compare_models_rejects_unknown_baseline(binary):
    y, yhat = binary
    with pytest.raises(InputError, match="not among the models"):
        compare_models(y, {"a": yhat}, baseline="b")


# ------------------------------------------------------------------ segments

def test_segments_are_flagged_when_too_small(rng):
    y = rng.binomial(1, 0.35, 1000).astype(float)
    scores = rng.normal(y, 1.0, 1000)
    segments = np.array(["big"] * 950 + ["small"] * 50)
    rows = rga_by_segment(y, scores, segments, min_size=100)
    flags = {row["segment"]: row for row in rows}
    assert flags["big"]["reliable"]
    assert not flags["small"]["reliable"]
    assert flags["small"]["ci_high"] - flags["small"]["ci_low"] > (
        flags["big"]["ci_high"] - flags["big"]["ci_low"]
    )


def test_a_segment_too_small_to_estimate_is_reported_not_raised(rng):
    """A two-row segment aborted the whole call.

    ``as_score_pair`` raises ``InsufficientDataError``, which is a *sibling* of
    ``InputError`` under ``RGBoxError`` rather than a subclass, so it escaped
    the handler and took every other segment down with it.
    """
    y = np.concatenate([rng.binomial(1, 0.4, 400), [0.0, 1.0]]).astype(float)
    scores = rng.normal(size=402)
    segments = np.array(["ok"] * 400 + ["sliver", "sliver"])
    rows = rga_by_segment(y, scores, segments, min_size=1)
    by_segment = {row["segment"]: row for row in rows}
    assert by_segment["sliver"]["rga"] is None
    assert by_segment["sliver"]["note"]
    # The point of the fix: the healthy segment still gets its number.
    assert by_segment["ok"]["rga"] is not None


def test_degenerate_segment_is_reported_not_raised(rng):
    y = np.concatenate([rng.binomial(1, 0.4, 400), np.zeros(60)]).astype(float)
    scores = rng.normal(size=460)
    segments = np.array(["ok"] * 400 + ["degenerate"] * 60)
    rows = rga_by_segment(y, scores, segments, min_size=10)
    degenerate = next(row for row in rows if row["segment"] == "degenerate")
    assert degenerate["rga"] is None
    assert "note" in degenerate


# ------------------------------------------------------------- contamination

def test_rga_is_far_more_outlier_stable_than_rmse(rng):
    """Quantifies the papers' robustness claim, which they never measured."""
    y = rng.lognormal(0, 1, 800)
    yhat = 0.7 * y + rng.normal(0, 0.3, 800)
    curve = contamination_curve(
        y, yhat, fractions=(0.0, 0.01, 0.05), magnitude=50.0,
        n_repeats=15, random_state=0,
    )
    for row in curve["curve"][1:]:
        assert row["rmse_relative_change"] > 5 * row["rga_relative_change"]
    assert curve["curve"][0]["rga_relative_change"] == pytest.approx(0.0, abs=1e-12)


def test_constant_target_still_raises_in_the_report(rng):
    with pytest.raises(UndefinedMetricError):
        accuracy_report(np.ones(50), rng.normal(size=50))
