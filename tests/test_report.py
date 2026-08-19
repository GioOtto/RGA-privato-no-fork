"""The report must be complete, deterministic and serialisable."""

from __future__ import annotations

import json

import pytest

from rgbox import rgbox_report

pytest.importorskip("pandas")
pytest.importorskip("sklearn")


@pytest.fixture
def report(fitted_logit):
    context = fitted_logit
    return rgbox_report(
        y=context["y_test"],
        X_test=context["X_test"],
        X_train=context["X_train"],
        model=context["model"],
        yhat=context["yhat"],
        variables=["income", "dti", "age", "noise"],
        protected="gender",
        model_name="PD scorecard",
        random_state=0,
    )


def test_covers_all_four_safe_principles(report):
    record = report.to_dict()
    assert record["accuracy"]["rga"]["rga"] > 0.5
    assert len(record["explainability"]) == 4
    assert len(record["robustness"]) == 4
    assert record["fairness"]["rga_parity"]["groups"]
    assert record["fairness"]["rgf"]["rgf"] <= 1.0
    assert record["fairness"]["proxy_leakage"]["proxies"]


def test_is_deterministic(fitted_logit):
    context = fitted_logit
    kwargs = {
        "y": context["y_test"],
        "X_test": context["X_test"],
        "X_train": context["X_train"],
        "model": context["model"],
        "yhat": context["yhat"],
        "variables": ["income", "dti"],
        "protected": "gender",
        "random_state": 7,
    }
    first, second = rgbox_report(**kwargs), rgbox_report(**kwargs)
    strip = lambda r: {k: v for k, v in r.to_dict().items() if k != "metadata"}  # noqa: E731
    assert strip(first) == strip(second)


def test_json_roundtrips(report):
    payload = json.loads(report.to_json())
    assert payload["schema_version"] == "1.0"
    assert payload["metadata"]["model_name"] == "PD scorecard"
    assert payload["metadata"]["n"] == report.metadata["n"]


def test_markdown_contains_the_numbers_and_the_caveat(report):
    text = report.to_markdown()
    assert "# Rank Graduation Box report" in text
    assert "## Accuracy" in text
    assert "## Explainability (RGE)" in text
    assert "## Robustness (AURGR)" in text
    assert "## Fairness" in text
    assert "Gini (2*RGA-1)" in text
    # The fairness caveat must survive into the rendered artefact.
    assert "not demographic parity" in text


def test_parity_p_value_says_which_one_it_is(report):
    """A bare "p = 0.03" is unreadable when two p-values exist.

    The rendered block must name the headline value as family-wise, say how
    many pairs it corrects for, and print the uncorrected one beside it so the
    size of the penalty is visible.
    """
    text = report.to_markdown()
    parity = report.fairness["rga_parity"]
    assert "family-wise" in text
    assert parity["multiplicity"] in text
    assert "Uncorrected p" in text
    assert f"{parity['gap_p_value']:.4g}" in text
    assert f"{parity['gap_p_value_unadjusted']:.4g}" in text


def test_html_is_self_contained(report):
    html = report.to_html()
    assert html.startswith("<!doctype html>")
    assert "<table>" in html and "</table>" in html
    assert "http://" not in html and "src=" not in html


def test_partial_specification_warns_instead_of_failing(fitted_logit):
    context = fitted_logit
    minimal = rgbox_report(
        y=context["y_test"],
        X_test=context["X_test"],
        model=context["model"],
        yhat=context["yhat"],
    )
    assert minimal.accuracy["rga"]["rga"] > 0.5
    assert minimal.explainability == []
    assert minimal.robustness == []
    assert minimal.fairness is None
    assert any("no 'variables'" in w for w in minimal.warnings)


def test_explainability_skipped_without_train_frame(fitted_logit):
    context = fitted_logit
    result = rgbox_report(
        y=context["y_test"],
        X_test=context["X_test"],
        model=context["model"],
        yhat=context["yhat"],
        variables=["income"],
    )
    assert result.explainability == []
    assert result.robustness  # robustness needs no training frame
    assert any("X_train was not supplied" in w for w in result.warnings)


def test_scores_are_computed_when_yhat_is_omitted(fitted_logit):
    context = fitted_logit
    result = rgbox_report(
        y=context["y_test"],
        X_test=context["X_test"],
        model=context["model"],
    )
    assert result.accuracy["rga"]["rga"] == pytest.approx(
        rgbox_report(
            y=context["y_test"],
            X_test=context["X_test"],
            model=context["model"],
            yhat=context["yhat"],
        ).accuracy["rga"]["rga"]
    )


def test_one_hot_protected_attribute_runs_end_to_end(fitted_logit):
    """A categorical attribute reaches the design matrix as dummies.

    Every fairness block must then read them as *one* attribute: rgf removes
    them as a unit, parity compares the reconstructed levels rather than each
    dummy's 0/1, and proxy_leakage does not score the dummies as proxies for
    themselves.
    """
    context = fitted_logit
    dummies = [c for c in context["X_test"].columns if c.startswith("region_")]
    assert len(dummies) == 2

    result = rgbox_report(
        y=context["y_test"],
        X_test=context["X_test"],
        X_train=context["X_train"],
        model=context["model"],
        yhat=context["yhat"],
        protected=dummies,
        random_state=0,
    )
    parity = result.fairness["rga_parity"]
    # Three levels, not two dummies: the omitted reference level is rebuilt.
    assert {g["group"] for g in parity["groups"]} == {*dummies, "reference"}
    assert result.fairness["rgf"]["attribute"] == dummies
    proxies = {row["variable"] for row in result.fairness["proxy_leakage"]["proxies"]}
    assert not proxies & set(dummies)
    json.loads(result.to_json())
    assert result.to_markdown()


def test_tiny_protected_group_is_flagged_in_warnings(fitted_logit):
    context = fitted_logit
    result = rgbox_report(
        y=context["y_test"],
        X_test=context["X_test"],
        X_train=context["X_train"],
        model=context["model"],
        yhat=context["yhat"],
        protected="gender",
        min_group_size=10_000,  # nothing can qualify
    )
    assert result.fairness["rga_parity"]["gap"] is None
    assert any("min_group_size" in w for w in result.warnings)
