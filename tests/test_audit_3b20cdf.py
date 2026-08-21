"""Regression tests for the defects found auditing 3b20cdf.

Every test here fails on the commit before the fix. The theme of this round is
narrower than the previous one and, for a validation library, worse: none of
these defects raised, and none of them produced an obviously wrong number.
They produced *interpretable-looking* ones - a percentage that is not a share
of anything, a group robustness figure measured partly on obligors who live in
two regions at once, a leakage score that changes when you renumber the levels,
a fairness section that reads as complete because nothing says it is not.
"""

from __future__ import annotations

import numpy as np
import pytest

from rgbox import (
    InputError,
    bootstrap_values,
    influence_values,
    jackknife_values,
    perturb,
    proxy_leakage,
    rga_ovr,
    rgbox_report,
    rgr,
    worst_cohort,
)

pd = pytest.importorskip("pandas")


# --------------------------------------------------------------------------
# F-02: marginal RGEs are not additive, so they have no "share of total"
# --------------------------------------------------------------------------


def test_the_rge_table_does_not_present_marginal_values_as_shares(fitted_logit):
    """Normalising non-additive marginals produced a fake decomposition.

    The table carried a "share of total" column, each RGE over the sum of the
    RGEs. Group RGE is not monotone - the explainability module measures a pair
    scoring below both of its members - so that sum is not a total and a
    percentage of it is not a share. It rendered as a decomposition of
    importance next to numbers that cannot decompose.
    """
    context = fitted_logit
    report = rgbox_report(
        y=context["y_test"],
        X_test=context["X_test"],
        X_train=context["X_train"],
        model=context["model"],
        yhat=context["yhat"],
        variables=["income", "dti"],
        random_state=0,
    )
    text = report.to_markdown()
    assert "share of total" not in text
    assert "%" not in text.split("## Robustness")[0].split("## Explainability")[1]
    assert "not additive" in text
    assert "rge_shapley" in text


# --------------------------------------------------------------------------
# F-03: a perturbed group must stay inside its own domain
# --------------------------------------------------------------------------


def test_group_shuffle_keeps_a_one_hot_encoding_valid(rng):
    """Independent shuffles put 12% of rows in two regions at once.

    ``rge(method="permute")`` drew one shared order for a group; robustness
    advanced a live Generator per column, so each dummy got its own. A group
    RGR was then partly a measurement of the model's response to rows that
    cannot exist.
    """
    n = 600
    level = rng.choice(["north", "south", "centre"], size=n, p=[0.45, 0.30, 0.25])
    frame = pd.DataFrame(
        {
            "region_north": (level == "north").astype(float),
            "region_south": (level == "south").astype(float),
            "x": rng.normal(size=n),
        }
    )
    y = frame["x"].to_numpy()

    def score(X):
        return np.asarray(X["x"]) + np.asarray(X["region_north"])

    seen = {}

    def capture(X):
        seen["both"] = int(((X["region_north"] == 1) & (X["region_south"] == 1)).sum())
        return score(X)

    rgr(
        frame,
        capture,
        ["region_north", "region_south"],
        yhat=y,
        kind="shuffle",
        group=True,
        random_state=0,
    )
    assert seen["both"] == 0, "a shuffled group fabricated an impossible level"


def test_a_single_column_shuffle_is_unchanged(rng):
    """The shared-order path must not touch the one-column case."""
    frame = pd.DataFrame({"a": np.arange(50.0)})
    once = perturb(frame, "a", kind="shuffle", random_state=11)
    twice = perturb(frame, "a", kind="shuffle", random_state=11)
    assert np.array_equal(once["a"].to_numpy(), twice["a"].to_numpy())


# --------------------------------------------------------------------------
# F-06: "the full box" must not omit outcome fairness silently
# --------------------------------------------------------------------------


def test_the_report_says_when_outcome_fairness_was_not_evaluated(fitted_logit):
    context = fitted_logit
    report = rgbox_report(
        y=context["y_test"],
        X_test=context["X_test"],
        X_train=context["X_train"],
        model=context["model"],
        yhat=context["yhat"],
        variables=["income"],
        protected="gender",
        random_state=0,
    )
    assert any("Outcome-based fairness was not evaluated" in w for w in report.warnings)
    assert "Not evaluated" in report.to_markdown()
    assert report.fairness is not None
    assert "outcome_parity" not in report.fairness


def test_the_report_computes_outcome_fairness_when_given_a_threshold(fitted_logit):
    context = fitted_logit
    report = rgbox_report(
        y=context["y_test"],
        X_test=context["X_test"],
        X_train=context["X_train"],
        model=context["model"],
        yhat=context["yhat"],
        variables=["income"],
        protected="gender",
        threshold=0.5,
        random_state=0,
    )
    assert report.fairness is not None
    block = report.fairness["outcome_parity"]
    assert "demographic_parity" in block["criteria"]
    assert block["threshold"] == 0.5
    text = report.to_markdown()
    assert "Outcome-based fairness" in text
    assert "Not evaluated" not in text
    assert not any("was not evaluated" in w for w in report.warnings)


def test_a_threshold_without_a_protected_attribute_is_not_silently_dropped(
    fitted_logit,
):
    context = fitted_logit
    report = rgbox_report(
        y=context["y_test"],
        X_test=context["X_test"],
        X_train=context["X_train"],
        model=context["model"],
        yhat=context["yhat"],
        variables=["income"],
        threshold=0.5,
        random_state=0,
    )
    assert any("outcome parity skipped" in w for w in report.warnings)


# --------------------------------------------------------------------------
# F-08: a nominal attribute has no order, whatever its codes are
# --------------------------------------------------------------------------


def test_proxy_leakage_does_not_depend_on_how_levels_are_numbered(rng):
    """One integer column with three levels was handed to RGA as a scale.

    Relabelling ``{north: 0, centre: 1, south: 2}`` to ``{centre: 0, north: 1,
    south: 2}`` is the same attribute and changed the answer.
    """
    n = 600
    level = rng.choice(["north", "south", "centre"], size=n)
    proxy = rng.normal(size=n) + (level == "south") * 2.0

    first = np.select([level == "north", level == "centre"], [0.0, 1.0], default=2.0)
    second = np.select([level == "centre", level == "north"], [0.0, 1.0], default=2.0)
    a = proxy_leakage(pd.DataFrame({"region": first, "proxy": proxy}), "region")
    b = proxy_leakage(pd.DataFrame({"region": second, "proxy": proxy}), "region")

    def leakage(result):
        return next(r["leakage"] for r in result["proxies"] if r["variable"] == "proxy")

    assert leakage(a) == pytest.approx(leakage(b))
    assert a["ordinal_note"]


def test_an_ordinal_attribute_can_still_be_scored_in_one_pass(rng):
    n = 400
    grade = rng.integers(0, 5, size=n).astype(float)
    frame = pd.DataFrame({"grade": grade, "proxy": grade + rng.normal(0, 0.1, n)})
    ordered = proxy_leakage(frame, "grade", ordered=True)
    assert ordered["ordered"] is True
    assert ordered["ordinal_note"] == ""
    assert ordered["proxies"][0]["level"] == "grade"


def test_a_binary_protected_column_is_untouched_by_the_decomposition(rng):
    n = 400
    sex = (rng.random(n) < 0.5).astype(float)
    frame = pd.DataFrame({"sex": sex, "proxy": sex + rng.normal(0, 0.3, n)})
    result = proxy_leakage(frame, "sex")
    assert result["ordinal_note"] == ""
    assert result["proxies"][0]["level"] == "sex"


# --------------------------------------------------------------------------
# F-09: the artefact is byte-identical only if the timestamp is pinned
# --------------------------------------------------------------------------


def test_a_pinned_timestamp_makes_the_whole_artefact_reproducible(fitted_logit):
    context = fitted_logit
    kwargs = {
        "y": context["y_test"],
        "X_test": context["X_test"],
        "X_train": context["X_train"],
        "model": context["model"],
        "yhat": context["yhat"],
        "variables": ["income"],
        "random_state": 7,
        "generated_at": "2026-03-31",
    }
    first, second = rgbox_report(**kwargs), rgbox_report(**kwargs)
    assert first.to_json() == second.to_json()
    assert first.to_markdown() == second.to_markdown()
    assert first.to_html() == second.to_html()
    assert first.metadata["generated_at"] == "2026-03-31"


# --------------------------------------------------------------------------
# F-10: a label is a value, not markup
# --------------------------------------------------------------------------


def test_a_pipe_in_a_group_level_does_not_split_a_markdown_cell(fitted_logit):
    """`"P&L | Q3"` is an ordinary segment name and it broke the table.

    Every row after the offending one shifted by a column, silently, in the
    artefact whose whole purpose is to be read.
    """
    rng = np.random.default_rng(0)
    n = 400
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "segment": np.where(
                np.arange(n) % 2 == 0, "retail | north", "retail | south"
            ),
        }
    )
    y = (rng.random(n) < 1 / (1 + np.exp(-frame["x"]))).astype(float).to_numpy()
    report = rgbox_report(
        y=y,
        X_test=frame,
        X_train=frame,
        model=lambda X: np.asarray(X["x"]),
        variables=["x"],
        protected="segment",
        min_group_size=10,
        random_state=0,
    )
    text = report.to_markdown()
    assert "retail \\| north" in text

    widths = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        # Count structural (unescaped) pipes only.
        unescaped = line.replace("\\\\", "").replace("\\|", "")
        widths.setdefault(unescaped.count("|"), 0)
        widths[unescaped.count("|")] += 1
    # The parity table has 5 columns and so 6 pipes on every one of its rows,
    # header, rule and body alike; a split cell would produce a 7.
    assert 7 not in widths, text

    html = report.to_html()
    assert "<td>retail | north</td>" in html
    assert "retail \\|" not in html


def test_a_newline_in_a_label_cannot_open_a_heading(fitted_logit):
    context = fitted_logit
    report = rgbox_report(
        y=context["y_test"],
        X_test=context["X_test"],
        X_train=context["X_train"],
        model=context["model"],
        yhat=context["yhat"],
        variables=["income"],
        model_name="v1\n## Injected heading",
        random_state=0,
    )
    text = report.to_markdown()
    # The words survive - they are part of the name the caller gave - but they
    # stay on the title line instead of opening a block of their own.
    assert not any(
        line.startswith("#") and "Injected" in line for line in text.splitlines()[1:]
    )
    assert "<h2>Injected heading</h2>" not in report.to_html()
    assert text.count("# Rank Graduation Box report") == 1
    assert text.splitlines()[0].endswith("v1 ## Injected heading")


# --------------------------------------------------------------------------
# F-11: the exported resampling helpers must honour the package's contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "helper", [jackknife_values, influence_values, bootstrap_values]
)
def test_the_public_resampling_helpers_accept_lists(helper):
    """`y.size` on a list raised AttributeError from inside the estimator."""
    y = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    yhat = [0.1, 0.9, 0.2, 0.8, 0.7, 0.3]
    values = helper(y, yhat)
    assert np.asarray(values).size > 0


@pytest.mark.parametrize(
    "helper", [jackknife_values, influence_values, bootstrap_values]
)
def test_the_public_resampling_helpers_type_their_errors(helper):
    """A length mismatch surfaced as a NumPy gufunc core-dimension message."""
    with pytest.raises(InputError, match="same length"):
        helper(np.array([0.0, 1.0, 0.0, 1.0]), np.array([0.1, 0.2, 0.3]))


def test_paired_bootstrap_validates_its_second_score(binary):
    y, yhat = binary
    with pytest.raises(InputError, match="paired_with"):
        bootstrap_values(y, yhat, n_resamples=10, paired_with=yhat[:-1])


# --------------------------------------------------------------------------
# F-12: two bins of one feature cannot intersect, so they must not be tried
# --------------------------------------------------------------------------


def test_bin_conditions_carry_the_feature_they_came_from(rng):
    """Without the id there is nothing to skip on, which is why nothing was."""
    from rgbox.cohorts import _bin_conditions

    n = 600
    frame = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    conditions = _bin_conditions(frame, ["a", "b"], 4)
    assert {feature_id for feature_id, _, _ in conditions} == {0, 1}
    for feature_id, name, _ in conditions:
        assert name.startswith(("a", "b")) or f"<= {'ab'[feature_id]} " in name


def test_the_search_never_pairs_two_bins_of_the_same_feature(rng):
    """Fed an overlapping same-feature pair, the search must still skip it.

    The bins a real feature produces are disjoint, so this cannot be shown on
    ordinary input - the intersection is empty either way and only the wasted
    work differs. Handing ``_search`` two *overlapping* conditions that share a
    feature id isolates the rule itself: before the fix the pair was
    intersected and returned as a depth-2 cohort.
    """
    from rgbox.cohorts import _search

    n = 400
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
    scores = rng.random(n)
    first = np.zeros(n, dtype=bool)
    first[:300] = True
    second = np.zeros(n, dtype=bool)
    second[100:] = True  # overlaps `first` on 200 rows

    same_feature = [(0, "left", first), (0, "right", second)]
    found = _search(y, scores, same_feature, min_size=50, max_depth=2)
    assert {names for names, _, _, _ in found} == {("left",), ("right",)}

    different_features = [(0, "left", first), (1, "right", second)]
    found = _search(y, scores, different_features, min_size=50, max_depth=2)
    assert ("left", "right") in {names for names, _, _, _ in found}


def test_the_search_still_finds_the_planted_cohort(rng):
    """Skipping empty intersections must not change any result."""
    n = 3000
    region = rng.choice(["north", "centre", "south"], n)
    channel = rng.choice(["branch", "online"], n)
    signal = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-signal))).astype(float)
    scores = 1 / (1 + np.exp(-(signal + rng.normal(0, 0.4, n))))
    broken = (region == "south") & (channel == "online")
    scores[broken] = rng.random(int(broken.sum()))
    frame = pd.DataFrame({"region": region, "channel": channel})

    result = worst_cohort(y, scores, frame, min_size=100, n_permutations=0)
    assert "region == 'south'" in result.cohorts[0].label
    assert "channel == 'online'" in result.cohorts[0].label


# --------------------------------------------------------------------------
# F-15: a one-vs-rest indicator needs equality, not an order
# --------------------------------------------------------------------------


def test_rga_ovr_accepts_string_class_labels():
    y = np.array(["cat", "dog", "bird", "cat", "dog", "bird", "cat", "dog"])
    proba = np.array(
        [
            [0.1, 0.7, 0.2],
            [0.1, 0.2, 0.7],
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
            [0.6, 0.3, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.1, 0.7],
        ]
    )
    result = rga_ovr(y, proba, classes=["bird", "cat", "dog"])
    assert result["n"] == 8
    assert [row["class"] for row in result["per_class"]] == ["bird", "cat", "dog"]
    assert all(row["rga"] is not None for row in result["per_class"])
    assert 0.0 <= result["rga"] <= 1.0


def test_rga_ovr_numeric_labels_are_unchanged():
    y = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    proba = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1],
            [0.3, 0.6, 0.1],
            [0.1, 0.3, 0.6],
        ]
    )
    assert rga_ovr(y, proba)["rga"] == pytest.approx(1.0)


def test_rga_ovr_still_rejects_a_two_dimensional_target():
    with pytest.raises(InputError, match="one-dimensional"):
        rga_ovr(np.zeros((4, 2)), np.zeros((4, 2)))
