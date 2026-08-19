"""Model adapters: duck typing, score direction, and clear failure."""

from __future__ import annotations

import numpy as np
import pytest

from rgbox import as_score_function, predict_scores, rga
from rgbox.exceptions import InputError, ModelAdapterError
from rgbox.predictors import resolve_columns


def test_plain_callable(rng):
    X = rng.normal(size=(50, 2))
    scores = predict_scores(lambda data: data[:, 0], X)
    assert np.allclose(scores, X[:, 0])


def test_regressor_uses_predict(rng):
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression

    X = rng.normal(size=(100, 3))
    y = X[:, 0] * 2 + rng.normal(size=100)
    model = LinearRegression().fit(X, y)
    assert np.allclose(predict_scores(model, X), model.predict(X))


def test_binary_classifier_uses_positive_class_probability(rng):
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    X = rng.normal(size=(200, 3))
    y = (X[:, 0] > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    assert np.allclose(predict_scores(model, X), model.predict_proba(X)[:, 1])


def test_pos_label_selects_the_right_column(rng):
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    X = rng.normal(size=(300, 3))
    y = rng.choice(["good", "bad"], 300)
    model = LogisticRegression().fit(X, y)
    index = list(model.classes_).index("bad")
    assert np.allclose(
        predict_scores(model, X, pos_label="bad"), model.predict_proba(X)[:, index]
    )


def test_unknown_pos_label_is_rejected(rng):
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    X = rng.normal(size=(100, 2))
    model = LogisticRegression().fit(X, (X[:, 0] > 0).astype(int))
    with pytest.raises(InputError, match="not one of the model's classes"):
        predict_scores(model, X, pos_label=9)


def test_decision_function_is_used_when_there_is_no_proba(rng):
    pytest.importorskip("sklearn")
    from sklearn.svm import LinearSVC

    X = rng.normal(size=(200, 3))
    y = (X[:, 0] > 0).astype(int)
    model = LinearSVC().fit(X, y)
    scores = predict_scores(model, X)
    assert np.allclose(scores, model.decision_function(X))
    # A margin and a probability rank identically, so RGA is unchanged.
    assert rga(y, scores) == pytest.approx(rga(y, model.decision_function(X)))


def test_greater_is_better_false_flips_the_ranking(rng):
    X = rng.normal(size=(100, 2))
    y = X[:, 0]
    forward = rga(y, predict_scores(lambda d: d[:, 0], X))
    reversed_ = rga(y, predict_scores(lambda d: d[:, 0], X, greater_is_better=False))
    assert forward == pytest.approx(1.0)
    assert reversed_ == pytest.approx(0.0, abs=1e-12)


def test_wrong_number_of_scores_is_caught(rng):
    X = rng.normal(size=(50, 2))
    with pytest.raises(ModelAdapterError, match="returned 3 scores for 50 rows"):
        predict_scores(lambda d: np.zeros(3), X)


def test_score_function_is_reusable(rng):
    X = rng.normal(size=(40, 2))
    fn = as_score_function(lambda d: d[:, 1])
    assert np.allclose(fn(X), fn(X))


def test_a_tuple_matching_a_column_label_is_that_one_column():
    """MultiIndex columns are tuples, so a tuple is a label before a group."""
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({("a", "b"): [1.0], "a": [2.0], "b": [3.0]})
    frame.columns = pd.MultiIndex.from_tuples([("a", "b"), ("a", ""), ("b", "")])
    assert resolve_columns(("a", "b"), frame, "variables") == [("a", "b")]


def test_a_tuple_that_is_not_a_column_label_is_still_a_group():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"a": [1.0], "b": [2.0]})
    assert resolve_columns(("a", "b"), frame, "variables") == ["a", "b"]


def test_a_list_is_never_read_as_a_label():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"a": [1.0], "b": [2.0]})
    assert resolve_columns(["a", "b"], frame, "variables") == ["a", "b"]


def test_empty_selection_is_reported_before_the_missing_labels_check(rng):
    with pytest.raises(InputError, match="is empty"):
        resolve_columns([], rng.normal(size=(3, 2)), "variables")
