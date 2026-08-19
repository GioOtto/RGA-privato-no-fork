"""RGA must be reachable from the model-selection tools teams already use."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from rgbox import rga
from rgbox.sklearn_api import gini_scorer, make_rga_scorer, rga_scorer


@pytest.fixture
def classification(rng):
    X = rng.normal(size=(400, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=400) > 0).astype(int)
    return X, y


def test_scorer_works_in_cross_val_score(classification):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X, y = classification
    scores = cross_val_score(
        LogisticRegression(), X, y, cv=4, scoring=make_rga_scorer()
    )
    assert scores.shape == (4,)
    assert np.all(scores > 0.5)


def test_callable_scorer_matches_a_manual_computation(classification):
    from sklearn.linear_model import LogisticRegression

    X, y = classification
    model = LogisticRegression().fit(X, y)
    assert rga_scorer(model, X, y) == pytest.approx(
        rga(y, model.predict_proba(X)[:, 1])
    )
    assert gini_scorer(model, X, y) == pytest.approx(2 * rga_scorer(model, X, y) - 1)


def test_scorer_drives_grid_search(classification):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV

    X, y = classification
    search = GridSearchCV(
        LogisticRegression(max_iter=500),
        {"C": [0.01, 1.0, 100.0]},
        scoring=rga_scorer,
        cv=3,
    ).fit(X, y)
    assert search.best_score_ > 0.5
    assert search.best_params_["C"] in (0.01, 1.0, 100.0)


def test_regression_scorer(rng):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    X = rng.normal(size=(300, 3))
    y = X[:, 0] * 3 + rng.normal(size=300)
    scores = cross_val_score(
        LinearRegression(), X, y, cv=4, scoring=make_rga_scorer(needs_proba=False)
    )
    assert np.all(scores > 0.7)


def test_scorer_is_invariant_to_monotone_recalibration(classification):
    """A raw margin and its Platt-scaled probability must score identically."""
    from sklearn.linear_model import LogisticRegression

    X, y = classification
    model = LogisticRegression().fit(X, y)
    margin = model.decision_function(X)
    probability = model.predict_proba(X)[:, 1]
    assert rga(y, margin) == pytest.approx(rga(y, probability), abs=1e-12)
