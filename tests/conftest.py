"""Shared fixtures.

Everything is seeded. A test suite for a statistical library that flakes is
worse than no test suite, because it trains people to re-run until green.
"""

from __future__ import annotations

import numpy as np
import pytest

SEED = 20260811


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


@pytest.fixture
def continuous(rng):
    y = rng.normal(size=400)
    return y, 0.7 * y + rng.normal(size=400)


@pytest.fixture
def binary(rng):
    y = rng.binomial(1, 0.3, 500).astype(float)
    return y, rng.normal(y * 1.2, 1.0, 500)


@pytest.fixture(
    params=["continuous", "binary", "ties_y", "ties_yhat", "ties_both",
            "counts", "negative", "skewed", "constant_yhat"]
)
def sample(request, rng):
    """A representative spread of (y, yhat) shapes, including degenerate ties."""
    n = 300
    if request.param == "continuous":
        y = rng.normal(size=n)
        return y, 0.5 * y + rng.normal(size=n)
    if request.param == "binary":
        y = rng.binomial(1, 0.35, n).astype(float)
        return y, rng.normal(y, 1.0, n)
    if request.param == "ties_y":
        return rng.integers(0, 4, n).astype(float), rng.normal(size=n)
    if request.param == "ties_yhat":
        return rng.normal(size=n), rng.integers(0, 5, n).astype(float)
    if request.param == "ties_both":
        return rng.integers(0, 3, n).astype(float), rng.integers(0, 4, n).astype(float)
    if request.param == "counts":
        return rng.poisson(3, n).astype(float), rng.gamma(2, 1, n)
    if request.param == "negative":
        return rng.normal(-50, 3, n), rng.normal(size=n)
    if request.param == "skewed":
        return rng.lognormal(0, 1.5, n), rng.normal(size=n)
    return rng.normal(size=n), np.ones(n)


@pytest.fixture
def credit_frame(rng):
    """A small, deliberately messy credit-scoring frame.

    Includes a string column, a near-duplicate of an informative predictor, a
    binary protected attribute and a genuinely irrelevant feature - the four
    shapes that broke the upstream implementation.
    """
    pd = pytest.importorskip("pandas")
    n = 900
    frame = pd.DataFrame({
        "income": rng.lognormal(10.0, 0.6, n),
        "age": rng.integers(21, 78, n).astype(float),
        "dti": rng.beta(2, 5, n),
        "noise": rng.normal(size=n),
        "gender": rng.binomial(1, 0.45, n).astype(float),
        "region": rng.choice(["north", "centre", "south"], n),
    })
    frame["income_copy"] = frame["income"] * (1 + rng.normal(0, 0.002, n))
    linear = (
        -0.9 * np.log(frame["income"])
        + 4.0 * frame["dti"]
        + 0.02 * frame["age"]
    )
    linear = linear - linear.mean()
    target = rng.binomial(1, 1.0 / (1.0 + np.exp(-linear))).astype(float)
    return frame, target


@pytest.fixture
def fitted_logit(credit_frame):
    """A fitted logistic regression plus its train/test split and scores."""
    pytest.importorskip("sklearn")
    pd = pytest.importorskip("pandas")
    from sklearn.linear_model import LogisticRegression

    frame, target = credit_frame
    design = pd.get_dummies(frame, columns=["region"], drop_first=True).astype(float)
    split = 600
    X_train, X_test = design.iloc[:split], design.iloc[split:]
    y_train, y_test = target[:split], target[split:]
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "yhat": scores,
        "raw_test": frame.iloc[split:],
    }
