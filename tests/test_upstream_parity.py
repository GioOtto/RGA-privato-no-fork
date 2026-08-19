"""The fork must return the upstream *numbers*, not just upstream-looking ones.

The reference implementation is re-derived inline from the published discrete
definition, so this test does not depend on ``upstream_reference/`` being
present and does not import the original package (which cannot even be
imported without CatBoost and XGBoost installed).
"""

from __future__ import annotations

import numpy as np
import pytest

from rgbox import rga


def upstream_rga(y, yhat):
    """The original algorithm, transcribed from safeaipackage 0.8.3.

    Rank ``yhat`` with ``method="min"``, replace each ``y`` by the mean of the
    ``y`` values sharing its predicted rank, reorder by ``yhat``, then take
    ``(conc - dec) / (inc - dec)`` over index-weighted sums. Uses pandas
    groupby/merge exactly as upstream did, so this is a faithful oracle rather
    than a paraphrase.
    """
    pd = pytest.importorskip("pandas")
    y = pd.DataFrame(y).reset_index(drop=True)
    yhat = pd.DataFrame(yhat).reset_index(drop=True)
    frame = pd.concat([y, yhat], axis=1)
    frame.columns = ["y", "yhat"]
    frame["ryhat"] = frame["yhat"].rank(method="min")
    support = frame.groupby("ryhat")["y"].mean().reset_index(name="support")
    frame = pd.merge(frame, support, on="ryhat", how="left")
    frame["rord"] = frame["support"]
    frame = frame.sort_values(by="yhat").reset_index(drop=True)
    ystar = frame["rord"].values
    index = np.arange(len(frame))
    conc = np.sum(index * ystar)
    ordered = np.sort(frame["y"])
    dec = np.sum(index * ordered[::-1])
    inc = np.sum(index * ordered)
    return (conc - dec) / (inc - dec)


def test_matches_upstream_to_machine_precision(sample):
    y, yhat = sample
    if np.ptp(y) == 0:
        pytest.skip("upstream returns nan here; the fork raises by design")
    assert rga(y, yhat) == pytest.approx(upstream_rga(y, yhat), abs=1e-12)


@pytest.mark.parametrize("n", [3, 4, 7, 15, 100])
def test_matches_upstream_at_small_n(rng, n):
    y = rng.normal(size=n)
    yhat = rng.normal(size=n)
    assert rga(y, yhat) == pytest.approx(upstream_rga(y, yhat), abs=1e-12)


def test_matches_upstream_with_extreme_ties(rng):
    """Every score tied: upstream's groupby collapses to one group."""
    y = rng.normal(size=50)
    yhat = np.full(50, 3.0)
    assert rga(y, yhat) == pytest.approx(upstream_rga(y, yhat), abs=1e-12)
    assert rga(y, yhat) == pytest.approx(0.5, abs=1e-12)


def test_closed_form_is_better_conditioned_than_upstream(rng):
    """Both forms lose precision on a hugely offset target; ours loses less.

    The information is destroyed in the float64 representation of ``y`` itself,
    so no algorithm recovers it - but the upstream form additionally sums terms
    of order ``n^2 * mean(y)`` before subtracting them, which costs roughly one
    to one and a half extra decimal digits.
    """
    y = rng.normal(100.0, 15.0, 500)
    yhat = y + rng.normal(0.0, 8.0, 500)
    baseline = rga(y, yhat)

    ours = abs(rga(y + 1e12, yhat) - baseline)
    theirs = abs(upstream_rga(y + 1e12, yhat) - baseline)
    assert ours < theirs
    assert ours < 1e-5
