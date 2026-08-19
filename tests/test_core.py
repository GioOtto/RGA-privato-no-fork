"""Input handling and error contracts."""

from __future__ import annotations

import numpy as np
import pytest

from rgbox import rga
from rgbox.exceptions import (
    InputError,
    InsufficientDataError,
    RGBoxError,
    UndefinedMetricError,
)


def test_accepts_lists_arrays_and_columns(rng):
    pd = pytest.importorskip("pandas")
    y = [0, 0, 1, 1, 0, 1]
    yhat = [0.1, 0.3, 0.6, 0.9, 0.2, 0.8]
    expected = rga(y, yhat)
    assert rga(np.array(y), np.array(yhat)) == pytest.approx(expected)
    assert rga(pd.Series(y), pd.Series(yhat)) == pytest.approx(expected)
    assert rga(pd.DataFrame(y), pd.DataFrame(yhat)) == pytest.approx(expected)
    assert rga(np.array(y).reshape(-1, 1), np.array(yhat)) == pytest.approx(expected)


def test_indexes_are_ignored_not_aligned(rng):
    """A misaligned index must not silently reorder or introduce NaN."""
    pd = pytest.importorskip("pandas")
    y = pd.Series([0.0, 1.0, 1.0, 0.0], index=[10, 11, 12, 13])
    yhat = pd.Series([0.2, 0.9, 0.7, 0.1], index=[0, 1, 2, 3])
    assert rga(y, yhat) == pytest.approx(rga(y.to_numpy(), yhat.to_numpy()))


def test_constant_target_raises_rather_than_returning_nan(rng):
    """Upstream returned nan here, which propagates silently into reports."""
    with pytest.raises(UndefinedMetricError, match="zero dispersion"):
        rga(np.ones(20), rng.normal(size=20))


def test_length_mismatch(rng):
    with pytest.raises(InputError, match="same length"):
        rga([1, 2, 3], [1, 2])


def test_too_few_observations():
    with pytest.raises(InsufficientDataError):
        rga([1.0], [2.0])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_values_rejected(bad):
    with pytest.raises(InputError, match=r"NaN|infinite"):
        rga([1.0, 2.0, bad], [1.0, 2.0, 3.0])
    with pytest.raises(InputError, match=r"NaN|infinite"):
        rga([1.0, 2.0, 3.0], [1.0, 2.0, bad])


def test_non_numeric_target_rejected_with_guidance():
    with pytest.raises(InputError, match="encode categories"):
        rga(["low", "high", "mid"], [1.0, 2.0, 3.0])


def test_boolean_targets_work():
    y = np.array([True, False, True, True, False])
    assert rga(y, [0.9, 0.1, 0.8, 0.7, 0.2]) == pytest.approx(1.0)


def test_two_observations_is_the_minimum(rng):
    assert rga([0.0, 1.0], [0.0, 1.0]) == pytest.approx(1.0)
    assert rga([0.0, 1.0], [1.0, 0.0]) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "weights, message",
    [
        ([1, 2], "length"),
        ([-1.0] * 6, "non-negative"),
        ([0.0] * 6, "positive sum"),
        ([1.0, 0, 0, 0, 0, 0], "non-zero weight"),
    ],
)
def test_bad_weights_rejected(weights, message):
    with pytest.raises((InputError, InsufficientDataError), match=message):
        rga([0.0, 1, 2, 3, 4, 5], [1.0, 2, 3, 4, 5, 6], weights=weights)


def test_all_errors_share_a_base_class():
    for error in (InputError, UndefinedMetricError, InsufficientDataError):
        assert issubclass(error, RGBoxError)
    # ...and stay catchable by legacy `except ValueError` code.
    assert issubclass(InputError, ValueError)
    assert issubclass(UndefinedMetricError, ValueError)


def test_large_sample_is_fast(rng):
    """A 200k-row hold-out must not need a coffee break."""
    import time

    y = rng.normal(size=200_000)
    yhat = 0.5 * y + rng.normal(size=200_000)
    start = time.perf_counter()
    rga(y, yhat)
    assert time.perf_counter() - start < 2.0
