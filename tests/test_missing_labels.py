"""Missing group/segment labels must be refused, not turned into empty levels.

Regression test for the defect where ``float('nan')`` labels produced one
"level" of size zero *per missing row*: ``labels.tolist()`` rebuilds a fresh
float object per element and ``nan != nan``, so ``dict.fromkeys`` never
deduplicated them, and ``labels == nan`` was all-False. A 2000-row report with
5% missing grew a 102-row parity table and silently analysed 1900 rows.
"""

from __future__ import annotations

import numpy as np
import pytest

from rgbox import rga, rga_by_segment, rga_parity
from rgbox.exceptions import InputError


@pytest.fixture
def paired(rng):
    y = (rng.random(400) < 0.5).astype(float)
    return y, rng.normal(size=400) + y


@pytest.fixture
def float_labels_with_nan():
    """A protected attribute as it arrives from a CSV: float64, some missing."""
    labels = np.zeros(400)
    labels[:120] = 1.0
    labels[120:200] = np.nan
    return labels


def test_float_nan_labels_are_rejected_not_expanded(paired, float_labels_with_nan):
    y, scores = paired
    with pytest.raises(InputError) as info:
        rga_parity(y, scores, float_labels_with_nan)
    assert "80 missing label(s)" in str(info.value)


def test_object_none_labels_are_rejected(paired):
    y, scores = paired
    labels = np.array(["a"] * 400, dtype=object)
    labels[:100] = None
    with pytest.raises(InputError, match="100 missing label"):
        rga_parity(y, scores, labels)


def test_segments_reject_missing_labels_too(paired, float_labels_with_nan):
    """rga_by_segment tolerates tiny segments, but not rows with no segment."""
    y, scores = paired
    with pytest.raises(InputError, match="missing label"):
        rga_by_segment(y, scores, float_labels_with_nan)


def test_pandas_na_is_recognised(paired):
    """A pandas nullable column carries pandas.NA, not float NaN."""
    pd = pytest.importorskip("pandas")
    y, scores = paired
    labels = pd.Series(["a"] * 400, dtype="string")
    labels[:60] = pd.NA
    with pytest.raises(InputError, match="60 missing label"):
        rga_parity(y, scores, labels)


def test_every_row_is_accounted_for_when_labels_are_clean(paired):
    """The property the bug broke: the table's counts add up to the sample."""
    y, scores = paired
    labels = np.where(np.arange(400) < 150, "a", "b")

    result = rga_parity(y, scores, labels, n_resamples=100, random_state=0)
    assert sum(group.n for group in result.groups) == 400
    assert len(result.groups) == 2

    rows = rga_by_segment(y, scores, labels)
    assert sum(row["n"] for row in rows) == 400
    assert len(rows) == 2


def test_a_missing_level_can_be_kept_by_naming_it(paired):
    """The documented escape hatch: encode 'missing' as a level of its own."""
    y, scores = paired
    labels = np.array(["a"] * 400, dtype=object)
    labels[:120] = "b"
    labels[120:200] = "missing"
    result = rga_parity(y, scores, labels, n_resamples=100, random_state=0)
    assert sum(group.n for group in result.groups) == 400
    assert {group.group for group in result.groups} == {"a", "b", "missing"}


def test_nullable_integer_target_reports_missing_not_dtype():
    """The error must name the real problem: NA, not 'encode categories'."""
    pd = pytest.importorskip("pandas")
    y = pd.Series([0, 1, None, 1, 1], dtype="Int64")
    with pytest.raises(InputError) as info:
        rga(y, [1.0, 2.0, 3.0, 4.0, 5.0])
    message = str(info.value)
    assert "missing value" in message
    assert "encode categories" not in message


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean"])
def test_nullable_columns_without_na_still_work(dtype):
    pd = pytest.importorskip("pandas")
    y = pd.Series([0, 1, 0, 1, 1], dtype=dtype)
    assert rga(y, [1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(5 / 6)
