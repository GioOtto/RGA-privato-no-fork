"""Input coercion: the missing-value predicate every grouping call goes through."""

from __future__ import annotations

import numpy as np
import pytest

from rgbox._validation import _count_missing, as_group_labels, is_missing
from rgbox.exceptions import InputError


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (np.array([1.0, np.nan, 3.0, np.nan]), 2),
        (np.array([1.0, 2.0, 3.0]), 0),
        (np.array([1, 2, 3]), 0),
        (np.array([True, False]), 0),
        (np.array(["a", "b", "nan"]), 0),  # the *text* "nan" is not missing
        (np.array(["a", None, np.nan], dtype=object), 2),
        (np.array([1.0 + 1j, complex("nan")]), 1),
        (np.array(["2020-01-01", "NaT"], dtype="datetime64[ns]"), 1),
        (np.array([], dtype=np.float64), 0),
    ],
)
def test_count_missing_agrees_with_the_element_wise_predicate(values, expected):
    """The vectorised paths must return exactly what asking each element does.

    _count_missing runs on every parity, segment and outcome call. The naive
    form - tolist() plus a Python predicate per element - was ~170x slower
    than the numpy one at n=1e6, so the dtype fast paths exist; they are only
    worth anything if they agree with the fallback.
    """
    assert _count_missing(values) == expected
    if values.dtype.kind not in "Mm":
        # tolist() on datetime64 yields datetime objects, which the predicate
        # reads differently; every other dtype must match element for element.
        assert expected == sum(1 for v in values.tolist() if is_missing(v))


def test_group_labels_still_reject_missing_after_vectorising():
    with pytest.raises(InputError, match="1 missing label"):
        as_group_labels(np.array([1.0, np.nan, 2.0]), "groups", 3)
    with pytest.raises(InputError, match="2 missing label"):
        as_group_labels(np.array(["a", None, np.nan], dtype=object), "groups", 3)


def test_group_labels_pass_a_clean_column_through():
    labels = np.array(["a", "b", "a"])
    assert as_group_labels(labels, "groups", 3).tolist() == ["a", "b", "a"]


def test_a_pandas_nullable_column_is_reported_as_missing_not_as_categorical():
    pd = pytest.importorskip("pandas")
    column = pd.array([1, None, 3], dtype="Int64")
    with pytest.raises(InputError, match="1 missing label"):
        as_group_labels(column, "groups", 3)
