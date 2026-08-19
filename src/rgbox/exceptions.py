"""Typed errors.

Upstream signalled every problem with a bare ``ValueError`` (or, in several
paths, with a silent ``nan`` or an ``UnboundLocalError``). A model-validation
pipeline in a regulated environment needs to tell "your inputs are malformed"
apart from "this metric is mathematically undefined on this sample" apart from
"this subgroup is too small to report", because the three demand different
operational responses.
"""

from __future__ import annotations

__all__ = [
    "RGBoxError",
    "InputError",
    "UndefinedMetricError",
    "InsufficientDataError",
    "ModelAdapterError",
]


class RGBoxError(Exception):
    """Base class for every error raised by rgbox."""


class InputError(RGBoxError, ValueError):
    """Malformed inputs: wrong shapes, NaNs, unknown column names, bad options.

    Subclasses ``ValueError`` so that ``except ValueError`` in legacy callers
    keeps working.
    """


class UndefinedMetricError(RGBoxError, ValueError):
    """The metric has no value on this sample.

    The canonical case is a constant target: the denominator of RGA is the Gini
    mean difference of ``y``, which is zero, so RGA is ``0/0``. Upstream
    returned ``nan`` here; we raise, because a ``nan`` that silently propagates
    into a fairness gap or a model-comparison table is a reporting incident.
    """


class InsufficientDataError(RGBoxError, ValueError):
    """Too few observations for the requested estimate to mean anything."""


class ModelAdapterError(RGBoxError, TypeError):
    """The object passed as ``model`` cannot be turned into a score function."""
