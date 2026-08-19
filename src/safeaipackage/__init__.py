"""Drop-in compatibility layer for the original ``safeaipackage`` API.

Existing notebooks keep working:

>>> from safeaipackage.core import rga                    # doctest: +SKIP
>>> from safeaipackage.check_explainability import compute_rge_values

Every function here delegates to :mod:`rgbox`, so you get the corrected
numerics, the typed errors and the missing dependency-free core - without
rewriting call sites. The original package left ``__init__.py`` empty; this one
also re-exports the four entry points so ``import safeaipackage`` is useful on
its own.

Where behaviour deviates
------------------------
The fixes are behaviour changes, deliberately. ``MIGRATION.md`` lists all of
them; the ones most likely to surface in existing code:

* ``rga`` **raises** ``UndefinedMetricError`` on a constant target instead of
  returning ``nan``;
* ``compute_rga_parity`` **uses** its ``yhat`` argument (upstream accepted it
  and then re-predicted, so passing random noise changed nothing) and returns a
  float subclass rather than a sentence;
* ``find_yhat`` **raises** on a multiclass model instead of silently scoring
  ``P(class == classes_[1])``;
* ``manipulate_testdata`` handles string/object columns via the mode instead of
  raising ``TypeError`` from ``.mean()``;
* ``perturb`` rejects non-numeric columns instead of sorting them
  lexicographically and returning a meaningless permutation.

Set ``SAFEAIPACKAGE_SILENCE_FORK_NOTICE=1`` to suppress the import-time notice.
"""

from __future__ import annotations

import os
import warnings

from rgbox import __version__ as _rgbox_version

from . import (
    check_accuracy,
    check_explainability,
    check_fairness,
    check_robustness,
    core,
)
from .check_explainability import compute_rge_values
from .check_fairness import compute_rga_parity
from .check_robustness import compute_rgr_values, perturb
from .core import rga

__version__ = _rgbox_version
__all__ = [
    "rga",
    "compute_rge_values",
    "compute_rgr_values",
    "compute_rga_parity",
    "perturb",
    "core",
    "check_accuracy",
    "check_explainability",
    "check_fairness",
    "check_robustness",
]

if not os.environ.get("SAFEAIPACKAGE_SILENCE_FORK_NOTICE"):
    warnings.warn(
        "This 'safeaipackage' is the rgbox fork's compatibility layer, not the "
        "upstream package. Numbers may differ where upstream had defects (see "
        "MIGRATION.md). New code should import from 'rgbox'.",
        UserWarning,
        stacklevel=2,
    )
