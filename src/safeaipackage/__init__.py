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

import importlib
import os
import warnings
from typing import Any

from rgbox import __version__ as _rgbox_version

from .core import rga

#: Attribute name -> submodule that defines it. Everything in here is reached
#: lazily, through ``__getattr__`` below, because every one of these submodules
#: imports pandas at module scope while the distribution declares numpy as its
#: only hard requirement. Importing them eagerly made ``import safeaipackage``
#: fail outright on a valid minimal install - the *other* top-level package in
#: the same wheel - even though ``import rgbox`` worked. ``safeaipackage.core``
#: is numpy-only and stays eager, so ``rga`` costs nothing.
_LAZY: dict[str, str] = {
    "check_accuracy": "check_accuracy",
    "check_explainability": "check_explainability",
    "check_fairness": "check_fairness",
    "check_robustness": "check_robustness",
    "compute_rge_values": "check_explainability",
    "compute_rga_parity": "check_fairness",
    "compute_rgr_values": "check_robustness",
    "perturb": "check_robustness",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(f".{module_name}", __name__)
    except ImportError as exc:
        raise ImportError(
            f"safeaipackage.{module_name} needs pandas, which is an optional "
            "dependency of this distribution: the DataFrame-based parts of the "
            "legacy API cannot work without it. Install it with "
            "`pip install safeaipackage-rgbox[pandas]`. The numpy-only core - "
            "`safeaipackage.core.rga`, and all of `rgbox.rga` / `rgbox.rga_ci` "
            "- does not need it."
        ) from exc
    value = module if name == module_name else getattr(module, name)
    globals()[name] = value  # import once, then resolve normally
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


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
