"""rgbox - the Rank Graduation Box, rebuilt for production model validation.

A fork of `safeaipackage <https://github.com/GolnooshBabaei/safeaipackage>`_,
the reference implementation of the S.A.F.E. AI metrics of Babaei, Giudici and
Raffinetti. Same measures, same published definitions; different engineering
and a statistics layer the original does not have.

Quick start
-----------
>>> import numpy as np
>>> from rgbox import rga, rga_ci, gini_score
>>> y = np.array([0, 0, 1, 1, 0, 1, 1, 0])
>>> scores = np.array([0.1, 0.3, 0.6, 0.9, 0.2, 0.8, 0.4, 0.5])
>>> round(rga(y, scores), 4)
0.8125
>>> round(gini_score(y, scores), 4)      # what a scorecard report calls "Gini"
0.625

Everything else hangs off that one measure:

* :func:`rga_ci` - standard error and confidence interval;
* :func:`rga_compare` - paired champion/challenger test;
* :func:`rge` / :func:`rge_shapley` - predictor importance;
* :func:`rgr_curve` - robustness with no free hyperparameter;
* :func:`rga_parity` - ranking quality across protected groups;
* :func:`outcome_parity` - demographic parity, equal opportunity, equalised
  odds and disparate impact, with intervals and a multiplicity correction;
* :func:`worst_cohort` - searches for the slice the model is worst on, instead
  of asking you to name it;
* :func:`rgbox_report` - the whole box as one serialisable artefact.

This code was written by an AI system. See the README before adopting it.
"""

from __future__ import annotations

__version__ = "1.0.2"
__upstream__ = "safeaipackage 0.8.3 (GolnooshBabaei/safeaipackage)"

from .accuracy import (
    AccuracyReport,
    accuracy_report,
    compare_models,
    contamination_curve,
    rga_by_segment,
    rga_ovr,
)
from .cohorts import Cohort, CohortSearch, worst_cohort
from .core import RGACurves, gini_score, rga, rga_curves, rga_score
from .exceptions import (
    InputError,
    InsufficientDataError,
    ModelAdapterError,
    RGBoxError,
    UndefinedMetricError,
)
from .explainability import RGEResult, rge, rge_group, rge_shapley
from .fairness import (
    GroupRGA,
    ParityResult,
    labels_from_dummies,
    proxy_leakage,
    rga_parity,
    rgf,
)
from .inference import (
    RGAComparison,
    RGAEstimate,
    bootstrap_values,
    influence_values,
    jackknife_values,
    rga_ci,
    rga_compare,
    rga_test,
)
from .outcomes import (
    CriterionResult,
    GroupRate,
    OutcomeParityResult,
    outcome_parity,
)
from .predictors import as_score_function, predict_scores
from .report import RGBoxReport, rgbox_report
from .robustness import RGRCurve, RGRResult, perturb, rgr, rgr_curve

__all__ = [
    "__version__",
    "__upstream__",
    # core
    "rga",
    "rga_score",
    "gini_score",
    "rga_curves",
    "RGACurves",
    # inference
    "rga_ci",
    "rga_compare",
    "rga_test",
    "RGAEstimate",
    "RGAComparison",
    "jackknife_values",
    "influence_values",
    "bootstrap_values",
    # accuracy
    "accuracy_report",
    "AccuracyReport",
    "compare_models",
    "rga_by_segment",
    "rga_ovr",
    "contamination_curve",
    # explainability
    "rge",
    "rge_group",
    "rge_shapley",
    "RGEResult",
    # robustness
    "rgr",
    "rgr_curve",
    "perturb",
    "RGRResult",
    "RGRCurve",
    # fairness
    "rga_parity",
    "rgf",
    "proxy_leakage",
    "labels_from_dummies",
    "ParityResult",
    "GroupRGA",
    # fairness - outcome-based criteria
    "outcome_parity",
    "OutcomeParityResult",
    "CriterionResult",
    "GroupRate",
    # cohort search
    "worst_cohort",
    "CohortSearch",
    "Cohort",
    # plumbing
    "predict_scores",
    "as_score_function",
    "rgbox_report",
    "RGBoxReport",
    # errors
    "RGBoxError",
    "InputError",
    "UndefinedMetricError",
    "InsufficientDataError",
    "ModelAdapterError",
]
