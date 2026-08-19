"""Legacy ``safeaipackage.check_robustness``."""

from __future__ import annotations

from typing import Any

import pandas as pd

from rgbox.robustness import perturb as _perturb
from rgbox.robustness import rgr as _rgr

__all__ = ["perturb", "compute_rgr_values"]


def perturb(
    data: pd.DataFrame,
    variable: str,
    perturbation_percentage: float = 0.05,
) -> pd.DataFrame:
    """Swap the lowest and highest ``perturbation_percentage`` of ``variable``.

    Same tail-swap scheme as upstream, vectorised. Deviation: a non-numeric
    column now raises instead of being sorted lexicographically and permuted to
    no purpose. Use ``rgbox.perturb(..., kind="shuffle")`` to model the loss of
    a categorical input, or ``kind="gaussian"`` for the noise-based
    perturbation used in the more recent literature.
    """
    return _perturb(data, variable, perturbation_percentage, kind="tailswap")


def compute_rgr_values(
    xtest: pd.DataFrame,
    yhat: list,
    model: Any,
    variables: list,
    perturbation_percentage: float = 0.05,
    group: bool = False,
) -> pd.DataFrame:
    """RGR for the given variables, individually or as one group.

    Identical signature, return shape and values to upstream.

    The result depends strongly on ``perturbation_percentage``, and nothing
    justifies the 0.05 default: on one model and one feature it sweeps from
    0.97 at 0.01 to 0.57 at 0.50. :func:`rgbox.rgr_curve` reports AURGR, the
    area under the whole curve, which has no such free parameter.
    """
    results = _rgr(
        xtest,
        model,
        list(variables),
        yhat=yhat,
        magnitude=perturbation_percentage,
        kind="tailswap",
        group=group,
    )
    if group:
        return pd.DataFrame(
            [results[0].rgr], index=[str(list(variables))], columns=["RGR"]
        )
    frame = pd.DataFrame(
        [item.rgr for item in results],
        index=[item.variables[0] for item in results],
        columns=["RGR"],
    )
    return frame.sort_values(by="RGR", ascending=False)
