"""One command, one validation artefact.

Model risk management wants a reproducible document, not a notebook: the same
inputs must produce the same numbers, every number must carry its uncertainty,
and the whole thing must serialise so it can be diffed between quarterly
re-validations. :func:`rgbox_report` runs the whole Rank Graduation Box and
returns a plain dictionary; :meth:`RGBoxReport.to_markdown` and
:meth:`RGBoxReport.to_html` render it.

Determinism: pass ``random_state`` and every resampling step is seeded, so two
runs on the same data produce byte-identical output.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .accuracy import accuracy_report
from .explainability import rge
from .fairness import labels_from_dummies, proxy_leakage, rga_parity, rgf
from .predictors import predict_scores, resolve_columns
from .robustness import rgr_curve

__all__ = ["RGBoxReport", "rgbox_report"]

_SCHEMA_VERSION = "1.0"


@dataclass
class RGBoxReport:
    """Serialisable results of a full Rank Graduation Box run."""

    metadata: dict[str, Any]
    accuracy: dict[str, Any]
    explainability: list[dict[str, Any]] = field(default_factory=list)
    robustness: list[dict[str, Any]] = field(default_factory=list)
    fairness: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "metadata": self.metadata,
            "accuracy": self.accuracy,
            "explainability": self.explainability,
            "robustness": self.robustness,
            "fairness": self.fairness,
            "warnings": self.warnings,
        }

    def to_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("indent", 2)
        kwargs.setdefault("default", str)
        return json.dumps(self.to_dict(), **kwargs)

    def to_markdown(self) -> str:
        out: list[str] = []
        add = out.append
        meta = self.metadata
        add(f"# Rank Graduation Box report - {meta.get('model_name', 'model')}")
        add("")
        add(f"- generated: {meta.get('generated_at')}")
        add(f"- evaluation sample: n = {meta.get('n')}")
        add(f"- rgbox version: {meta.get('rgbox_version')}")
        add(f"- random_state: {meta.get('random_state')!r}")
        add("")

        acc = self.accuracy
        rga_block = acc["rga"]
        add("## Accuracy")
        add("")
        add("| metric | value | 95% CI |")
        add("|---|---|---|")
        add(
            f"| RGA | {rga_block['rga']:.4f} | "
            f"{rga_block['ci_low']:.4f} - {rga_block['ci_high']:.4f} |"
        )
        add(
            f"| Gini (2*RGA-1) | {acc['gini']:+.4f} | "
            f"{acc['gini_ci_low']:+.4f} - {acc['gini_ci_high']:+.4f} |"
        )
        for name, value in acc.get("reference_metrics", {}).items():
            if value is not None:
                add(f"| {name} | {value:+.4f} | |")
        add("")

        if self.explainability:
            add("## Explainability (RGE)")
            add("")
            add("| variable | RGE | share of total |")
            add("|---|---|---|")
            total = sum(row["rge"] for row in self.explainability) or 1.0
            for row in self.explainability:
                add(
                    f"| {row['label']} | {row['rge']:.4f} | "
                    f"{100 * row['rge'] / total:.1f}% |"
                )
            add("")

        if self.robustness:
            add("## Robustness (AURGR)")
            add("")
            add("| variable | AURGR | least robust at |")
            add("|---|---|---|")
            for row in self.robustness:
                worst = min(
                    zip(row["magnitudes"], row["rgr"]), key=lambda pair: pair[1]
                )
                add(
                    f"| {row['label']} | {row['aurgr']:.4f} | "
                    f"magnitude {worst[0]:g} -> RGR {worst[1]:.4f} |"
                )
            add("")

        if self.fairness:
            add("## Fairness")
            add("")
            parity = self.fairness.get("rga_parity")
            if parity:
                add(f"Protected attribute: `{parity.get('attribute')}`")
                add("")
                add("| group | n | RGA | 95% CI | in gap |")
                add("|---|---|---|---|---|")
                for group in parity["groups"]:
                    if group["rga"] is None:
                        add(f"| {group['group']} | {group['n']} | - | - | no |")
                    else:
                        add(
                            f"| {group['group']} | {group['n']} | "
                            f"{group['rga']:.4f} | {group['ci_low']:.4f} - "
                            f"{group['ci_high']:.4f} | "
                            f"{'yes' if group['included'] else 'no'} |"
                        )
                add("")
                if parity["gap"] is not None:
                    ci = ""
                    if parity["gap_ci_low"] is not None:
                        ci = (
                            f" (95% CI {parity['gap_ci_low']:.4f} - "
                            f"{parity['gap_ci_high']:.4f})"
                        )
                    add(
                        f"**RGA parity gap: {parity['gap']:.4f}{ci}**, "
                        f"p = {parity['gap_p_value']:.4g} "
                        f"(family-wise, {parity['multiplicity']})"
                    )
                    if parity.get("gap_bias_corrected") is not None:
                        add("")
                        add(f"Bias-corrected gap: {parity['gap_bias_corrected']:.4f}")
                    # Naming the uncorrected value is what makes the correction
                    # auditable: a reader who only ever sees the adjusted number
                    # cannot tell how much of it was the multiplicity penalty.
                    if parity.get("gap_p_value_unadjusted") is not None:
                        add("")
                        add(
                            f"Uncorrected p (selects the widest of "
                            f"{len(parity['pairwise'])} pairs, so it "
                            f"over-rejects): "
                            f"{parity['gap_p_value_unadjusted']:.4g}"
                        )
                    add("")
                    add(f"> {parity['gap_ci_note']}")
                    add("")
            if self.fairness.get("rgf"):
                block = self.fairness["rgf"]
                add(
                    f"RGF (ranking reliance on `{block['attribute']}`): "
                    f"{block['rgf']:.4f} - RGE {block['rge']:.4f}"
                )
                add("")
            add(
                "> RGA parity is *AUC parity*: equal ranking quality per group. "
                "It is not demographic parity or equalised odds, and does not "
                "imply either. Report it alongside outcome-based criteria."
            )
            add("")

        if self.warnings:
            add("## Warnings")
            add("")
            for item in self.warnings:
                add(f"- {item}")
            add("")
        return "\n".join(out)

    def to_html(self) -> str:
        """Minimal self-contained HTML rendering of the markdown."""
        body: list[str] = []
        in_table = False
        for line in self.to_markdown().splitlines():
            stripped = line.strip()
            if stripped.startswith("|"):
                cells = [c.strip() for c in stripped.strip("|").split("|")]
                if set("".join(cells)) <= set("-: "):
                    continue
                if not in_table:
                    body.append("<table>")
                    in_table = True
                    tag = "th"
                else:
                    tag = "td"
                row = "".join(f"<{tag}>{c}</{tag}>" for c in cells)
                body.append(f"<tr>{row}</tr>")
                continue
            if in_table:
                body.append("</table>")
                in_table = False
            if stripped.startswith("### "):
                body.append(f"<h3>{stripped[4:]}</h3>")
            elif stripped.startswith("## "):
                body.append(f"<h2>{stripped[3:]}</h2>")
            elif stripped.startswith("# "):
                body.append(f"<h1>{stripped[2:]}</h1>")
            elif stripped.startswith("> "):
                body.append(f"<blockquote>{stripped[2:]}</blockquote>")
            elif stripped.startswith("- "):
                body.append(f"<li>{stripped[2:]}</li>")
            elif stripped:
                body.append(f"<p>{stripped}</p>")
        if in_table:
            body.append("</table>")
        style = (
            "body{font-family:system-ui,sans-serif;max-width:60rem;margin:2rem "
            "auto;padding:0 1rem;line-height:1.5}"
            "table{border-collapse:collapse;margin:1rem 0}"
            "td,th{border:1px solid #ccc;padding:.35rem .6rem;text-align:left}"
            "th{background:#f4f4f4}"
            "blockquote{border-left:3px solid #888;margin:1rem 0;padding:.2rem "
            "1rem;color:#444}"
        )
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>Rank Graduation Box report</title><style>{style}</style>"
            f"</head><body>{''.join(body)}</body></html>"
        )


def rgbox_report(
    *,
    y: Any,
    X_test: Any,
    model: Any,
    X_train: Any = None,
    yhat: Any = None,
    variables: Sequence[Any] | None = None,
    protected: Any = None,
    model_name: str = "model",
    level: float = 0.95,
    ci_method: str = "jackknife",
    perturbation_kind: str = "tailswap",
    rge_method: str = "mean",
    min_group_size: int = 50,
    n_resamples: int = 2000,
    random_state: Any = 0,
    pos_label: Any = None,
    greater_is_better: bool = True,
) -> RGBoxReport:
    """Run the full box - accuracy, explainability, robustness, fairness.

    Only ``y``, ``X_test`` and ``model`` are required. Explainability needs
    ``X_train`` (the source of replacement values) and ``variables``;
    robustness needs ``variables``; fairness needs ``protected``. Anything not
    requestable is skipped and noted in ``warnings`` rather than raising, so a
    partially specified call still yields a usable report.

    ``protected`` may be a **list** of one-hot dummies, which is how a
    categorical attribute usually reaches a design matrix. The attribute is
    then treated as one thing throughout: :func:`rgbox.rgf` removes its dummies
    as a unit, and the parity table compares its reconstructed levels (see
    :func:`rgbox.labels_from_dummies`) rather than each dummy's 0/1.
    """
    from . import __version__

    warnings: list[str] = []
    if yhat is None:
        yhat = predict_scores(
            model, X_test, pos_label=pos_label, greater_is_better=greater_is_better
        )

    accuracy = accuracy_report(
        y,
        yhat,
        method=ci_method,
        level=level,
        n_resamples=n_resamples,
        random_state=random_state,
    ).to_dict()

    explainability: list[dict[str, Any]] = []
    if variables and X_train is not None:
        explainability = [
            item.to_dict()
            for item in rge(
                X_train,
                X_test,
                model,
                list(variables),
                yhat=yhat,
                method=rge_method,
                pos_label=pos_label,
                greater_is_better=greater_is_better,
                random_state=random_state,
            )
        ]
    elif variables:
        warnings.append("explainability skipped: X_train was not supplied.")

    robustness: list[dict[str, Any]] = []
    if variables:
        robustness = [
            curve.to_dict()
            for curve in rgr_curve(
                X_test,
                model,
                list(variables),
                yhat=yhat,
                kind=perturbation_kind,
                pos_label=pos_label,
                greater_is_better=greater_is_better,
                random_state=random_state,
            )
        ]
    else:
        warnings.append("explainability and robustness skipped: no 'variables'.")

    fairness: dict[str, Any] | None = None
    if protected is not None:
        block: dict[str, Any] = {}
        protected_columns = resolve_columns(protected, X_test, "protected")
        # A one-hot encoded attribute is one attribute: rgf removes its dummies
        # as a unit, so parity must compare its levels, not each dummy's 0/1.
        group_values = (
            X_test[protected_columns[0]]
            if len(protected_columns) == 1
            else labels_from_dummies(X_test, protected_columns)
        )
        block["rga_parity"] = rga_parity(
            y,
            yhat,
            group_values,
            min_group_size=min_group_size,
            level=level,
            method=ci_method,
            n_resamples=n_resamples,
            random_state=random_state,
            attribute=protected,
        ).to_dict()
        if X_train is not None:
            block["rgf"] = rgf(
                X_train,
                X_test,
                model,
                protected,
                yhat=yhat,
                method=rge_method,
                pos_label=pos_label,
                greater_is_better=greater_is_better,
                random_state=random_state,
            )
            block["proxy_leakage"] = proxy_leakage(
                X_train,
                X_test,
                model,
                protected,
                yhat=yhat,
                pos_label=pos_label,
                greater_is_better=greater_is_better,
                random_state=random_state,
            )
        gap = block["rga_parity"]["gap"]
        if gap is None:
            warnings.append(
                "RGA parity gap not computable: fewer than two groups met "
                f"min_group_size={min_group_size}."
            )
        fairness = block

    metadata = {
        "model_name": model_name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n": len(X_test),
        "rgbox_version": __version__,
        "random_state": random_state,
        "level": level,
        "ci_method": ci_method,
        "rge_method": rge_method,
        "perturbation_kind": perturbation_kind,
        "protected_attribute": protected,
        "variables": list(variables) if variables else [],
    }
    return RGBoxReport(
        metadata=metadata,
        accuracy=accuracy,
        explainability=explainability,
        robustness=robustness,
        fairness=fairness,
        warnings=warnings,
    )
