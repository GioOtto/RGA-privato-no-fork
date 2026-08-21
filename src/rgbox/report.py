"""One command, one validation artefact.

Model risk management wants a reproducible document, not a notebook: the same
inputs must produce the same numbers, every number must carry its uncertainty,
and the whole thing must serialise so it can be diffed between quarterly
re-validations. :func:`rgbox_report` runs the whole Rank Graduation Box and
returns a plain dictionary; :meth:`RGBoxReport.to_markdown` and
:meth:`RGBoxReport.to_html` render it.

Determinism: pass ``random_state`` and every resampling step is seeded, so two
runs on the same data produce the same *numbers*. The artefact as a whole is
byte-identical only if ``generated_at`` is also pinned - by default it is
``datetime.now(timezone.utc)``, which differs between runs. Pass
``generated_at="2026-03-31"`` (or any string, or ``""``) to get a genuinely
diffable file; see :func:`rgbox_report`.
"""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .accuracy import accuracy_report
from .exceptions import RGBoxError
from .explainability import rge
from .fairness import labels_from_dummies, proxy_leakage, rga_parity, rgf
from .outcomes import outcome_parity
from .predictors import predict_scores, resolve_columns
from .robustness import rgr_curve

__all__ = ["RGBoxReport", "rgbox_report"]

_SCHEMA_VERSION = "1.1"

#: Said in the artefact itself whenever no threshold or decision vector was
#: supplied, because "the full box" reads as a complete fairness audit and,
#: without this section, it is not one.
_NO_OUTCOME_FAIRNESS = (
    "Outcome-based fairness was not evaluated: no 'decisions' or 'threshold' "
    "was supplied, so this report contains no demographic parity, equal "
    "opportunity, equalised odds or disparate impact. RGA parity is AUC "
    "parity and does not stand in for any of them."
)


def _escape(text: str) -> str:
    """HTML-escape one text fragment, quotes included."""
    return html.escape(str(text), quote=True)


def _md_cell(value: Any) -> str:
    """One Markdown *table cell*, structurally inert.

    ``to_html`` escapes, so a hostile group level could no longer become
    markup - but ``to_markdown`` interpolated caller-supplied text straight
    into pipe-delimited rows. A model name, column label or protected-attribute
    level containing ``|`` split one cell into two and misaligned every
    subsequent column; one containing a newline ended the table outright and
    left the remaining rows as body text. Neither needs malice: ``"P&L | Q3"``
    is an ordinary segment name.

    A backslash-escaped pipe renders as a literal ``|`` in every CommonMark
    implementation, and newlines collapse to spaces because a table cell has
    nowhere to put them.
    """
    text = str(value)
    text = text.replace("\\", "\\\\").replace("|", "\\|")
    return " ".join(text.split())


def _split_md_row(line: str) -> list[str]:
    """Split one ``|``-delimited Markdown row on its *unescaped* pipes."""
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for character in line.strip().strip("|"):
        if escaped:
            current.append(character if character in "|\\" else "\\" + character)
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "|":
            cells.append("".join(current))
            current = []
        else:
            current.append(character)
    if escaped:
        current.append("\\")
    cells.append("".join(current))
    return cells


def _md_text(value: Any) -> str:
    """One Markdown text fragment that cannot open a new block.

    Same problem as :func:`_md_cell` outside a table: a model name containing a
    newline followed by ``## `` emitted a heading into the report it was
    supposed to title, and any caller-supplied string could add arbitrary block
    structure the same way - including into ``to_html``, which renders the
    markdown and so promoted the injected line to a real ``<h2>``. Collapsing
    the whitespace keeps every fragment on the line it was interpolated into,
    where it can only ever be text.

    Pipes are left alone here: they are structural inside a table row and
    nowhere else, so escaping them outside one would only put backslashes into
    prose. Table cells go through :func:`_md_cell` instead.
    """
    return " ".join(str(value).split())


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

    @property
    def _ci_label(self) -> str:
        """``"95% CI"``, or whatever level was actually used.

        This was the literal string ``"95% CI"`` in every heading and every
        parenthesis, while ``level=`` was passed faithfully to the estimators.
        A report computed at 90% was therefore labelled as a 95% one - the
        numbers right, the caption wrong, in the artefact whose whole purpose
        is to be quoted.
        """
        level = self.metadata.get("level", 0.95)
        percent = level * 100.0
        text = (
            f"{percent:.0f}" if abs(percent - round(percent)) < 1e-9 else f"{percent:g}"
        )
        return f"{text}% CI"

    def to_markdown(self) -> str:
        out: list[str] = []
        add = out.append
        meta = self.metadata
        ci_label = self._ci_label
        add(
            f"# Rank Graduation Box report - {_md_text(meta.get('model_name', 'model'))}"
        )
        add("")
        add(f"- generated: {_md_text(meta.get('generated_at'))}")
        add(f"- evaluation sample: n = {meta.get('n')}")
        add(f"- rgbox version: {_md_text(meta.get('rgbox_version'))}")
        add(f"- random_state: {_md_text(repr(meta.get('random_state')))}")
        add("")

        acc = self.accuracy
        rga_block = acc["rga"]
        add("## Accuracy")
        add("")
        add(f"| metric | value | {ci_label} |")
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
                add(f"| {_md_cell(name)} | {value:+.4f} | |")
        add("")

        if self.explainability:
            add("## Explainability (RGE)")
            add("")
            # This table used to carry a "share of total" column: each RGE
            # divided by the sum of the RGEs. Marginal RGEs are not additive
            # and not monotone in the coalition - the module docstring of
            # rgbox.explainability measures a pair whose joint RGE (0.072) is
            # below *both* of its members (0.638, 0.246) - so their sum is not
            # a total, and a percentage of it is not a share of anything. It
            # rendered as a decomposition of importance, which is exactly what
            # it is not. rge_shapley() produces contributions that do add up,
            # by construction; this table reports the marginal sensitivity it
            # actually measures.
            add("| variable | RGE | RGA(full vs reduced) |")
            add("|---|---|---|")
            for row in self.explainability:
                add(
                    f"| {_md_cell(row['label'])} | {row['rge']:.4f} | "
                    f"{row['rga_reduced']:.4f} |"
                )
            add("")
            add(
                "> RGE is a *marginal* ranking sensitivity: how far the ranking "
                "moves when this predictor alone is removed. These values are "
                "not additive and not monotone in the group - a pair can score "
                "below both of its members - so they do not decompose the "
                "model's importance and must not be read as shares of a total. "
                "Use `rge_shapley()` when contributions that sum to the total "
                "are needed."
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
                    f"| {_md_cell(row['label'])} | {row['aurgr']:.4f} | "
                    f"magnitude {worst[0]:g} -> RGR {worst[1]:.4f} |"
                )
            add("")

        if self.fairness:
            add("## Fairness")
            add("")
            parity = self.fairness.get("rga_parity")
            if parity:
                add(f"Protected attribute: `{_md_text(parity.get('attribute'))}`")
                add("")
                add(f"| group | n | RGA | {ci_label} | in gap |")
                add("|---|---|---|---|---|")
                for group in parity["groups"]:
                    if group["rga"] is None:
                        add(
                            f"| {_md_cell(group['group'])} | {group['n']} "
                            "| - | - | no |"
                        )
                    else:
                        add(
                            f"| {_md_cell(group['group'])} | {group['n']} | "
                            f"{group['rga']:.4f} | {group['ci_low']:.4f} - "
                            f"{group['ci_high']:.4f} | "
                            f"{'yes' if group['included'] else 'no'} |"
                        )
                add("")
                if parity["gap"] is not None:
                    ci = ""
                    if parity["gap_ci_low"] is not None:
                        ci = (
                            f" ({ci_label} {parity['gap_ci_low']:.4f} - "
                            f"{parity['gap_ci_high']:.4f})"
                        )
                    add(
                        f"**RGA parity gap: {parity['gap']:.4f}{ci}**, "
                        f"p = {parity['gap_p_value']:.4g} "
                        f"(family-wise, {parity['multiplicity']})"
                    )
                    if parity.get("gap_excess_over_noise") is not None:
                        add("")
                        add(
                            f"Gap in excess of sampling noise: "
                            f"{parity['gap_excess_over_noise']:+.4f} "
                            f"(noise floor at these group sizes: "
                            f"{parity['gap_noise_floor']:.4f}). "
                            "Zero or below means the spread is no larger than "
                            "exact parity would produce by itself."
                        )
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
                    add(f"> {_md_text(parity['gap_ci_note'])}")
                    add("")
            if self.fairness.get("rgf"):
                block = self.fairness["rgf"]
                add(
                    f"RGF (ranking reliance on `{_md_text(block['attribute'])}`): "
                    f"{block['rgf']:.4f} - RGE {block['rge']:.4f}"
                )
                add("")
            leakage = self.fairness.get("proxy_leakage")
            if leakage:
                add("### Proxy leakage")
                add("")
                model_row = leakage.get("model_leakage")
                if model_row and model_row.get("leakage") is not None:
                    add(
                        f"The model's own score ranks "
                        f"`{_md_text(model_row['level'])}` "
                        f"with RGA {model_row['rga']:.4f} "
                        f"(leakage {model_row['leakage']:.4f})."
                    )
                    add("")
                scored = [
                    row for row in leakage["proxies"] if row.get("rga") is not None
                ]
                if scored:
                    add("| predictor | level | RGA | leakage |")
                    add("|---|---|---|---|")
                    for row in scored[:10]:
                        add(
                            f"| {_md_cell(row['variable'])} | "
                            f"{_md_cell(row['level'])} | "
                            f"{row['rga']:.4f} | {row['leakage']:.4f} |"
                        )
                    add("")
                if leakage.get("ordinal_note"):
                    add(f"> {_md_text(leakage['ordinal_note'])}")
                    add("")
                add(
                    "> Leakage is `|2*RGA - 1|` between the predictor and the "
                    "protected attribute: 0 means the predictor carries none "
                    "of it, 1 means it reconstructs it exactly."
                )
                add("")
            self._add_outcome_parity(add, ci_label)
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
                add(f"- {_md_text(item)}")
            add("")
        return "\n".join(out)

    def _add_outcome_parity(self, add, ci_label: str) -> None:
        """Render the outcome-based block, or say plainly that there is none.

        The silent-omission case is the one that matters. ``rgbox_report`` is
        advertised as "the full box", and a fairness section holding only RGA
        parity, RGF and proxy leakage reads as a completed fairness audit while
        containing no outcome-based criterion at all - the exact misreading the
        fairness module's own docstring warns against. When no threshold or
        decision vector was supplied, that absence is now stated in the
        artefact rather than left to be inferred from a missing heading.
        """
        assert self.fairness is not None
        block = self.fairness.get("outcome_parity")
        if not block:
            add("### Outcome-based fairness")
            add("")
            add(f"> **Not evaluated.** {_md_text(_NO_OUTCOME_FAIRNESS)}")
            add("")
            return

        threshold = block.get("threshold")
        cutoff = "supplied 0/1 decisions" if threshold is None else f"{threshold:g}"
        add(f"### Outcome-based fairness (decision cut-off: {_md_text(cutoff)})")
        add("")
        add(f"| criterion | conditioned on | gap | {ci_label} | p (max-T) |")
        add("|---|---|---|---|---|")
        for criterion in block["criteria"].values():
            if criterion["gap"] is None:
                add(
                    f"| {_md_cell(criterion['name'])} | "
                    f"{_md_cell(criterion['conditioned_on'] or 'all rows')} "
                    "| - | - | - |"
                )
                continue
            interval = (
                "-"
                if criterion["gap_ci_low"] is None
                else f"{criterion['gap_ci_low']:+.4f} - {criterion['gap_ci_high']:+.4f}"
            )
            p_value = (
                "-"
                if criterion["gap_p_value"] is None
                else f"{criterion['gap_p_value']:.4g}"
            )
            add(
                f"| {_md_cell(criterion['name'])} | "
                f"{_md_cell(criterion['conditioned_on'] or 'all rows')} | "
                f"{criterion['gap']:.4f} | {interval} | {p_value} |"
            )
        add("")
        if block.get("equalized_odds") is not None:
            add(
                f"Equalised odds (the larger of the TPR and FPR gaps): "
                f"{block['equalized_odds']:.4f}"
            )
            add("")
        if block.get("disparate_impact") is not None:
            met = block.get("four_fifths_rule_met")
            verdict = "meets" if met else "fails"
            add(
                f"**Disparate impact ratio: {block['disparate_impact']:.4f}** - "
                f"{verdict} the four-fifths rule."
            )
            add("")
        for note in block.get("notes", []):
            add(f"> {_md_text(note)}")
            add("")
        add(f"> {_md_text(block['interpretation'])}")
        add("")

    def to_html(self) -> str:
        """Minimal self-contained HTML rendering of the markdown.

        Every piece of text is escaped. Almost all of it is numbers this module
        formatted itself, but not all: model names, column labels, group levels
        and warning text come from the caller's data, and none of that is under
        this package's control. Interpolated raw, a level spelled
        ``<img src=x onerror=...>`` in a protected attribute stopped being a
        table cell and became markup the moment somebody opened the report -
        stored HTML injection, in a document produced to be circulated.

        Table rows are split on *unescaped* pipes only, and each cell is then
        un-escaped, so a label containing ``|`` survives the round trip as one
        cell holding a literal pipe. Splitting on every pipe - which is what a
        plain ``split("|")`` does - would have turned ``PD \\| v3.1`` into two
        cells and shifted every column after it, reintroducing in HTML the
        misalignment :func:`_md_cell` exists to prevent in Markdown.
        """
        body: list[str] = []
        in_table = False
        for line in self.to_markdown().splitlines():
            stripped = line.strip()
            if stripped.startswith("|"):
                cells = [c.strip() for c in _split_md_row(stripped)]
                if set("".join(cells)) <= set("-: "):
                    continue
                if not in_table:
                    body.append("<table>")
                    in_table = True
                    tag = "th"
                else:
                    tag = "td"
                row = "".join(f"<{tag}>{_escape(c)}</{tag}>" for c in cells)
                body.append(f"<tr>{row}</tr>")
                continue
            if in_table:
                body.append("</table>")
                in_table = False
            if stripped.startswith("### "):
                body.append(f"<h3>{_escape(stripped[4:])}</h3>")
            elif stripped.startswith("## "):
                body.append(f"<h2>{_escape(stripped[3:])}</h2>")
            elif stripped.startswith("# "):
                body.append(f"<h1>{_escape(stripped[2:])}</h1>")
            elif stripped.startswith("> "):
                body.append(f"<blockquote>{_escape(stripped[2:])}</blockquote>")
            elif stripped.startswith("- "):
                body.append(f"<li>{_escape(stripped[2:])}</li>")
            elif stripped:
                body.append(f"<p>{_escape(stripped)}</p>")
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
    decisions: Any = None,
    threshold: float | None = None,
    model_name: str = "model",
    generated_at: Any = None,
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

    Outcome-based fairness
    ----------------------
    ``decisions`` and ``threshold`` drive :func:`rgbox.outcome_parity` -
    demographic parity, equal opportunity, predictive equality, equalised odds
    and disparate impact. Pass ``decisions`` already at 0/1, or pass
    ``threshold`` to cut ``yhat`` (or ``decisions``, if that is a score vector)
    at an explicit value. There is deliberately no default cut-off.

    Without one of them the section is **absent, and said to be absent**, in
    ``warnings`` and in the rendered artefact. Until 1.0.2 the report offered
    no way to ask for these criteria at all, so "the full box" shipped a
    fairness section made entirely of ranking measures - and the fairness
    module's own text says RGA parity must be reported *alongside* outcome
    criteria, never instead of them.

    Determinism
    -----------
    ``generated_at`` fixes the timestamp written into the metadata. The default
    is the current UTC time, which means two runs of an otherwise fully seeded
    report are **not** byte-identical - the README, this module and the worked
    example all claimed they were, and the package's own determinism test had
    to strip the metadata before comparing. Pass any string (a value date, or
    ``""``) to get an artefact that really does diff cleanly between quarterly
    re-validations.
    """
    from . import __version__

    warnings: list[str] = []

    def section(name: str, build):
        """Run one section; on a typed failure, note it and carry on.

        Accuracy is deliberately *not* wrapped: it is the report, and a report
        with no accuracy block is not a partial report but a failed one.
        Everything else is a section that can be missing.

        This boundary existed only around proxy leakage, so any other section
        raising took the whole run with it - a single string column among
        ``variables`` reached ``perturb``'s numeric check and destroyed the
        accuracy, explainability and fairness blocks that had already been
        computed. Only :class:`rgbox.RGBoxError` is caught: a TypeError or a
        MemoryError from inside a user-supplied model is a bug, not a section
        that does not apply, and swallowing it would be worse than the crash.
        """
        try:
            return build()
        except RGBoxError as exc:
            warnings.append(f"{name} skipped: {exc}")
            return None

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
        explainability = (
            section(
                "explainability",
                lambda: [
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
                ],
            )
            or []
        )
    elif variables:
        warnings.append("explainability skipped: X_train was not supplied.")

    robustness: list[dict[str, Any]] = []
    if variables:
        robustness = (
            section(
                "robustness",
                lambda: [
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
                ],
            )
            or []
        )
    else:
        warnings.append("explainability and robustness skipped: no 'variables'.")

    fairness: dict[str, Any] | None = None
    if protected is None and (decisions is not None or threshold is not None):
        warnings.append(
            "outcome parity skipped: 'decisions'/'threshold' were supplied but "
            "'protected' was not, and every outcome criterion is a comparison "
            "between the levels of a protected attribute."
        )
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
        parity = section(
            "RGA parity",
            lambda: rga_parity(
                y,
                yhat,
                group_values,
                min_group_size=min_group_size,
                level=level,
                method=ci_method,
                n_resamples=n_resamples,
                random_state=random_state,
                attribute=protected,
            ).to_dict(),
        )
        if parity is not None:
            block["rga_parity"] = parity
            if parity["gap"] is None:
                warnings.append(
                    "RGA parity gap not computable: fewer than two groups met "
                    f"min_group_size={min_group_size}."
                )
        if X_train is not None:
            rgf_block = section(
                "RGF",
                lambda: rgf(
                    X_train,
                    X_test,
                    model,
                    protected,
                    yhat=yhat,
                    method=rge_method,
                    pos_label=pos_label,
                    greater_is_better=greater_is_better,
                    random_state=random_state,
                ),
            )
            if rgf_block is not None:
                block["rgf"] = rgf_block
        # Proxy leakage needs the protected attribute to be *numeric*, because
        # it ranks it with RGA. The rest of the fairness block does not - a
        # string column is a perfectly ordinary grouping variable for
        # rga_parity - so a categorical attribute must cost the report this
        # one section, not the whole run.
        leakage = section(
            "proxy leakage", lambda: proxy_leakage(X_test, protected, yhat=yhat)
        )
        if leakage is not None:
            block["proxy_leakage"] = leakage

        # Outcome-based criteria: what the model *does* at a cut-off, as
        # opposed to how well it ranks. Not derivable from anything above.
        if decisions is not None or threshold is not None:
            outcomes = section(
                "outcome parity",
                lambda: outcome_parity(
                    y,
                    yhat if decisions is None else decisions,
                    group_values,
                    threshold=threshold,
                    min_group_size=min_group_size,
                    level=level,
                    n_resamples=n_resamples,
                    random_state=random_state,
                    attribute=protected,
                ).to_dict(),
            )
            if outcomes is not None:
                block["outcome_parity"] = outcomes
        else:
            warnings.append(_NO_OUTCOME_FAIRNESS)
        fairness = block

    metadata = {
        "model_name": model_name,
        "generated_at": (
            datetime.now(timezone.utc).isoformat(timespec="seconds")
            if generated_at is None
            else str(generated_at)
        ),
        "n": len(X_test),
        "rgbox_version": __version__,
        "random_state": random_state,
        "level": level,
        "ci_method": ci_method,
        "rge_method": rge_method,
        "perturbation_kind": perturbation_kind,
        "protected_attribute": protected,
        "decision_threshold": threshold,
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
