"""End-to-end model validation for a PD scorecard.

    python examples/bank_model_validation.py

Walks the questions a model-risk function actually asks, in order:

1. How good is the ranking, and how sure are we?           -> accuracy_report
2. Is the challenger genuinely better than the champion?   -> rga_compare
3. Does it hold up per portfolio segment?                  -> rga_by_segment
4. What drives it, and does that survive collinearity?     -> rge / rge_shapley
5. What happens under a data shock?                        -> rgr_curve
6. Does it rank equally well across protected groups?      -> rga_parity
7. Give me a document.                                     -> rgbox_report

Synthetic data, but shaped like the real thing: a lognormal income, a
near-duplicate feature (the collinearity that breaks naive importance), a
string region column, a protected attribute, and a segment where the model
genuinely underperforms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rgbox import (
    accuracy_report,
    compare_models,
    rga_by_segment,
    rga_parity,
    rgbox_report,
    rge,
    rge_shapley,
    rgr_curve,
)

RANDOM_STATE = 20260811


def build_portfolio(n: int = 6000) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(RANDOM_STATE)
    frame = pd.DataFrame({
        "income": rng.lognormal(10.4, 0.55, n),
        "age": rng.integers(21, 78, n).astype(float),
        "dti": rng.beta(2, 5, n),
        "utilisation": rng.beta(2, 3, n),
        "months_on_book": rng.integers(1, 200, n).astype(float),
        "prior_defaults": rng.poisson(0.3, n).astype(float),
        "marketing_channel_id": rng.normal(size=n),        # genuinely irrelevant
        "gender": rng.binomial(1, 0.46, n).astype(float),  # protected
        "region": rng.choice(["north", "centre", "south"], n, p=[.45, .3, .25]),
    })
    # A near-duplicate of an informative predictor: the shape that makes naive
    # variable importance unreadable.
    frame["income_declared"] = frame["income"] * (1 + rng.normal(0, 0.03, n))

    log_odds = (
        -1.1 * np.log(frame["income"] / 30_000)
        + 3.2 * frame["dti"]
        + 1.4 * frame["utilisation"]
        + 0.55 * frame["prior_defaults"]
        - 0.004 * frame["months_on_book"]
    )
    # The southern book is genuinely harder to rank: extra unmodelled noise.
    log_odds = log_odds + np.where(
        frame["region"] == "south", rng.normal(0, 2.2, n), 0.0
    )
    log_odds -= log_odds.mean()
    default = rng.binomial(1, 1.0 / (1.0 + np.exp(-log_odds))).astype(float)
    return frame, default


def main() -> None:
    frame, default = build_portfolio()
    design = pd.get_dummies(frame, columns=["region"], drop_first=True).astype(float)

    split = 4000
    X_train, X_test = design.iloc[:split], design.iloc[split:]
    y_train, y_test = default[:split], default[split:]
    regions_test = frame["region"].iloc[split:]

    # Scaled, because `income` runs to six figures and lbfgs would not
    # converge. The pipeline is also the point: rgbox scores it exactly like a
    # bare estimator, because models are duck-typed rather than matched against
    # a list of supported classes.
    champion = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=3000)
    ).fit(X_train, y_train)
    challenger = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, random_state=RANDOM_STATE
    ).fit(X_train, y_train)

    champion_scores = champion.predict_proba(X_test)[:, 1]
    challenger_scores = challenger.predict_proba(X_test)[:, 1]

    rule = "=" * 78

    print(rule)
    print("1. DISCRIMINATORY POWER")
    print(rule)
    print(accuracy_report(y_test, champion_scores))
    print()
    print("   The Gini line is the number the scorecard report already quotes.")
    print("   Unlike AUROC it would still be defined if the target were an LGD")
    print("   or a loss amount rather than a 0/1 default flag.")

    print()
    print(rule)
    print("2. CHAMPION vs CHALLENGER (paired)")
    print(rule)
    comparison = compare_models(
        y_test,
        {"logistic (champion)": champion_scores, "gbm (challenger)": challenger_scores},
        baseline="logistic (champion)",
        random_state=RANDOM_STATE,
    )
    for row in comparison["ranking"]:
        print(f"   {row['model']:<22} RGA {row['rga']:.4f}  "
              f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]   Gini {row['gini']:+.4f}")
    for row in comparison["vs_baseline"]:
        verdict = "REPLACE" if row["significant"] and row["difference"] > 0 else "KEEP"
        print(f"\n   {row['model']} - champion = {row['difference']:+.4f} "
              f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}], p = {row['p_value']:.4g}")
        print(f"   -> decision: {verdict} the champion")
    print("\n   Paired, not two independent intervals: both models score the same")
    print("   rows, so their errors are correlated and the difference is far")
    print("   better determined than either level.")

    print()
    print(rule)
    print("3. PERFORMANCE BY SEGMENT")
    print(rule)
    print(f"   {'segment':<10}{'n':>7}{'RGA':>9}{'95% CI':>20}{'reliable':>10}")
    for row in rga_by_segment(y_test, champion_scores, regions_test, min_size=200):
        if row.get("rga") is None:
            print(f"   {row['segment']:<10}{row['n']:>7}{'n/a':>9}")
            continue
        interval = f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
        print(f"   {row['segment']:<10}{row['n']:>7}{row['rga']:>9.4f}{interval:>20}"
              f"{row['reliable']!s:>10}")
    print("\n   The southern book is genuinely harder - and the intervals show")
    print("   the gap is not an artefact of its smaller size.")

    print()
    print(rule)
    print("4. WHAT DRIVES THE RANKING")
    print(rule)
    drivers = [c for c in design.columns if c != "gender"]
    print("   Marginal RGE (each predictor removed on its own):")
    for item in rge(X_train, X_test, champion, drivers, yhat=champion_scores)[:6]:
        print(f"      {item.label:<24}{item.rge:.4f}")
    print()
    print("   Note income vs income_declared: two copies of the same information.")
    print("   Marginal importance cannot split credit between them, and the pair")
    print("   evaluated jointly can score *below* either member. Shapley values")
    print("   are efficient by construction, so they can:")
    shapley = rge_shapley(
        X_train, X_test, champion, drivers[:8], yhat=champion_scores,
        normalize=True, random_state=RANDOM_STATE,
    )
    for name, value in shapley.ranked()[:6]:
        print(f"      {name!s:<24}{value:+.4f}")
    print(f"      {'sum':<24}{sum(shapley.values.values()):+.4f}"
          f"   (= coalition worth {shapley.total:.4f})")

    print()
    print(rule)
    print("5. ROBUSTNESS TO A DATA SHOCK")
    print(rule)
    print(f"   {'variable':<24}{'AURGR':>8}   RGR across the shock grid")
    for curve in rgr_curve(
        X_test, champion, ["income", "dti", "utilisation", "marketing_channel_id"],
        yhat=champion_scores,
    ):
        trace = " ".join(f"{v:.2f}" for v in curve.values)
        print(f"   {curve.label:<24}{curve.aurgr:>8.4f}   {trace}")
    print("\n   AURGR summarises the whole curve, so nobody has to defend the")
    print("   choice of a single perturbation size.")

    print()
    print(rule)
    print("6. FAIRNESS: EQUAL RANKING QUALITY ACROSS GROUPS")
    print(rule)
    parity = rga_parity(
        y_test, champion_scores, X_test["gender"],
        attribute="gender", random_state=RANDOM_STATE,
    )
    for group in parity.groups:
        print(f"   gender={group.group:<6} n={group.n:<6} RGA {group.rga:.4f} "
              f"[{group.ci_low:.4f}, {group.ci_high:.4f}]")
    print(f"\n   {parity}")
    print(f"\n   {parity.GAP_CI_NOTE}")
    print("\n   And the caveat that belongs in the report: this is AUC parity -")
    print("   equal ranking quality per group. It is not demographic parity and")
    print("   not equalised odds, and implies neither.")

    print()
    print(rule)
    print("7. THE ARTEFACT")
    print(rule)
    report = rgbox_report(
        y=y_test, X_test=X_test, X_train=X_train, model=champion,
        yhat=champion_scores,
        variables=["income", "dti", "utilisation", "marketing_channel_id"],
        protected="gender", model_name="PD scorecard v3.1",
        random_state=RANDOM_STATE,
    )
    from pathlib import Path

    out = Path("reports")
    out.mkdir(exist_ok=True)
    (out / "validation.json").write_text(report.to_json(), encoding="utf-8")
    (out / "validation.md").write_text(report.to_markdown(), encoding="utf-8")
    (out / "validation.html").write_text(report.to_html(), encoding="utf-8")
    print("   wrote reports/validation.{json,md,html}")
    print("   Seeded throughout, so re-running produces byte-identical output -")
    print("   which is what makes quarterly re-validations diffable.")


if __name__ == "__main__":
    main()
