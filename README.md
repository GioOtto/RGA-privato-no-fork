# rgbox — the Rank Graduation Box, rebuilt for production model validation

A fork of [`safeaipackage`](https://github.com/GolnooshBabaei/safeaipackage), the
reference implementation of the S.A.F.E. AI metrics of Babaei, Giudici and
Raffinetti. **Same measures, same published definitions, same numbers** — plus
the statistics layer they never had, and without the defects that made the
original hard to deploy.

```python
from rgbox import rga_ci, rga_compare

rga_ci(y_test, champion_scores)
# RGA = 0.6961 (95% CI 0.6733-0.7189, SE 0.0116, jackknife, n=2000)

rga_compare(y_test, challenger_scores, champion_scores)
# RGA(A) = 0.6855, RGA(B) = 0.6961, difference = -0.0106
# (95% CI -0.0232..+0.0019), p = 0.0966 [not significant, jackknife, n=2000]
```

(Real output of `examples/bank_model_validation.py`, not an illustration.)

That second line is the whole point: the challenger *looks* worse, but the gap
is not distinguishable from noise, so the decision is "keep the champion" and
you can say why. Upstream would have told you `0.6855` and `0.6961` and left
you to guess.

---

> ## ⚠️ This code was written by an AI
>
> Every line in `src/`, `tests/`, `benchmarks/` and `docs/` was written by
> Claude (Anthropic), including this README. A human directed the work, chose
> the scope and reviewed the output; a human did not type the code.
>
> The mathematics was derived and then **checked numerically** rather than
> asserted: 354 tests, agreement with the original implementation to ~1e-16,
> agreement of the analytic standard error with DeLong's estimator to 0.3%,
> and confidence-interval coverage measured by simulation. What is verified is
> stated as verified; what is not is listed in
> [§ Open questions](docs/THEORY.md#5-open-questions).
>
> **If you would rather not run AI-generated code, this fork is not for you** —
> use [the original](https://github.com/GolnooshBabaei/safeaipackage) or the
> authors' newer [`safe-ai-metrics`](https://safeai.readthedocs.io). No hard
> feelings, and no argument: that is a legitimate policy position, particularly
> in a regulated environment where code provenance is itself an audit item.
>
> <details><summary><b>In italiano</b></summary>
>
> Tutto il codice in `src/`, `tests/`, `benchmarks/` e `docs/` — questo README
> compreso — è stato scritto da un'intelligenza artificiale (Claude, Anthropic).
> Una persona ha diretto il lavoro, definito lo scopo e revisionato il
> risultato; non lo ha scritto a mano.
>
> La matematica è stata derivata e poi **verificata numericamente**, non
> asserita: 354 test, accordo con l'implementazione originale a ~1e-16, accordo
> dello standard error analitico con lo stimatore di DeLong allo 0.3%, e
> copertura degli intervalli di confidenza misurata per simulazione. Ciò che è
> verificato è dichiarato come tale; ciò che non lo è sta in
> [§ Open questions](docs/THEORY.md#5-open-questions).
>
> **Se preferisci non usare codice generato da un'AI, questo fork non fa per
> te**: usa [l'originale](https://github.com/GolnooshBabaei/safeaipackage) o il
> più recente [`safe-ai-metrics`](https://safeai.readthedocs.io) degli stessi
> autori. È una posizione legittima, a maggior ragione in ambito regolamentato
> dove la provenienza del codice è essa stessa oggetto di audit.
> </details>

> ## ⚠️ Licence warning — read before deploying
>
> The upstream project **has no licence**, so it is "all rights reserved" by
> default and cannot normally clear third-party-software review at a bank.
> This fork's own code is MIT, but the inherited material
> (`upstream_reference/`, `R_codes/`, `examples/employee.xlsx`) is not.
> **[NOTICE.md](NOTICE.md) explains what to do about it** — in short, ask the
> authors for an explicit licence; it is a one-email fix.

---

## Install

```bash
pip install safeaipackage-rgbox[all]      # everything
pip install safeaipackage-rgbox           # numpy only — core + full inference
```

`numpy` is the **only** hard requirement. `pandas` and `scikit-learn` are
extras; `xgboost` and `catboost` are never needed (though both are fully
supported — they are duck-typed, and a CI job asserts neither is imported).

For comparison, upstream declared `install_requires=[]` while importing all
five, and its `core.py` could not be imported without CatBoost and XGBoost
installed.

---

## Why fork at all

Three reasons, in order of importance.

### 1. No published implementation reports uncertainty

Not the original package, not the authors' newer `safe-ai-metrics`, not any
paper in the family. The most recent one (arXiv:2511.23100, Nov 2025) reports
only cross-validation standard deviations. The R simulation scripts in this
repository `library(bootstrap)` and then never call it — they report the
Monte-Carlo `sd()` across 1000 *synthetic* datasets, which is not a quantity
you have when validating one model on one hold-out sample.

So a validation function reading "challenger 0.812 vs champion 0.804" has no
way to know whether to act. And a fairness gap between a 300-obligor segment
and a 12,000-obligor one is mostly noise: at n=300 the RGA standard error is
around 0.03.

This fork supplies standard errors, confidence intervals, a **paired**
champion/challenger test and a permutation test — by three independent methods
that agree with each other and, on binary targets, with DeLong's AUC estimator.

### 2. Verified defects

Nine of them, each reproduced and each now covered by a regression test. See
[MIGRATION.md](MIGRATION.md) for the full list with escape hatches. The four
that would matter most in production:

- `compute_rga_parity` accepted a `yhat` argument, validated it, and **never
  used it** — passing random noise instead of the real scores returned a
  bit-identical result.
- It also returned a **string** (`"The RGA-based imparity between the protected
  gorups is 0.0188."`), unusable in a threshold check or a loop.
- A **multiclass** model was silently scored as `P(class == classes_[1])`, every
  other class discarded, no warning.
- A **constant target** returned `nan` rather than raising — easy to hit on a
  small fairness subgroup that turns out to be single-class, and it propagates.

### 3. Undocumented properties that change how you read the output

Not bugs, but they invalidate the natural reading of the numbers:

- **RGE's grand total is 0.5, not 1.** Remove every predictor and the reduced
  score is constant, so RGA is *exactly* 0.5 and RGE is *exactly* 0.5 — for any
  model on any data. The README's "RGE equal to 1 shows a high contribution"
  describes an unreachable value. (`normalize=True` fixes the scale.)
- **Group RGE is not monotone.** Measured on the test fixture:
  `RGE({income}) = 0.638`, `RGE({income_copy}) = 0.246`,
  `RGE({income, income_copy}) = 0.072`. The pair scores below *both* members.
  So individual RGE values are not additive contributions and must not be
  charted as if they were.
- **RGR depends entirely on an arbitrary default.** Same model, same feature:
  0.966 at magnitude 0.01, 0.568 at 0.50. Nothing justifies 0.05.
- **The paper's own R code defines fairness differently from the Python
  package.** `RGF = RGA(ŷ_full, ŷ_without_protected)` in the R scripts;
  `max−min` RGA across groups in Python. Both are implemented here, under names
  that say which is which.

---

## The one identity everything rests on

Expanding the published definition, the `(n+1)·Σy` terms cancel and RGA
collapses to a ratio of covariances-with-ranks:

```math
\mathrm{RGA} = \frac12 + \frac{\mathrm{cov}\big(y, R(\hat y)\big)}{2\,\mathrm{cov}\big(y, R(y)\big)}
```

i.e. `RGA = (1 + γ)/2` where γ is the **Schechtman–Yitzhaki Gini correlation**.
This is algebra, not approximation — the test suite asserts agreement with the
upstream implementation to 1e-16 on continuous, binary, tied, count, negative
and all-tied data.

It buys three things: speed (one sort per argument, no pandas groupby/merge
round-trip), conditioning (the original sums terms of order `n²·ȳ` and then
subtracts them), and — decisively — **access to inference**, because a Gini
correlation is a smooth functional of the empirical distribution.

It also explains the behaviour: the asymmetry (`rga(a,b) ≠ rga(b,a)`), the 0.5
under independence, and the fact that RGA is Pearson-like in `y` and
Spearman-like in `ŷ`, which is what makes it *not* Spearman's ρ.

**And the practical hook:** for binary targets `RGA == AUROC` exactly, so
`2·RGA − 1` is the *Gini coefficient / Accuracy Ratio* every scorecard
validation report already quotes — but now defined for targets that are not
binary. An LGD, a loss amount, an exposure or a rating notch gets a number on
the scale your committee already reads. That framing appears nowhere in the
source literature and is, for a bank, the reason to adopt the measure at all.

The equality is *checked*, not asserted: `accuracy_report` computes its `auroc`
independently, as a Mann-Whitney rank sum with average ranks for ties, so the
two agreeing is evidence. (It used to be a copy of the RGA estimate, which made
the comparison a tautology — see the changelog.)

Full derivations, including the influence function and the exact leave-one-out
recursion, are in **[docs/THEORY.md](docs/THEORY.md)**.

---

## What's in the box

### Accuracy

```python
from rgbox import rga, gini_score, rga_ci, rga_compare, rga_test, accuracy_report

rga(y, scores)                          # the measure; weights= supported
gini_score(y, scores)                   # 2·RGA − 1, the banking scale
rga_ci(y, scores, method="jackknife")   # estimate + SE + CI
rga_compare(y, a, b)                    # paired test — the DeLong analogue
rga_test(y, scores)                     # H0: no ranking information
accuracy_report(y, scores)              # RGA beside AUROC/Spearman/RMSE/τ
```

Three inference methods, all agreeing to ~1%:

| method | cost | notes |
|---|---|---|
| `"jackknife"` *(default)* | `O(n log n)` | **exact** delete-one values, not the `O(n²)` loop. Deterministic. Also gives pseudo-values and BCa acceleration. |
| `"influence"` | `O(n log n)` | one pass; reproduces DeLong's AUC SE to 0.3% on binary targets |
| `"bootstrap"` | `O(B·n)` | exact multinomial resampling, fully vectorised; percentile / basic / BCa |

Measured 95% coverage: 0.92–0.95 across Gaussian, binary, heavy-tie and
heavy-tailed designs.

### Explainability

```python
from rgbox import rge, rge_shapley

rge(X_train, X_test, model, ["ltv", "dti"], normalize=True, ci=True)
rge(X_train, X_test, model, cols, method="retrain", refit=my_refit)  # the R-code definition
rge_shapley(X_train, X_test, model, cols)   # importances that actually add up
```

Removal strategies: `"mean"` (upstream, but string/categorical columns no
longer crash), `"median"`, `"mode"`, `"permute"`, `"retrain"`.

`"permute"` on a group of columns applies one shared permutation to all of
them, so the rows move as a block and a one-hot encoded attribute stays a valid
one-hot. Independent per-column shuffles leave two dummies both at 1 at rate
`p_i·p_j` — 11.25% of rows on the three-level attribute in the test suite —
scoring the model on data that cannot exist.

### Robustness

```python
from rgbox import rgr, rgr_curve

rgr_curve(X_test, model, ["income"])                     # AURGR — no hyperparameter
rgr(X_test, model, ["income"], kind="gaussian", n_repeats=20)
```

Perturbations: `"tailswap"` (upstream's, vectorised), `"gaussian"` (the newer
literature's `N(0, (0.5σ)²)`), `"shuffle"`.

A stochastic perturbation has two sources of uncertainty — the sampling noise of
RGA and the spread across draws — and reporting only the first understates the
interval. With `n_repeats > 1` and `ci=True` the per-draw intervals are pooled
by **Rubin's rules** (`T = W + (1 + 1/m)·B`), so the interval and the point
estimate describe the same quantity. Seeds are drawn once per repeat and shared
across variables — common random numbers, so two variables are compared under
the same shocks rather than under independent ones.

### Fairness

```python
from rgbox import rga_parity, rgf, proxy_leakage

parity = rga_parity(y, scores, groups)   # per-group RGA + CIs, gap + CI + p-value
rgf(X_train, X_test, model, "gender")    # the R code's definition
proxy_leakage(X_test, "gender", yhat=scores)      # which features proxy it

# A one-hot encoded attribute: pass the dummies as a list, and it is treated
# as one attribute everywhere — removed as a unit, scored level by level.
rgf(X_train, X_test, model, ["region_centre", "region_south"])
```

Multi-level attributes, a `min_group_size` floor so a 12-row segment can't drive
the headline, and an explicit note that `max−min` is non-negative by
construction so its interval never contains 0 — test parity with the p-value.

`gap_noise_floor` says what `max−min` would average under *exact* parity at
these group sizes, and `gap_excess_over_noise` is the gap net of it. It averages
0 under exact parity and can be negative, which reads as "no more spread than
noise alone produces". (It replaces a `gap_bias_corrected` that subtracted the
*bootstrap* mean instead — see the changelog for why that removed only a quarter
to a third of the bias, and grew the gap more often than it shrank it.)

**The gap's p-value is corrected for multiplicity.** `max−min` selects the
widest of `k(k−1)/2` pairs, so referring it to a normal — which is what every
implementation we know of does — rejects far too often. Under *exact* parity,
at a nominal 5%:

| groups | pairs | uncorrected | `gap_p_value` |
|---|---|---|---|
| 2 | 1 | 4.3% | 4.7% |
| 3 | 3 | 13.3% | 6.0% |
| 5 | 10 | **27.3%** | 4.3% |

A library whose whole argument is "do not act on noise" cannot ship a headline
test that fires on a quarter of perfectly fair five-level attributes. The
correction is **max-T**, not Bonferroni: the groups are disjoint samples, so
under H0 their RGA estimates are independent normals with the standard errors
already computed, and the joint null of every pairwise statistic can be
simulated from them directly — no re-estimation, and it exploits the real
correlation between pairs that share a group, so it costs much less power.
A real 0.30 gap across five groups is still detected at p = 5e-4.

Being simulated, `gap_p_value` bottoms out at `1 / (n_resamples + 1)` — read a
saturated value as "smaller than the simulation can measure", and raise
`n_resamples` for a finer figure. `gap_p_value_unadjusted` keeps the raw value,
every entry in `pairwise` carries both, and `multiplicity` records what was
corrected for.

A categorical attribute reaches a design matrix as dummies, so `protected`
accepts a list of them — in `rgf`, in `proxy_leakage`, and in `rgbox_report`.
Removing a single dummy would answer a different question ("does the model use
*this level*"), and for a `drop_first` encoding it is not even well posed for
the reference level. `labels_from_dummies` inverts the encoding so the parity
table compares levels rather than each dummy's 0/1. A tuple is read as a single
`MultiIndex` column label, never as a group; use a list for a group.

**RGA parity is AUC parity**: equal *ranking quality* per group. It is not
demographic parity, not equalised odds, and implies neither. That sentence is
printed in the generated report, not just here.

### Outcome-based fairness

Because the sentence above ends "report it alongside outcome-based criteria,
never instead of them", and until 1.0.1 there was no way to do so:

```python
from rgbox import outcome_parity

result = outcome_parity(y, scores, groups, threshold=0.5)
result["demographic_parity"].gap        # + CI, + family-wise p-value
result["equal_opportunity"].gap         # TPR parity
result["predictive_equality"].gap       # FPR parity
result.equalized_odds                   # the worse of the two
result.disparate_impact                 # min/max selection rate, + four-fifths flag
```

Per-group **Wilson** intervals, a normal interval for each gap, a log-scale
interval for the disparate-impact ratio — all closed form, so they cost nothing
and are deterministic. Fairlearn has had bootstrap intervals for these since
0.11; what is not in it is that the headline p-value carries the **same max-T
correction** as `rga_parity`, because `max−min` over `k` groups selects the
widest of `k(k−1)/2` pairs here exactly as it does there.

**There is no default threshold, deliberately.** Every one of these criteria is
defined on a decision, not a score, and the rest of this package is
threshold-free by design. Pass decisions at 0/1, or pass scores with an explicit
`threshold=`; passing scores without one raises. A fairness number computed at a
cut-off nobody chose is worse than no number, because it looks authoritative.

### Finding the bad cohort instead of guessing it

`rga_by_segment` asks "how does the model do on *these* slices", which presumes
you already know where to look. The failure mode of monitoring is the slice
nobody thought to cut:

```python
from rgbox import worst_cohort

search = worst_cohort(y, scores, X_test, features=["region", "channel", "vintage"])
search.cohorts[0].label     # "region == 'south' AND channel == 'online'"
search.p_value              # accounts for having searched every cohort
```

Exhaustive over single bin conditions and their pairwise intersections, so there
is no heuristic to tune. The selection effect is much larger than the parity
table's — thousands of cohorts, not ten pairs — so the p-value comes from
permuting the **cohort definitions** against the (target, score) pairs, which
keeps every cohort size and the overall RGA intact and tests homogeneity.
Permuting the *score* instead would test "the model has no signal at all", drive
every cohort to 0.5 and leave the test powerless against the case it exists for.

One documented caveat: slicing on a predictor the model **uses** lowers RGA
inside every slice by range restriction alone, and the test correctly reports
that as heterogeneity. Slice on variables the model never saw for a clean read.

### The report

```python
from rgbox import rgbox_report

report = rgbox_report(y=y_test, X_test=X_test, X_train=X_train, model=model,
                      variables=[...], protected="gender", random_state=0)
report.to_json(); report.to_markdown(); report.to_html()
```

Seeded end to end, so re-running gives byte-identical output — which is what
makes quarterly re-validations diffable.

### scikit-learn

```python
from rgbox.sklearn_api import make_rga_scorer, rga_scorer
cross_val_score(model, X, y, scoring=make_rga_scorer())
GridSearchCV(model, grid, scoring=rga_scorer)
```

---

## Performance

One core, pure NumPy, no compiled extension:

| n | `rga` | vs upstream | jackknife CI | bootstrap B=200 |
|---|---|---|---|---|
| 1 000 | 0.15 ms | 23× faster | 0.20 ms | 20 ms |
| 10 000 | 1.4 ms | 6.0× | 1.9 ms | 243 ms |
| 100 000 | 22 ms | 3.5× | 28 ms | 2.7 s |
| 1 000 000 | 311 ms | 3.5× | — | — |

Measured on one machine; the speedup against upstream at small `n` is dominated
by pandas' fixed overhead and so varies a lot with machine and pandas version.
The inference columns got 3.5–4× faster in 1.0.1, in pure NumPy: the exact
jackknife was performing **six** sorts where the derivation says two, because
each rank aggregate re-sorted the same array. A shared sort order fixed it, and
a test now asserts the count.

### Why there is still no compiled extension

Because the profile says it would not pay. `rga` is two sorts and two dot
products, and the sorts are 58–70% of the total, so even an *infinitely fast*
sort caps the gain at 2.4–3.3×:

| n | 2× `argsort` | `rga` total | sort share | ceiling if the sort were free |
|---|---|---|---|---|
| 10 000 | 1.3 ms | 2.3 ms | 58% | 2.4× |
| 100 000 | 18 ms | 28 ms | 65% | 2.9× |
| 1 000 000 | 280 ms | 400 ms | 70% | 3.3× |

A Rust/PyO3 sort is perhaps 2–3× faster than NumPy's, not infinite, so the
realistic gain is under 2× — against giving up the universal pure-Python wheel
for `cibuildwheel` on 3 operating systems × 5 Python versions, on a package
that today installs anywhere with only numpy. Sharing one sort between the
aggregates delivered more than that, for free. `python benchmarks/bench.py`
reproduces every number here, including this argument.

---

## Backward compatibility

The old API still works, call-site unchanged:

```python
from safeaipackage.core import rga
from safeaipackage.check_explainability import compute_rge_values
from safeaipackage.check_fairness import compute_rga_parity
from safeaipackage.check_accuracy import accuracy_table   # restored — see below
```

Same signatures, same return shapes. Some **numbers change** where upstream had
defects — all nine listed in [MIGRATION.md](MIGRATION.md). Notably
`compute_rga_parity` now returns a `float` subclass that still *prints* the
legacy sentence, so `if gap > 0.05:` works and `print(gap)` still reads
naturally.

`check_accuracy` is restored. Its compiled `.pyc` files are still committed
upstream but the source was deleted, leaving the "A" of S.A.F.E. as the only
principle with no module.

---

## Verification

```bash
pip install -e ".[dev]"
pytest                                    # 354 tests
python benchmarks/bench.py
python examples/bank_model_validation.py
```

Tested on CPython 3.11 and 3.12, against pandas 2.1 / 3.0 and scikit-learn
1.4 / 1.9. CI covers 3.9–3.13 on Linux, macOS and Windows, plus a
**numpy-only** job that asserts pandas, scikit-learn, XGBoost, CatBoost and
SciPy are absent and that the core and all three inference methods still run.

Notable test files:

- `tests/test_upstream_parity.py` — re-derives the original algorithm inline and
  asserts agreement to 1e-12
- `tests/test_properties.py` — bounds, monotone invariance, asymmetry,
  `RGA == AUROC == WMW`, the Gini-correlation identity, curve areas
- `tests/test_inference.py` — exact jackknife vs `O(n²)`, exact bootstrap vs
  physical resampling, DeLong agreement, interval coverage, p-value uniformity,
  and the sort count the derivation claims
- `tests/test_upstream_regressions.py` — one test per upstream defect
- `tests/test_missing_labels.py` — a missing group label must be refused, not
  turned into a group of size zero
- `tests/test_gap_noise_floor.py` — the parity gap net of noise must average 0
  under exact parity, which is what the old bias correction never did
- `tests/test_outcomes.py` — outcome criteria against hand-computed rates, plus
  the family-wise error rate of their gap test
- `tests/test_cohorts.py` — the cohort search must find a planted weakness and
  stay quiet on pure noise
- `tests/test_docs_math.py` — every formula in the Markdown must survive
  GitHub's renderer

---

## Citing

The measures are not this fork's invention. Cite the original work:

- Babaei, G., Giudici, P., & Raffinetti, E. (2025). *A Rank Graduation Box for
  SAFE AI.* Expert Systems with Applications, 259, 125239.
  [doi:10.1016/j.eswa.2024.125239](https://doi.org/10.1016/j.eswa.2024.125239)
- Giudici, P., & Raffinetti, E. (2025). *RGA: a unified measure of predictive
  accuracy.* Advances in Data Analysis and Classification, 19(1), 67–93.
  [doi:10.1007/s11634-023-00574-2](https://doi.org/10.1007/s11634-023-00574-2)
- Raffinetti, E. (2023). *A rank graduation accuracy measure to mitigate
  artificial intelligence risks.* Quality & Quantity, 57(Suppl 2), 131–150.
  [doi:10.1007/s11135-023-01613-y](https://doi.org/10.1007/s11135-023-01613-y)

The inference layer in `rgbox/inference.py` is **not** from those papers. It is
derived in [docs/THEORY.md](docs/THEORY.md) and validated numerically; describe
it as such rather than attributing it to the original authors.

## Related

- [`safeaipackage`](https://github.com/GolnooshBabaei/safeaipackage) — upstream
- [`safe-ai-metrics`](https://safeai.readthedocs.io) — the same group's newer
  package (RGA/RGR/RGE with curves and plots; no fairness, no inference)

## Licence

MIT for this fork's own code — but **read [NOTICE.md](NOTICE.md) first**: the
upstream project it derives from carries no licence at all.
