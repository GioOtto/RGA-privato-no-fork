# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning restarts at 1.0.0 for the fork; upstream's last release was 0.8.3
(14 May 2025).

## [1.0.1]

### Fixed — statistics

- **`outcome_parity` called the most extreme disparity in the data "not
  significant".** Every test was divided by the *observed* plug-in standard
  error of the group rates. A rate of exactly 0 or exactly 1 has a plug-in
  variance of exactly 0, so a group nobody was selected from compared against
  one where everybody was gave `0 / 0` — read as "no evidence" and reported as
  `p = 1.0` (adjusted and unadjusted alike), with a zero-width confidence
  interval `(1.0, 1.0)`, for a selection-rate gap of 100 points. The max-T
  simulation degenerated with it, because it was fed the same zero errors.

  Tests are now computed under the null they are testing: the common rate is
  pooled across the eligible rows of the included groups, each group's null
  error is `sqrt(p_pooled (1 - p_pooled) / n_g)`, and the pairwise statistic is
  the standard two-proportion score test. The `hypot` of two pooled errors is
  exactly that test's denominator, so the max-T draws now simulate the same
  quantity the statistic is referred to. The 0-vs-100 case goes from `p = 1.0`
  to `p < 1e-40` unadjusted and to the `1 / (n_resamples + 1)` floor adjusted;
  type I error under exact parity stays at nominal.

  Intervals are *not* pooled — a confidence interval must not assume the null —
  and move from Wald to **Agresti-Caffo**, the two-sample analogue of the
  Wilson intervals already used for the per-group rates. The degenerate gap
  interval becomes `(0.953, 1.0)`. `pairwise` records gain a
  `null_standard_error` key; `standard_error` keeps its old unpooled meaning.
- **`rga_test` accepted any `alternative`.** Anything other than `"greater"` or
  `"less"` fell through to the two-sided branch, so `alternative="grater"` ran
  a different test and returned a different p-value (0.265 against 0.868 on the
  suite's own fixture) without complaint. Unknown values now raise.
- **`rga_test` disagreed with itself on a constant score.** The analytic branch
  divided a zero deviation by a zero null SD, called the result 0, and reported
  `p = 0.5` for a one-sided test — halfway to significance for a score carrying
  no information at all. The Monte-Carlo branch raised `ZeroDivisionError`
  computing its own statistic on the same input. The permutation null of a
  constant score is a point mass, so the exact p-value is 1 for every
  alternative; both branches now return that.
- **RGA was not invariant to the scale of `weights`.** Multiplying every weight
  by a positive constant cannot change a ratio whose numerator and denominator
  both scale as its square, but the zero-denominator guard compares against a
  tolerance derived from `y` alone, which does not scale. `weights = [1] * 4`
  was computed; `weights = [1e-8] * 4` — the same relative weights, equally
  legal under every check — was rejected as numerically degenerate. Weights are
  now normalised to mean 1 on the way in; RGA is now identical across 16 orders
  of magnitude of weight scale.
- **`rga_curves` raised `ZeroDivisionError` on a constant positive target.**
  `sum(y) > 0` does not imply dispersion. `rga()` catches this with a typed
  `UndefinedMetricError`; `rga_curves` now applies the same guard to the same
  denominator instead of dividing by a hard zero.
- **`bootstrap_values(block_size=0)` never returned.** `take = min(block_size,
  remaining)` is 0 on every iteration, so the loop counter never advanced — a
  mistyped tuning parameter hung the worker rather than raising.

### Fixed — inverted and silently wrong results

- **`pos_label` was ignored by the `decision_function` adapter.** scikit-learn
  orients a binary margin towards `classes_[1]`; asking for `classes_[0]` means
  the same margin negated. The argument was accepted and the orientation never
  changed, so RGA came back as `1 - RGA` with no error — measured at 0.964
  reported as 0.036 on a `LinearSVC`. A `pos_label` naming a class the model was
  never fitted on was accepted here too, while the `predict_proba` branch
  rejected it.
- **`make_rga_scorer(pos_label=...)` dropped the argument entirely.** It reached
  neither `make_scorer` nor the metric, so `pos_label=0` and `pos_label=1`
  produced scorers that behaved identically. Honouring it needs both the score
  column *and* `y_true` re-expressed relative to the chosen class, so passing it
  now returns a callable scorer built on `predict_scores`.
- **`rge(method=...)` accepted anything.** `RemovalMethod` is a type hint, and
  four methods had explicit branches while *every other string* fell through to
  mean substitution: `method="meam"` returned the mean-substitution RGE, and
  `method="retrian"` returned it instead of demanding the `refit` callable that
  `"retrain"` requires. Now validated against `REMOVAL_METHODS`.
- **`rgr_curve(kind="shuffle")` built a curve over a parameter that does not
  exist.** A shuffle ignores `magnitude`; the call fell back to the gaussian
  grid and, because `rgr` re-derives its seeds from `random_state` on every
  call, evaluated the identical experiment at all six points — six bit-for-bit
  equal values, and an AURGR that averaged one number. Now rejected, with a
  pointer to `rgr(..., kind="shuffle")` for the single benchmark it is.
- **`outcome_parity(threshold=float("nan"))` reported perfect parity.**
  `values >= nan` is False on every row, so the sample became "nobody
  selected", every rate 0 and every gap 0, with no error. Thresholds must now
  be finite.
- **`equalized_odds` is `None` when only one of its two gaps is computable**,
  instead of the survivor plus a note. It is defined as the larger of the TPR
  and FPR gaps; with one missing there is no such quantity, and a consumer
  reading the JSON number would never see the note.

### Fixed — validation and error typing

- `check_count` and `check_finite_scalar` centralise what each counter used to
  get wrong on its own. Newly rejected, having previously hung, crashed inside
  NumPy, or silently returned something plausible:
  `outcome_parity(n_resamples=0)` (returned `p = 1.0` for *every* gap, since
  `(1 + 0) / (0 + 1)` is the answer when the max-T sample is empty),
  `rge_shapley(n_permutations=0)` (`ZeroDivisionError`) and `-1` (every Shapley
  value came back `-0.0`), `worst_cohort(top=-1)` (returned every cohort
  *except the best one*), `worst_cohort(n_permutations=-1)`
  (`ZeroDivisionError` on `-1 + 1`), `worst_cohort(n_bins<2)` (numeric features
  produced no bins at all, and the search reported finding nothing having
  looked at nothing), `contamination_curve(n_repeats=0)` (NaN in every
  contaminated row) and fractions outside `[0, 1]` (raw NumPy `ValueError`),
  and `rga_test(n_permutations=0)` (NaN statistic).
- **`rge_shapley` crashed on MultiIndex columns in the sampled path.** Building
  an object array from a list of equal-length tuples — which is how pandas
  spells MultiIndex columns, and which `resolve_columns` explicitly supports —
  produces a 2-D array, so permuting it yielded unhashable ndarrays where the
  coalition sets need labels. Positions are permuted now, not labels.
- **`as_group_labels` silently `ravel()`ed multidimensional input.** A 2×2 block
  passed by mistake alongside four observations became four group labels in
  row-major order, while the numeric arguments of the same call rejected it.
  Both now use one policy: `(n,)`, `(n, 1)` or `(1, n)`.
- `perturb` validates `kind` and requires a finite `magnitude`.

### Fixed — the report artefact

- **`RGBoxReport.to_html` performed no escaping.** Model names, column labels,
  group levels and warning text come from the caller's data and were
  interpolated raw into `<h1>`, `<td>`, `<p>` and `<blockquote>`, so a level
  spelled `<img src=x onerror=...>` became markup as soon as the report was
  opened — stored HTML injection in a document produced to be circulated.
- **The report printed "95% CI" whatever `level` was.** The level was passed
  faithfully to the estimators and recorded correctly in `metadata` and the
  JSON; only the captions were hard-coded, so a 90% report was labelled as a
  95% one.
- **One failing section destroyed the whole report.** The docstring's promise
  that what cannot be computed is skipped and noted held only for
  `proxy_leakage`. A single string column among `variables` reached `perturb`'s
  numeric check and took the accuracy, explainability and fairness blocks down
  with it. Every section now has its own boundary. Only `RGBoxError` is caught:
  an arbitrary exception from a user-supplied model is a bug, not an
  inapplicable section.

### Fixed — packaging

- **`import safeaipackage` failed on a valid minimal install.** The
  distribution declares numpy as its only hard requirement, but the
  compatibility layer eagerly imported four submodules that import pandas at
  module scope — so `import rgbox` worked and the other top-level package in
  the same wheel did not. Those submodules are now reached lazily, with an
  error naming the missing extra. `safeaipackage.core` is numpy-only and stays
  eager.
- **`py.typed` was missing** despite the `Typing :: Typed` classifier. Added for
  both packages, and a test now asserts it ships in the built wheel.
- **The version is 1.0.1.** Six source files documented behaviour changes "in
  1.0.1" while `pyproject.toml` and `rgbox.__version__` both said 1.0.0 and the
  changelog kept them under `[Unreleased]`. A test now ties the three together.
- **The declared dependency floors were never tested, and two of them were
  wrong.** CI installed whatever pip resolved on the day, which is always the
  newest, so `>=` in `pyproject.toml` was an assertion nobody checked. A
  `minimum-dependencies` job now installs exactly the declared lower bounds,
  and found on its first run:
  - `accuracy_report` raised `AttributeError` on the declared `scipy>=1.8`.
    `kendalltau(...).statistic` is SciPy 1.9 and later; before that the field
    is `.correlation`, which is still kept as an alias. Now reads whichever
    exists, so the 1.8 floor is true.
  - **every `make_rga_scorer()` score was NaN below scikit-learn 1.4.** The
    `response_method` keyword arrived in 1.4; older `make_scorer` forwards
    unrecognised keywords to the *metric* rather than rejecting them, so the
    scorer was constructed successfully and then died at scoring time with
    `rga() got an unexpected keyword argument 'response_method'` — which
    `cross_val_score` converts into NaN. The `except TypeError` fallback that
    was meant to cover this could never fire, because nothing raised at
    construction. Capability is now detected from `make_scorer`'s signature.
  - the scikit-learn floor moves to **`>=1.4`**. Below it, lbfgs used a looser
    stopping rule and a logistic fit on unscaled data terminates at a different
    point (35 iterations against 2611 on the suite's own fixture), so the
    explainability results are not reproducible there. Verified green on
    Python 3.10 with numpy 1.22.4, pandas 1.5.3, scikit-learn 1.4.2 and
    scipy 1.8.1.
- **CI actions are pinned to commit SHAs** rather than to `@v5`-style major
  tags, which the action owner can repoint at any time, and `ruff`, `build` and
  `twine` are installed at pinned versions. The workflow declares
  `permissions: contents: read`.
- **`twine check` rejected the wheel the build job had just produced.** The
  build backend is uncapped (`hatchling>=1.21`), so each run resolves the
  newest hatchling, and the core-metadata version it writes tracks its own
  releases: 1.21 emits `Metadata-Version: 2.1`, 1.27 emits 2.4, 1.32 emits
  2.5. `twine check` refuses any metadata version not on a list held inside
  twine itself — a current transitive `packaging` does not help — so the
  pinned twine 6.1.0 failed the `build` job with `'2.5' is not a valid
  metadata version` on a commit that touched neither the package nor the
  workflow. Pinned checker, floating backend: a break on a timer. twine moves
  to **7.0.0**, the first release accepting 2.5, and the `build` job now
  prints the generating backend and the metadata version *before* checking,
  so the next such mismatch names the stale pin instead of only quoting a
  number. Hatchling is deliberately still uncapped — it is also what users
  build the sdist with.
- The `minimal-install` job now also asserts `import safeaipackage` works — it
  only ever tested `import rgbox`, which is why the pandas-at-module-scope
  regression above went unnoticed — and the `build` job asserts `py.typed`
  is present in the built wheel, not merely in the source tree.

### Added

- **`outcome_parity`** — demographic parity, disparate impact, equal
  opportunity, predictive equality and equalised odds, each with per-group
  Wilson intervals, a gap interval, and a **max-T family-wise p-value**. This
  closes a gap the package had itself declared: `fairness.py` said "report it
  alongside outcome-based criteria, never instead of them" and then offered no
  way to do so. Everything is closed-form — no resampling, no SciPy, no pandas.
  Against Fairlearn (which has had bootstrap intervals since 0.11) the two
  differences are that the intervals are analytic, hence free and
  deterministic, and that the headline p-value is corrected for having selected
  the widest of `k(k−1)/2` pairs.
  - **There is no default threshold, deliberately.** Every one of these
    criteria is defined on a decision rather than a score, and the rest of the
    package is threshold-free by design. Pass `decisions` already at 0/1, or
    pass scores with an explicit `threshold=`; passing scores without one
    raises. A fairness figure computed at a cut-off nobody chose is worse than
    no figure, because it looks authoritative.
  - `equalized_odds` is the larger of the TPR and FPR gaps; `disparate_impact`
    is the min/max selection-rate ratio with a log-scale interval and a
    `four_fifths_rule_met` flag.
- **`worst_cohort`** — searches for the slice the model ranks worst on, instead
  of requiring you to name it first (`rga_by_segment` presumes you already know
  where to look). Exhaustive over single bin conditions and their pairwise
  intersections, so there is no search heuristic to tune.
  - The selection effect here is far larger than the one `rga_parity` faced —
    thousands of cohorts rather than ten pairs — so a **permutation p-value**
    accounts for the whole search. What is permuted matters: the *cohort
    definitions* are permuted against the (target, score) pairs, which keeps
    every cohort size, every overlap and the overall RGA intact and tests
    homogeneity. Permuting the score instead would test "the model has no
    signal at all", drive every cohort to 0.5 and leave the test with no power
    against exactly the case it exists for.
  - Documented, with the measured example: slicing on a predictor the model
    **uses** depresses RGA inside every slice by range restriction alone, and
    the test correctly reports that as heterogeneity. Prefer slicing on
    variables the model never saw.
- `proxy_leakage` gained a `model_leakage` entry when `yhat` is supplied: how
  well the model's *own score* ranks the protected attribute, which is the
  question a reviewer asks straight after seeing which features proxy it. The
  generated report now renders the proxy table, which it previously computed
  and discarded.
- `rgbox._ranks.SortedIndex` — one sort, shared by every aggregate built on it.

### Fixed

- **`rga_parity` raised a bare `StopIteration` when every group scored the same
  RGA.** The same `max()`/`min()` collapse that was fixed in `outcome_parity`
  survived here untouched: both return the *first* extremal element, so a tie
  put `best` on top of `worst`, `{best, worst}` became a one-element set, and
  the widest-pair lookup found no pair. A model that ranks perfectly inside
  every group (RGA 1.0 throughout) reaches it, and `rgbox_report` calls
  `rga_parity` unguarded, so the whole report died with it. The ends now come
  from an RGA-ordered list and the gap is reported as 0.
- **A `pandas.NA` in a categorical column took `worst_cohort` down.**
  `_bin_conditions` computed the `present` mask correctly and then still
  broadcast `arr == level` over the whole column. `NA == "x"` is `NA`, not
  `False`, so numpy asking for its truth value raised `TypeError: boolean value
  of NA is ambiguous` — on `string` dtype, on `category`, and on a plain object
  column, directly contradicting the docstring promise that `pandas.NA` joins
  the `"<column> is missing"` bin. Levels are now grouped in a single pass over
  the present rows, which never touches the missing ones. (`None`, float `NaN`
  and Categorical `NaN` always worked.)
- **A feature that was entirely missing, or constant, was reported as a
  cohort.** Its single bin covered every row, so it scored a shortfall of 0
  against itself and then intersected with every other bin at depth 2 to
  produce an exact duplicate of that bin — 7 "cohorts searched" where 3 were
  distinct, and a `top=10` list half-filled with copies. A bin holding the
  whole sample is the sample, not a cohort, and is now dropped.
- **One NaN turned a numeric feature into a categorical one in `worst_cohort`.**
  `as_1d_float` raises the same `InputError` for "this column is not numeric"
  and for "this numeric column contains a NaN", and `_bin_conditions` read the
  second as the first. The level-per-value branch then emitted **one bin per
  distinct float** — `n` bins instead of `n_bins` — so no bin could clear
  `min_size`, the search reported nothing, and depth 2 spent O(n²) mask
  intersections getting there: measured at n=4000, **34.6 s to search zero
  cohorts**, down to 0.003 s. Rows whose value is missing now join an explicit
  `"<column> is missing"` bin rather than falling out of every bin, so a cohort
  defined by the absence of a field is searched like any other. The same
  treatment is applied on the categorical branch, where NaN levels had the
  `nan != nan` deduplication problem fixed for group labels above.
- **`outcome_parity` raised a bare `StopIteration` when every included group
  had the same rate.** `max()` and `min()` both return the *first* extremal
  element, so `best` collapsed onto `worst`, `{best, worst}` became a
  one-element set, and the widest-pair lookup found nothing. Two ordinary
  inputs reach it: a threshold above every score (every rate is 0), and two
  groups whose counts reduce to the same proportion (60/200 against 120/400).
  The two ends are now taken from a rate-ordered list, which keeps them
  distinct; the gap is reported as 0.
- **A categorical protected column aborted the whole report.** `proxy_leakage`
  ranks the protected attribute with RGA and so needs it numeric; nothing else
  in the fairness block does, and `rga_parity` groups on strings happily. The
  call sat outside any guard, so `rgbox_report(..., protected="region")` raised
  instead of producing a report — against the documented contract that anything
  not computable is skipped and noted in `warnings`. It now costs that one
  section and a warning.
- `Cohort` is a frozen dataclass with an `np.ndarray` field, which poisoned the
  derived `__eq__` and `__hash__`: `a == b` raised on the ambiguous array truth
  value and `hash(a)` on unhashability, taking `CohortSearch` equality and
  `set(result.cohorts)` down with them. The mask is excluded from comparison —
  it is a derived view of `conditions` against the same frame.
- `CohortSearch.SELECTION_NOTE`, which is serialised into `to_dict()` and
  quoted in reports, described a test the code does not run: it said the
  p-value permutes the *score* and takes the *best*-found cohort, where the
  code permutes the cohort definitions and takes the worst. The module
  docstring argues at length that permuting the score is the wrong null.
- `_count_missing` ran `tolist()` plus a Python-level predicate per element on
  every parity, segment and outcome call. Vectorised per dtype, with the object
  path kept as the fallback: **136 ms → 0.8 ms** for a million labels, and the
  fast paths are tested to agree with the fallback element for element.
- **`rga_parity` and `rga_by_segment` invented one empty group per missing
  label, and silently dropped those rows.** `dict.fromkeys(labels.tolist())`
  rebuilds a fresh `float` object per element and `nan != nan`, so NaN labels
  never deduplicated; `labels == nan` was then all-False, giving each one
  `n = 0`. A 2000-row report with 5% missing produced a **102-row** parity
  table, 100 rows of it `nan | 0`, and analysed 1900 rows without a warning.
  Missing labels (`None`, NaN, `pandas.NA`, `NaT`) are now **rejected** with a
  message naming the count, consistent with the package's NaN policy
  everywhere else. Encode missing as an explicit level if it should be
  reported. Object-dtype `None` labels and unused `Categorical` categories
  were unaffected and still are.
- **`gap_bias_corrected` did not remove the bias it claimed to**, and is
  replaced by `gap_excess_over_noise` plus `gap_noise_floor`. The old formula
  subtracted the *bootstrap* mean excess, but the stratified bootstrap
  resamples inside each group around the observed per-group RGAs, so its world
  already contains the observed spread and its mean sits on top of the gap
  rather than above a true zero. Measured under exact parity it removed only
  24% (k=2), 32% (k=3) and 37% (k=5) of the selection bias; with a real gap
  present it came out **larger** than the raw gap in 83 runs out of 150. The
  replacement subtracts the expectation of `max − min` under the null of exact
  parity, simulated from the same independent normals the max-T p-value
  already uses — so it costs nothing, it averages 0 under exact parity (now a
  regression test), and it may be negative, which means the observed spread is
  no larger than noise alone would produce.
- **`proxy_leakage` accepted seven arguments and read none of them.**
  `X_train`, `model`, `yhat`, `method`, `pos_label`, `greater_is_better` and
  `random_state` never appeared in the body: passing a junk training frame, the
  *string* `"not a model at all"` as `model` and random noise as `yhat`
  returned a bit-identical result. That is verbatim the upstream defect this
  fork leads its migration notes with. The signature is now
  `proxy_leakage(X, protected, candidates=None, *, yhat=None)`, `yhat` does
  something, and the old call shape raises with an explanation instead of
  binding a frame to `protected`.
- **`compute_rga_parity` (compatibility layer) was unseeded**, so `gap_ci` and
  `gap_p_value` drifted between two runs on identical data — measured
  `(0.00170, 0.11277)` against `(0.00164, 0.11600)` and p-values 0.7046 against
  0.7136. It now passes `random_state=0`. The gap itself was always
  deterministic.
- **A pandas nullable column with missing values reported the wrong problem.**
  `Int64`/`Float64`/`boolean`/`string` with `pandas.NA` arrives as an object
  array, so the float conversion failed and the error read "has non-numeric
  dtype … encode categories to numbers first". The package already had a good
  message for missing values; it just was not reachable. It is now.
- **`resolve_columns` accepted a `set`, whose iteration order is not stable
  across processes.** `rge_shapley(..., {"a","b","c","d"})` returned its values
  in four different orders under `PYTHONHASHSEED` 0/1/2/3, against a package
  whose headline claim is byte-identical re-runs. Sets are now rejected, with
  the sorted list to pass instead named in the message.
- **The maths in `README.md` and `docs/THEORY.md` did not render on GitHub**,
  in two independent ways, both verified against GitHub's own renderer
  (`POST /markdown`, `mode=gfm`, which returns the payload MathJax receives).
  All 28 formulas now come back intact.
  - `\operatorname` is not on GitHub's MathJax allow-list, so those blocks
    displayed "The following macros are not allowed: operatorname" instead of a
    formula — including the central `RGA = ½ + cov(y,R(ŷ)) / 2cov(y,R(y))`
    identity on the README's front page. Nine occurrences, replaced by
    `\mathrm` with an explicit `\,` where the operator used to supply the
    spacing itself.
  - The larger half: GitHub runs its Markdown pass **before** the maths pass,
    and that pass strips any backslash preceding a non-letter — inside maths as
    much as outside, because `\,` is a legal CommonMark escape. `THEORY.md` had
    **57** such sequences on 22 lines (18 `\,`, 16 `\;`, 8 `\{`, 8 `\}`, 4
    `\!`, 2 `\_`, 1 `\ `), so `\mathbb 1\{Z \ge z_0\}` lost its set braces and
    `\texttt{income\_copy}` acquired a subscript. It was invisible wherever
    `\operatorname` was present, because the banner replaced the whole block —
    fixing the macro alone would have exposed it. Display maths now uses
    ```` ```math ```` fences and fragile inline maths uses `` $`…`$ ``, whose
    contents are code to the Markdown pass.
  - One inline `$…$` region spanned a line break (`THEORY.md` 346–347), which
    GitHub does not recognise as maths at all; it rendered as literal LaTeX
    source. Rewrapped.
  - `tests/test_docs_math.py` now enforces all of this, so it cannot regress.
- **The exact jackknife performed six sorts, not the two `docs/THEORY.md` §2.2
  derives.** `average_ranks`, `suffix_sums_strictly_greater` and
  `tie_group_sums` each re-sorted the same array, and `_leave_one_out_sums`
  called all three, twice over. A shared `SortedIndex` brings
  `jackknife_values` and `influence_values` to 2 sorts and `rga_ci` to 4 (from
  8). `bootstrap_values` also re-sorted once per block and now hoists it. The
  sort count is asserted by a test rather than left to drift.

### Changed — behaviour changes, numbers move

- `rga_parity`'s stratified bootstrap for `gap_ci` now runs through the exact
  vectorised multinomial bootstrap instead of a Python loop over
  `n_resamples × k` calls to `rga`. **The draws differ, so `gap_ci` moves**
  (it is a percentile interval of a resampling distribution; the estimate,
  the gap and the p-value are unchanged). Measured wall time at the defaults:
  4.62 s → 2.08 s for five groups of 1000.
- `ParityResult.gap_bias_corrected` is **gone**, replaced by
  `gap_excess_over_noise` and `gap_noise_floor` — see *Fixed*. The report
  prints the new pair with its interpretation.
- `proxy_leakage`'s signature changed — see *Fixed*. `rgbox_report` no longer
  needs `X_train` to produce a proxy table.
- Missing group/segment labels now raise instead of being silently dropped —
  see *Fixed*.
- A `set` passed as `variables`/`protected`/`features` now raises — see
  *Fixed*.

### Documented — properties that were stated wrongly

- **RGR and AURGR live on `[0, 1]`, not `[0.5, 1]`.** The module claimed 0.5 as
  the floor and "collapses immediately scores near 0.5". A tail swap at
  `magnitude = 0.5` exchanges every value with its mirror, i.e. **reverses the
  column outright** (measured rank correlation with the original: +0.02 at
  0.10, −0.58 at 0.25, −1.00 at 0.50), so a model monotone in that feature
  scores `RGR = 0` at the top of the default grid and `AURGR = 0.217`. Below
  0.5 means the perturbation *inverted* the ranking, which is information, not
  a floor violation. The grid is unchanged — it spans the perturbation's whole
  legal domain, which is what makes AURGR free of a hyperparameter — but the
  scale is now described correctly and AURGR should be compared across
  variables and model versions, not against an absolute 0.5.

### Changed — three behaviour changes, numbers move

- **`rga_parity(...).gap_p_value` is now family-wise corrected.** The gap is
  `max − min` over groups, so its statistic is the largest of `k(k−1)/2`
  pairwise ones and is *selected*, not fixed in advance; referring it to a
  normal rejected far too often. Measured type I error under exact parity, at a
  nominal 5%: 4.3% → 4.7% at two groups, **13.3% → 6.0%** at three, **27.3% →
  4.3%** at five. The correction is **max-T**: groups are disjoint samples, so
  under H0 their RGA estimates are independent normals with the standard errors
  already computed, and the joint null of every pairwise statistic is simulated
  from them directly — no re-estimation, and far less conservative than
  Bonferroni because it uses the real correlation between pairs sharing a
  group. Power is retained: a real 0.30 gap across five groups still gives
  p = 5e-4.
  - The raw value survives as **`gap_p_value_unadjusted`**, and every entry in
    `pairwise` now carries both `p_value` and `p_value_adjusted`.
  - New **`multiplicity`** field records the pair count and draw count.
  - **The adjusted value is simulated, so it cannot fall below
    `1 / (n_resamples + 1)`** — about 5e-4 at the default 2000 draws. It
    saturates on strong disparities where the unadjusted value would report
    1e-15. That is a resolution limit, not a weaker verdict; raise
    `n_resamples` if a specific figure is needed.
  - With two groups there is one pair and nothing to correct, so the two agree
    up to simulation noise.
- **`rgr` and `rgr_curve` draw a different random stream for a given int
  seed**, as a consequence of the `Generator` fix below. One seed is now drawn
  per repeat up front and shared across variables — common random numbers, so
  RGR stays comparable between variables — and a live `Generator` is threaded
  down instead of a per-column integer seed. Stochastic `kind` values
  (`"gaussian"`) move; the default `"tailswap"` is deterministic and unaffected.
- **`rge(..., method="permute", group=True)` now applies a single shared
  permutation to the whole group** instead of an independent shuffle per
  column. Rows are reordered as a block, so every removed column still comes
  from the same original row. Independent shuffles fabricated rows that cannot
  exist — two dummies both land on 1 at rate `p_i·p_j`, 11.25% of rows on the
  three-level attribute in the test suite — and destroyed the joint structure a
  group of correlated columns exists to hold fixed. **Group `permute` values
  from 1.0.0 are not comparable with these.** Single columns and every other
  `method` are unaffected. `rge_shapley` with `method="permute"` moves too, on
  its coalitions of two or more predictors; its singleton coalitions already
  drew exactly this permutation, so the two paths were and remain consistent.

### Added

- `protected` accepts a **list of one-hot dummies** in `rgf`, `proxy_leakage`
  and `rgbox_report`, and the attribute is then treated as one attribute
  throughout: removed as a unit by `rgf`, scored level by level by
  `proxy_leakage`, and compared by reconstructed level in the parity table.
  Removing one dummy at a time answers a different question, and under a
  `drop_first` encoding it is not well posed for the reference level.
- `labels_from_dummies` — inverts a one-hot encoding back to one label per row,
  which is what `rga_parity` needs. All-zero rows become the `reference` level
  a `drop_first` encoding omits, rather than being dropped, which would have
  silently excluded the largest group from the gap.
- `proxy_leakage` rows carry a `level` key naming the level that produced the
  reported figure, and the protected columns are excluded from the default
  candidate list.
- The report's parity block now names its headline p-value as family-wise, says
  how many pairs it corrects for, and prints the uncorrected value beside it —
  a bare `p = 0.03` is unreadable once two p-values exist, and showing only the
  adjusted one hides the size of the penalty.

### Fixed

- **`accuracy_report` reported `auroc` by copying the RGA estimate**, so the
  agreement between the two was a tautology and the figure was not an
  independent check of anything. It is now computed on its own as a
  Mann-Whitney rank sum over the positives, with average ranks for ties.
  Verified against `sklearn.roc_auc_score` to 1e-12, including ties and
  non-0/1 labels. Values are unchanged where the identity holds — that is the
  point — but the number is now evidence rather than a restatement.
- **`rgr(n_repeats>1, ci=True)` reported the confidence interval of the last
  draw next to a point estimate that was the mean of all of them.** The two
  now describe the same quantity: `_pool_across_draws` combines the per-draw
  intervals by **Rubin's rules** (`T = W + (1 + 1/m)·B`), so the interval
  carries both the sampling uncertainty of RGA and the spread across
  perturbation draws, and `estimate.method` records the pooling. With `m == 1`
  it reduces exactly to the previous single interval, so the default path is
  untouched.
- **`rgr` and `rgr_curve` raised `TypeError` when `random_state` was a
  `numpy.random.Generator`** — the documented contract, and the same defect
  already fixed in `explainability.py` but never carried across. The
  per-column seed was computed as `int(random_state) + offset`.
- **`rga_by_segment` aborted the whole call on a segment with fewer than three
  rows**, taking every other segment's result down with it. It caught
  `(UndefinedMetricError, InputError)`, but `as_score_pair` raises
  `InsufficientDataError`, which is a *sibling* of `InputError` under
  `RGBoxError` rather than a subclass, so it escaped the handler. Such
  segments are now reported with `rga: None` and a note, like every other
  unusable segment. Audited the package for the same gap; this was the only
  instance.
- `rgf` expanded a tuple column label into one column per element, so a pandas
  `MultiIndex` frame raised `InputError: column(s) [...] are not in the data`.
  `resolve_columns` now resolves an exact label match first, everywhere: a
  tuple that *is* one of the frame's columns is that column, and a list is the
  only way to spell a group.
- `rge(..., method="permute")` raised `TypeError` when `random_state` was a
  `numpy.random.Generator` rather than an int, because the per-column seed was
  computed as `int(random_state) + offset`.

## [1.0.0] — 2026-08-11

Fork of `safeaipackage` 0.8.3. Written by an AI system; see the README.

### Added — statistical inference (no prior implementation has any)

- `rga_ci` — standard error and confidence interval by three independent
  methods: exact `O(n log n)` jackknife (default), plug-in influence function,
  and an exact vectorised multinomial bootstrap with percentile / basic / BCa
  intervals. They agree to ~1%; the influence-function SE reproduces DeLong's
  AUC estimator to 0.3% on binary targets. Measured 95% coverage 0.92–0.95.
- `rga_compare` — **paired** champion/challenger test. Differences the
  per-observation contributions (pseudo-values, influence values, or replicates
  under shared resampling weights) so the correlation between two models scored
  on the same rows is used, not discarded.
- `rga_test` — test of "no ranking information", with the exact permutation
  moments in closed form (no simulation) plus a Monte-Carlo variant.
- `jackknife_values`, `influence_values`, `bootstrap_values` as public building
  blocks.

### Added — measures and reporting

- `gini_score` — `2·RGA − 1`, the Gini coefficient / Accuracy Ratio of
  scorecard validation, generalised to non-binary targets.
- `rga_curves` — the Lorenz / dual Lorenz / concordance curves, with areas that
  reproduce the closed form exactly.
- `rge_shapley` — Shapley values of the game `v(S) = RGE(S)`; efficient by
  construction, so importances add up under collinearity. Exact below 12
  predictors, permutation-sampled above.
- `rge(..., method="retrain")` — the definition used by the paper's own R code,
  alongside `"mean"`, `"median"`, `"mode"` and `"permute"`.
- `rge(..., normalize=True)` — rescales so the attainable maximum really is 1.
- `rgr_curve` / **AURGR** — robustness summarised over a grid of perturbation
  magnitudes, removing the arbitrary `perturbation_percentage`.
- `perturb(kind="gaussian" | "shuffle")` — the noise-based scheme of the more
  recent literature, and a total-loss-of-input reference.
- `rga_parity` — per-group RGA with confidence intervals, gap with a stratified
  bootstrap interval and a bias-corrected point estimate, all pairwise tests,
  and a `min_group_size` floor.
- `rgf` — the R code's fairness measure, `RGA(ŷ_full, ŷ_without_protected)`,
  which the Python package never implemented.
- `proxy_leakage` — ranks predictors by how strongly they proxy a protected
  attribute.
- `rga_ovr` — one-vs-rest RGA for multiclass targets.
- `rga_by_segment` — per-portfolio performance with reliability flags.
- `contamination_curve` — quantifies the papers' unmeasured claim that RGA is
  more outlier-robust than RMSE (at 1% contamination: RGA moves 23%, RMSE 780%).
- `rgbox_report` — the whole box as a deterministic, seeded JSON / Markdown /
  HTML artefact.
- `rgbox.sklearn_api` — `make_rga_scorer`, `rga_scorer`, `gini_scorer` for
  `cross_val_score` and `GridSearchCV`.
- Sample weights on `rga`; integer weights reproduce the replicated sample
  exactly.
- `safeaipackage.check_accuracy`, restored: its `.pyc` files are still committed
  upstream but the source was deleted.

### Fixed — defects verified against upstream 0.8.3

- `compute_rga_parity` accepted a `yhat` argument and never used it, re-predicting
  internally. Passing random noise returned a bit-identical result.
- `compute_rga_parity` returned a formatted string with a typo
  (`"...protected gorups is 0.0188."`) instead of a number.
- `compute_rga_parity` enumerated group levels from `xtrain` but filtered
  `xtest`, so a train-only level produced an empty slice and a `ValueError`
  from inside the estimator.
- `rga` returned `nan` on a constant target; now raises `UndefinedMetricError`.
- `find_yhat` scored multiclass models as `P(class == classes_[1])`, silently
  discarding every other class; now raises, with `pos_label=` and `rga_ovr` as
  the documented paths.
- `find_yhat` had no `else` branch, so unsupported models produced
  `UnboundLocalError` (or `AttributeError: __sklearn_tags__` on scikit-learn
  ≥ 1.6); now `ModelAdapterError`, and plain callables are accepted.
- `manipulate_testdata` only recognised `pandas.CategoricalDtype` as
  non-numeric, so string/object columns raised
  `TypeError: Could not perform reduction 'mean' with string dtype`.
- `perturb` sorted non-numeric columns lexicographically and returned a
  meaningless permutation; now rejected, with `kind="shuffle"` as the
  alternative.
- `check_nan` documented a `TypeError` it never raised.
- NumPy arrays passed where column names were needed produced the misleading
  `"'a' is not in the variables"`; the message now names the real problem.

### Changed

- **Dependencies.** `numpy` is the only hard requirement. Upstream declared
  `install_requires=[]` while importing pandas, scikit-learn, XGBoost and
  CatBoost — and `core.py` imported `util.utils`, which imported the two
  boosting libraries at module scope, so `rga(y, yhat)` on two arrays needed
  both installed. XGBoost and CatBoost models remain fully supported by duck
  typing; a CI job asserts neither is ever imported.
- **Core rewritten in closed form.** `RGA = ½ + cov(y, R(ŷ)) / (2 cov(y, R(y)))`
  — the Schechtman–Yitzhaki Gini correlation, up to affine rescaling. Identical
  values (~1e-16), 2.2× to 73× faster, and better conditioned: on a target
  offset by 1e12 the upstream form loses about 1.5 more decimal digits.
- **Typed errors.** `RGBoxError` and subclasses replace bare `ValueError`,
  silent `nan` and `UnboundLocalError`. All subclass `ValueError` where
  relevant, so `except ValueError` in existing code keeps working.
- Models are duck-typed rather than matched against an enumerated list of
  classes; any callable `X -> scores` works.
- `safeaipackage` is now a thin compatibility layer over `rgbox`, emitting a
  one-time notice (silence with `SAFEAIPACKAGE_SILENCE_FORK_NOTICE=1`).

### Documented — properties that change how the output should be read

- RGE's grand coalition is *exactly* 0.5, never 1, for any model on any data:
  a constant reduced score has all ranks tied, so RGA is exactly 0.5. Individual
  values are **not** capped at 0.5 — 0.638 is observed in the test suite when
  removing a predictor inverts rather than flattens the ranking.
- Group RGE is not monotone. On the test fixture `RGE({income}) = 0.638`,
  `RGE({income_copy}) = 0.246`, `RGE({income, income_copy}) = 0.072` — the pair
  scores below both of its members.
- RGR sweeps 0.966 → 0.568 across the legal range of `perturbation_percentage`;
  the 0.05 default is unjustified.
- Tail-swapping a binary column leaves its mean unchanged and swaps arbitrary
  tied values, so RGR on indicator variables carries little information.
- The paper's R code defines fairness as `RGA(ŷ_full, ŷ_without_protected)`,
  not as the `max−min` gap the Python package computes.
- RGA parity is *AUC parity*, not demographic parity or equalised odds; the
  caveat is printed in the generated report.
- `max−min` is non-negative by construction, so its percentile interval never
  contains 0 even under exact parity.

### Repository hygiene

- `.gitignore` rewritten — upstream's had every rule commented out, so
  `__pycache__/`, `.ipynb_checkpoints/` and a stray `fastapi.cpython-38.pyc` in
  the repository root were all tracked. Those are now untracked.
- `pyproject.toml` (hatchling, src layout) replaces the `setup.py` that existed
  only inside the PyPI tarball and never in the repository.
- `LICENSE` added for this fork's code, and `NOTICE.md` explains that the
  upstream project has none — a hard stop for third-party-software review at a
  regulated institution, and worth an email to the authors.
- GitHub Actions CI: 3.9–3.13 on Linux/macOS/Windows, a numpy-only install job,
  `ruff`, and a packaging check.
- Upstream sources preserved verbatim under `upstream_reference/` for diffing;
  nothing in `src/` imports them.
