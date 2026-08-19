# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning restarts at 1.0.0 for the fork; upstream's last release was 0.8.3
(14 May 2025).

## [Unreleased]

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
  bank, and worth an email to the authors.
- GitHub Actions CI: 3.9–3.13 on Linux/macOS/Windows, a numpy-only install job,
  `ruff`, and a packaging check.
- Upstream sources preserved verbatim under `upstream_reference/` for diffing;
  nothing in `src/` imports them.
