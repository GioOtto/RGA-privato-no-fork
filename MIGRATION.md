# Migrating from `safeaipackage`

## The short version

```bash
pip uninstall safeaipackage
pip install safeaipackage-rgbox[all]
```

Existing code keeps running unchanged — `import safeaipackage` still works and
every function keeps its signature and return shape. Some **numbers will
change**, because several upstream behaviours were defects. They are listed
below, each with the reason and the escape hatch.

New code should use `import rgbox`.

To silence the import-time notice from the compatibility layer:

```bash
export SAFEAIPACKAGE_SILENCE_FORK_NOTICE=1
```

---

## Behaviour changes that can alter your results

### 1. `compute_rga_parity` now uses `yhat`

**Upstream:** the argument was accepted, converted, NaN-checked — and never
read. The function re-predicted internally with `find_yhat`. Passing a vector
of random numbers in place of the real scores returned a bit-identical result.

**Now:** `yhat` is used. If you were relying on the old number, it was the
number for `model.predict_proba(xtest)`, which you can reproduce by passing
`yhat=None`.

**Why it matters:** any calibrated, clipped, post-processed or challenger score
you passed was being discarded.

### 2. `compute_rga_parity` returns a number

**Upstream:** `"The RGA-based imparity between the protected gorups is 0.0188."`
— a string, with a typo.

**Now:** an `ImparityScore`, a `float` subclass. `result < 0.05` works,
`str(result)` still prints a sentence (typo fixed), and `result.result` carries
the full analysis: per-group RGA, confidence intervals, pairwise tests,
subgroup sizes.

```python
gap = compute_rga_parity(xtrain, xtest, ytest, yhat, model, "gender")
if gap > 0.05:                      # would have raised TypeError upstream
    escalate()
for group in gap.result.groups:     # new
    print(group.group, group.n, group.rga, group.ci_low, group.ci_high)
```

### 3. Group levels come from the test set

**Upstream:** levels were enumerated from `xtrain` and used to filter `xtest`.
A level present in training and absent from the test split produced an empty
slice and `ValueError: Found array with 0 sample(s) (shape=(0, k))` raised from
inside scikit-learn.

**Now:** levels come from the evaluation data. Absent levels simply do not
appear; degenerate subgroups are reported with a note instead of crashing.

### 4. A constant target raises instead of returning `nan`

**Upstream:** `rga(np.ones(n), scores)` → `nan`, silently. This is easy to hit
on small fairness subgroups, where a segment turns out to be single-class.

**Now:** `UndefinedMetricError`. It subclasses `ValueError`, so
`except ValueError` still catches it.

```python
from rgbox import UndefinedMetricError
try:
    value = rga(y_segment, scores_segment)
except UndefinedMetricError:
    value = None        # explicit, not silently propagated into a mean
```

### 5. Multiclass models raise instead of being silently mis-scored

**Upstream:** `find_yhat` did `[x[1] for x in model.predict_proba(X)]` — column
1, unconditionally. A three-class model was scored as
`P(class == classes_[1])` with every other class discarded, and no warning.

**Now:** `ModelAdapterError`, with three named ways forward:

```python
predict_scores(model, X, pos_label="default")   # explicit one-vs-rest
rga_ovr(y, model.predict_proba(X), classes=model.classes_)   # OvR average
rga(y, my_own_score_function(X))                # anything you like
```

### 6. Unsupported model types raise a useful error

**Upstream:** `find_yhat` had no `else` branch, so `yhat` stayed unbound and
the caller saw `UnboundLocalError: cannot access local variable 'yhat'` — or,
on scikit-learn ≥ 1.6, `AttributeError: __sklearn_tags__`.

**Now:** `ModelAdapterError` naming what the object is missing. And plain
callables are accepted, so a wrapped internal scoring engine works:

```python
rge(X_train, X_test, lambda frame: my_engine.score(frame), variables=["ltv"])
```

### 7. String and object columns no longer crash variable removal

**Upstream:** `manipulate_testdata` only recognised `pandas.CategoricalDtype`
as non-numeric, so a plain string column fell through to `.mean()` and raised
`TypeError: Could not perform reduction 'mean' with string dtype`.

**Now:** non-numeric columns are replaced by the training **mode**. If you had
worked around this by casting to `category`, you can drop the workaround; the
result is the same.

### 8. `perturb` rejects non-numeric columns

**Upstream:** sorted the column's labels lexicographically and swapped its
"tails" — a permutation with no interpretation, which on a categorical column
typically left the distribution unchanged and so reported near-perfect
robustness.

**Now:** `InputError`. Use `kind="shuffle"` to model a total loss of a
categorical input, or encode it numerically first.

Note also that tail-swapping a **binary** column is near-meaningless even
though it is numeric: it exchanges arbitrary tied values and leaves the mean
untouched. RGR on indicator variables should be read with that in mind.

### 9. `check_nan` raises the `TypeError` its docstring promised

Upstream documented `TypeError` for non-DataFrame input and never raised it.

---

## Things that did *not* change

* `rga(y, yhat)` returns the same value to ~1e-16, on every input shape tested
  — continuous, binary, ties in either or both arguments, counts, negatives.
  The implementation is different; the number is not.
* `compute_rge_values` and `compute_rgr_values` return the same DataFrame
  shape, index and sort order.
* Default `perturbation_percentage=0.05` and the tail-swap scheme are
  unchanged for numeric columns.

---

## Dependency changes

Upstream `setup.py` declared `install_requires=[]` while importing pandas,
numpy, scikit-learn, xgboost and catboost — and `core.py` imported
`util.utils`, which imported CatBoost and XGBoost at module scope. So computing
`rga(y, yhat)`, twenty lines of NumPy on two arrays, required two
gradient-boosting libraries.

| | upstream (undeclared) | this fork |
|---|---|---|
| `numpy` | required | **required** |
| `pandas` | required | extra `[pandas]` — needed for DataFrame workflows and the compat layer |
| `scikit-learn` | required | extra `[sklearn]` — only for `rgbox.sklearn_api` |
| `xgboost` | required | **not needed**, ever |
| `catboost` | required | **not needed**, ever |
| `scipy` | — | optional (adds Kendall's tau to the accuracy report) |

XGBoost and CatBoost models are fully supported — they are duck-typed like
everything else. The test suite exercises `XGBClassifier`, `XGBRegressor`,
`CatBoostClassifier`, `CatBoostRegressor` and CatBoost with string
`cat_features`, and confirms neither library is ever imported.

---

## What you gain by moving to `rgbox`

| Upstream call | `rgbox` equivalent | What is new |
|---|---|---|
| `rga(y, yhat)` | `rga_ci(y, yhat)` | standard error, confidence interval |
| — | `rga_compare(y, a, b)` | paired champion/challenger test |
| — | `rga_test(y, yhat)` | test against "no ranking information" |
| — | `gini_score(y, yhat)` | the `2·AUROC−1` scale, for any target |
| `compute_rge_values(...)` | `rge(..., normalize=True)` | a scale that reaches 1 |
| — | `rge_shapley(...)` | importances that add up under collinearity |
| — | `rge(..., method="retrain")` | the paper's own R-code definition |
| `compute_rgr_values(...)` | `rgr_curve(...)` | AURGR: no arbitrary hyperparameter |
| `compute_rga_parity(...)` | `rga_parity(y, yhat, groups)` | per-group CIs, no model needed |
| — | `rgf(...)` | the R code's fairness measure, absent from the Python package |
| — | `proxy_leakage(...)` | which predictors proxy the protected attribute |
| — | `accuracy_report(...)` | RGA beside AUROC/Spearman/RMSE, with intervals |
| — | `rga_by_segment(...)` | per-portfolio performance with reliability flags |
| — | `rgbox_report(...)` | the whole box as JSON / Markdown / HTML |
| — | `make_rga_scorer()` | `cross_val_score`, `GridSearchCV` |
| — | `rga(y, yhat, weights=w)` | sample weights |
