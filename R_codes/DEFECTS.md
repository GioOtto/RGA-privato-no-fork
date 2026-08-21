# Known defects in the inherited R simulation scripts

**Do not use `Simulation_experiment_{A,B,C,D}.R` as an independent reference
without applying the corrections below.** The scripts run to completion and
produce plausible tables; several of the numbers they produce are not the
quantities the code's own comments say they are.

The four scripts are **verbatim upstream material** and, per
[`NOTICE.md`](../NOTICE.md), are not covered by this fork's MIT licence. They
are therefore left exactly as received: this file documents the defects instead
of patching them, so that the diff against upstream stays empty and the
findings are still on the record. This file is original work and is MIT.

Verified against commit `3b20cdf` of this repository. Line numbers are for
`Simulation_experiment_A.R`; B, C and D carry the same defects within a few
lines of the same positions.

---

## R-01 — The train/test split is stratified on the wrong `Y`

**Severity: high — every downstream number in all four scripts.**

The generation loop binds `Y` as a side effect on each iteration:

```r
for (i in 1:s)          # s = 1000
{
  set.seed(i)
  dataframe <- as.data.frame(mvrnorm(n, mu = mu, Sigma = S))
  Y  <- dataframe[,1]           # <- global, overwritten 1000 times
  ...
  dfs[[i]] <- data
}
```

After the loop, `Y` holds the target of dataset **1000** and nothing else. The
split loop then stratifies on it (line 72):

```r
for (t in 1:s)
{
  set.seed(t)
  training.samples <- createDataPartition(Y, p = 0.8, list = FALSE, times = 1)
  train[[t]] <- dfs[[t]][training.samples,]
  test[[t]]  <- dfs[[t]][-training.samples,]
}
```

`createDataPartition` on a continuous vector splits it into quantile groups and
samples within each, so the indices it returns depend on the *values* of its
argument. Dataset `t` is therefore partitioned using the quantile structure of
dataset 1000. Nothing errors, because all 1000 datasets have `n = 100` rows and
the indices are always in range — the split is simply not the stratified split
the code claims to perform. It is, in effect, an arbitrary fixed index set
reused for all 1000 replicates, which also removes the between-replicate
variation in the split that the simulation design assumes.

Correct form:

```r
training.samples <- createDataPartition(dfs[[t]]$Y, p = 0.8, list = FALSE, times = 1)
```

## R-02 — The split comment says 70/30, the code does 80/20

**Severity: low — documentation, but it is the line a reader quotes.**

Line 68 reads `## Split dataset into train and test (70% train and 30% test)`
directly above `p = 0.8`. The realised split is 80/20.

## R-03 — The ROBUSTNESS section mutates `train[[r]]` in place

**Severity: high — makes the results order-dependent across sections.**

Lines 175–188 overwrite the training data itself rather than a perturbed copy:

```r
train[[r]]$X1 <- replace(train[[r]]$X1, train[[r]]$X1 > quantile(train[[r]]$X1, 0.85), runif(n = 15, min = 15, max = 22))
train[[r]]$X1 <- replace(train[[r]]$X1, train[[r]]$X1 < quantile(train[[r]]$X1, 0.15), runif(n = 15, min = -6, max = -4))
# ... same for X2, X3, X4
```

`train` is never restored. The EXPLAINABILITY section (from line 210) and the
FAIRNESS section (from line 381) then read the *perturbed* training sets, so
the explainability and fairness figures do not describe the same experiment as
the accuracy figures computed before line 129. Running the sections in a
different order, or re-running one of them in a fresh session, gives different
answers.

The FAIRNESS section compounds it, binarising `X1` in place as well (lines
418–419):

```r
train[[r]]$X1[train[[r]]$X1 <  mean(train[[r]]$X1)] <- 0
train[[r]]$X1[train[[r]]$X1 >= mean(train[[r]]$X1)] <- 1
```

Correct form: perturb a local copy and leave the base data immutable.

```r
train_pert <- train[[r]]
train_pert$X1 <- replace(train_pert$X1, ...)
```

Note also that `runif(n = 15, ...)` hard-codes 15 replacement draws while the
number of rows past the 0.85 quantile of an 80-row training set is 12.
`replace()` assigns positionally, so the last three draws are discarded and R
emits *"number of items to replace is not a multiple of replacement length"* on
every one of the 1000 iterations. The count should be derived from the mask,
not written as a literal.

## R-04 — Ranking after `round(yhat, 4)` manufactures ties

**Severity: medium — biases RGA upward on tightly-spaced predictions.**

The in-script `RGA` function ranks the *rounded* predictions:

```r
ryhat <- rank(round(yhat, 4), ties.method = "min")
```

Predictions closer together than 1e-4 collapse into a tie group and are
averaged, which is a different estimator from RGA on the raw scores. On a
regression fit whose predictions span a few units this is harmless; on a
probability output concentrated near 0 or 1 it is not. `rgbox.rga` ranks the
values as given.

## R-05 — `library(bootstrap)` is loaded for a method the scripts never run

**Severity: low — misleading provenance.**

Line 11 reads `library(bootstrap) # package needed for implementing the
Jackknife method`. Neither `jackknife()` nor `bootstrap()` is called anywhere
in any of the four scripts, and no resampling-based standard error is computed.
The only dispersion reported is the Monte-Carlo standard deviation *across the
1000 synthetic datasets* — a quantity nobody has when validating one model on
one hold-out sample, and a different thing from a standard error for RGA. That
gap is what `rgbox.inference` exists to fill.

---

## If these scripts are to be kept as a scientific reference

They need to be rewritten as pure pipelines, which is a larger change than
fixing the five items above and cannot be done to the verbatim copies without
losing their value as a diff target:

1. one replicate generates, splits, fits and scores **its own** immutable
   dataset — no global `Y`, no shared `train` list mutated between sections;
2. separate `generate → split → fit → metric → aggregate`, so a section cannot
   depend on whether an earlier one has run;
3. record the seed and the package versions with the results;
4. add regression tests comparing the R and Python figures on a fixed input.
