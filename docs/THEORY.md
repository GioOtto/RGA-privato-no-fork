# Theory notes

Everything this package computes follows from one algebraic identity and one
smoothness argument. This file states both, derives the inference layer, and
records what was checked numerically. It is written to be checkable, not to be
taken on trust: every claim has a corresponding test.

Notation: $n$ observations $(y_i, z_i)$ with $z = \hat y$ the model score,
$R(\cdot)$ the average-rank function (ties get the group mean), $\bar y$ the
sample mean.

---

## 1. RGA, and why it is a Gini correlation

### 1.1 The published definition

Rank the sample by $z$. Let $y^{*}$ be $y$ reordered accordingly, with tied
scores replaced by the conditional mean of $y$ within the tie group. Then

```math
\mathrm{RGA} \;=\; \frac{\sum_i i\,y^{*}_i - \sum_i i\,y_{(n+1-i)}}
                        {\sum_i i\,y_{(i)} - \sum_i i\,y_{(n+1-i)}}
```

with $y_{(\cdot)}$ the ascending order statistics. Geometrically: the area
between the dual Lorenz curve and the concordance curve, over the area between
the Lorenz and dual Lorenz curves.

### 1.2 Collapsing the sums

Write $\Psi_z = \sum_i y_i R(z_i)$ and $\Psi_y = \sum_i y_i R(y_i)$.

*Concordance term.* Within a tie group of $z$ all members carry the same value
$y^{*}$ (the group mean), so summing $` i\,y^{*}_i `$ over the group equals the
group's $y$-mass times the group's mean position, and the mean 0-based position
of a tie group is $R - 1$. Hence

```math
\textstyle\sum_i i\,y^{*}_i \;=\; \Psi_z - \sum_i y_i .
```

*Lorenz term.* The same argument with $v = y$ (here it is exact rather than an
averaging step, because tied $y$ values are literally equal):

```math
\textstyle\sum_i i\,y_{(i)} \;=\; \Psi_y - \sum_i y_i .
```

*Dual term.* Reversing the order maps position $i \mapsto n-1-i$, so
$`\sum_i i\,y_{(n+1-i)} = (n-1)\sum_i y_i - \sum_i i\,y_{(i)}`$.

Substituting, the $\sum_i y_i$ terms collect into $(n+1)\sum_i y_i$ in both
numerator and denominator:

```math
\mathrm{RGA}
= \frac{\Psi_z + \Psi_y - (n+1)\sum_i y_i}{2\Psi_y - (n+1)\sum_i y_i}.
```

Since $(n+1)/2 = \bar R$ is the mean rank,
$`(n+1)\sum_i y_i = 2n\,\bar y\,\bar R`$, and both
$`\Psi - n\bar y\bar R = n\,\mathrm{cov}(y, R)`$. Therefore

```math
\boxed{\;\mathrm{RGA} \;=\; \frac12 \;+\;
\frac{\mathrm{cov}\!\big(y,\,R(\hat y)\big)}
     {2\,\mathrm{cov}\!\big(y,\,R(y)\big)}\;}
```

Equivalently $\mathrm{RGA} = (1+\gamma)/2$ with $\gamma$ the **Schechtman–Yitzhaki
Gini correlation** of $y$ on $\hat y$. Implementation:

```python
centred = y - y.mean()
rga = 0.5 + centred @ rank(yhat) / (2 * (centred @ rank(y)))
```

*Checked:* `tests/test_upstream_parity.py` asserts agreement with the original
pandas implementation to $10^{-12}$ across continuous, binary, tied-in-$y$,
tied-in-$\hat y$, tied-in-both, Poisson-count, negative-valued and
all-tied samples, and at $n = 3, 4, 7, 15, 100$.

### 1.3 What the identity explains

| Observed property | Explanation |
|---|---|
| $\mathrm{RGA}(a,b) \neq \mathrm{RGA}(b,a)$ | Gini correlations are asymmetric by construction: Pearson-like in the first argument, Spearman-like in the second. |
| $\mathrm{RGA} = 0.5$ under independence | $`\mathrm{cov}(y, R(\hat y)) = 0`$. |
| Bounds $[0,1]$ | Rearrangement inequality: $\sum y_i R(z_i)$ is maximised by the concordant arrangement (value $\Psi_y$) and minimised by the antithetic one. Tie-averaging is a convex combination, so it stays inside. |
| Invariance to increasing transforms of $\hat y$ | Only $R(\hat y)$ enters. |
| *Non*-invariance to monotone transforms of $y$ | $y$ enters through its values, not only its ranks. This is what distinguishes RGA from Spearman's $\rho$. |
| Numerical conditioning | The published form sums terms of order $n^2\bar y$ and subtracts them; centring first removes that cancellation. |

### 1.4 Relation to measures already in use

For binary $`y \in \{0,1\}`$, $\mathrm{RGA} = \mathrm{AUROC}$ exactly, ties
included, hence also the normalised Wilcoxon–Mann–Whitney statistic. So

```math
\text{Gini (Accuracy Ratio)} = 2\,\mathrm{RGA} - 1 = 2\,\mathrm{AUROC} - 1 = D_{yx},
```

Somers' D. The practical content: **RGA is the credit-risk Gini coefficient,
extended to targets that are not binary.** A loss amount, an LGD, an exposure or
a rating notch gets a number on the same scale the validation report already
uses. That is the reason to adopt it, and it is not stated this way anywhere in
the source literature.

---

## 2. Inference

No published work on RGA supplies a standard error, a confidence interval or a
hypothesis test. The most recent paper in the family (arXiv:2511.23100,
November 2025) reports only cross-validation standard deviations; the R
simulation scripts in this repository load the `bootstrap` package and never
call it, reporting instead the Monte-Carlo `sd()` across 1000 *synthetic*
datasets — a quantity unavailable when validating one model on one hold-out
sample.

RGA is a smooth functional of the empirical distribution, so the standard
machinery applies.

### 2.1 Influence function

Let $G$ be the CDF of $Z=\hat y$, $F$ that of $Y$, $\mu = E[Y]$, and

```math
N = \mathrm{cov}(Y, G(Z)) = \psi - \mu/2, \qquad
  D = \mathrm{cov}(Y, F(Y)) = \varphi - \mu/2,
```

with $\psi = E[YG(Z)]$, $\varphi = E[YF(Y)]$, so $\theta = 1/2 + N/(2D)$.

Perturbing $P_\varepsilon = (1-\varepsilon)P + \varepsilon\delta_{(y_0,z_0)}$
and differentiating at $\varepsilon = 0$ — the CDF is itself estimated, which
contributes the second term —

```math
\mathrm{IF}_\psi(y_0,z_0) = y_0G(z_0) + E\!\left[Y\mathbb 1\{Z \ge z_0\}\right] - 2\psi,
```

and $\mathrm{IF}_\mu = y_0-\mu$, giving

```math
\mathrm{IF}_N = y_0G(z_0) + E[Y\mathbb 1\{Z\ge z_0\}] - 2\psi - \tfrac{y_0-\mu}{2},
```

$\mathrm{IF}_D$ identically with $F$ and $Y$ in place of $G$ and $Z$, and by the
delta method

```math
\mathrm{IF}_\theta = \frac{\mathrm{IF}_N}{2D} - \frac{N\,\mathrm{IF}_D}{2D^2},
\qquad \widehat{\mathrm{SE}} = \frac{\mathrm{sd}(\mathrm{IF})}{\sqrt n}.
```

**Mid-distribution convention.** The plug-in uses
$`\hat G(z_i) = (R(z_i) - \tfrac12)/n`$, i.e. $P(Z<z)+P(Z=z)/2$, rather than
$R(z_i)/n$. The two differ by $1/(2n)$. That is asymptotically irrelevant but
not harmless: with the naive version a constant (perfectly uninformative) score
picks up a spurious $O(1/n)$ variance instead of the exactly zero it must have.
With the mid-CDF the constant-score standard error comes out at $5\times10^{-18}$.

*Checked:* on binary targets RGA is the AUC, so DeLong's estimator is an
external oracle. Ratio of our SE to DeLong's: 0.998 at $n=200$, 0.999 at
$n=600$, 0.9997 at $n=3000$. Naive $R/n$ gave 1.10 / 1.08 / 1.01 — the
convention matters more than the sample size.

### 2.2 Exact jackknife in $O(n\log n)$

Delete-one is usually $O(n^2)$: recompute the statistic $n$ times. But deleting
observation $k$ changes ranks in a completely predictable way — every value
strictly above $z_k$ drops one rank, every value tied with $z_k$ drops half a
rank, everything below is unaffected:

```math
R^{-k}(z_i) = R(z_i) - \mathbb 1\{z_i > z_k\} - \tfrac12\mathbb 1\{z_i = z_k,\, i \neq k\}.
```

Therefore, with $T_k = \sum_{z_i > z_k} y_i$ and $U_k = \sum_{z_i = z_k} y_i$
(a suffix sum and a segment sum over one sort),

```math
S_k \;=\; \sum_{i\neq k} y_i R^{-k}(z_i) \;=\; \Psi_z - y_kR(z_k) - T_k - \tfrac12(U_k - y_k),
```

and since average ranks over $n-1$ elements always sum to $(n-1)n/2$,

```math
\mathrm{RGA}^{-k} = \frac12 + \frac{S_k^{(z)} - \bar y_{-k}\frac{(n-1)n}{2}}
                                {2\left(S_k^{(y)} - \bar y_{-k}\frac{(n-1)n}{2}\right)}.
```

The whole delete-one family costs two sorts — one per argument, shared between
the rank, suffix-sum and tie-group aggregates by `rgbox._ranks.SortedIndex`.
It used to cost six: each aggregate re-sorted the same array, which is the kind
of thing a derivation states and an implementation quietly does not do.
`tests/test_inference.py` now asserts the sort count, so the claim and the code
cannot drift apart again. *Checked* against the naive $O(n^2)$ computation: max
absolute deviation $\le 10^{-11}$ on every sample shape in the fixture set,
including heavy ties.

This buys the default standard error, the bias estimate, the pseudo-values used
for paired comparison, and the acceleration constant for BCa intervals.

### 2.3 Exact bootstrap without resampling

A bootstrap replicate is a multinomial reweighting of the original sample, and
weighted average ranks have a closed form: for a tie group of total weight $W$
preceded by cumulative weight $C$, every member ranks at $C + (W+1)/2$. So a
replicate never materialises a resampled array, and — decisively — **the sort
order does not depend on the replicate**. Sorting once and applying
`np.add.reduceat` over a $(B, n)$ weight block computes $B$ replicates in one
vectorised pass.

*Checked:* `_multinomial_rga` reproduces `rga(np.repeat(y, counts), np.repeat(z, counts))`
to $10^{-11}$, on data with ties in both arguments.

### 2.4 Paired model comparison

Champion versus challenger is a *paired* problem: both scores are computed on
the same rows, so their RGA estimates are strongly positively correlated and
comparing two independent intervals is badly conservative. All three methods
difference the per-observation contributions before taking a variance:
pseudo-values, influence values, or replicates under *shared* resampling
weights. On a binary target this is the rank-graduation analogue of DeLong's
test for correlated AUCs.

*Checked:* on two models sharing most of their signal, the paired SE is under
half the naive $\sqrt{\mathrm{SE}_A^2 + \mathrm{SE}_B^2}$.

### 2.5 A permutation test with exact moments

Under $H_0$ (the score carries no ranking information) the ranks are
exchangeable with respect to $y$. The permutation variance of a sum of paired
products is known in closed form:

```math
\mathrm{Var}_\pi\!\left(\sum_i a_i b_{\pi(i)}\right)
= \frac{\sum_i (a_i-\bar a)^2 \sum_i (b_i-\bar b)^2}{n-1},
```

so with $a = y - \bar y$ and $b = R(\hat y)$ the null standard deviation of RGA
is available without simulating anything. *Checked:* agrees with a 4000-draw
Monte-Carlo permutation to within 0.03 in p-value; the p-value distribution is
uniform under the null (rejection rate 0.02–0.10 at the 5% level over 400
replications).

### 2.6 Coverage

Nominal 95% intervals, 400–500 replications per design:

| design | coverage |
|---|---|
| Gaussian, $n=200$ | 0.92 |
| Gaussian, $n=1000$ | 0.94 |
| binary, $n=500$ | 0.95 |
| lognormal, $n=300$ | 0.95 |

Slightly conservative-to-nominal. The $n=200$ Gaussian case is the weakest;
prefer the BCa bootstrap below $n \approx 300$.

---

## 3. The derived measures, and their pathologies

### 3.1 RGE

$\mathrm{RGE}(S) = 1 - \mathrm{RGA}(\hat y, \hat y^{-S})$.

**The grand coalition is pinned to 0.5.** Remove every predictor and the reduced
score is constant; all its ranks tie; $\mathrm{RGA} = 0.5$ *exactly*, so
$\mathrm{RGE} = 0.5$ — never 1, for any model on any data. The documented
$[0,1]$ scale therefore misrepresents what the numbers can do, and raw RGE
values are not comparable across datasets. `normalize=True` rescales by 2.

**Individual values are not capped at 0.5.** If removing a predictor *inverts*
the ranking rather than flattening it, $\mathrm{RGA} < 0.5$ and
$\mathrm{RGE} > 0.5$. Measured on the test fixture: 0.638 for a single
predictor.

**It is not monotone.** On the same fixture:

| coalition | RGE |
|---|---|
| $`\{\texttt{income}\}`$ | 0.638 |
| $`\{\texttt{income\_copy}\}`$ | 0.246 |
| $`\{\texttt{income}, \texttt{income\_copy}\}`$ | **0.072** |

The pair scores below *both* of its members. Two near-collinear predictors are
fitted with opposing coefficients; deleting one destroys the balance, deleting
both cancels the damage. Consequence: individual RGE values are marginal, not
additive, and a bar chart of them is misleading under collinearity.

**Shapley fixes the additivity.** Treat $v(S) = \mathrm{RGE}(S)$ as a
cooperative game. Shapley values are efficient by construction — they sum to
$v(\text{all predictors}) = 0.5$, or 1 normalised — and give symmetric players
equal credit. Because the game is non-monotone, individual Shapley values may
be negative; that is information, not a bug.

**Mean substitution is not retraining.** The R code accompanying the paper
retrains (`lm(Y ~ . - X1)`); the Python package substitutes the mean. These
estimate different things: substitution holds the other coefficients at values
fitted *in the presence of* the removed predictor, and pushes rows off the data
manifold when features are correlated (the Hooker–Mundler critique). Both are
available: `method="mean"` and `method="retrain"`.

### 3.2 RGR

$\mathrm{RGR} = \mathrm{RGA}(\hat y, \hat y^{\text{perturbed}})$, and the answer
depends entirely on how hard you push. On one fitted model and one feature:

| tail fraction | 0.01 | 0.05 | 0.10 | 0.20 | 0.50 |
|---|---|---|---|---|---|
| RGR | 0.966 | 0.860 | 0.781 | 0.652 | 0.568 |

The upstream default of 0.05 is unjustified and makes results incomparable
across studies. **AURGR** — the trapezoidal area under
magnitude $\mapsto$ RGR, anchored at $\mathrm{RGR}(0)=1$ and normalised by the
grid width — is a single number with no free parameter.

**The scale is $[0,1]$, and 0.5 is the middle of it, not the bottom.** RGR is
an RGA, so a perturbation that *inverts* the ranking scores below 0.5, and the
tail swap reaches that by construction: at magnitude $m$ it exchanges the lowest
$m$ fraction with the highest, so at $m = 0.5$ it exchanges every value with its
mirror — a verbatim reversal of the column. Measured rank correlation with the
original, on a 10-row column: $+0.02$ at $m = 0.10$, $-0.58$ at $0.25$, $-1.00$
at $0.50$. Consequently, for a model monotone in the perturbed feature, the top
of the default grid gives $\mathrm{RGR} = 0$ and AURGR lands near 0.22.

That is the measure behaving correctly over a perturbation that runs from a mild
tail shock to a sign flip, not a floor being violated — but it means AURGR must
be compared across variables and across model versions, never against an
absolute threshold of 0.5. Earlier versions of this file and of the module
docstring described the scale as $[0.5, 1]$; that was wrong.

The tail-swap scheme is also meaningless on discrete inputs: on a 0/1 indicator
it exchanges arbitrary tied values and leaves the mean unchanged; on a string
column the original sorted lexicographically. Numeric columns only, or use
`kind="shuffle"` / `kind="gaussian"` (the latter matching the newer literature's
$\varepsilon \sim N(0,(0.5\sigma_{\hat y})^2)$).

### 3.3 Fairness

The Python package computes $\max_g \mathrm{RGA}_g - \min_g \mathrm{RGA}_g$.
**The R code accompanying the paper computes something else entirely**:
$\mathrm{RGF} = \mathrm{RGA}(\hat y_{\text{full}}, \hat y_{\text{without the protected variable}})$,
structurally an RGE on the protected attribute. Both are implemented here,
under names that say which is which (`rga_parity`, `rgf`).

What the gap measures is **AUC parity**: equal *ranking quality* per group. It
is not demographic parity, not equalised odds, and implies neither. A model can
rank perfectly inside both groups while assigning one group systematically
worse scores; a model can be equal-opportunity-fair while ranking better in the
group it had more data for.

Three statistical cautions:

1. The gap is a difference of noisy estimates and protected subgroups are
   small. At $n=200$ the RGA standard error is around 0.04, so an observed gap
   of 0.08 against a large reference group is about one standard error.
2. $\max - \min$ is non-negative *by construction*, so its percentile interval
   never contains 0, not even under exact parity. Test parity with the signed
   pairwise interval or the p-value; read the gap interval as "how large could
   the worst-case spread be". See *The gap's noise floor* below for the point
   estimate net of that inflation.
3. The same selection effect afflicts the *p-value*, and this is the one that
   silently produces false findings of discrimination. See below.

#### Multiplicity in the gap's p-value

The gap statistic is the largest of the $\binom{k}{2}$ pairwise ones, chosen
*after* seeing the data. Referring it to a standard normal — the unadjusted
p-value — therefore over-rejects, and badly. Measured type I error under exact
parity, at a nominal 5%:

| $k$ | pairs | unadjusted | max-T |
|---|---|---|---|
| 2 | 1 | 4.3% | 4.7% |
| 3 | 3 | 13.3% | 6.0% |
| 5 | 10 | 27.3% | 4.3% |

The correction exploits a structural fact specific to this setting. The groups
partition the sample, so under

```math
H_0:\ \mathrm{RGA}_1 = \cdots = \mathrm{RGA}_k
```

the estimates $\widehat{\mathrm{RGA}}_g$ are computed on **disjoint** rows and
are therefore *independent*, each asymptotically
$`N(\mathrm{RGA}_g, \sigma_g^2)`$ with $\sigma_g$ already available from the
per-group jackknife. The joint null distribution of the whole family of
pairwise statistics

```math
T_{ij} = \frac{\widehat{\mathrm{RGA}}_i - \widehat{\mathrm{RGA}}_j}
{\sqrt{\sigma_i^2 + \sigma_j^2}}
```

is then simulable *directly* — draw $Z_g \sim N(0, \sigma_g^2)$ once per group,
form every pair, take $\max_{i<j} |T_{ij}|$ — with **no re-estimation of RGA**
and no resampling of the data. The family-wise p-value is the tail probability
of that maximum, by the max-T (single-step) principle of Westfall and Young.

Because it uses the true correlation between pairs that share a group —
$T_{12}$ and $T_{13}$ both carry $Z_1$, so the family is far from independent —
this is materially less conservative than Bonferroni, which would treat all
$\binom{k}{2}$ as separate. Power is retained: a real 0.30 gap across five
groups still gives $p = 5 \times 10^{-4}$. At $k=2$ there is one pair, nothing
is selected, and the adjustment is the identity up to simulation error.

Two practical notes. The simulated p-value cannot fall below $1/(m+1)$ for $m$
draws, so it saturates at about $5\times10^{-4}$ at the default $m = 2000$;
that is a resolution limit of the simulation, not a weaker verdict. And the
normal approximation is the same one underlying the per-group intervals, so it
inherits their reliability — which is why `min_group_size` exists.

#### The gap's noise floor

The same simulation answers a second question for free. Under $H_0$ the
expected value of $\max - \min$ is not 0 — it grows with $k$ and with the
per-group standard errors — so the *point estimate* is inflated by exactly the
effect the p-value corrects for. Averaging $\max_g Z_g - \min_g Z_g$ over the
draws already in hand gives that expectation, and

```math
\mathrm{excess} \;=\; \mathrm{gap} \;-\; E_0\!\left[\max - \min\right]
```

reported as `gap_excess_over_noise`, is the observed spread net of it. Measured $E_0$: 0.034 for two groups of 300,
0.084 for five groups of 200. The quantity averages 0 under exact parity — a
regression test asserts this — and may be negative, which reads as "no more
spread than sampling noise produces by itself".

**What this replaces, and why.** Until 1.0.1 the reported figure was
$`2\,\mathrm{gap} - E^{*}[\max-\min]`$, a bootstrap bias correction using the
stratified bootstrap that produces `gap_ci`. It does not work here, for a
structural reason worth stating: the bootstrap resamples *inside* each group
around the **observed** per-group RGAs, so the bootstrap world already contains
the observed spread, and $E^{*}[\max-\min]$ sits just above the observed gap
rather than above a true zero. It therefore measures resampling noise given
already-separated centres, not the inflation of the gap relative to parity.
This is the standard failure of bootstrap bias correction for a maximum at a
boundary. Measured: it removed 24% of the bias at $k=2$, 32% at $k=3$ and 37%
at $k=5$, and with a genuine gap present it came out *larger* than the raw gap
in 83 of 150 replications.

### 3.4 Outcome-based criteria

RGA parity is AUC parity, and the module that computes it has always said to
report it beside outcome-based criteria. `outcome_parity` supplies those:
selection rate $P(D=1\mid g)$, its min/max ratio (disparate impact), the true
positive rate $P(D=1\mid Y=1, g)$ (equal opportunity), the false positive rate
$P(D=1\mid Y=0, g)$ (predictive equality), and the larger of the last two gaps
(equalised odds).

Each is a within-group proportion, so everything is closed form: **Wilson**
score intervals per group — chosen over Wald because subgroup counts are small
and rates sit near 0 or 1, precisely where Wald degenerates to zero width — a
two-proportion normal interval per gap, and a delta-method interval on
$\log(p_{\min}/p_{\max})$ for the ratio, which keeps it inside $(0,\infty)$.

The multiplicity argument transfers unchanged: $\max-\min$ over $k$ groups
selects the widest of $\binom{k}{2}$ pairs here exactly as it does for RGA, the
groups are again disjoint so the proportions are again independent under $H_0$,
and the same max-T simulation applies with $\sigma_g^2 = p_g(1-p_g)/n_g$.

These criteria need a **threshold**, which the rest of this package deliberately
avoids. There is no default: scores must be accompanied by an explicit
`threshold=`, or the decision must be passed already at 0/1. A fairness figure
computed at a cut-off nobody selected carries an authority it has not earned.
Note also that the criteria are mutually incompatible in general — demographic
parity and equalised odds cannot both hold when prevalence differs between
groups — so the table is a set of trade-offs to be read together, not a
checklist to be passed.

### 3.5 Searching for the worst cohort

`rga_by_segment` requires you to name the slices. `worst_cohort` searches them:
quantile-bin every candidate feature, enumerate every cohort definable by one
bin condition or an intersection of two, score each one large enough, rank
ascending.

The selection problem is then severe — thousands of cohorts rather than the ten
pairs of a five-level parity table — and the minimum of many noisy estimates
sits far below their common mean even under perfect homogeneity. The max-T
construction used above does not transfer, because it relies on the groups being
disjoint and hence independent, and cohorts overlap heavily.

The null is therefore obtained by permutation, and *what* is permuted carries
the whole argument. The hypothesis worth rejecting is **homogeneity** — "the
model ranks equally well everywhere" — not "the model has no ranking
information". Permuting $\hat y$ against $y$ would test the second: it drives
every cohort's RGA to 0.5, so an observed worst cohort of 0.44 stops looking
unusual, and the test loses all power against the case it exists for. Permuting
the **cohort definitions** against the $(y, \hat y)$ pairs instead preserves
every cohort's size, every overlap and the overall RGA, and destroys only the
association between membership and being ranked badly. The p-value is the share
of permutations whose worst-found cohort was at least as bad as the observed
one, which accounts for the entire search.

*Checked:* on twenty replications with pure-noise slicing features the p-value
was below 0.05 in at most 15% of runs with a median above 0.2; a planted cohort
scored by pure noise is detected at $p < 0.05$.

**Range restriction is the one true positive that is not a defect.** Slicing on
a predictor the model *uses* lowers RGA inside every slice by construction:
conditioning on a narrow band removes the between-band variation the score was
exploiting, the same reason a within-decile AUC is always below the overall one.
Measured on a clean logistic model in one predictor $x$: overall RGA 0.72,
middle quartile of $x$ crossed with a level of an unrelated attribute 0.45,
$p = 0.005$. Ranking quality genuinely is not homogeneous there; it simply has
a benign cause. Prefer slicing on variables the model never saw, and when
slicing on model features compare a band against its siblings rather than
against the overall figure.

### 3.6 Outlier robustness, measured

The papers assert RGA is more robust than RMSE without quantifying it.
Replacing a fraction of targets with values 50 standard deviations away
(mean of 15 draws, lognormal target):

| contaminated | $\Delta$RGA | $\Delta$RMSE |
|---|---|---|
| 1% | 23% | 780% |
| 2% | 31% | 1141% |
| 5% | 39% | 1859% |
| 10% | 43% | 2664% |

RGA reads ranks, so a corrupted observation can move it by at most its rank
displacement; RMSE reads values, so it diverges with the contamination
magnitude. The claim holds, and `contamination_curve` reproduces the table.

---

## 4. Performance

Measured on one core, `numpy` only, no compiled extension:

| $n$ | `rga` | jackknife CI | bootstrap, $B=200$ |
|---|---|---|---|
| 1 000 | 0.15 ms | 0.20 ms | 20 ms |
| 10 000 | 1.4 ms | 1.9 ms | 243 ms |
| 100 000 | 22 ms | 28 ms | 2.7 s |

The inference columns are 3.5–4× faster than in 1.0.0, and the change was
arithmetic rather than cleverness: §2.2 derives the whole delete-one family as
*two* sorts, but the implementation performed **six**, because `average_ranks`,
`suffix_sums_strictly_greater` and `tie_group_sums` each re-sorted the same
array and `_leave_one_out_sums` called all three, once per argument. The
bootstrap re-sorted once per replicate block on top of that. Sharing one
`SortedIndex` restored the derived cost; a test asserts the sort count so the
derivation and the code cannot drift apart again.

### Why not a compiled extension

`rga` is two sorts and two dot products, and the sorts dominate:

| $n$ | 2× `argsort` | `rga` total | sort share | ceiling if the sort were free |
|---|---|---|---|---|
| 10 000 | 1.3 ms | 2.3 ms | 58% | 2.4× |
| 100 000 | 18 ms | 28 ms | 65% | 2.9× |
| 1 000 000 | 280 ms | 400 ms | 70% | 3.3× |

So even an *infinitely fast* sort caps the speedup at about 3×, and a Rust
implementation would deliver perhaps 2–3× on the sort itself, i.e. well under
2× overall. Against that: losing the universal pure-Python wheel and moving to
`cibuildwheel` across three operating systems and five Python versions, for a
package that today installs anywhere with numpy alone. Fixing the redundant
sorts gave more than a compiled sort could have, and cost nothing to audit.

The default inference method is the jackknife precisely because it is exact,
deterministic *and* $O(n\log n)$ — the bootstrap is there for BCa intervals and
for people who want a resampling distribution to look at. `rga_parity`'s gap
interval now goes through the same vectorised multinomial bootstrap instead of
a Python loop (4.6 s → 2.1 s for five groups of 1000 at the defaults), which
was the one place a resampling loop was genuinely the bottleneck.

---

## 5. Open questions

Things this fork deliberately does not settle:

* **Coverage below $n \approx 200$.** The intervals are asymptotic. A
  small-sample correction (or a systematic study of BCa versus $t$-bootstrap in
  this setting) is unfinished work.
* **RGE with conditional imputation.** `method="mean"` and `"permute"` both
  break the joint distribution; `"retrain"` is expensive. Knockoffs or
  conditional sampling would be the principled middle ground. A group
  `"permute"` shares one permutation across the group's columns, which
  preserves the joint distribution *within* the group — enough to keep a
  one-hot encoding valid — but still severs it from every other predictor,
  which is the part that matters here.
* **Censored targets.** RGA and Harrell's C-index are both concordance
  measures; a formal comparison, and a censoring-aware RGA, is an obvious
  extension.
* **Multivariate / ordinal responses.** arXiv:2511.23100 proposes a whitening +
  weighted-average construction; it is not implemented here.
* **Inference for RGE and AURGR.** The intervals reported for RGE reflect
  evaluation-set sampling only, with the model held fixed. Uncertainty from
  *retraining* (the `"retrain"` path) is a different and larger quantity.
* **A search-aware interval for the worst cohort.** `worst_cohort` reports a
  family-wise p-value that accounts for the search, and a per-cohort interval
  that does not. A selection-adjusted *interval* — the analogue of post-
  selection inference for the minimum — is not implemented.
* **Range restriction, quantified.** Slicing on a model predictor lowers RGA
  inside every slice by a knowable amount under a parametric model; subtracting
  that expectation would separate "this cohort is genuinely worse" from "this
  cohort is narrower", and would make cohort search usable on the model's own
  features. Not attempted here.
* **Threshold selection for the outcome criteria.** `outcome_parity` requires
  the threshold rather than choosing one, deliberately. Reporting a whole
  threshold *curve* per criterion — the fairness analogue of AURGR's escape
  from `perturbation_percentage` — would remove the free parameter the same
  way, and is the obvious next step.
