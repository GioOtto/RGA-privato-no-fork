# Provenance and licensing notice

**Read this before redistributing anything in this repository, and before
putting it into a regulated production environment.**

## The upstream project has no licence

[`GolnooshBabaei/safeaipackage`](https://github.com/GolnooshBabaei/safeaipackage)
contains no `LICENSE` file, and the PyPI distribution `safeaipackage` (latest
0.8.3, 14 May 2025) ships no licence metadata either — its `requires_dist` and
`license` fields are both empty.

Under the Berne Convention and essentially every national copyright statute,
code published without a licence is **"all rights reserved" by default**.
Posting source on GitHub grants viewing and forking *within GitHub* under
GitHub's Terms of Service, but it grants no right to use, modify, redistribute
or run the code commercially outside that platform.

Practical consequences:

- A regulated institution cannot clear unlicensed third-party code through
  third-party-software review. This is normally a hard stop, not a formality.
- Publishing this fork to PyPI as a derivative of unlicensed code would be a
  redistribution the upstream authors have not authorised.
- A thesis that ships derived code has the same problem, in a milder form.

**Recommended action: email the authors and ask them to add an explicit licence
(MIT or Apache-2.0) to the upstream repository.** Contact:
`golnoosh.babaei@unipv.it` (from the upstream README). It is a small request
and authors almost always agree; a one-line reply on record is what an
audit trail needs.

## What is what in this repository

| Path | Origin | Status |
|---|---|---|
| `src/rgbox/` | Written from scratch for this fork | MIT (see `LICENSE`) |
| `src/safeaipackage/` | Written from scratch; reproduces the upstream *API surface* (function names and signatures), which is not itself copyrightable expression | MIT |
| `tests/`, `benchmarks/`, `docs/` | Written from scratch | MIT |
| `README.md`, `MIGRATION.md`, `CHANGELOG.md`, this file | Written from scratch | MIT |
| `upstream_reference/safeaipackage/` | **Verbatim upstream source**, kept only so the fork's behaviour can be diffed against it | Unlicensed — rights retained by the original authors |
| `R_codes/*.R` | **Verbatim upstream source** | Unlicensed — rights retained by the original authors |
| `examples/`, `examples/employee.xlsx` | Upstream | Unlicensed |

No upstream source file was copied into `src/`. The implementations here were
derived from the published mathematical definitions (see `docs/THEORY.md`) and
from the algebraic identity documented in `rgbox/core.py`, then checked for
numerical agreement against the upstream algorithm. Agreement of *output* is
the point; it is not evidence of copied *code*.

If you need a repository that is unambiguously clean, delete
`upstream_reference/`, `R_codes/` and `examples/`. Nothing in `src/` or
`tests/` imports them, and the test suite passes without them — except
`tests/test_upstream_parity.py`, which re-derives the upstream algorithm
inline from its published formula precisely so that the comparison does not
depend on those directories being present.

## Academic attribution

The measures are not this fork's invention. Cite the original work:

- Babaei, G., Giudici, P., & Raffinetti, E. (2025). *A Rank Graduation Box for
  SAFE AI.* Expert Systems with Applications, 259, 125239.
  <https://doi.org/10.1016/j.eswa.2024.125239>
- Giudici, P., & Raffinetti, E. (2025). *RGA: a unified measure of predictive
  accuracy.* Advances in Data Analysis and Classification, 19(1), 67–93.
  <https://doi.org/10.1007/s11634-023-00574-2>
- Raffinetti, E. (2023). *A rank graduation accuracy measure to mitigate
  artificial intelligence risks.* Quality & Quantity, 57(Suppl 2), 131–150.
  <https://doi.org/10.1007/s11135-023-01613-y>

The statistical-inference layer (`rgbox/inference.py`) is not from those
papers; it is derived here and validated numerically. It should be described
as such rather than attributed to the original authors.

## Related, newer official software

The same research group has since released a separate package,
**`safe-ai-metrics`** (docs: <https://safeai.readthedocs.io>), covering RGA,
RGR and RGE with curve/area variants and plotting. It is *not* a fork of
`safeaipackage` and, as of this writing, also reports no standard errors,
confidence intervals or hypothesis tests, and covers no fairness metric.
Evaluate it alongside this fork before committing to either.
