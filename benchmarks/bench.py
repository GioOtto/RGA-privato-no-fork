"""Timing and accuracy benchmarks.

    python benchmarks/bench.py

Answers two questions: is the closed form actually faster than the upstream
pandas pipeline, and is the inference layer cheap enough to run by default in a
validation pipeline. (Spoiler for the second: yes, which is why there is no C
extension in this package.)
"""

from __future__ import annotations

import sys
import time

import numpy as np

from rgbox import rga, rga_ci
from rgbox.inference import bootstrap_values, influence_values, jackknife_values

SIZES = (1_000, 10_000, 100_000, 1_000_000)


def timeit(fn, *args, repeats: int = 3, **kwargs) -> float:
    """Best-of-N wall time in milliseconds."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args, **kwargs)
        best = min(best, time.perf_counter() - start)
    return best * 1000.0


def upstream_rga(y, yhat):
    """The original pandas implementation, for a like-for-like comparison."""
    import pandas as pd

    frame = pd.concat(
        [pd.DataFrame(y).reset_index(drop=True),
         pd.DataFrame(yhat).reset_index(drop=True)], axis=1
    )
    frame.columns = ["y", "yhat"]
    frame["ryhat"] = frame["yhat"].rank(method="min")
    support = frame.groupby("ryhat")["y"].mean().reset_index(name="support")
    frame = pd.merge(frame, support, on="ryhat", how="left")
    frame["rord"] = frame["support"]
    frame = frame.sort_values(by="yhat").reset_index(drop=True)
    index = np.arange(len(frame))
    conc = np.sum(index * frame["rord"].values)
    ordered = np.sort(frame["y"])
    return (conc - np.sum(index * ordered[::-1])) / (
        np.sum(index * ordered) - np.sum(index * ordered[::-1])
    )


def main() -> int:
    rng = np.random.default_rng(0)
    try:
        import pandas  # noqa: F401
        have_pandas = True
    except ImportError:
        have_pandas = False

    print("=" * 78)
    print("point estimate: closed form vs upstream pandas pipeline")
    print("=" * 78)
    print(f"{'n':>10} {'rgbox (ms)':>12} {'upstream (ms)':>15} {'speedup':>9}")
    for n in SIZES:
        y = rng.normal(size=n)
        yhat = 0.6 * y + rng.normal(size=n)
        ours = timeit(rga, y, yhat)
        if have_pandas and n <= 1_000_000:
            theirs = timeit(upstream_rga, y, yhat, repeats=1)
            print(f"{n:>10,} {ours:>12.2f} {theirs:>15.2f} {theirs / ours:>8.1f}x")
        else:
            print(f"{n:>10,} {ours:>12.2f} {'-':>15} {'-':>9}")

    print()
    print("=" * 78)
    print("inference cost")
    print("=" * 78)
    print(f"{'n':>10} {'jackknife':>11} {'influence':>11} {'boot B=200':>12} "
          f"{'boot B=2000':>13}")
    for n in SIZES[:3]:
        y = rng.normal(size=n)
        yhat = 0.6 * y + rng.normal(size=n)
        jack = timeit(jackknife_values, y, yhat)
        infl = timeit(influence_values, y, yhat)
        b200 = timeit(bootstrap_values, y, yhat, repeats=1,
                      n_resamples=200, random_state=0)
        b2000 = timeit(bootstrap_values, y, yhat, repeats=1,
                       n_resamples=2000, random_state=0) if n <= 10_000 else float("nan")
        print(f"{n:>10,} {jack:>10.2f}ms {infl:>10.2f}ms {b200:>11.1f}ms "
              f"{b2000:>12.1f}ms")

    print()
    print("=" * 78)
    print("exactness: fast jackknife vs naive O(n^2)")
    print("=" * 78)
    for n in (200, 800):
        y = rng.integers(0, 4, n).astype(float)      # heavy ties, the hard case
        yhat = rng.integers(0, 6, n).astype(float)
        fast = jackknife_values(y, yhat)
        naive = np.array(
            [rga(np.delete(y, k), np.delete(yhat, k)) for k in range(n)]
        )
        print(f"  n={n:<6d} max |fast - naive| = {np.max(np.abs(fast - naive)):.2e}")

    print()
    print("=" * 78)
    print("agreement of the three standard errors")
    print("=" * 78)
    print(f"  {'design':<22}{'jackknife':>11}{'influence':>11}{'bootstrap':>11}")
    designs = {
        "gaussian n=2000": (lambda r: (lambda a: (a, 0.7 * a + r.normal(size=2000)))(
            r.normal(size=2000))),
        "binary n=2000": (lambda r: (lambda b: (b.astype(float),
                                                r.normal(b, 1.0, 2000)))(
            r.binomial(1, 0.3, 2000))),
        "heavy ties n=2000": (lambda r: (r.integers(0, 3, 2000).astype(float),
                                         r.integers(0, 4, 2000).astype(float))),
        "lognormal n=2000": (lambda r: (lambda a: (np.exp(a),
                                                   0.7 * a + r.normal(size=2000)))(
            r.normal(size=2000))),
    }
    for label, build in designs.items():
        y, yhat = build(rng)
        errors = [
            rga_ci(y, yhat, method=m, n_resamples=2000, random_state=0).standard_error
            for m in ("jackknife", "influence", "bootstrap")
        ]
        print(f"  {label:<22}" + "".join(f"{e:>11.5f}" for e in errors))
    return 0


if __name__ == "__main__":
    sys.exit(main())
