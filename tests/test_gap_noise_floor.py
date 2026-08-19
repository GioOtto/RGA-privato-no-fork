"""The gap's noise floor, and why the old bootstrap "bias correction" was not one.

``gap_bias_corrected`` used to be ``max(0, 2*gap - mean(bootstrap gaps))``. The
bootstrap resamples *inside* each group around the observed per-group RGAs, so
its world already contains the observed spread and its mean sits on top of the
gap rather than above a true zero. Measured under exact parity that removed
only 24-37% of the selection bias, and with a real gap present it exceeded the
raw gap in 83 runs out of 150 - it added noise instead of removing bias.

The replacement subtracts the expectation of ``max - min`` under the *null* of
exact parity, simulated from the same independent normals the max-T p-value
already uses. These tests pin both the property the old one failed (it must
average to zero under exact parity) and the arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from rgbox import rga_parity


def _parity_under_exact_parity(seed, sizes, n_resamples=400):
    """Groups that differ only by sampling noise: the true gap is exactly 0."""
    rng = np.random.default_rng(seed)
    ys, scores, groups = [], [], []
    for index, size in enumerate(sizes):
        x = rng.normal(size=size)
        ys.append((rng.random(size) < 1 / (1 + np.exp(-x))).astype(float))
        scores.append(x)
        groups.append(np.full(size, index))
    return rga_parity(
        np.concatenate(ys),
        np.concatenate(scores),
        np.concatenate(groups),
        n_resamples=n_resamples,
        random_state=seed,
    )


@pytest.mark.parametrize("sizes", [[300, 300], [300, 300, 300], [200] * 5])
def test_excess_over_noise_averages_zero_under_exact_parity(sizes):
    """The property the bootstrap "correction" never had."""
    excess, raw = [], []
    for seed in range(60):
        result = _parity_under_exact_parity(seed, sizes)
        if result.gap is None:
            continue
        excess.append(result.gap_excess_over_noise)
        raw.append(result.gap)
    excess, raw = np.array(excess), np.array(raw)

    # max - min is inflated by construction and stays inflated...
    assert raw.mean() > 0.02
    # ...while the corrected figure is centred on the truth, which is 0.
    assert abs(excess.mean()) < 0.25 * raw.mean()
    # and it is genuinely two-sided, not clipped at zero like the old one.
    assert (excess < 0).any()


def test_noise_floor_grows_with_the_number_of_groups():
    """More groups means a wider max - min under the very same null."""
    floors = {}
    for k in (2, 3, 5):
        result = _parity_under_exact_parity(7, [250] * k)
        floors[k] = result.gap_noise_floor
    assert floors[2] < floors[3] < floors[5]


def test_excess_is_exactly_gap_minus_floor(rng):
    y = rng.binomial(1, 0.4, 1200).astype(float)
    scores = rng.normal(y, 1.0, 1200)
    groups = rng.choice(["a", "b", "c"], 1200)
    result = rga_parity(y, scores, groups, n_resamples=300, random_state=0)
    assert result.gap_excess_over_noise == pytest.approx(
        result.gap - result.gap_noise_floor
    )
    assert result.gap_noise_floor > 0.0


def test_a_real_gap_survives_the_correction(rng):
    """Subtracting the floor must not erase a disparity that is genuinely there."""
    n = 1200
    strong = rng.normal(size=n)
    weak = rng.normal(size=n)
    y = np.concatenate(
        [
            (rng.random(n) < 1 / (1 + np.exp(-1.8 * strong))).astype(float),
            (rng.random(n) < 1 / (1 + np.exp(-0.3 * weak))).astype(float),
        ]
    )
    scores = np.concatenate([strong, weak])
    groups = np.concatenate([np.zeros(n), np.ones(n)])
    result = rga_parity(y, scores, groups, n_resamples=400, random_state=1)
    assert result.gap > 0.15
    assert result.gap_excess_over_noise > 0.10
    assert result.gap_p_value < 0.01


def test_the_dictionary_carries_both_numbers_and_explains_them(rng):
    y = rng.binomial(1, 0.4, 800).astype(float)
    scores = rng.normal(y, 1.0, 800)
    groups = rng.binomial(1, 0.5, 800)
    record = rga_parity(y, scores, groups, n_resamples=200, random_state=0).to_dict()
    assert "gap_excess_over_noise" in record
    assert "gap_noise_floor" in record
    assert "gap_bias_corrected" not in record
    assert "gap_excess_over_noise" in record["gap_ci_note"]
