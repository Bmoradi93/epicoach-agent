"""Scoring-rule and comparison-test sanity checks."""

import numpy as np

from garmin_lab.replay.scoring import (
    block_bootstrap_ci,
    crps_gaussian,
    crps_samples,
    diebold_mariano,
    ece_from_pit,
    pit_gaussian,
)


def test_crps_gaussian_matches_sample_estimate():
    rng = np.random.default_rng(0)
    y, mu, sd = 1.3, 1.0, 0.5
    closed = crps_gaussian(np.array([y]), np.array([mu]), np.array([sd]))[0]
    sampled = crps_samples(y, rng.normal(mu, sd, 20000))
    assert abs(closed - sampled) < 0.01


def test_crps_is_proper_prefers_true_sigma():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 5000)
    good = crps_gaussian(y, np.zeros_like(y), np.ones_like(y)).mean()
    over = crps_gaussian(y, np.zeros_like(y), np.full_like(y, 3.0)).mean()
    under = crps_gaussian(y, np.zeros_like(y), np.full_like(y, 0.3)).mean()
    assert good < over and good < under


def test_pit_calibrated_has_low_ece():
    rng = np.random.default_rng(2)
    y = rng.normal(0, 1, 5000)
    assert ece_from_pit(pit_gaussian(y, np.zeros_like(y), np.ones_like(y))) < 0.03
    assert ece_from_pit(pit_gaussian(y, np.zeros_like(y), np.full_like(y, 0.3))) > 0.2


def test_diebold_mariano_detects_difference_and_sign():
    rng = np.random.default_rng(3)
    d = rng.normal(-0.5, 1.0, 60)  # A better than B
    stat, p = diebold_mariano(d, lag=4)
    assert stat < 0 and p < 0.01
    stat0, p0 = diebold_mariano(rng.normal(0, 1, 60), lag=4)
    assert p0 > 0.05 or abs(stat0) < 2.5


def test_block_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(4)
    x = rng.normal(2.0, 1.0, 200)
    m, lo, hi = block_bootstrap_ci(x, block=14, n_boot=500)
    assert lo < 2.0 < hi and abs(m - x.mean()) < 1e-12
