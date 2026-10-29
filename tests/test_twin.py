"""Unit tests for twin building blocks (fast; no MCMC)."""

import jax.numpy as jnp
import numpy as np

from garmin_lab.twin.features import edwards_trimp, exp_trimp, virtual_power
from garmin_lab.twin.ffm import _filter, bhattacharyya


def test_filter_matches_closed_form():
    w = jnp.ones(50)
    tau = 10.0
    out = np.asarray(_filter(w, jnp.exp(-1.0 / tau)))
    k = np.arange(1, 51)
    a = np.exp(-1.0 / tau)
    expected = (1 - a**k) / (1 - a)
    assert np.allclose(out, expected, atol=1e-5)


def test_exp_trimp_monotone_in_intensity():
    dt = np.full(600, 1.0)
    lo = exp_trimp(np.full(600, 120.0), dt, 58.0, 200.0)
    hi = exp_trimp(np.full(600, 170.0), dt, 58.0, 200.0)
    assert 0 < lo < hi
    # 10 minutes at full reserve = 10 * 0.64 * e^1.92
    full = exp_trimp(np.full(600, 200.0), dt, 58.0, 200.0)
    assert abs(full - 10 * 0.64 * np.exp(1.92)) < 1e-6


def test_edwards_zones():
    dt = np.full(60, 1.0)
    assert edwards_trimp(np.full(60, 90.0), dt, 200.0) == 0.0  # 45% HRmax -> zone 0
    assert (
        abs(edwards_trimp(np.full(60, 190.0), dt, 200.0) - 5.0) < 1e-9
    )  # 95% -> zone 5, 1 minute


def test_virtual_power_climb_vs_flat():
    n = 100
    t = np.arange(n, dtype=float)
    v = np.full(n, 3.0)
    dist = np.cumsum(v)
    flat, g0 = virtual_power(v, np.zeros(n), dist, t, 80.0)
    climb, g1 = virtual_power(v, 0.10 * dist, dist, t, 80.0)  # 10% grade
    assert np.nanmean(climb[20:-20]) > np.nanmean(flat[20:-20]) + 150
    assert abs(np.nanmean(g1[20:-20]) - 0.10) < 0.01


def test_bhattacharyya_identity_and_disjoint():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20000)
    assert bhattacharyya(x, x) > 0.99
    assert bhattacharyya(x, x + 20) < 0.01
