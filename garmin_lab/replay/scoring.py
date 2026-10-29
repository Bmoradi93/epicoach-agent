"""Proper scoring rules and forecast-comparison tests for one long dependent series."""

from __future__ import annotations

import numpy as np
from scipy import stats


# ------------------------------------------------------------------ scores
def crps_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Closed-form CRPS of N(mu, sigma) at y (Gneiting & Raftery 2007)."""
    sigma = np.maximum(sigma, 1e-9)
    z = (y - mu) / sigma
    return sigma * (
        z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi)
    )


def crps_samples(y: float, samples: np.ndarray) -> float:
    """Energy-form CRPS from predictive samples."""
    s = np.sort(np.asarray(samples, dtype=float))
    n = len(s)
    term1 = np.mean(np.abs(s - y))
    term2 = np.sum(np.abs(s[:, None] - s[None, :])) / (2 * n * n)
    return float(term1 - term2)


def log_score_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return -stats.norm.logpdf(y, mu, np.maximum(sigma, 1e-9))


def pit_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return stats.norm.cdf(y, mu, np.maximum(sigma, 1e-9))


def ece_from_pit(pit: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error of a PIT histogram against uniform."""
    h, _ = np.histogram(pit, bins=bins, range=(0, 1))
    return float(np.abs(h / len(pit) - 1 / bins).sum() / 2)


def coverage(
    y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, level: float = 0.9
) -> float:
    z = stats.norm.ppf(0.5 + level / 2)
    return float(np.mean(np.abs(y - mu) <= z * sigma))


def brier(p: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    return (np.asarray(p) - np.asarray(outcome)) ** 2


# ------------------------------------------------------------------- tests
def diebold_mariano(d: np.ndarray, lag: int = 4) -> tuple[float, float]:
    """DM test on loss differential d_t = loss_A - loss_B with Newey-West HAC variance.

    Returns (statistic, two-sided p). Negative statistic favours A.
    """
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 8:
        return float("nan"), float("nan")
    dbar = d.mean()
    e = d - dbar
    gamma0 = np.mean(e * e)
    var = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        w = 1 - k / (lag + 1)
        var += 2 * w * np.mean(e[k:] * e[:-k])
    var = max(var, 1e-12)
    stat = dbar / np.sqrt(var / n)
    # Harvey-Leybourne-Newbold small-sample correction
    h = 1
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return float(stat), float(p)


def block_bootstrap_ci(
    x: np.ndarray,
    block: int = 14,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Moving-block bootstrap CI for the mean of a dependent series."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < block:
        block = max(1, n // 2)
    nb = int(np.ceil(n / block))
    starts = np.arange(0, n - block + 1)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, nb)])[
            :n
        ]
        means[b] = x[idx].mean()
    return (
        float(x.mean()),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )
