"""Forecasters for the replay benchmark.

Every forecaster has the signature::

    forecast(hist: pl.DataFrame, future: pl.DataFrame, origin_day: int) -> (mu, sigma)

where ``hist`` holds traversal_features rows with ``day_index <= origin_day``,
``future`` the rows to predict, and ``(mu, sigma)`` describe a Gaussian
predictive distribution per future row. Two targets are supported through the
``target`` attribute: ``"log_moving_s"`` (segment time) and ``"hr_mean"``
(climb heart rate conditional on realized virtual power, temperature and
elapsed time).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import polars as pl

Forecast = tuple[np.ndarray, np.ndarray]


@dataclass
class Forecaster:
    name: str
    target: str
    fn: Callable[[pl.DataFrame, pl.DataFrame, int], Forecast]
    family: str = "baseline"

    def __call__(
        self, hist: pl.DataFrame, future: pl.DataFrame, origin_day: int
    ) -> Forecast:
        return self.fn(hist, future, origin_day)


def _y(df: pl.DataFrame, target: str) -> np.ndarray:
    if target == "log_moving_s":
        return np.log(df["moving_s"].to_numpy().astype(float))
    return df[target].to_numpy().astype(float)


def _pooled_sd(hist: pl.DataFrame, target: str, floor: float) -> float:
    y = _y(hist, target)
    seg = hist["segment_id"].to_numpy()
    resid = []
    for s in np.unique(seg):
        ys = y[seg == s]
        if len(ys) >= 3:
            resid.extend(ys - ys.mean())
    return (
        float(max(np.std(resid), floor))
        if len(resid) >= 5
        else float(max(np.std(y), floor))
    )


# ------------------------------------------------------------- persistence
def persistence(target: str, floor: float) -> Forecaster:
    def fn(hist, future, origin_day):
        y = _y(hist, target)
        seg = hist["segment_id"].to_numpy()
        day = hist["day_index"].to_numpy()
        sd = _pooled_sd(hist, target, floor)
        mu = np.empty(future.height)
        for i, s in enumerate(future["segment_id"]):
            m = seg == s
            if m.any():
                mu[i] = y[m][np.argmax(day[m])]
            else:
                mu[i] = y.mean()
        return mu, np.full(future.height, sd * 1.2)

    return Forecaster("persistence", target, fn)


def ewma(target: str, floor: float, half_life: float = 4.0) -> Forecaster:
    lam = 0.5 ** (1 / half_life)

    def fn(hist, future, origin_day):
        y = _y(hist, target)
        seg = hist["segment_id"].to_numpy()
        day = hist["day_index"].to_numpy()
        sd = _pooled_sd(hist, target, floor)
        mu = np.empty(future.height)
        for i, s in enumerate(future["segment_id"]):
            m = seg == s
            if m.any():
                ys = y[m][np.argsort(day[m])]
                w = lam ** np.arange(len(ys))[::-1]
                mu[i] = np.sum(w * ys) / w.sum()
            else:
                mu[i] = y.mean()
        return mu, np.full(future.height, sd)

    return Forecaster("ewma", target, fn)


# --------------------------------------------------------- linear / ridge
def _design(
    df: pl.DataFrame, target: str, segs: list[str], day_ref: float
) -> np.ndarray:
    n = df.height
    cols = [np.ones(n)]
    for s in segs[1:]:
        cols.append((df["segment_id"].to_numpy() == s).astype(float))
    if target == "hr_mean":
        cols += [
            (df["vp_mean_w"].to_numpy().astype(float) - 140) / 50,
            df["temp_c20"].to_numpy().astype(float),
            df["elapsed_h"].to_numpy().astype(float),
        ]
    else:
        cols += [
            (df["hr_mean"].fill_null(140).to_numpy().astype(float) - 145) / 10,
            df["temp_c20"].to_numpy().astype(float),
            df["elapsed_h"].to_numpy().astype(float),
        ]
    return np.column_stack(cols)


def ridge(target: str, floor: float, alpha: float = 1.0) -> Forecaster:
    def fn(hist, future, origin_day):
        segs = sorted(
            set(hist["segment_id"].to_list()) | set(future["segment_id"].to_list())
        )
        X = _design(hist, target, segs, origin_day)
        y = _y(hist, target)
        A = X.T @ X + alpha * np.eye(X.shape[1])
        beta = np.linalg.solve(A, X.T @ y)
        resid = y - X @ beta
        sd = max(float(resid.std()), floor)
        Xf = _design(future, target, segs, origin_day)
        return Xf @ beta, np.full(future.height, sd * 1.1)

    return Forecaster("ridge_covariates", target, fn)


# ------------------------------------------------------------------- GP
def gp(target: str, floor: float) -> Forecaster:
    """GP with RBF on calendar time (length-scale 60 d) + linear covariates + segment offsets."""

    def fn(hist, future, origin_day):
        segs = sorted(
            set(hist["segment_id"].to_list()) | set(future["segment_id"].to_list())
        )
        X = _design(hist, target, segs, origin_day)
        Xf = _design(future, target, segs, origin_day)
        y = _y(hist, target)
        # linear mean via ridge, GP on residual over time
        beta = np.linalg.solve(X.T @ X + np.eye(X.shape[1]), X.T @ y)
        r = y - X @ beta
        t = hist["day_index"].to_numpy().astype(float)
        tf = future["day_index"].to_numpy().astype(float)
        ell, s2 = 60.0, max(r.var() * 0.5, floor**2)
        noise = max(r.var() * 0.5, floor**2)
        K = s2 * np.exp(-0.5 * ((t[:, None] - t[None, :]) / ell) ** 2) + noise * np.eye(
            len(t)
        )
        Ks = s2 * np.exp(-0.5 * ((tf[:, None] - t[None, :]) / ell) ** 2)
        L = np.linalg.cholesky(K + 1e-6 * np.eye(len(t)))
        a = np.linalg.solve(L.T, np.linalg.solve(L, r))
        mu = Xf @ beta + Ks @ a
        v = np.linalg.solve(L, Ks.T)
        var = s2 - np.sum(v * v, axis=0) + noise
        return mu, np.sqrt(np.maximum(var, floor**2))

    return Forecaster("gp_time_covariates", target, fn)


# --------------------------------------------------- Banister with defaults
def banister_default(load_daily: np.ndarray, floor: float) -> Forecaster:
    """FFM with population tau=(42,7); k1,k2 by least squares on HR-adjusted log speed."""
    import jax.numpy as jnp

    from garmin_lab.twin.ffm import _filter

    F = np.asarray(_filter(jnp.asarray(load_daily / 100.0), jnp.exp(-1 / 42.0)))
    G = np.asarray(_filter(jnp.asarray(load_daily / 100.0), jnp.exp(-1 / 7.0)))

    def fn(hist, future, origin_day):
        segs = sorted(
            set(hist["segment_id"].to_list()) | set(future["segment_id"].to_list())
        )
        X = _design(hist, "log_moving_s", segs, origin_day)
        d = hist["day_index"].to_numpy()
        X = np.column_stack([X, F[d], G[d]])
        y = _y(hist, "log_moving_s")
        beta = np.linalg.solve(X.T @ X + 1e-3 * np.eye(X.shape[1]), X.T @ y)
        sd = max(float((y - X @ beta).std()), floor)
        df = future["day_index"].to_numpy()
        # future load unknown: propagate from origin with zero load
        Ff = F[origin_day] * np.exp(-(df - origin_day) / 42.0)
        Gf = G[origin_day] * np.exp(-(df - origin_day) / 7.0)
        Xf = np.column_stack(
            [_design(future, "log_moving_s", segs, origin_day), Ff, Gf]
        )
        return Xf @ beta, np.full(future.height, sd * 1.1)

    return Forecaster("banister_default", "log_moving_s", fn)


# ------------------------------------------------------------ static twin
def twin_hr(floor: float, samples: int = 400) -> Forecaster:
    """Hierarchical HR response model refit at each origin (NUTS, short run)."""
    from garmin_lab.twin.hr_model import VP_REF, VP_SCALE, fit_hr, load_hr_data

    def fn(hist, future, origin_day):
        d = load_hr_data(hist)
        idata = fit_hr(d, warmup=400, samples=samples, chains=1)
        post = idata.posterior
        cw = post["c_week"].values.reshape(-1, d.n_week)[:, d.week.max()]
        beta = post["beta_vp50"].values.ravel()
        gT = post["gamma_T"].values.ravel()
        gd = post["gamma_d"].values.ravel()
        gdT = post["gamma_dT"].values.ravel()
        u = post["u_seg"].values.reshape(-1, d.n_seg) * post["sigma_u"].values.reshape(
            -1, 1
        )
        sig_a = post["sigma_ride"].values.ravel()
        sig = post["sigma"].values.ravel()
        nu = post["nu"].values.ravel() + 2
        mu_out, sd_out = np.empty(future.height), np.empty(future.height)
        for i, row in enumerate(future.iter_rows(named=True)):
            useg = (
                u[:, d.seg_names.index(row["segment_id"])]
                if row["segment_id"] in d.seg_names
                else np.zeros_like(beta)
            )
            m = (
                cw
                + beta * (row["vp_mean_w"] - VP_REF) / VP_SCALE
                + gT * row["temp_c20"]
                + gd * row["elapsed_h"]
                + gdT * row["temp_c20"] * row["elapsed_h"]
                + useg
            )
            var = m.var() + np.mean(sig_a**2) + np.mean(sig**2 * nu / (nu - 2))
            mu_out[i], sd_out[i] = m.mean(), np.sqrt(var)
        return mu_out, np.maximum(sd_out, floor)

    return Forecaster("twin_hr_model", "hr_mean", fn, family="twin")


def twin_ffm(floor: float, samples: int = 400) -> Forecaster:
    """Static twin for segment time: bounded FFM with learned tau on climbs (speed -> time via
    segment distance), EWMA on descents/rolling. Load after the origin is unknown and decays.
    """
    from garmin_lab.twin.ffm import HR_REF, fit_ffm, load_ffm_data

    fallback = ewma("log_moving_s", floor)

    def fn(hist, future, origin_day):
        mu_out, sd_out = fallback(hist, future, origin_day)
        is_climb = (future["kind"] == "climb").to_numpy()
        if not is_climb.any():
            return mu_out, sd_out
        d = load_ffm_data(until_day=origin_day)
        if len(d.log_v) < 15:
            return mu_out, sd_out
        idata = fit_ffm(d, warmup=500, samples=samples, chains=1)
        post = idata.posterior
        p = post["p_day"].values.reshape(-1, len(d.w))
        tau1 = post["tau1"].values.ravel()
        a = post["a_seg"].values.reshape(-1, d.n_seg)
        b_hr = post["b_hr10"].values.ravel()
        b_T = post["b_T"].values.ravel()
        b_el = post["b_elapsed"].values.ravel()
        b_s = post["b_season_sin"].values.ravel()
        b_c = post["b_season_cos"].values.ravel()
        sig = post["sigma"].values.ravel()
        last = len(d.w) - 1
        for i, row in enumerate(future.iter_rows(named=True)):
            if not is_climb[i] or row["segment_id"] not in d.seg_names:
                continue
            h = max(row["day_index"] - last, 0)
            p_f = p[:, last] * np.exp(-h / tau1)
            doy = row["date"].timetuple().tm_yday
            seg_a = a[:, d.seg_names.index(row["segment_id"])]
            log_v = (
                seg_a
                + p_f
                + b_hr * (row["hr_mean"] - HR_REF) / 10
                + b_T * row["temp_c20"]
                + b_el * row["elapsed_h"]
                + b_s * np.sin(2 * np.pi * doy / 365.25)
                + b_c * np.cos(2 * np.pi * doy / 365.25)
            )
            log_t = np.log(row["dist_m"]) - log_v
            mu_out[i] = log_t.mean()
            sd_out[i] = max(np.sqrt(log_t.var() + np.mean(sig**2) * 1.3), floor)
        return mu_out, sd_out

    return Forecaster("twin_ffm_learned_tau", "log_moving_s", fn, family="twin")
