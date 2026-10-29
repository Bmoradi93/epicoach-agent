"""Segment-aggregate heart-rate response model on climb traversals.

    HR_i = c_{w(i)} + beta * (VP_i - VP_ref)/50 + gamma_T * (T_i - 20)
           + gamma_d * elapsed_h_i + gamma_dT * (T_i - 20) * elapsed_h_i
           + u_{seg(i)} + a_{ride(i)} + eps_i,     eps_i ~ StudentT(nu, 0, sigma)

``c_w`` is a Gaussian random walk over weeks: heart rate at reference virtual
power on a reference climb. A falling ``c_w`` means improved aerobic response
("climbing response improved" hypothesis). ``gamma_d`` is cardiac drift
(bpm/h), ``gamma_dT`` its heat interaction, ``gamma_T`` the heat offset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
from numpyro.infer import MCMC, NUTS

from garmin_lab.ingest.client import REPO_ROOT

log = logging.getLogger("garmin_lab.twin.hr")
DATASET = REPO_ROOT / "data" / "dataset"
TWIN_DIR = REPO_ROOT / "data" / "twin"
VP_REF = 140.0
VP_SCALE = 50.0
TRAIL_HIGHWAYS = ("path", "track", "cycleway", "footway", "bridleway", "service")


@dataclass
class HRData:
    hr: np.ndarray
    vp: np.ndarray
    temp20: np.ndarray
    elapsed_h: np.ndarray
    seg: np.ndarray
    ride: np.ndarray
    week: np.ndarray
    n_seg: int
    n_ride: int
    n_week: int
    seg_names: list[str]
    traversal_index: np.ndarray
    frame: pl.DataFrame | None = None


def load_hr_data(
    features: pl.DataFrame | None = None, until_day: int | None = None
) -> HRData:
    f = (
        features
        if features is not None
        else pl.read_parquet(DATASET / "traversal_features.parquet")
    )
    f = f.filter(
        (pl.col("kind") == "climb")
        & pl.col("hr_mean").is_not_null()
        & pl.col("vp_mean_w").is_not_null()
        & (pl.col("hr_ok_frac") >= 0.7)
        & pl.col("temp_c20").is_not_null()
    )
    if "osm_highway" in f.columns:
        f = f.filter(pl.col("osm_highway").is_in(TRAIL_HIGHWAYS))
    if until_day is not None:
        f = f.filter(pl.col("day_index") <= until_day)
    f = f.with_row_index("_i")
    segs = sorted(f["segment_id"].unique().to_list())
    seg_map = {s: i for i, s in enumerate(segs)}
    rides = sorted(f["activity_id"].unique().to_list())
    ride_map = {r: i for i, r in enumerate(rides)}
    return HRData(
        hr=f["hr_mean"].to_numpy().astype(float),
        vp=f["vp_mean_w"].to_numpy().astype(float),
        temp20=f["temp_c20"].to_numpy().astype(float),
        elapsed_h=f["elapsed_h"].to_numpy().astype(float),
        seg=np.array([seg_map[s] for s in f["segment_id"]]),
        ride=np.array([ride_map[r] for r in f["activity_id"]]),
        week=f["week"].to_numpy().astype(int),
        n_seg=len(segs),
        n_ride=len(rides),
        n_week=int(f["week"].max()) + 1,
        seg_names=segs,
        traversal_index=f["_i"].to_numpy(),
        frame=f,
    )


Extra = tuple[np.ndarray, float] | None  # (design n x k, prior sd) for template terms


def extra_term(extra: Extra):
    """Additive linear term for compiled template specs: X @ theta, theta ~ N(0, sd)."""
    if extra is None:
        return 0.0
    X, sd = extra
    theta = numpyro.sample("theta_extra", dist.Normal(0.0, sd).expand([X.shape[1]]))
    return jnp.asarray(X) @ theta


def hr_model(d: HRData, hr_obs: np.ndarray | None = None, extra: Extra = None) -> None:
    # weekly random walk for HR at reference power (non-centred)
    c0 = numpyro.sample("c0", dist.Normal(145.0, 15.0))
    sigma_c = numpyro.sample("sigma_c", dist.HalfNormal(0.75))
    z = numpyro.sample("z_c", dist.Normal(0.0, 1.0).expand([d.n_week]))
    c = numpyro.deterministic("c_week", c0 + jnp.cumsum(z) * sigma_c)

    beta = numpyro.sample("beta_vp50", dist.Normal(10.0, 5.0))  # bpm per 50 W
    gamma_T = numpyro.sample("gamma_T", dist.Normal(0.5, 0.5))  # bpm per degC above 20
    gamma_d = numpyro.sample("gamma_d", dist.Normal(4.0, 3.0))  # bpm per hour
    gamma_dT = numpyro.sample("gamma_dT", dist.Normal(0.0, 0.5))

    sigma_u = numpyro.sample("sigma_u", dist.HalfNormal(6.0))
    u = numpyro.sample("u_seg", dist.Normal(0.0, 1.0).expand([d.n_seg])) * sigma_u
    sigma_a = numpyro.sample("sigma_ride", dist.HalfNormal(5.0))
    a = numpyro.sample("a_ride", dist.Normal(0.0, 1.0).expand([d.n_ride])) * sigma_a

    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))
    sigma = numpyro.sample("sigma", dist.HalfNormal(4.0))

    mu = (
        c[d.week]
        + beta * (d.vp - VP_REF) / VP_SCALE
        + gamma_T * d.temp20
        + gamma_d * d.elapsed_h
        + gamma_dT * d.temp20 * d.elapsed_h
        + u[d.seg]
        + a[d.ride]
        + extra_term(extra)
    )
    numpyro.deterministic("mu", mu)
    numpyro.sample("hr", dist.StudentT(nu + 2.0, mu, sigma), obs=hr_obs)


def fit_hr(
    d: HRData,
    seed: int = 0,
    warmup: int = 1000,
    samples: int = 1000,
    chains: int = 2,
    extra: Extra = None,
) -> az.InferenceData:
    numpyro.set_host_device_count(chains)
    kernel = NUTS(hr_model, target_accept_prob=0.9)
    mcmc = MCMC(
        kernel,
        num_warmup=warmup,
        num_samples=samples,
        num_chains=chains,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), d, hr_obs=d.hr, extra=extra)
    idata = az.from_numpyro(mcmc)
    return idata


def summarize(idata: az.InferenceData, d: HRData) -> pl.DataFrame:
    post = idata.posterior
    rows = []
    for v in (
        "beta_vp50",
        "gamma_T",
        "gamma_d",
        "gamma_dT",
        "sigma_u",
        "sigma_ride",
        "sigma",
        "nu",
        "sigma_c",
        "c0",
    ):
        x = post[v].values.ravel()
        rows.append(
            {
                "param": v,
                "mean": float(x.mean()),
                "sd": float(x.std()),
                "q05": float(np.quantile(x, 0.05)),
                "q95": float(np.quantile(x, 0.95)),
            }
        )
    cw = post["c_week"].values.reshape(-1, d.n_week)
    first, last = cw[:, d.week.min()], cw[:, d.week.max()]
    delta = last - first
    rows.append(
        {
            "param": "c_week_last_minus_first",
            "mean": float(delta.mean()),
            "sd": float(delta.std()),
            "q05": float(np.quantile(delta, 0.05)),
            "q95": float(np.quantile(delta, 0.95)),
        }
    )
    rows.append(
        {
            "param": "P(c_week decreased)",
            "mean": float((delta < 0).mean()),
            "sd": 0.0,
            "q05": 0.0,
            "q95": 0.0,
        }
    )
    return pl.DataFrame(rows)


def ppc_stats(idata: az.InferenceData, d: HRData) -> dict[str, float]:
    mu = idata.posterior["mu"].values.reshape(-1, len(d.hr))
    pred = mu.mean(0)
    resid = d.hr - pred
    sd = idata.posterior["sigma"].values.mean()
    lo, hi = np.quantile(mu, [0.05, 0.95], axis=0)
    return {
        "n": int(len(d.hr)),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": float(np.mean(np.abs(resid))),
        "resid_sd": float(resid.std()),
        "sigma_post": float(sd),
        "coverage_90_mu": float(
            np.mean((d.hr >= lo - 1.64 * sd) & (d.hr <= hi + 1.64 * sd))
        ),
        "r2": float(1 - resid.var() / d.hr.var()),
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    TWIN_DIR.mkdir(parents=True, exist_ok=True)
    d = load_hr_data()
    log.info(
        "HR model data: %d traversals, %d segments, %d rides, %d weeks",
        len(d.hr),
        d.n_seg,
        d.n_ride,
        d.n_week,
    )
    idata = fit_hr(d)
    div = int(idata.sample_stats["diverging"].values.sum())
    rhat = az.rhat(
        idata,
        var_names=[
            "beta_vp50",
            "gamma_T",
            "gamma_d",
            "gamma_dT",
            "sigma_c",
            "sigma_u",
            "sigma",
        ],
    )
    log.info(
        "divergences=%d  max rhat=%.3f",
        div,
        max(float(v.values.max()) for v in rhat.data_vars.values()),
    )
    summ = summarize(idata, d)
    print(summ.to_pandas().to_string(index=False))
    stats = ppc_stats(idata, d)
    print("PPC:", stats)
    seg_u = (
        idata.posterior["u_seg"].values.reshape(-1, d.n_seg).mean(0)
        * idata.posterior["sigma_u"].values.mean()
    )
    print(
        pl.DataFrame({"segment": d.seg_names, "u_seg_bpm": np.round(seg_u, 1)})
        .to_pandas()
        .to_string(index=False)
    )
    idata.to_netcdf(str(TWIN_DIR / "hr_model.nc"))
    summ.write_parquet(TWIN_DIR / "hr_model_summary.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
