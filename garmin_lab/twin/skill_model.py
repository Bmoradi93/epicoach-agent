"""Latent technical-descending skill from matched descent traversals.

    log T_i = mu_{seg(i)} + lambda_{seg(i)} * S_{w(i)} + delta_wet * wet_i
              + delta_sleep * (sleep_h_i - 7) + delta_elapsed * elapsed_h_i + eps_i

``S_w`` is a weekly Gaussian random walk (skill; lower log-time = faster);
``lambda_seg >= 0`` is how strongly a segment's time depends on skill
(technical segments load more). ``T`` is moving time in seconds.
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
from garmin_lab.twin.hr_model import Extra, extra_term

log = logging.getLogger("garmin_lab.twin.skill")
DATASET = REPO_ROOT / "data" / "dataset"
TWIN_DIR = REPO_ROOT / "data" / "twin"


@dataclass
class SkillData:
    log_t: np.ndarray
    seg: np.ndarray
    week: np.ndarray
    wet: np.ndarray
    sleep_dev: np.ndarray
    sleep_obs: np.ndarray
    elapsed_h: np.ndarray
    n_seg: int
    n_week: int
    seg_names: list[str]
    frame: pl.DataFrame | None = None


def load_skill_data(
    features: pl.DataFrame | None = None, until_day: int | None = None
) -> SkillData:
    f = (
        features
        if features is not None
        else pl.read_parquet(DATASET / "traversal_features.parquet")
    )
    f = f.filter(
        (pl.col("kind") == "descent")
        & (pl.col("moving_s") > 30)
        & (~pl.col("has_stop"))
    )
    if until_day is not None:
        f = f.filter(pl.col("day_index") <= until_day)
    segs = sorted(f["segment_id"].unique().to_list())
    seg_map = {s: i for i, s in enumerate(segs)}
    sleep = f["sleep_h"].to_numpy().astype(float)
    sleep_obs = ~np.isnan(sleep)
    return SkillData(
        log_t=np.log(f["moving_s"].to_numpy().astype(float)),
        seg=np.array([seg_map[s] for s in f["segment_id"]]),
        week=f["week"].to_numpy().astype(int),
        wet=f["wet_now"].to_numpy().astype(float),
        sleep_dev=np.where(sleep_obs, sleep - 7.0, 0.0),
        sleep_obs=sleep_obs.astype(float),
        elapsed_h=f["elapsed_h"].to_numpy().astype(float),
        n_seg=len(segs),
        n_week=int(f["week"].max()) + 1,
        seg_names=segs,
        frame=f,
    )


def skill_model(
    d: SkillData, obs: np.ndarray | None = None, extra: Extra = None
) -> None:
    mu_seg = numpyro.sample("mu_seg", dist.Normal(5.0, 1.0).expand([d.n_seg]))
    lam = numpyro.sample("lambda_seg", dist.HalfNormal(0.1).expand([d.n_seg]))
    sigma_S = numpyro.sample("sigma_S", dist.HalfNormal(0.3))
    z = numpyro.sample("z_S", dist.Normal(0.0, 1.0).expand([d.n_week]))
    S = numpyro.deterministic(
        "S_week", jnp.cumsum(z) * sigma_S
    )  # S_0 = z_0*sigma; scale fixed by lambda prior
    delta_wet = numpyro.sample("delta_wet", dist.Normal(0.0, 0.05))
    delta_sleep = numpyro.sample(
        "delta_sleep", dist.Normal(0.0, 0.03)
    )  # per hour of sleep deviation
    delta_el = numpyro.sample("delta_elapsed", dist.Normal(0.0, 0.05))
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.1))
    mu = (
        mu_seg[d.seg]
        + lam[d.seg] * S[d.week]
        + delta_wet * d.wet
        + delta_sleep * d.sleep_dev * d.sleep_obs
        + delta_el * d.elapsed_h
        + extra_term(extra)
    )
    numpyro.deterministic("mu", mu)
    numpyro.sample("log_t", dist.StudentT(nu + 2.0, mu, sigma), obs=obs)


def fit_skill(
    d: SkillData,
    seed: int = 0,
    warmup: int = 1000,
    samples: int = 1000,
    chains: int = 2,
    extra: Extra = None,
) -> az.InferenceData:
    numpyro.set_host_device_count(chains)
    mcmc = MCMC(
        NUTS(skill_model, target_accept_prob=0.9),
        num_warmup=warmup,
        num_samples=samples,
        num_chains=chains,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), d, obs=d.log_t, extra=extra)
    return az.from_numpyro(mcmc)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    TWIN_DIR.mkdir(parents=True, exist_ok=True)
    d = load_skill_data()
    log.info(
        "skill data: %d traversals, %d segments, sleep observed for %.0f%%",
        len(d.log_t),
        d.n_seg,
        100 * d.sleep_obs.mean(),
    )
    idata = fit_skill(d)
    div = int(idata.sample_stats["diverging"].values.sum())
    log.info("divergences=%d", div)
    post = idata.posterior
    rows = []
    for v in ("delta_wet", "delta_sleep", "delta_elapsed", "sigma_S", "sigma", "nu"):
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
    # percent effect of 1 h less sleep on descent time
    ds = post["delta_sleep"].values.ravel()
    rows.append(
        {
            "param": "pct_slower_per_hour_less_sleep",
            "mean": float(100 * (np.exp(-ds) - 1).mean()),
            "sd": float(100 * (np.exp(-ds) - 1).std()),
            "q05": float(100 * np.quantile(np.exp(-ds) - 1, 0.05)),
            "q95": float(100 * np.quantile(np.exp(-ds) - 1, 0.95)),
        }
    )
    summ = pl.DataFrame(rows)
    print(summ.to_pandas().to_string(index=False))
    lam = post["lambda_seg"].values.reshape(-1, d.n_seg).mean(0)
    mu_seg = post["mu_seg"].values.reshape(-1, d.n_seg).mean(0)
    print(
        pl.DataFrame(
            {
                "segment": d.seg_names,
                "lambda": np.round(lam, 3),
                "median_moving_s": np.round(np.exp(mu_seg)),
                "n": [int((d.seg == i).sum()) for i in range(d.n_seg)],
            }
        )
        .to_pandas()
        .to_string(index=False)
    )
    mu = post["mu"].values.reshape(-1, len(d.log_t)).mean(0)
    resid = d.log_t - mu
    print(
        "PPC: rmse(log)=%.3f  ->  %.1f%% timing error; r2=%.2f"
        % (
            np.sqrt((resid**2).mean()),
            100 * np.sqrt((resid**2).mean()),
            1 - resid.var() / d.log_t.var(),
        )
    )
    idata.to_netcdf(str(TWIN_DIR / "skill_model.nc"))
    summ.write_parquet(TWIN_DIR / "skill_model_summary.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
