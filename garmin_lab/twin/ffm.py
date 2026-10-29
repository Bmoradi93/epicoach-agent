"""Fitness-fatigue (Banister) model with learned time constants, measured
through heart-rate-adjusted climb speed on repeated segments.

Latent states (daily, load w_d in exp-TRIMP/100):

    F_d = exp(-1/tau1) F_{d-1} + w_d        fitness
    G_d = exp(-1/tau2) G_{d-1} + w_d        fatigue
    p_d = k1 F_d - k2 G_d                   performance (log-speed units)

Measurement on climb traversal i (day d(i), segment s(i)):

    log v_i = a_s + p_d + b_hr (HR_i - 145)/10 + b_T (T_i - 20) + b_el elapsed_h_i + eps_i

i.e. speed at matched internal cost. ``k1`` sets the scale of p (b is fixed
to 1), so k1, k2 are in log-speed per unit of filtered load.
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
from garmin_lab.twin.hr_model import TRAIL_HIGHWAYS, Extra, extra_term

log = logging.getLogger("garmin_lab.twin.ffm")
DATASET = REPO_ROOT / "data" / "dataset"
TWIN_DIR = REPO_ROOT / "data" / "twin"
LOAD_SCALE = 100.0
TAU1_LO, TAU1_HI = 15.0, 90.0
HR_REF = 145.0
PRIOR = {
    "tau1_mu": np.log(42.0),
    "tau1_sd": 0.3,
    "tau2_mu": np.log(7.0),
    "tau2_sd": 0.4,
    "ratio_mu": np.log(2.0),
    "ratio_sd": 0.3,
}


@dataclass
class FFMData:
    w: np.ndarray  # daily load (scaled), length D
    day0: object
    log_v: np.ndarray
    day: np.ndarray  # index into w
    seg: np.ndarray
    hr10: np.ndarray
    temp20: np.ndarray
    elapsed_h: np.ndarray
    n_seg: int
    seg_names: list[str]
    dates: np.ndarray
    season_sin: np.ndarray
    season_cos: np.ndarray
    frame: pl.DataFrame | None = None


def load_ffm_data(
    load_col: str = "exp_trimp",
    until_day: int | None = None,
    features: pl.DataFrame | None = None,
) -> FFMData:
    load = pl.read_parquet(DATASET / "daily_load.parquet").sort("date")
    f = (
        features
        if features is not None
        else pl.read_parquet(DATASET / "traversal_features.parquet")
    )
    f = f.filter(
        (pl.col("kind") == "climb")
        & pl.col("hr_mean").is_not_null()
        & (pl.col("hr_ok_frac") >= 0.7)
        & pl.col("temp_c20").is_not_null()
        & pl.col("osm_highway").is_in(TRAIL_HIGHWAYS)
        & (pl.col("moving_speed_ms") > 0.5)
    )
    day0 = load["date"].min()
    f = f.with_columns((pl.col("date") - day0).dt.total_days().alias("d"))
    if until_day is not None:
        f = f.filter(pl.col("d") <= until_day)
        load = load.filter((pl.col("date") - day0).dt.total_days() <= until_day)
    segs = sorted(f["segment_id"].unique().to_list())
    seg_map = {s: i for i, s in enumerate(segs)}
    return FFMData(
        w=load[load_col].to_numpy().astype(float) / LOAD_SCALE,
        day0=day0,
        log_v=np.log(f["moving_speed_ms"].to_numpy().astype(float)),
        day=f["d"].to_numpy().astype(int),
        seg=np.array([seg_map[s] for s in f["segment_id"]]),
        hr10=(f["hr_mean"].to_numpy().astype(float) - HR_REF) / 10.0,
        temp20=f["temp_c20"].to_numpy().astype(float),
        elapsed_h=f["elapsed_h"].to_numpy().astype(float),
        n_seg=len(segs),
        seg_names=segs,
        dates=f["date"].to_numpy(),
        season_sin=np.sin(2 * np.pi * f["date"].dt.ordinal_day().to_numpy() / 365.25),
        season_cos=np.cos(2 * np.pi * f["date"].dt.ordinal_day().to_numpy() / 365.25),
        frame=f,
    )


def _filter(w: jnp.ndarray, decay: jnp.ndarray) -> jnp.ndarray:
    def step(carry, wd):
        new = decay * carry + wd
        return new, new

    _, out = jax.lax.scan(step, 0.0, w)
    return out


def ffm_model(
    d: FFMData,
    obs: np.ndarray | None = None,
    with_fatigue: bool = True,
    with_season: bool = True,
    extra: Extra = None,
) -> None:
    # tau1 bounded to the literature range [15, 90] d; prior mass centred near 45 d
    u1 = numpyro.sample("u_tau1", dist.Beta(2.2, 3.0))
    tau1 = numpyro.deterministic("tau1", TAU1_LO + (TAU1_HI - TAU1_LO) * u1)
    k1 = numpyro.sample("k1", dist.HalfNormal(0.02))
    F = _filter(jnp.asarray(d.w), jnp.exp(-1.0 / tau1))
    if with_fatigue:
        # physiological ordering: fatigue decays faster than fitness, and weighs more per unit load
        r = numpyro.sample(
            "tau_ratio", dist.Beta(2.0, 8.0)
        )  # tau2/tau1, prior mean 0.2 (~7/42)
        tau2 = numpyro.deterministic("tau2", tau1 * r)
        excess = numpyro.sample("k_excess", dist.LogNormal(0.0, 0.5))
        ratio = numpyro.deterministic("k2_over_k1", 1.0 + excess)
        k2 = numpyro.deterministic("k2", k1 * ratio)
        G = _filter(jnp.asarray(d.w), jnp.exp(-1.0 / tau2))
        p = k1 * F - k2 * G
    else:
        p = k1 * F
    p = p - jnp.mean(p)  # centre so a_seg is identifiable
    numpyro.deterministic("p_day", p)

    a = numpyro.sample(
        "a_seg", dist.Normal(0.8, 0.5).expand([d.n_seg])
    )  # log m/s ~ 2.2 m/s
    b_hr = numpyro.sample("b_hr10", dist.Normal(0.05, 0.05))  # log-speed per 10 bpm
    b_T = numpyro.sample("b_T", dist.Normal(0.0, 0.01))
    b_el = numpyro.sample("b_elapsed", dist.Normal(0.0, 0.05))
    # seasonal nuisance (trail conditions): annual harmonic of day-of-year
    if with_season:
        b_s = numpyro.sample("b_season_sin", dist.Normal(0.0, 0.05))
        b_c = numpyro.sample("b_season_cos", dist.Normal(0.0, 0.05))
    else:
        b_s = b_c = 0.0
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.1))
    mu = (
        a[d.seg]
        + p[d.day]
        + b_hr * d.hr10
        + b_T * d.temp20
        + b_el * d.elapsed_h
        + b_s * d.season_sin
        + b_c * d.season_cos
        + extra_term(extra)
    )
    numpyro.deterministic("mu", mu)
    numpyro.sample("log_v", dist.StudentT(nu + 2.0, mu, sigma), obs=obs)


def fit_ffm(
    d: FFMData,
    with_fatigue: bool = True,
    with_season: bool = True,
    seed: int = 0,
    warmup: int = 1500,
    samples: int = 1500,
    chains: int = 2,
    extra: Extra = None,
) -> az.InferenceData:
    numpyro.set_host_device_count(chains)
    mcmc = MCMC(
        NUTS(ffm_model, target_accept_prob=0.95),
        num_warmup=warmup,
        num_samples=samples,
        num_chains=chains,
        progress_bar=False,
    )
    mcmc.run(
        jax.random.PRNGKey(seed),
        d,
        obs=d.log_v,
        with_fatigue=with_fatigue,
        with_season=with_season,
        extra=extra,
    )
    return az.from_numpyro(mcmc, log_likelihood=True)


def bhattacharyya(prior: np.ndarray, post: np.ndarray, bins: int = 60) -> float:
    lo, hi = min(prior.min(), post.min()), max(prior.max(), post.max())
    hp, _ = np.histogram(prior, bins=bins, range=(lo, hi), density=True)
    hq, _ = np.histogram(post, bins=bins, range=(lo, hi), density=True)
    width = (hi - lo) / bins
    return float(np.sum(np.sqrt(hp * hq)) * width)


def tau_report(idata: az.InferenceData, rng: np.random.Generator) -> pl.DataFrame:
    post = idata.posterior
    rows = []
    n = post["tau1"].values.size
    tau1_prior = TAU1_LO + (TAU1_HI - TAU1_LO) * rng.beta(2.2, 3.0, size=n)
    priors = {
        "tau1": tau1_prior,
        "tau2": tau1_prior * rng.beta(2.0, 8.0, size=n),
        "k2_over_k1": 1.0 + rng.lognormal(0.0, 0.5, size=n),
    }
    for name, prior in priors.items():
        if name not in post:
            continue
        x = post[name].values.ravel()
        mu, sd = float(np.log(prior).mean()), float(np.log(prior).std())
        rows.append(
            {
                "param": name,
                "prior_median": float(np.exp(mu)),
                "post_median": float(np.median(x)),
                "post_q05": float(np.quantile(x, 0.05)),
                "post_q95": float(np.quantile(x, 0.95)),
                "prior_sd_log": sd,
                "post_sd_log": float(np.log(x).std()),
                "overlap_BC": bhattacharyya(np.log(prior), np.log(x)),
            }
        )
    x = post["tau1"].values.ravel()
    rows.append(
        {
            "param": "P(tau1 < 42)",
            "prior_median": 0.5,
            "post_median": float((x < 42).mean()),
            "post_q05": np.nan,
            "post_q95": np.nan,
            "prior_sd_log": np.nan,
            "post_sd_log": np.nan,
            "overlap_BC": np.nan,
        }
    )
    return pl.DataFrame(rows)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    TWIN_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    d = load_ffm_data()
    log.info(
        "FFM data: %d days of load, %d climb traversals on %d segments",
        len(d.w),
        len(d.log_v),
        d.n_seg,
    )

    full = fit_ffm(d, with_fatigue=True)
    log.info("full: divergences=%d", int(full.sample_stats["diverging"].values.sum()))
    print(tau_report(full, rng).to_pandas().to_string(index=False))
    post = full.posterior
    for v in (
        "k1",
        "k2",
        "b_hr10",
        "b_T",
        "b_elapsed",
        "b_season_sin",
        "b_season_cos",
        "sigma",
    ):
        if v not in post:
            continue
        x = post[v].values.ravel()
        print(
            f"{v:10s} mean={x.mean():+.4f} sd={x.std():.4f} 90%=[{np.quantile(x,0.05):+.4f},{np.quantile(x,0.95):+.4f}]"
        )
    p = post["p_day"].values.reshape(-1, len(d.w))
    pm = p.mean(0)
    print(
        "performance p_day: range of posterior mean = %.3f log-speed (%.1f%% speed swing); mean posterior sd = %.3f"
        % (
            pm.max() - pm.min(),
            100 * (np.exp(pm.max() - pm.min()) - 1),
            p.std(0).mean(),
        )
    )

    fit_only = fit_ffm(d, with_fatigue=False)
    no_season = fit_ffm(d, with_fatigue=True, with_season=False)
    print("\n-- sensitivity: no seasonal nuisance --")
    print(tau_report(no_season, rng).to_pandas().to_string(index=False))
    pn = no_season.posterior["p_day"].values.reshape(-1, len(d.w)).mean(0)
    print(
        "no-season performance swing: %.1f%%"
        % (100 * (np.exp(pn.max() - pn.min()) - 1))
    )
    cmp = az.compare(
        {
            "fitness+fatigue": full,
            "fitness_only": fit_only,
            "fitness+fatigue, no season": no_season,
        }
    )
    print(cmp.to_string())
    no_season.to_netcdf(str(TWIN_DIR / "ffm_no_season.nc"))

    full.to_netcdf(str(TWIN_DIR / "ffm_full.nc"))
    fit_only.to_netcdf(str(TWIN_DIR / "ffm_fitness_only.nc"))
    tau_report(full, rng).write_parquet(TWIN_DIR / "ffm_tau_report.parquet")
    pl.DataFrame(
        {
            "date": pl.date_range(
                d.day0, d.day0 + pl.duration(days=len(d.w) - 1), "1d", eager=True
            ),
            "p_mean": pm,
            "p_sd": p.std(0),
            "load": d.w * LOAD_SCALE,
        }
    ).write_parquet(TWIN_DIR / "ffm_p_day.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
