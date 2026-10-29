"""Synthetic parameter-recovery check for the fitness-fatigue model.

Simulates HR-adjusted climb speeds from known (tau1, tau2, k1, k2) using the
athlete's *real* load history and *real* observation design (days, segments,
heart rates), then refits. Answers: could this sampling design recover the
time constants even if the model were exactly right?

    pixi run -e research python -m garmin_lab.twin.recovery
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
import numpy as np
import polars as pl

from garmin_lab.twin.ffm import (
    TWIN_DIR,
    FFMData,
    _filter,
    fit_ffm,
    load_ffm_data,
)

log = logging.getLogger("garmin_lab.twin.recovery")

TRUE = {
    "tau1": 35.0,
    "tau2": 7.0,
    "k1": 0.010,
    "k2_over_k1": 2.0,
    "b_hr10": 0.025,
    "sigma": 0.06,
}


def simulate(
    d: FFMData, rng: np.random.Generator, truth: dict[str, float]
) -> np.ndarray:
    w = jnp.asarray(d.w)
    F = np.asarray(_filter(w, jnp.exp(-1.0 / truth["tau1"])))
    G = np.asarray(_filter(w, jnp.exp(-1.0 / truth["tau2"])))
    p = truth["k1"] * F - truth["k1"] * truth["k2_over_k1"] * G
    p = p - p.mean()
    a_seg = np.log(
        np.array([np.exp(d.log_v[d.seg == s]).mean() for s in range(d.n_seg)])
    )
    mu = a_seg[d.seg] + p[d.day] + truth["b_hr10"] * d.hr10
    return mu + rng.standard_t(6, size=len(mu)) * truth["sigma"], p


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    d = load_ffm_data()
    rows = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        y, p_true = simulate(d, rng, TRUE)
        d_sim = FFMData(**{**d.__dict__, "log_v": y})
        idata = fit_ffm(d_sim, with_fatigue=True, with_season=True, seed=seed)
        post = idata.posterior
        for name in ("tau1", "tau2", "k2_over_k1", "k1"):
            x = post[name].values.ravel()
            rows.append(
                {
                    "seed": seed,
                    "param": name,
                    "true": TRUE[name],
                    "post_median": float(np.median(x)),
                    "q05": float(np.quantile(x, 0.05)),
                    "q95": float(np.quantile(x, 0.95)),
                    "covered": bool(
                        np.quantile(x, 0.05) <= TRUE[name] <= np.quantile(x, 0.95)
                    ),
                }
            )
        p_hat = post["p_day"].values.reshape(-1, len(d.w)).mean(0)
        rows.append(
            {
                "seed": seed,
                "param": "corr(p_true,p_hat)",
                "true": 1.0,
                "post_median": float(np.corrcoef(p_true, p_hat)[0, 1]),
                "q05": np.nan,
                "q95": np.nan,
                "covered": True,
            }
        )
        log.info(
            "seed %d done (true swing %.1f%%)",
            seed,
            100 * (np.exp(p_true.max() - p_true.min()) - 1),
        )
    df = pl.DataFrame(rows)
    print(df.to_pandas().to_string(index=False))
    df.write_parquet(TWIN_DIR / "ffm_recovery.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
