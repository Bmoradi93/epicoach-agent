"""Seed the registry with the core pre-specified hypotheses and the evidence the
twin fits provide for them, then pre-register ledger predictions for the next
instrument ride.

    pixi run -e research python -m garmin_lab.registry.seed               # wipe + reseed
    pixi run -e research python -m garmin_lab.registry.seed --ledger-only # re-issue predictions only

Evidence is a Savage-Dickey-style log Bayes factor for a directional claim:
log BF = log(posterior odds) - log(prior odds), both computed from samples.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
import math

import arviz as az
import numpy as np
import polars as pl

from garmin_lab.ingest.client import REPO_ROOT
from garmin_lab.registry.store import Registry
from garmin_lab.segments.routes import INSTRUMENT_CLIMB, INSTRUMENT_DESCENT
from garmin_lab.twin.ffm import TAU1_HI, TAU1_LO, bhattacharyya

MIN_SHRINKAGE = (
    0.10  # posterior sd must shrink >= 10% relative to the prior for the data to count
)

log = logging.getLogger("garmin_lab.registry.seed")
TWIN = REPO_ROOT / "data" / "twin"
REG_PATH = REPO_ROOT / "data" / "registry.sqlite"
INSTRUMENT = {"climb": INSTRUMENT_CLIMB, "descent": INSTRUMENT_DESCENT}


def directional_log_bf(
    post: np.ndarray, prior: np.ndarray, op: str, value: float
) -> tuple[float, float, float]:
    f = (lambda x: x > value) if op == ">" else (lambda x: x < value)
    pp = float(np.clip(f(post).mean(), 1e-4, 1 - 1e-4))
    pq = float(np.clip(f(prior).mean(), 1e-4, 1 - 1e-4))
    return math.log(pp / (1 - pp)) - math.log(pq / (1 - pq)), pp, pq


def register_ledger(
    reg: Registry,
    feats: pl.DataFrame,
    fits: dict[str, tuple[str, object]],
    h_climb: int,
    h_descent: int,
    days_ahead: int = 14,
) -> int:
    """Pre-register the two instrument predictions (climb moving time at 160 bpm from the
    FFM; descent moving time from the skill model) for the next ride within ``days_ahead``.
    Returns the number of predictions registered."""
    from garmin_lab.agents.tools import _descent_prediction, _instrument_prediction

    target_time = dt.datetime.combine(
        dt.date.today() + dt.timedelta(days=days_ahead),
        dt.time(23, 59),
        tzinfo=dt.timezone.utc,
    )
    n = 0
    open_targets = {
        p["target"]
        for p in reg.ledger()
        if not p["scored_at"] and not p["withdrawn_at"]
    }
    ip = _instrument_prediction(feats)
    if f"moving_s@{INSTRUMENT['climb']}" in open_targets:
        log.info("ledger: climb prediction already open, not re-issued")
    elif "instrument_predicted_moving_s_at_160" in ip:
        reg.predict(
            h_climb,
            f"moving_s@{INSTRUMENT['climb']}",
            target_time,
            {
                "family": "normal",
                "mu": ip["instrument_predicted_moving_s_at_160"],
                "sigma": ip["instrument_predicted_sd_s"],
                "conditions": {
                    "hr_bpm": 160,
                    "protocol": "TCT east gate -> University Drive, lap at the road",
                },
            },
        )
        n += 1
    dp = _descent_prediction(feats)
    if f"moving_s@{INSTRUMENT['descent']}" in open_targets:
        log.info("ledger: descent prediction already open, not re-issued")
    elif "descent_predicted_log_mu" in dp:
        reg.predict(
            h_descent,
            f"moving_s@{INSTRUMENT['descent']}",
            target_time,
            {
                "family": "lognormal",
                "mu": dp["descent_predicted_log_mu"],
                "sigma": dp["descent_predicted_log_sd"],
                "conditions": dp["descent_conditions"],
            },
        )
        n += 1
    elif dp.get("descent_note"):
        log.warning("ledger: no descent prediction: %s", dp["descent_note"])
    log.info("ledger: %d predictions registered for before %s", n, target_time.date())
    return n


def load_fits(reg: Registry, cutoff) -> dict[str, tuple[str, object]]:
    """Register the stored twin fits (content-hashed) and return {name: (fit_id, posterior)}."""
    fits = {}
    for name in ("hr_model", "skill_model", "ffm_full"):
        p = TWIN / f"{name}.nc"
        fid = f"{name}:{hashlib.sha1(p.read_bytes()).hexdigest()[:10]}"
        reg.register_fit(
            fid, model_spec_hash=name, data_cutoff=cutoff, posterior_path=str(p)
        )
        fits[name] = (fid, az.from_netcdf(p).posterior)
    return fits


def ledger_only() -> int:
    """Re-issue the ledger predictions on the EXISTING registry (no wipe), e.g. after the
    instrument changed. The hypotheses are found by their formal spec params."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    feats = pl.read_parquet(
        REPO_ROOT / "data" / "dataset" / "traversal_features.parquet"
    )
    reg = Registry(REG_PATH)
    fits = load_fits(reg, feats["date"].max())
    by_param = {
        (
            (h["formal_spec"] or {}).get("param")
            if isinstance(h["formal_spec"], dict)
            else None
        ): h["id"]
        for h in reg.list()
    }
    h_climb, h_descent = by_param.get("gamma_d"), by_param.get("delta_sleep")
    if h_climb is None or h_descent is None:
        log.error("could not find gamma_d / delta_sleep hypotheses in the registry")
        return 1
    register_ledger(reg, feats, fits, h_climb=h_climb, h_descent=h_descent)
    for p in reg.ledger():
        if p["withdrawn_at"] is None and p["scored_at"] is None:
            print(
                f"  pred {p['id']} (H{p['hypothesis_id']}) {p['target']}  by {p['target_time'][:10]}  {p['distribution'][:90]}"
            )
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    rng = np.random.default_rng(0)
    feats = pl.read_parquet(
        REPO_ROOT / "data" / "dataset" / "traversal_features.parquet"
    )
    cutoff = feats["date"].max()
    if REG_PATH.exists():
        REG_PATH.unlink()
    reg = Registry(REG_PATH)

    fits = load_fits(reg, cutoff)
    n = 4000

    def ev(hid, name, post_var, prior, op, value, label, log_scale=False):
        """Directional evidence, gated by identifiability: if the posterior sd did not shrink by at
        least MIN_SHRINKAGE relative to the prior, the data did not speak and 0 evidence is recorded
        (the test still counts toward the budget, so the hypothesis can reach 'undetermined').
        """
        fid, post = fits[name]
        x = post[post_var].values.ravel()
        lbf, pp, pq = directional_log_bf(x, prior, op, value)
        xs, ps = (np.log(x), np.log(prior)) if log_scale else (x, prior)
        shrink = 1.0 - xs.std() / ps.std()
        bc = bhattacharyya(ps, xs)
        method = f"savage-dickey {label} (shrinkage {shrink:.2f}, BC {bc:.2f})"
        if shrink < MIN_SHRINKAGE:
            method = f"identifiability gate: shrinkage {shrink:.2f} < {MIN_SHRINKAGE}; raw logBF {lbf:+.2f} not admitted"
            lbf = 0.0
        b = reg.record_evidence(hid, lbf, fid, method=method, as_of=cutoff)
        log.info(
            "H%d %-20s post P=%.3f prior P=%.3f shrink=%+.2f BC=%.2f logBF=%+.2f -> belief %.2f",
            hid,
            label,
            pp,
            pq,
            shrink,
            bc,
            lbf,
            b,
        )

    # H1 cardiac drift exists
    h1 = reg.register(
        "My heart rate drifts upward within a ride at matched power (cardiac drift > 0 bpm/h).",
        {"model": "hr_model", "param": "gamma_d", "op": ">", "value": 0.0},
        prior_prob=0.5,
        actor="pre-specified",
    )
    ev(h1, "hr_model", "gamma_d", rng.normal(4, 3, n), ">", 0.0, "gamma_d > 0")

    # H2 drift is worse in heat
    h2 = reg.register(
        "My cardiac drift is larger on hot days (heat x drift interaction > 0).",
        {"model": "hr_model", "param": "gamma_dT", "op": ">", "value": 0.0},
        prior_prob=0.5,
        actor="pre-specified",
    )
    ev(h2, "hr_model", "gamma_dT", rng.normal(0, 0.5, n), ">", 0.0, "gamma_dT > 0")

    # H3 short sleep slows technical descents
    h3 = reg.register(
        "Less sleep the night before slows my technical descents (delta_sleep < 0 per hour of sleep).",
        {"model": "skill_model", "param": "delta_sleep", "op": "<", "value": 0.0},
        prior_prob=0.5,
        actor="pre-specified",
    )
    ev(
        h3,
        "skill_model",
        "delta_sleep",
        rng.normal(0, 0.03, n),
        "<",
        0.0,
        "delta_sleep < 0",
    )

    # H4 fitness time constant shorter than the textbook 42 d
    h4 = reg.register(
        "My fitness time constant is shorter than the population default (tau1 < 42 d).",
        {"model": "ffm", "param": "tau1", "op": "<", "value": 42.0},
        prior_prob=0.5,
        actor="pre-specified",
        test_budget=4,
    )
    prior_tau1 = TAU1_LO + (TAU1_HI - TAU1_LO) * rng.beta(2.2, 3.0, n)
    ev(h4, "ffm_full", "tau1", prior_tau1, "<", 42.0, "tau1 < 42", log_scale=True)

    # H5 aerobic response improved over the record
    h5 = reg.register(
        "My heart rate at a reference climbing power has decreased over the 18-month record (aerobic response improved).",
        {
            "model": "hr_model",
            "param": "c_week_last_minus_first",
            "op": "<",
            "value": 0.0,
        },
        prior_prob=0.5,
        actor="pre-specified",
    )
    fid, post = fits["hr_model"]
    cw = post["c_week"].values
    cw = cw.reshape(-1, cw.shape[-1])
    delta = cw[:, -1] - cw[:, 0]
    lbf, pp, pq = directional_log_bf(delta, rng.normal(0, 1, n), "<", 0.0)
    reg.record_evidence(
        h5, lbf, fid, method="savage-dickey c_week delta < 0", as_of=cutoff
    )
    log.info(
        "H%d c_week decreased: post P=%.3f  logBF=%+.2f -> belief %.2f",
        h5,
        pp,
        lbf,
        reg.belief(h5),
    )

    register_ledger(reg, feats, fits, h_climb=h1, h_descent=h3)

    print("\nRegistry state:")
    for h in reg.list():
        print(
            f"  H{h['id']}  belief={h['belief']:.2f}  state={h['state']:<10s}  {h['text']}"
        )
    print("\nLedger:")
    for p in reg.ledger():
        print(
            f"  pred {p['id']} (H{p['hypothesis_id']}) {p['target']}  by {p['target_time'][:10]}  {p['distribution'][:90]}"
        )
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(ledger_only() if "--ledger-only" in sys.argv else main())
