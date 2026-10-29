"""Deterministic tools exposed to the ADK agents.

Everything that produces a number the system acts on lives here. LLM agents
may call these; they cannot set beliefs, compute evidence, or bypass safety.
All tools return JSON-serialisable dicts.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import polars as pl

from garmin_lab.agents.compile import (
    build_terms,
    is_structural,
    savage_dickey_at_zero,
)
from garmin_lab.agents.eig import EIGScorer, default_menu
from garmin_lab.agents.safety import AthleteState, check_week
from garmin_lab.agents.templates import (
    TemplateError,
    validate_spec,
    vocabulary_text,
)
from garmin_lab.ingest.client import REPO_ROOT
from garmin_lab.registry.store import Registry, RegistryError
from garmin_lab.segments.routes import INSTRUMENT_CLIMB, INSTRUMENT_DESCENT
from garmin_lab.twin.ffm import TWIN_DIR, bhattacharyya

DATASET = REPO_ROOT / "data" / "dataset"
REG_PATH = REPO_ROOT / "data" / "registry.sqlite"
CYCLE_DIR = REPO_ROOT / "data" / "cycles"
INSTRUMENT_HR = 160.0
INSTRUMENT_LAP_NOTE = "lap where the Trans Canada Trail gravel meets University Drive"
FIT_NAME = {"hr_model": "hr_model", "skill_model": "skill_model", "ffm": "ffm_full"}
SHRINK_GATE = 0.10


@dataclass
class Context:
    """Where the tools look. The live cycle uses the stored twin and the full dataset; the replay
    points this at a per-origin twin directory, the features visible at that origin, a separate
    registry and a frozen clock, so every tool is causal with respect to the origin."""

    twin_dir: Path = TWIN_DIR
    features: pl.DataFrame | None = None  # None -> full traversal_features.parquet
    registry_path: Path = REG_PATH
    as_of: dt.date | None = None  # None -> today
    fit_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"warmup": 400, "samples": 400, "chains": 1}
    )
    test_budget: int = 6
    gate: bool = True


CTX = Context()
_registry: Registry | None = None
_scorer: EIGScorer | None = None


def set_context(ctx: Context) -> None:
    global CTX, _registry
    CTX = ctx
    _registry = None


def _features() -> pl.DataFrame:
    if CTX.features is not None:
        return CTX.features
    return pl.read_parquet(DATASET / "traversal_features.parquet")


def _as_of() -> dt.date:
    return CTX.as_of or dt.date.today()


def _until(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("date") <= _as_of()) if CTX.as_of else df


def _clean(obj: Any) -> Any:
    """Make a tool result JSON-safe: NaN/inf -> None, numpy scalars -> python, dates -> str."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if (math.isnan(obj) or math.isinf(obj)) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (dt.date, dt.datetime, Path)):
        return str(obj)
    return obj


def _jsonsafe(fn):
    def wrapper(*a, **k):
        return _clean(fn(*a, **k))

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    wrapper.__signature__ = __import__("inspect").signature(fn)
    wrapper.__annotations__ = fn.__annotations__
    return wrapper


def _reg() -> Registry:
    global _registry
    if _registry is None:
        clock = (
            (lambda: dt.datetime.combine(CTX.as_of, dt.time(12)).isoformat())
            if CTX.as_of
            else None
        )
        _registry = Registry(CTX.registry_path, clock=clock)
    return _registry


def _posterior(name: str):
    return az.from_netcdf(CTX.twin_dir / f"{name}.nc").posterior


# ----------------------------------------------------------------- data views
@_jsonsafe
def athlete_snapshot() -> dict[str, Any]:
    """Current athlete state: recent load, readiness, sleep, HRV, last rides, instrument history."""
    daily = _until(pl.read_parquet(DATASET / "daily.parquet").sort("date")).tail(14)
    load = _until(pl.read_parquet(DATASET / "daily_load.parquet").sort("date"))
    feats = _features()
    last28 = load.tail(28)
    inst = feats.filter(pl.col("segment_id") == INSTRUMENT_CLIMB).sort("date").tail(6)
    pred = _instrument_prediction(feats)
    dpred = _descent_prediction(feats)
    return {
        **pred,
        **dpred,
        "as_of": str(load["date"].max()),
        "weekly_load_target_exp_trimp": round(float(last28["exp_trimp"].sum() / 4), 1),
        "acute_7d_mean": round(float(load.tail(7)["exp_trimp"].mean()), 1),
        "chronic_28d_mean": round(float(last28["exp_trimp"].mean()), 1),
        "rest_days_last_14": int((load.tail(14)["n_activities"] == 0).sum()),
        "latest_readiness": daily["readiness_score"].drop_nulls().tail(1).to_list(),
        "mean_sleep_h_14d": (
            round(float(daily["sleep_seconds"].drop_nulls().mean() / 3600), 2)
            if daily["sleep_seconds"].drop_nulls().len()
            else None
        ),
        "mean_hrv_14d": daily["hrv_last_night"].drop_nulls().mean(),
        "instrument_climb_recent": inst.select(
            ["date", "moving_s", "hr_mean", "vp_mean_w", "temp_c20"]
        )
        .with_columns(pl.col("date").cast(pl.Utf8))
        .to_dicts(),
    }


def _instrument_prediction(
    feats: pl.DataFrame, hr_target: float = INSTRUMENT_HR
) -> dict[str, Any]:
    """Twin prediction for the instrument climb ridden at the protocol heart rate: moving time from the FFM
    (log speed at matched HR, current performance state, segment offset), or None if the segment is not in
    the stored fit yet."""
    from garmin_lab.twin.ffm import HR_REF, load_ffm_data

    path = CTX.twin_dir / "ffm_full.nc"
    if not path.exists():
        return {}
    d = load_ffm_data(features=feats)
    if INSTRUMENT_CLIMB not in d.seg_names:
        return {
            "instrument_note": "instrument climb not yet in the fitness model (needs >= 1 clean traversal)"
        }
    post = az.from_netcdf(path).posterior
    # The stored fit is deliberately stale (Golden Rule: predict from a fit that has not seen
    # the ride). Features may have more days / segments than the fit, so shape every reshape
    # from the posterior's own dims, never from the current design.
    n_seg_fit = int(post["a_seg"].shape[-1])
    n_day_fit = int(post["p_day"].shape[-1])
    if n_seg_fit != d.n_seg:
        return {
            "instrument_note": (
                f"segment set changed since the last fit ({n_seg_fit} fitted, {d.n_seg} now); "
                "refit the FFM before issuing instrument predictions"
            )
        }
    k = d.seg_names.index(INSTRUMENT_CLIMB)
    a = post["a_seg"].values.reshape(-1, n_seg_fit)[:, k]
    p_last = post["p_day"].values.reshape(-1, n_day_fit)[:, -1]
    seg = feats.filter(pl.col("segment_id") == INSTRUMENT_CLIMB)
    dist = float(seg["dist_m"].median())
    doy = _as_of().timetuple().tm_yday
    log_v = (
        a
        + p_last
        + post["b_hr10"].values.ravel() * (hr_target - HR_REF) / 10
        + post["b_elapsed"].values.ravel() * 0.0
        + post["b_season_sin"].values.ravel() * np.sin(2 * np.pi * doy / 365.25)
        + post["b_season_cos"].values.ravel() * np.cos(2 * np.pi * doy / 365.25)
    )
    log_t = np.log(dist) - log_v
    sd = float(np.sqrt(log_t.var() + post["sigma"].values.mean() ** 2))
    mu_s = float(np.exp(log_t.mean()))
    return {
        "instrument_climb": INSTRUMENT_CLIMB,
        "instrument_protocol": f"hold {hr_target:.0f} +- 5 bpm up the Trans Canada Trail from the east gate; {INSTRUMENT_LAP_NOTE}",
        "instrument_predicted_moving_s_at_160": round(mu_s),
        "instrument_predicted_sd_s": round(mu_s * sd),
        "instrument_n_baseline_rides": seg.height,
    }


def _descent_prediction(
    feats: pl.DataFrame, sleep_h: float | None = None, wet: float = 0.0
) -> dict[str, Any]:
    """Twin prediction for the instrument descent from the stored skill-model fit: log moving time =
    segment level + skill state (random walk, advanced one step for the coming week) + covariates at
    protocol conditions (dry, habitual sleep unless given, elapsed_h = the athlete's median at the
    gate). Returns the lognormal (mu, sigma) in log-seconds, or {} if the segment is not fitted yet."""
    from garmin_lab.twin.skill_model import load_skill_data

    path = CTX.twin_dir / "skill_model.nc"
    if not path.exists():
        return {}
    d = load_skill_data(features=feats)
    if INSTRUMENT_DESCENT not in d.seg_names:
        return {"descent_note": "instrument descent not yet in the skill model"}
    post = az.from_netcdf(path).posterior
    n_seg_fit = int(post["mu_seg"].shape[-1])
    if n_seg_fit != d.n_seg:
        return {
            "descent_note": (
                f"descent segment set changed since the last fit ({n_seg_fit} fitted, {d.n_seg} now); "
                "refit the skill model before issuing descent predictions"
            )
        }
    k = d.seg_names.index(INSTRUMENT_DESCENT)
    mu_seg = post["mu_seg"].values.reshape(-1, n_seg_fit)[:, k]
    lam = post["lambda_seg"].values.reshape(-1, n_seg_fit)[:, k]
    s_last = post["S_week"].values.reshape(-1, post["S_week"].shape[-1])[:, -1]
    sigma_s = post["sigma_S"].values.ravel()
    rng = np.random.default_rng(0)
    s_next = s_last + sigma_s * rng.standard_normal(len(s_last))  # one RW step ahead
    seg = d.frame.filter(pl.col("segment_id") == INSTRUMENT_DESCENT)
    elapsed_h = float(seg["elapsed_h"].median())
    sleep_dev = 0.0 if sleep_h is None else sleep_h - 7.0
    log_t = (
        mu_seg
        + lam * s_next
        + post["delta_wet"].values.ravel() * wet
        + post["delta_sleep"].values.ravel() * sleep_dev
        + post["delta_elapsed"].values.ravel() * elapsed_h
    )
    nu = post["nu"].values.ravel() + 2.0
    obs_var = float((post["sigma"].values.ravel() ** 2 * nu / (nu - 2.0)).mean())
    mu_log = float(log_t.mean())
    sd_log = float(np.sqrt(log_t.var() + obs_var))
    return {
        "instrument_descent": INSTRUMENT_DESCENT,
        "descent_protocol": "Poplar->Nicoles, self-selected effort, dry; lap at the Poplar/Nicoles junction and at Pipeline",
        "descent_predicted_moving_s": round(math.exp(mu_log)),
        "descent_predicted_log_mu": mu_log,
        "descent_predicted_log_sd": sd_log,
        "descent_conditions": {
            "sleep_h": "habitual" if sleep_h is None else sleep_h,
            "wet": wet,
            "elapsed_h": round(elapsed_h, 2),
        },
        "descent_n_clean_baseline": seg.height,
    }


@_jsonsafe
def residual_report() -> dict[str, Any]:
    """Structured diagnostics from the stored twin fits: parameter posteriors, identifiability,
    and residual patterns by covariate. This is what the Hypothesizer reasons over."""
    out: dict[str, Any] = {}
    hr = pl.read_parquet(CTX.twin_dir / "hr_model_summary.parquet").to_dicts()
    sk = pl.read_parquet(CTX.twin_dir / "skill_model_summary.parquet").to_dicts()
    tau = pl.read_parquet(CTX.twin_dir / "ffm_tau_report.parquet").to_dicts()
    out["hr_model_params"] = [
        {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
        for r in hr
    ]
    out["skill_model_params"] = [
        {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
        for r in sk
    ]
    out["ffm_identifiability"] = [
        {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
        for r in tau
    ]
    # residual patterns on the HR model
    post = _posterior("hr_model")
    from garmin_lab.twin.hr_model import load_hr_data

    d = load_hr_data(_features())
    mu = post["mu"].values.reshape(-1, len(d.hr)).mean(0)
    resid = d.hr - mu
    f = d.frame.with_columns(pl.Series("resid", resid))
    pats = {}
    for col, cut in (
        ("temp_c20", 7.0),
        ("elapsed_h", 1.0),
        ("sleep_h", 6.5),
        ("hour_local", 12.0),
        ("wx_humidity", 70.0),
    ):
        if col in f.columns:
            hi = f.filter(pl.col(col) > cut)["resid"]
            lo = f.filter(pl.col(col) <= cut)["resid"]
            if hi.len() >= 5 and lo.len() >= 5:
                pats[f"{col} > {cut}"] = {
                    "n_hi": hi.len(),
                    "mean_resid_hi": round(float(hi.mean()), 2),
                    "n_lo": lo.len(),
                    "mean_resid_lo": round(float(lo.mean()), 2),
                }
    by_seg = (
        f.group_by("segment_id")
        .agg(pl.col("resid").mean().round(2), pl.len())
        .sort("len", descending=True)
        .to_dicts()
    )
    out["hr_residual_patterns"] = pats
    out["hr_residual_by_segment"] = by_seg
    out["hr_residual_sd_bpm"] = round(float(resid.std()), 2)
    # autocorrelation of residuals by ride date (fatigue carry-over signal)
    byday = f.group_by("date").agg(pl.col("resid").mean()).sort("date")
    r = byday["resid"].to_numpy()
    out["hr_resid_lag1_autocorr_by_ride"] = (
        round(float(np.corrcoef(r[:-1], r[1:])[0, 1]), 3) if len(r) > 5 else None
    )
    return out


@_jsonsafe
def model_vocabulary() -> dict[str, str]:
    """The only base models, parameters, covariates and operations a model spec may use."""
    return {"vocabulary": vocabulary_text()}


# ------------------------------------------------------------------ registry
@_jsonsafe
def registry_list(state: str = "") -> dict[str, Any]:
    """List hypotheses (optionally filtered by state) with beliefs, states and lineage."""
    rows = _reg().list(state or None)
    return {
        "hypotheses": [
            {k: v for k, v in r.items() if k not in ("created_at",)} for r in rows
        ]
    }


@_jsonsafe
def registry_register(
    text: str, formal_spec_json: str, prior_prob: float, actor: str = "hypothesizer"
) -> dict[str, Any]:
    """Register a new hypothesis. formal_spec_json must be a valid model spec (see model_vocabulary).
    Prior must be in [0.05, 0.95]. Returns the id or an error."""
    try:
        spec = validate_spec(json.loads(formal_spec_json))
        hid = _reg().register(
            text,
            {
                "base": spec.base,
                "operation": spec.operation,
                "args": spec.args,
                "notes": spec.notes,
            },
            prior_prob,
            actor=actor,
            test_budget=CTX.test_budget,
        )
        return {"ok": True, "hypothesis_id": hid, "warnings": spec.warnings}
    except (TemplateError, RegistryError, json.JSONDecodeError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}


@_jsonsafe
def registry_refine(
    parent_id: int, text: str, formal_spec_json: str, prior_prob: float
) -> dict[str, Any]:
    """Refine an existing hypothesis that already has evidence or a verdict."""
    try:
        spec = validate_spec(json.loads(formal_spec_json))
        hid = _reg().refine(
            parent_id,
            text,
            {
                "base": spec.base,
                "operation": spec.operation,
                "args": spec.args,
                "notes": spec.notes,
            },
            prior_prob,
        )
        return {"ok": True, "hypothesis_id": hid, "warnings": spec.warnings}
    except (TemplateError, RegistryError, json.JSONDecodeError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}


@_jsonsafe
def registry_retire(hypothesis_id: int, reason: str) -> dict[str, Any]:
    """Retire a hypothesis (e.g. duplicate or untestable)."""
    try:
        _reg().retire(hypothesis_id, reason)
        return {"ok": True}
    except RegistryError as exc:
        return {"ok": False, "error": str(exc)}


# --------------------------------------------------------------- statistician
def _declared_priors(
    n: int, rng: np.random.Generator
) -> dict[tuple[str, str], np.ndarray]:
    """Prior samples for every parameter a directional claim may be made about (mirrors the models)."""
    return {
        ("hr_model", "gamma_d"): rng.normal(4, 3, n),
        ("hr_model", "gamma_dT"): rng.normal(0, 0.5, n),
        ("hr_model", "gamma_T"): rng.normal(0.5, 0.5, n),
        ("hr_model", "beta_vp50"): rng.normal(10, 5, n),
        ("skill_model", "delta_sleep"): rng.normal(0, 0.03, n),
        ("skill_model", "delta_wet"): rng.normal(0, 0.05, n),
        ("skill_model", "delta_elapsed"): rng.normal(0, 0.05, n),
        ("ffm", "tau1"): 15 + 75 * rng.beta(2.2, 3.0, n),
        ("ffm", "b_T"): rng.normal(0, 0.01, n),
        ("ffm", "b_elapsed"): rng.normal(0, 0.05, n),
        ("ffm", "b_hr10"): rng.normal(0.05, 0.05, n),
    }


def _record(
    reg: Registry,
    h: dict[str, Any],
    lbf: float,
    shrink: float,
    fit_id: str,
    method: str,
) -> dict[str, Any]:
    """Incremental, gated evidence. ``cum_log_bf`` tracks the latest full-data log BF, so the
    increment recorded now is (current - previous); a gated evaluation records 0."""
    gated = CTX.gate and shrink < SHRINK_GATE
    increment = 0.0 if gated else lbf - float(h["cum_log_bf"])
    cutoff = _features()["date"].max()
    reg.register_fit(fit_id, method.split(" ")[0], cutoff, str(CTX.twin_dir))
    belief = reg.record_evidence(
        h["id"], increment, fit_id, method + (" GATED" if gated else ""), as_of=cutoff
    )
    _maybe_verdict(reg, h["id"])
    return {
        "ok": True,
        "log_bf_full": round(lbf, 3),
        "log_bf_increment": round(increment, 3),
        "admitted": not gated,
        "shrinkage": round(float(shrink), 3),
        "new_belief": round(belief, 3),
        "state": reg.get(h["id"])["state"],
    }


def _maybe_verdict(reg: Registry, hid: int) -> None:
    h = reg.get(hid)
    if h["state"] not in ("proposed", "under_test"):
        return
    for state, ok in (
        ("supported", h["belief"] >= 0.8),
        ("refuted", h["belief"] <= 0.2),
        (
            "undetermined",
            h["n_tests"] >= h["test_budget"] and abs(h["cum_log_bf"]) < math.log(3),
        ),
    ):
        if ok:
            reg.transition(hid, state)
            return


@_jsonsafe
def evaluate_directional_claim(hypothesis_id: int) -> dict[str, Any]:
    """Statistician tool: for a hypothesis whose spec is a directional_claim on a parameter of a
    stored fit, compute Savage-Dickey evidence with the identifiability gate and record it.
    Only this tool moves beliefs."""
    reg = _reg()
    h = reg.get(hypothesis_id)
    spec = h["formal_spec"]
    if not isinstance(spec, dict) or spec.get("operation") != "directional_claim":
        return {
            "ok": False,
            "error": "not a directional_claim; use evaluate_structural_spec",
        }
    base, args = spec["base"], spec["args"]
    name = FIT_NAME[base]
    path = CTX.twin_dir / f"{name}.nc"
    post = az.from_netcdf(path).posterior
    param, op, value = args["param"], args["op"], float(args["value"])
    if param not in post:
        return {"ok": False, "error": f"{param} not in posterior of {name}"}
    x = post[param].values.reshape(-1)
    priors = _declared_priors(len(x), np.random.default_rng(0))
    if (base, param) not in priors:
        return {
            "ok": False,
            "error": f"no declared prior for {base}.{param}; add it to the statistician before testing",
        }
    prior = priors[(base, param)]
    f = (lambda z: z > value) if op == ">" else (lambda z: z < value)
    pp = float(np.clip(f(x).mean(), 1e-4, 1 - 1e-4))
    pq = float(np.clip(f(prior).mean(), 1e-4, 1 - 1e-4))
    lbf = math.log(pp / (1 - pp)) - math.log(pq / (1 - pq))
    log_scale = param in ("tau1", "tau2")
    xs, ps = (np.log(x), np.log(prior)) if log_scale else (x, prior)
    shrink = 1 - xs.std() / ps.std()
    method = f"{name} savage-dickey {param}{op}{value} BC={bhattacharyya(ps, xs):.2f}"
    fit_id = f"{name}:{hashlib.sha1(path.read_bytes()).hexdigest()[:10]}"
    out = _record(reg, h, lbf, shrink, fit_id, method)
    return {**out, "posterior_prob": pp, "prior_prob": pq}


def _load_base_data(base: str):
    feats = _features()
    if base == "hr_model":
        from garmin_lab.twin.hr_model import fit_hr, load_hr_data

        return load_hr_data(feats), fit_hr
    if base == "skill_model":
        from garmin_lab.twin.skill_model import fit_skill, load_skill_data

        return load_skill_data(feats), fit_skill
    from garmin_lab.twin.ffm import fit_ffm, load_ffm_data

    return load_ffm_data(features=feats), fit_ffm


@_jsonsafe
def evaluate_structural_spec(hypothesis_id: int) -> dict[str, Any]:
    """Statistician tool for template specs that change the model (add_covariate, add_interaction,
    add_changepoint, split_by_condition): compile the spec into one extra linear term, refit the
    base model on the data visible now, and record Savage-Dickey evidence for the term being
    non-zero, with the identifiability gate. Only this tool and evaluate_directional_claim move beliefs.
    """
    reg = _reg()
    h = reg.get(hypothesis_id)
    raw = h["formal_spec"]
    if not isinstance(raw, dict) or not is_structural(raw):
        return {"ok": False, "error": "spec is not structural"}
    spec = validate_spec(raw)
    d, fit = _load_base_data(spec.base)
    tb = build_terms([spec], spec.base, d.frame)
    if not tb.terms:
        return {
            "ok": False,
            "error": "spec not compilable on current data (no contrast yet or unsupported parameter)",
        }
    idata = fit(d, extra=tb.extra(d.frame), **CTX.fit_kwargs)
    theta = idata.posterior["theta_extra"].values.reshape(-1)
    sd = savage_dickey_at_zero(theta, tb.prior_sd)
    name = FIT_NAME[spec.base]
    fit_id = f"{name}+{hashlib.sha1(spec.key().encode()).hexdigest()[:8]}:{_as_of()}"
    method = f"{name} savage-dickey theta[{spec.key()}]!=0 post={sd['post_mean']:+.3f}+-{sd['post_sd']:.3f}"
    out = _record(reg, h, sd["log_bf"], sd["shrinkage"], fit_id, method)
    return {**out, "term_post_mean": sd["post_mean"], "term_post_sd": sd["post_sd"]}


def evaluate_open_hypotheses(max_structural: int = 6) -> list[dict[str, Any]]:
    """Tool-only statistician pass: every open hypothesis is (re)evaluated on the data visible now.
    Directional claims are cheap; structural specs need a refit, so at most ``max_structural`` of
    the most recent are evaluated per pass."""
    out, n_struct = [], 0
    rows = [h for h in _reg().list() if h["state"] in ("proposed", "under_test")]
    for h in sorted(rows, key=lambda r: -r["id"]):
        spec = h["formal_spec"]
        if not isinstance(spec, dict):
            continue
        if spec.get("operation") == "directional_claim":
            out.append({"id": h["id"], **evaluate_directional_claim(h["id"])})
        elif is_structural(spec) and n_struct < max_structural:
            n_struct += 1
            out.append({"id": h["id"], **evaluate_structural_spec(h["id"])})
    return out


def working_specs(base: str, threshold: float = 0.5):
    """Structural specs the registry currently believes (belief >= threshold, not retired/refuted):
    these are admitted into the working model used for forecasting."""
    specs = []
    for h in _reg().list():
        raw = h["formal_spec"]
        if (
            isinstance(raw, dict)
            and is_structural(raw)
            and raw.get("base") == base
            and h["state"] not in ("retired", "refuted")
            and h["belief"] >= threshold
            and h["n_tests"] > 0
        ):
            specs.append(validate_spec(raw))
    return specs


@_jsonsafe
def ledger_predict(
    hypothesis_id: int,
    target: str,
    days_ahead: int,
    mu: float,
    sigma: float,
    family: str = "normal",
    conditions_json: str = "{}",
) -> dict[str, Any]:
    """Pre-register a predictive distribution for an outcome within days_ahead days."""
    try:
        tt = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=days_ahead)
        pid = _reg().predict(
            hypothesis_id,
            target,
            tt,
            {
                "family": family,
                "mu": mu,
                "sigma": sigma,
                "conditions": json.loads(conditions_json),
            },
        )
        return {"ok": True, "prediction_id": pid, "target_time": tt.isoformat()}
    except (RegistryError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": str(exc)}


# ------------------------------------------------------------------ design
@_jsonsafe
def score_workout_menu(lambda_info: float = 0.6) -> dict[str, Any]:
    """Score the candidate training plans by expected information gain about the fitness-fatigue
    parameters (measured on the instrument climb) traded against expected performance change.
    lambda_info in [0,1] weights information vs performance."""
    global _scorer
    snap = athlete_snapshot()
    if _scorer is None:
        _scorer = EIGScorer(n_outer=800, n_inner=800, test_sigma=0.035)
    rows = _scorer.score_menu(
        default_menu(snap["weekly_load_target_exp_trimp"]),
        lam=float(np.clip(lambda_info, 0, 1)),
    )
    for r in rows:
        r["loads"] = [round(x) for x in r["loads"]]
        r["first_week"] = [round(x) for x in r["first_week"]]
        for k in ("eig_nats", "eig_per_week", "exp_perf_change", "utility"):
            r[k] = round(r[k], 4)
    return {"weekly_target": snap["weekly_load_target_exp_trimp"], "ranked": rows}


@_jsonsafe
def safety_check(
    first_week_loads_json: str,
    hr_target_max: float = 165.0,
    unavailable_days_json: str = "[]",
    illness: bool = False,
) -> dict[str, Any]:
    """Hard constraint check for a 7-day plan. Returns violations; an empty list means feasible."""
    load = pl.read_parquet(DATASET / "daily_load.parquet").sort("date")
    daily = pl.read_parquet(DATASET / "daily.parquet").sort("date")
    readiness = daily["readiness_score"].drop_nulls().tail(1).to_list()
    hr_max = float(
        pl.read_parquet(DATASET / "activities.parquet")["maxHR"].quantile(0.995)
    )
    st = AthleteState(
        load["exp_trimp"].to_numpy().astype(float),
        hr_max=hr_max,
        readiness=readiness[0] if readiness else None,
        illness_reported=illness,
        unavailable_days=json.loads(unavailable_days_json),
    )
    try:
        plan = json.loads(first_week_loads_json)
    except json.JSONDecodeError as exc:
        return {"feasible": False, "violations": [f"bad plan json: {exc}"]}
    v = check_week(st, plan, hr_target_max)
    return {
        "feasible": not v,
        "violations": v,
        "hr_cap": round(0.92 * hr_max),
        "readiness": readiness,
    }


def _instrument_workout(
    plan_name: str, label: str, day_offset: int, hr_target_bpm: float
) -> dict[str, Any]:
    return {
        "workoutName": f"EpiCoach {plan_name}: {label} test",
        "sportType": {"sportTypeKey": "cycling"},
        "scheduled_day_offset": day_offset,
        "steps": [
            {
                "type": "warmup",
                "duration_min": 5,
                "target": {"hr_zone": 2},
                "note": "roll in from the car to the Trans Canada Trail east gate",
            },
            {
                "type": "interval",
                "segment": INSTRUMENT_CLIMB,
                "duration": "lap_button",
                "target": {"hr_bpm": [hr_target_bpm - 5, hr_target_bpm + 5]},
                "note": f"hold the band all the way up; {INSTRUMENT_LAP_NOTE}",
            },
            {"type": "recovery", "duration_min": 3, "target": {"hr_zone": 1}},
            {
                "type": "interval",
                "segment": INSTRUMENT_DESCENT,
                "duration": "lap_button",
                "target": "self-selected",
                "note": "enter via Poplar Trail; timed from the Nicoles junction to Pipeline",
            },
            {"type": "cooldown", "duration_min": 10},
        ],
    }


@_jsonsafe
def build_workout(
    plan_name: str,
    first_week_loads_json: str,
    hr_target_bpm: float = INSTRUMENT_HR,
) -> dict[str, Any]:
    """Serialise the chosen plan's two protocol test rides (FRESH on day 1, LOADED on day 5) into
    Garmin-workout JSON (dry run: written to data/cycles, not uploaded)."""
    from garmin_lab.agents.eig import TEST_DAYS

    CYCLE_DIR.mkdir(parents=True, exist_ok=True)
    loads = json.loads(first_week_loads_json)
    workouts = [
        _instrument_workout(plan_name, label, day, hr_target_bpm)
        for label, day in zip(("fresh", "loaded"), TEST_DAYS)
    ]
    bundle = {
        "plan": plan_name,
        "plan_first_week_exp_trimp": loads,
        "workouts": workouts,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    path = CYCLE_DIR / f"workout_{dt.date.today().isoformat()}_{plan_name}.json"
    path.write_text(json.dumps(bundle, indent=2))
    return {
        "ok": True,
        "path": str(path),
        "n_workouts": len(workouts),
        "test_days": [d + 1 for d in TEST_DAYS],
    }


@_jsonsafe
def save_report(markdown: str) -> dict[str, Any]:
    """Persist the weekly report."""
    CYCLE_DIR.mkdir(parents=True, exist_ok=True)
    path = CYCLE_DIR / f"report_{dt.date.today().isoformat()}.md"
    path.write_text(markdown)
    return {"ok": True, "path": str(path)}
