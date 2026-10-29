"""Agentic replay: the hypothesis loop run causally at every weekly origin.

    pixi run -e research python -m garmin_lab.replay.agentic [--step 7] [--max-origins N]

At each origin ``o`` (same grid as ``harness.py``):

1. refit the base twin (HR, skill, FFM) on traversals with ``day_index <= o``
   and write it to ``data/replay/agentic/origin_{o}/`` (this is the only
   twin the tools can see);
2. run the Hypothesizer and the Modeler/Critic loop against that twin and a
   replay registry whose clock is frozen at the origin date;
3. the statistician tool (re)evaluates every open hypothesis: directional
   claims by Savage-Dickey on the stored fit, structural specs by compiling
   the term and refitting; evidence is incremental and gated;
4. the *working model* = base + structural specs with belief >= 0.5 is
   refit and used to forecast the traversals in ``(o, o+7]``.

Three methods are emitted, all from the same base fits so they differ only in
what the registry admits: ``agentic_twin`` (gated), ``agentic_twin_nogate``
(identifiability gate off) and ``agentic_twin_noregistry`` (hypotheses never
admitted, i.e. the static twin). The run is resumable: finished origins are
recorded in ``progress.json`` and skipped; LLM transcripts are kept per origin.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import polars as pl
from google.adk.agents import SequentialAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

from garmin_lab.agents import tools as T
from garmin_lab.agents.compile import TermBuilder, build_terms
from garmin_lab.agents.cycle import DEFAULT_MODEL, build_hypothesis_stage
from garmin_lab.agents.templates import ModelSpec, validate_spec
from garmin_lab.ingest.client import REPO_ROOT
from garmin_lab.replay import forecasters as F
from garmin_lab.replay.harness import (
    HORIZON_DAYS,
    MIN_HISTORY_PER_SEGMENT,
    STEP_DAYS,
    WARMUP_DAYS,
    load_targets,
)
from garmin_lab.replay.scoring import (
    crps_gaussian,
    log_score_gaussian,
    pit_gaussian,
)
from garmin_lab.twin.ffm import HR_REF, fit_ffm, load_ffm_data, tau_report
from garmin_lab.twin.hr_model import VP_REF, VP_SCALE, fit_hr, load_hr_data
from garmin_lab.twin.hr_model import summarize as hr_summary
from garmin_lab.twin.skill_model import fit_skill, load_skill_data

log = logging.getLogger("garmin_lab.replay.agentic")
DATASET = REPO_ROOT / "data" / "dataset"
WORK = REPO_ROOT / "data" / "replay" / "agentic"
FIT_KW = {"warmup": 400, "samples": 400, "chains": 1}
METHODS = ("agentic_twin", "agentic_twin_nogate", "agentic_twin_noregistry")
ADMIT = 0.5

REPLAY_PREAMBLE = """REPLAY MODE. Today is {date} (day {day} of the record); you can only see data up to today.
Propose AT MOST 2 new hypotheses this week, and propose NONE if the registry already covers the
patterns you see (say so and stop). Prefer refining a hypothesis that has evidence over adding a
near-duplicate."""


# ------------------------------------------------------------------ base twin
def fit_base_twin(feats: pl.DataFrame, out: Path) -> dict[str, Any]:
    """Fit HR, skill and FFM on the visible features and store what the tools read."""
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    d_hr = load_hr_data(feats)
    hr = fit_hr(d_hr, **FIT_KW)
    hr.to_netcdf(str(out / "hr_model.nc"))
    hr_summary(hr, d_hr).write_parquet(out / "hr_model_summary.parquet")

    d_sk = load_skill_data(feats)
    sk = fit_skill(d_sk, **FIT_KW)
    sk.to_netcdf(str(out / "skill_model.nc"))
    _skill_summary(sk).write_parquet(out / "skill_model_summary.parquet")

    d_ffm = load_ffm_data(features=feats)
    ffm = fit_ffm(d_ffm, warmup=500, samples=400, chains=1)
    ffm.to_netcdf(str(out / "ffm_full.nc"))
    tau_report(ffm, rng).write_parquet(out / "ffm_tau_report.parquet")
    return {"hr": (hr, d_hr), "skill": (sk, d_sk), "ffm": (ffm, d_ffm)}


def _skill_summary(idata: az.InferenceData) -> pl.DataFrame:
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
    return pl.DataFrame(rows)


# ------------------------------------------------------------------ forecasts
def forecast_hr(
    idata: az.InferenceData,
    d,
    future: pl.DataFrame,
    tb: TermBuilder | None,
    floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior-predictive Gaussian summary of the (possibly augmented) HR model."""
    post = idata.posterior
    cw = post["c_week"].values.reshape(-1, d.n_week)[:, d.week.max()]
    beta = post["beta_vp50"].values.ravel()
    gT, gd, gdT = (post[k].values.ravel() for k in ("gamma_T", "gamma_d", "gamma_dT"))
    u = post["u_seg"].values.reshape(-1, d.n_seg) * post["sigma_u"].values.reshape(
        -1, 1
    )
    sig_a, sig = post["sigma_ride"].values.ravel(), post["sigma"].values.ravel()
    nu = post["nu"].values.ravel() + 2
    theta = (
        post["theta_extra"].values.reshape(len(beta), -1) if tb and tb.terms else None
    )
    X = tb.design(future) if theta is not None else None
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
        if theta is not None:
            m = m + theta @ X[i]
        var = m.var() + np.mean(sig_a**2) + np.mean(sig**2 * nu / (nu - 2))
        mu_out[i], sd_out[i] = m.mean(), np.sqrt(var)
    return mu_out, np.maximum(sd_out, floor)


def forecast_seg_time(
    idata: az.InferenceData,
    d,
    hist: pl.DataFrame,
    future: pl.DataFrame,
    tb: TermBuilder | None,
    floor: float,
    origin_day: int,
) -> tuple[np.ndarray, np.ndarray]:
    """FFM (possibly augmented) on climbs, EWMA elsewhere: mirrors ``forecasters.twin_ffm``."""
    mu_out, sd_out = F.ewma("log_moving_s", floor)(hist, future, origin_day)
    is_climb = (future["kind"] == "climb").to_numpy()
    if not is_climb.any() or len(d.log_v) < 15:
        return mu_out, sd_out
    post = idata.posterior
    p = post["p_day"].values.reshape(-1, len(d.w))
    tau1 = post["tau1"].values.ravel()
    a = post["a_seg"].values.reshape(-1, d.n_seg)
    b = {
        k: post[k].values.ravel()
        for k in ("b_hr10", "b_T", "b_elapsed", "b_season_sin", "b_season_cos")
    }
    sig = post["sigma"].values.ravel()
    theta = (
        post["theta_extra"].values.reshape(len(tau1), -1) if tb and tb.terms else None
    )
    X = tb.design(future) if theta is not None else None
    last = len(d.w) - 1
    for i, row in enumerate(future.iter_rows(named=True)):
        if not is_climb[i] or row["segment_id"] not in d.seg_names:
            continue
        h = max(row["day_index"] - last, 0)
        doy = row["date"].timetuple().tm_yday
        log_v = (
            a[:, d.seg_names.index(row["segment_id"])]
            + p[:, last] * np.exp(-h / tau1)
            + b["b_hr10"] * (row["hr_mean"] - HR_REF) / 10
            + b["b_T"] * row["temp_c20"]
            + b["b_elapsed"] * row["elapsed_h"]
            + b["b_season_sin"] * np.sin(2 * np.pi * doy / 365.25)
            + b["b_season_cos"] * np.cos(2 * np.pi * doy / 365.25)
        )
        if theta is not None:
            log_v = log_v + theta @ X[i]
        log_t = np.log(row["dist_m"]) - log_v
        mu_out[i] = log_t.mean()
        sd_out[i] = max(np.sqrt(log_t.var() + np.mean(sig**2) * 1.3), floor)
    return mu_out, sd_out


# --------------------------------------------------------------- LLM stage
async def _run_stage(model: str, preamble: str, msg: str) -> list[dict[str, Any]]:
    agent = SequentialAgent(
        name="HypothesisStage", sub_agents=build_hypothesis_stage(model, preamble)
    )
    runner = InMemoryRunner(agent=agent, app_name="epicoach-replay")
    session = await runner.session_service.create_session(
        app_name="epicoach-replay", user_id="athlete"
    )
    transcript: list[dict[str, Any]] = []
    async for ev in runner.run_async(
        user_id="athlete",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=msg)]),
    ):
        if not (ev.content and ev.content.parts):
            continue
        for part in ev.content.parts:
            if part.text:
                transcript.append({"agent": ev.author, "text": part.text})
            elif part.function_call:
                transcript.append(
                    {
                        "agent": ev.author,
                        "call": part.function_call.name,
                        "args": dict(part.function_call.args or {}),
                    }
                )
    return transcript


def hypothesis_stage(
    model: str, origin_day: int, as_of: dt.date, retries: int = 3
) -> list[dict[str, Any]]:
    preamble = REPLAY_PREAMBLE.format(date=as_of.isoformat(), day=origin_day)
    for attempt in range(retries):
        try:
            return asyncio.run(
                _run_stage(model, preamble, "Run this week's hypothesis review.")
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "LLM stage failed (attempt %d): %s", attempt + 1, str(exc)[:200]
            )
            time.sleep(20 * (attempt + 1))
    return [{"agent": "system", "text": "LLM stage unavailable this origin"}]


# ------------------------------------------------------------- admission
def _ungated_belief(h: dict[str, Any], lbf_full: dict[str, float]) -> float:
    """Belief the registry would hold without the identifiability gate: prior + latest full log BF."""
    prior_logit = math.log(h["belief"] / (1 - h["belief"])) - float(h["cum_log_bf"])
    z = prior_logit + lbf_full.get(str(h["id"]), 0.0)
    return 1 / (1 + math.exp(-z))


def admitted_specs(base: str, lbf_full: dict[str, float]) -> dict[str, list[ModelSpec]]:
    """Structural specs admitted into the working model under each method."""
    gated = T.working_specs(base, ADMIT)
    nogate = []
    for h in T._reg().list():
        raw = h["formal_spec"]
        if (
            isinstance(raw, dict)
            and raw.get("base") == base
            and h["state"] not in ("retired",)
            and h["n_tests"] > 0
            and str(h["id"]) in lbf_full
            and _ungated_belief(h, lbf_full) >= ADMIT
        ):
            spec = validate_spec(raw)
            if spec.operation != "directional_claim":
                nogate.append(spec)
    return {
        "agentic_twin": gated,
        "agentic_twin_nogate": nogate,
        "agentic_twin_noregistry": [],
    }


def working_fit(
    base: str, specs: list[ModelSpec], base_fit: tuple, cache: dict
) -> tuple[az.InferenceData, TermBuilder | None]:
    """Refit base + specs (cached by spec keys); no specs -> the base fit."""
    if not specs:
        return base_fit[0], None
    key = "|".join(sorted(s.key() for s in specs))
    if key in cache:
        return cache[key]
    idata0, d = base_fit
    tb = build_terms(specs, base, d.frame)
    if not tb.terms:
        return idata0, None
    fit = {"hr_model": fit_hr, "ffm": fit_ffm}[base]
    idata = fit(d, extra=tb.extra(d.frame), **FIT_KW)
    cache[key] = (idata, tb)
    return cache[key]


# ------------------------------------------------------------------ origin
def _future(data: pl.DataFrame, o: int) -> pl.DataFrame:
    hist = data.filter(pl.col("day_index") <= o)
    seen = (
        hist.group_by("segment_id")
        .len()
        .filter(pl.col("len") >= MIN_HISTORY_PER_SEGMENT)["segment_id"]
    )
    return data.filter(
        (pl.col("day_index") > o)
        & (pl.col("day_index") <= o + HORIZON_DAYS)
        & pl.col("segment_id").is_in(seen.implode())
    )


def _rows(
    target: str,
    method: str,
    o: int,
    fut: pl.DataFrame,
    y: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
) -> list[dict]:
    crps, ls, pit = (
        crps_gaussian(y, mu, sd),
        log_score_gaussian(y, mu, sd),
        pit_gaussian(y, mu, sd),
    )
    return [
        {
            "target": target,
            "method": method,
            "family": "agent",
            "origin_day": o,
            "segment_id": fut["segment_id"][i],
            "day_index": int(fut["day_index"][i]),
            "y": float(y[i]),
            "mu": float(mu[i]),
            "sigma": float(sd[i]),
            "crps": float(crps[i]),
            "logscore": float(ls[i]),
            "pit": float(pit[i]),
        }
        for i in range(fut.height)
    ]


def run_origin(
    o: int,
    feats: pl.DataFrame,
    seg_time: pl.DataFrame,
    climb_hr: pl.DataFrame,
    model: str,
    lbf_full: dict[str, float],
    use_llm: bool,
) -> tuple[list[dict], list[dict], list[dict]]:
    """One origin: fit -> hypothesise -> evaluate -> admit -> forecast. Returns (pred rows, registry snapshot, transcript)."""
    out = WORK / f"origin_{o}"
    hist = feats.filter(pl.col("day_index") <= o)
    as_of = hist["date"].max()
    T.set_context(
        T.Context(
            twin_dir=out,
            features=hist,
            registry_path=WORK / "registry.sqlite",
            as_of=as_of,
            fit_kwargs=FIT_KW,
            test_budget=12,
            gate=True,
        )
    )
    t0 = time.time()
    fits = fit_base_twin(hist, out)
    log.info("origin %d (%s): base twin fit in %.0fs", o, as_of, time.time() - t0)

    transcript = hypothesis_stage(model, o, as_of) if use_llm else []
    evidence = T.evaluate_open_hypotheses(max_structural=4)
    for e in evidence:
        if e.get("ok"):
            lbf_full[str(e["id"])] = e["log_bf_full"]
    log.info("origin %d: %d hypotheses evaluated", o, len(evidence))

    rows: list[dict] = []
    cache: dict = {}
    fut_hr, fut_t = _future(climb_hr, o), _future(seg_time, o)
    hist_t = seg_time.filter(pl.col("day_index") <= o)
    for method, specs in admitted_specs("hr_model", lbf_full).items():
        if fut_hr.height:
            idata, tb = working_fit("hr_model", specs, fits["hr"], cache)
            mu, sd = forecast_hr(idata, fits["hr"][1], fut_hr, tb, 3.0)
            rows += _rows(
                "climb_hr", method, o, fut_hr, F._y(fut_hr, "hr_mean"), mu, sd
            )
    for method, specs in admitted_specs("ffm", lbf_full).items():
        if fut_t.height and hist_t.height >= 15:
            idata, tb = working_fit("ffm", specs, fits["ffm"], cache)
            mu, sd = forecast_seg_time(
                idata, fits["ffm"][1], hist_t, fut_t, tb, 0.03, o
            )
            rows += _rows(
                "seg_time", method, o, fut_t, F._y(fut_t, "log_moving_s"), mu, sd
            )
    snapshot = [
        {
            "origin_day": o,
            "as_of": str(as_of),
            "id": h["id"],
            "text": h["text"],
            "state": h["state"],
            "belief": h["belief"],
            "cum_log_bf": h["cum_log_bf"],
            "n_tests": h["n_tests"],
            "operation": (
                h["formal_spec"].get("operation")
                if isinstance(h["formal_spec"], dict)
                else None
            ),
            "base": (
                h["formal_spec"].get("base")
                if isinstance(h["formal_spec"], dict)
                else None
            ),
        }
        for h in T._reg().list()
    ]
    return (
        rows,
        snapshot,
        transcript + [{"agent": "Statistician", "evidence": evidence}],
    )


# -------------------------------------------------------------------- main
def _load_progress() -> dict[str, Any]:
    p = WORK / "progress.json"
    return json.loads(p.read_text()) if p.exists() else {"done": [], "lbf_full": {}}


def _save(progress: dict, rows: list[dict], snaps: list[dict]) -> None:
    (WORK / "progress.json").write_text(json.dumps(progress))
    for name, new in (
        ("predictions_agentic.parquet", rows),
        ("registry_trajectory.parquet", snaps),
    ):
        if not new:
            continue
        path = WORK / name
        df = pl.DataFrame(new)
        if path.exists():
            df = pl.concat([pl.read_parquet(path), df], how="vertical_relaxed")
        df.write_parquet(path)


def merge_into_benchmark() -> None:
    """Replace the agentic rows in data/replay/predictions.parquet and refresh the summary tables."""
    from garmin_lab.replay.harness import OUT, summarize

    pred_path, mine = OUT / "predictions.parquet", WORK / "predictions_agentic.parquet"
    if not (pred_path.exists() and mine.exists()):
        return
    old = pl.read_parquet(pred_path).filter(~pl.col("method").is_in(list(METHODS)))
    pred = pl.concat([old, pl.read_parquet(mine)], how="vertical_relaxed")
    pred.write_parquet(pred_path)
    summ, dm = summarize(pred)
    summ.write_parquet(OUT / "summary.parquet")
    dm.write_parquet(OUT / "dm_tests.parquet")
    with pl.Config(tbl_rows=60, tbl_cols=20, float_precision=3):
        print(summ)
        print(dm)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--step", type=int, default=STEP_DAYS)
    ap.add_argument(
        "--max-origins", type=int, default=0, help="stop after N origins (smoke test)"
    )
    ap.add_argument(
        "--no-llm", action="store_true", help="skip the LLM stage (statistician only)"
    )
    ap.add_argument(
        "--merge-only",
        action="store_true",
        help="only merge finished predictions into the benchmark",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    WORK.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        merge_into_benchmark()
        return 0
    lock = WORK / "run.lock"
    if lock.exists() and _pid_alive(int(lock.read_text() or 0)):
        log.error(
            "another agentic replay is running (pid %s); refusing to start",
            lock.read_text(),
        )
        return 2
    lock.write_text(str(os.getpid()))
    seg_time, climb_hr, _ = load_targets()
    feats = pl.read_parquet(DATASET / "traversal_features.parquet")
    max_day = int(max(seg_time["day_index"].max(), climb_hr["day_index"].max()))
    origins = list(range(WARMUP_DAYS, max_day - HORIZON_DAYS + 1, args.step))
    progress = _load_progress()
    todo = [o for o in origins if o not in progress["done"]]
    if args.max_origins:
        todo = todo[: args.max_origins]
    log.info(
        "%d origins total, %d done, %d to run",
        len(origins),
        len(progress["done"]),
        len(todo),
    )
    for o in todo:
        t0 = time.time()
        rows, snaps, transcript = run_origin(
            o,
            feats,
            seg_time,
            climb_hr,
            args.model,
            progress["lbf_full"],
            not args.no_llm,
        )
        (WORK / f"origin_{o}" / "transcript.json").write_text(
            json.dumps(transcript, default=str, indent=1)
        )
        progress["done"].append(o)
        _save(progress, rows, snaps)
        log.info(
            "origin %d done in %.0fs: %d predictions, %d hypotheses in registry",
            o,
            time.time() - t0,
            len(rows),
            len(snaps),
        )
    lock.unlink(missing_ok=True)
    merge_into_benchmark()
    return 0


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
