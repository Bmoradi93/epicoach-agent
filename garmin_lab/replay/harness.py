"""Rolling-origin replay benchmark.

    pixi run -e research python -m garmin_lab.replay.harness [--fast]

At each weekly origin t (after a warm-up), every forecaster sees traversal
features with ``day_index <= t`` and predicts the traversals that occur in
(t, t+7]. Predictions are scored with CRPS, log score, PIT-based ECE and 90%
coverage; methods are compared with Diebold-Mariano (HAC) tests on per-origin
mean CRPS differences and moving-block bootstrap CIs.

Outputs ``data/replay/predictions.parquet``, ``data/replay/summary.parquet``,
``data/replay/dm_tests.parquet`` and prints a results table.
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import polars as pl

from garmin_lab.ingest.client import REPO_ROOT
from garmin_lab.replay import forecasters as F
from garmin_lab.replay.scoring import (
    block_bootstrap_ci,
    coverage,
    crps_gaussian,
    diebold_mariano,
    ece_from_pit,
    log_score_gaussian,
    pit_gaussian,
)
from garmin_lab.twin.hr_model import TRAIL_HIGHWAYS

log = logging.getLogger("garmin_lab.replay")
DATASET = REPO_ROOT / "data" / "dataset"
OUT = REPO_ROOT / "data" / "replay"
WARMUP_DAYS = 120
STEP_DAYS = 7
HORIZON_DAYS = 7
MIN_HISTORY_PER_SEGMENT = (
    3  # a segment must have been traversed >= 3 times before it is a forecast target
)


def load_targets() -> tuple[pl.DataFrame, pl.DataFrame, np.ndarray]:
    f = pl.read_parquet(DATASET / "traversal_features.parquet").filter(
        pl.col("osm_highway").is_in(TRAIL_HIGHWAYS)
    )
    seg_time = f.filter(
        (pl.col("moving_s") > 30)
        & (~pl.col("has_stop"))
        & pl.col("hr_mean").is_not_null()
        & pl.col("temp_c20").is_not_null()
    )
    climb_hr = f.filter(
        (pl.col("kind") == "climb")
        & pl.col("hr_mean").is_not_null()
        & (pl.col("hr_ok_frac") >= 0.7)
        & pl.col("vp_mean_w").is_not_null()
        & pl.col("temp_c20").is_not_null()
    )
    load = pl.read_parquet(DATASET / "daily_load.parquet").sort("date")
    # align daily load to traversal day_index (both start at the first activity date)
    t0 = f["date"].min()
    load = load.filter(pl.col("date") >= t0)
    return seg_time, climb_hr, load["exp_trimp"].to_numpy().astype(float)


def run_target(
    name: str, data: pl.DataFrame, fcs: list[F.Forecaster], origins: list[int]
) -> pl.DataFrame:
    rows = []
    for fc in fcs:
        t_start = time.time()
        for o in origins:
            hist = data.filter(pl.col("day_index") <= o)
            seen = (
                hist.group_by("segment_id")
                .len()
                .filter(pl.col("len") >= MIN_HISTORY_PER_SEGMENT)["segment_id"]
            )
            fut = data.filter(
                (pl.col("day_index") > o)
                & (pl.col("day_index") <= o + HORIZON_DAYS)
                & pl.col("segment_id").is_in(seen.implode())
            )
            if fut.height == 0 or hist.height < 15:
                continue
            if fc.family == "agent":
                log.info("%s origin %d (%d targets)", fc.name, o, fut.height)
            try:
                mu, sd = fc(hist, fut, o)
            except Exception as exc:  # noqa: BLE001
                log.warning("%s @ origin %d failed: %s", fc.name, o, str(exc)[:100])
                continue
            y = F._y(fut, fc.target)
            crps = crps_gaussian(y, mu, sd)
            ls = log_score_gaussian(y, mu, sd)
            pit = pit_gaussian(y, mu, sd)
            for i in range(fut.height):
                rows.append(
                    {
                        "target": name,
                        "method": fc.name,
                        "family": fc.family,
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
                )
        log.info("%-24s %-14s %5.1fs", fc.name, name, time.time() - t_start)
    return pl.DataFrame(rows)


def summarize(pred: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    summ_rows, dm_rows = [], []
    for tgt in pred["target"].unique().sort():
        p = pred.filter(pl.col("target") == tgt)
        methods = p["method"].unique().sort().to_list()
        # per-origin mean CRPS per method (for DM)
        per_origin = (
            p.group_by(["method", "origin_day"])
            .agg(pl.col("crps").mean())
            .pivot(index="origin_day", on="method", values="crps")
            .sort("origin_day")
        )
        ref = "twin_hr_model" if tgt == "climb_hr" else "twin_ffm_learned_tau"
        for m in methods:
            q = p.filter(pl.col("method") == m)
            y, mu, sd = q["y"].to_numpy(), q["mu"].to_numpy(), q["sigma"].to_numpy()
            mean_crps, lo, hi = block_bootstrap_ci(q["crps"].to_numpy(), block=14)
            summ_rows.append(
                {
                    "target": tgt,
                    "method": m,
                    "n": q.height,
                    "n_origins": q["origin_day"].n_unique(),
                    "crps": mean_crps,
                    "crps_lo": lo,
                    "crps_hi": hi,
                    "logscore": float(q["logscore"].mean()),
                    "ece": ece_from_pit(q["pit"].to_numpy()),
                    "cov90": coverage(y, mu, sd),
                    "mae": float(np.mean(np.abs(y - mu))),
                }
            )
            if m != ref and ref in per_origin.columns and m in per_origin.columns:
                both = per_origin.select(["origin_day", ref, m]).drop_nulls()
                d = both[ref].to_numpy() - both[m].to_numpy()  # negative favours twin
                stat, pval = diebold_mariano(d, lag=4)
                dm_rows.append(
                    {
                        "target": tgt,
                        "reference": ref,
                        "method": m,
                        "n_origins": both.height,
                        "dm_stat": stat,
                        "dm_p": pval,
                        "mean_diff": float(d.mean()),
                    }
                )
    return pl.DataFrame(summ_rows).sort(["target", "crps"]), pl.DataFrame(dm_rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fast", action="store_true", help="skip NUTS-based twin forecasters"
    )
    ap.add_argument("--step", type=int, default=STEP_DAYS)
    ap.add_argument(
        "--react",
        action="store_true",
        help="include the PHIA-style ReAct LLM baseline (API calls, cached)",
    )
    ap.add_argument(
        "--react-only",
        action="store_true",
        help="run only the ReAct baseline and merge into existing predictions.parquet",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    seg_time, climb_hr, load = load_targets()
    max_day = int(max(seg_time["day_index"].max(), climb_hr["day_index"].max()))
    origins = list(range(WARMUP_DAYS, max_day - HORIZON_DAYS + 1, args.step))
    log.info(
        "targets: seg_time %d rows, climb_hr %d rows; %d weekly origins",
        seg_time.height,
        climb_hr.height,
        len(origins),
    )

    fcs_time = [
        F.persistence("log_moving_s", 0.03),
        F.ewma("log_moving_s", 0.03),
        F.ridge("log_moving_s", 0.03),
        F.gp("log_moving_s", 0.03),
        F.banister_default(load, 0.03),
    ]
    fcs_hr = [
        F.persistence("hr_mean", 3.0),
        F.ewma("hr_mean", 3.0),
        F.ridge("hr_mean", 3.0),
        F.gp("hr_mean", 3.0),
    ]
    if not args.fast:
        fcs_time.append(F.twin_ffm(0.03))
        fcs_hr.append(F.twin_hr(3.0))
    if args.react or args.react_only:
        from garmin_lab.agents.react_baseline import react

        if args.react_only:
            fcs_time = [react("log_moving_s", 0.03)]
            fcs_hr = [react("hr_mean", 3.0)]
        else:
            fcs_time.append(react("log_moving_s", 0.03))
            fcs_hr.append(react("hr_mean", 3.0))

    pred = pl.concat(
        [
            run_target("seg_time", seg_time, fcs_time, origins),
            run_target("climb_hr", climb_hr, fcs_hr, origins),
        ]
    )
    if args.react_only and (OUT / "predictions.parquet").exists():
        old = pl.read_parquet(OUT / "predictions.parquet").filter(
            pl.col("method") != "react_agent"
        )
        pred = pl.concat([old, pred], how="vertical_relaxed")
    pred.write_parquet(OUT / "predictions.parquet")
    summ, dm = summarize(pred)
    summ.write_parquet(OUT / "summary.parquet")
    dm.write_parquet(OUT / "dm_tests.parquet")
    with pl.Config(tbl_rows=50, tbl_cols=20, float_precision=3):
        print(summ)
        print(dm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
