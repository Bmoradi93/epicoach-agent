"""Score open ledger predictions against instrument traversals.

Rules (RIDE_CHECKLIST.md §C.6):
  * only PROTOCOL traversals qualify: the ride must have a lap press within LAP_TOL_S of
    the traversal's gate entry AND gate exit (the athlete's 4-lap protocol). Rides through
    the gates without laps are training data, not tests;
  * a prediction is scored by the FIRST qualifying traversal of its target segment whose entry
    timestamp is strictly after the prediction was sealed (``registered_at``) and not
    after ``target_time``;
  * one traversal scores at most one prediction (I10, enforced by the registry);
  * predictions are processed in ``target_time`` order so the earlier-due one claims
    the earlier ride;
  * a prediction is scored AS SEALED, always.

    python -m garmin_lab.registry.score            # score + print
    python -m garmin_lab.registry.score --dry-run  # print only
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm

from garmin_lab.ingest.client import REPO_ROOT
from garmin_lab.registry.store import Registry, RegistryError

log = logging.getLogger("garmin_lab.registry.score")
DATASET = REPO_ROOT / "data" / "dataset"
LAP_TOL_S = 60.0


def crps_normal(x: float, mu: float, sd: float) -> float:
    z = (x - mu) / sd
    return float(sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / math.sqrt(math.pi)))


def score_one(y: float, dist: dict) -> tuple[float, float]:
    """Return (crps, point prediction in the outcome's units)."""
    if dist["family"] == "normal":
        return crps_normal(y, dist["mu"], dist["sigma"]), dist["mu"]
    if dist["family"] == "lognormal":
        return crps_normal(math.log(y), dist["mu"], dist["sigma"]), math.exp(dist["mu"])
    raise ValueError(f"unknown family {dist['family']}")


def _lap_protocol(tr: pl.DataFrame, laps: pl.DataFrame, tol_s: float = LAP_TOL_S) -> pl.Series:
    """True where the ride has a lap boundary within tol_s of both the gate entry and exit."""
    # lap boundaries = every lap start plus the end of the last lap
    starts = laps.select(
        pl.col("activity_id"),
        pl.col("start_time").dt.epoch("s").cast(pl.Float64).alias("t"),
    )
    ends = laps.select(
        pl.col("activity_id"),
        (pl.col("start_time").dt.epoch("s").cast(pl.Float64) + pl.col("total_elapsed_time")).alias("t"),
    )
    marks = pl.concat([starts, ends]).group_by("activity_id").agg(pl.col("t").alias("marks"))
    j = tr.join(marks, on="activity_id", how="left")
    out = []
    for r in j.select(["t_entry", "t_exit", "marks"]).to_dicts():
        m = np.asarray(r["marks"] or [], dtype=float)
        ok = (
            len(m) > 0
            and np.abs(m - r["t_entry"]).min() <= tol_s
            and np.abs(m - r["t_exit"]).min() <= tol_s
        )
        out.append(bool(ok))
    return pl.Series("protocol", out)


def load_traversals(features: Path = DATASET / "traversal_features.parquet") -> pl.DataFrame:
    f = pl.read_parquet(features)
    tr = pl.read_parquet(DATASET / "traversals.parquet").select(
        ["activity_id", "segment_id", "t_entry", "t_exit"]
    )
    laps = pl.read_parquet(DATASET / "laps.parquet")
    tr = tr.with_columns(_lap_protocol(tr, laps))
    return f.join(tr, on=["activity_id", "segment_id"], how="left")


def score_open(reg: Registry, f: pl.DataFrame, dry_run: bool = False) -> list[dict]:
    claimed = {
        p["outcome_ref"]
        for p in reg.ledger()
        if p["scored_at"] and p.get("outcome_ref")
    }
    out: list[dict] = []
    for p in sorted(reg.ledger(), key=lambda p: p["target_time"]):
        if p["scored_at"] or p["withdrawn_at"]:
            continue
        seg = p["target"].split("@", 1)[1]
        t0 = dt.datetime.fromisoformat(p["registered_at"]).timestamp()
        t1 = dt.datetime.fromisoformat(p["target_time"]).timestamp()
        cands = (
            f.filter(
                (pl.col("segment_id") == seg)
                & (pl.col("t_entry") > t0)
                & (pl.col("t_entry") <= t1)
                & pl.col("protocol")
            )
            .sort("t_entry")
            .to_dicts()
        )
        cands = [c for c in cands if f"{c['activity_id']}@{seg}" not in claimed]
        if not cands:
            continue
        r = cands[0]
        ref = f"{r['activity_id']}@{seg}"
        y = float(r["moving_s"])
        c, mu = score_one(y, json.loads(p["distribution"]))
        row = {
            "pred": p["id"], "hyp": p["hypothesis_id"], "segment": seg.split(":")[1],
            "ride": str(r["date"]), "activity_id": r["activity_id"], "actual_s": y,
            "predicted_s": mu, "crps": c, "hr": r["hr_mean"], "sleep_h": r["sleep_h"],
        }
        if not dry_run:
            try:
                reg.score_prediction(p["id"], outcome=y, crps=c, outcome_ref=ref)
            except RegistryError as e:
                log.warning("%s", e)
                continue
        claimed.add(ref)
        out.append(row)
        print(
            f"scored pred {row['pred']} (H{row['hyp']}) {row['segment']:20s} ride {row['ride']}: "
            f"actual {y:.0f}s vs predicted {mu:.0f}s  crps={c:.3f}  hr={row['hr']:.0f}"
        )
        if len(cands) > 1:
            print(f"   later unscored protocol rides in window: {[str(x['date']) for x in cands[1:]]}")
    skipped = f.filter(~pl.col("protocol") & (pl.col("date") >= pl.date(2026, 10, 1)))
    if skipped.height:
        print("non-protocol (training) traversals, not scored:",
              [f"{r['date']} {r['segment_id'].split(':')[1]} {r['moving_s']:.0f}s@{r['hr_mean']:.0f}" for r in skipped.to_dicts()])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--registry", default=str(REPO_ROOT / "data" / "registry.sqlite"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    reg = Registry(a.registry)
    rows = score_open(reg, load_traversals(), dry_run=a.dry_run)
    if not rows:
        print("nothing to score")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
