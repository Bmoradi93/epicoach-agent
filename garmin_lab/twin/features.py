"""Build the modelling tables from the dataset.

Usage::

    pixi run -e research python -m garmin_lab.twin.features

Outputs (``data/dataset/``)::

    traversal_features.parquet   one row per traversal: virtual power, grade, weather,
                                 elapsed, prior-night sleep/HRV/readiness, week index
    daily_load.parquet           one row per calendar day: exp-TRIMP, Edwards TRIMP,
                                 Garmin load, ride minutes, rest flag
"""

from __future__ import annotations

import json
import logging

import duckdb
import numpy as np
import polars as pl

from garmin_lab.ingest.client import REPO_ROOT
from garmin_lab.signals.hr_qc import hr_quality_flags

log = logging.getLogger("garmin_lab.features")
DATASET = REPO_ROOT / "data" / "dataset"

# Athlete constants (from Garmin profile + data); override via profile if present
G = 9.81
RHO = 1.20
CDA = 0.40  # MTB upright, irrelevant at climbing speeds
CRR = 0.020  # nominal; absorbed by segment random effect in the model
ETA = 0.976
BIKE_KG = 14.0
HR_MAX_DEFAULT = 200.0
HR_REST_DEFAULT = 58.0
RIDE_TYPES = ("mountain_biking", "cycling", "gravel_cycling", "road_biking")


def athlete_constants() -> dict[str, float]:
    prof = REPO_ROOT / "data" / "raw" / "profile" / "user_profile.json"
    mass = 66.0
    if prof.exists():
        ud = json.loads(prof.read_text()).get("userData", {})
        if ud.get("weight"):
            mass = float(ud["weight"]) / 1000.0
    con = duckdb.connect(str(DATASET / "garmin.duckdb"), read_only=True)
    hr_max = (
        con.execute("SELECT quantile_cont(maxHR,0.995) FROM rides").fetchone()[0]
        or HR_MAX_DEFAULT
    )
    hr_rest = (
        con.execute("SELECT avg(resting_hr) FROM daily").fetchone()[0]
        or HR_REST_DEFAULT
    )
    con.close()
    return {
        "mass_kg": mass + BIKE_KG,
        "rider_kg": mass,
        "hr_max": float(hr_max),
        "hr_rest": float(hr_rest),
    }


def virtual_power(
    speed: np.ndarray, alt: np.ndarray, dist: np.ndarray, t: np.ndarray, mass: float
) -> np.ndarray:
    """Martin et al. power balance per sample (W). Grade from altitude smoothed over ~30 m."""
    v = np.nan_to_num(speed.astype(float))
    # smooth altitude with a centred moving average over 15 samples (~30-45 m at climb speed)
    k = 15
    kern = np.ones(k) / k
    alt_s = np.convolve(np.nan_to_num(alt, nan=np.nanmean(alt)), kern, mode="same")
    dd = np.gradient(dist)
    dd[dd <= 0.1] = np.nan
    grade = np.gradient(alt_s) / dd
    grade = np.clip(np.nan_to_num(grade), -0.4, 0.4)
    theta = np.arctan(grade)
    a = np.gradient(v, t)
    a = np.clip(np.nan_to_num(a), -2, 2)
    p = (
        mass * G * v * (np.sin(theta) + CRR * np.cos(theta))
        + 0.5 * RHO * CDA * v**3
        + mass * a * v
    ) / ETA
    return np.clip(p, 0, 1500), grade


def exp_trimp(hr: np.ndarray, dt: np.ndarray, hr_rest: float, hr_max: float) -> float:
    hrr = np.clip((hr - hr_rest) / (hr_max - hr_rest), 0, 1)
    return float(np.nansum(dt / 60.0 * hrr * 0.64 * np.exp(1.92 * hrr)))


def edwards_trimp(hr: np.ndarray, dt: np.ndarray, hr_max: float) -> float:
    frac = hr / hr_max
    zones = np.digitize(frac, [0.5, 0.6, 0.7, 0.8, 0.9])  # 0..5
    return float(np.nansum(dt / 60.0 * zones))


def build_traversal_features(
    con: duckdb.DuckDBPyConnection, consts: dict[str, float]
) -> pl.DataFrame:
    tr = con.execute(
        """
        SELECT t.*, s.kind, s.trail_name, s.osm_highway, s.area, s.length_m, s.grade_pct,
               a.date, a.start_utc, a.wx_temp_c, a.wx_humidity, a.wx_wind_speed, a.wx_desc,
               a.start_local, a.activityTrainingLoad AS garmin_load
        FROM traversals t JOIN segments s USING(segment_id) JOIN activities a USING(activity_id)
        ORDER BY t.t_entry
        """
    ).pl()
    # per-traversal virtual power from samples
    vp_mean, vp_med, grade_mean, speed_mean = [], [], [], []
    cache: dict[int, dict] = {}
    for row in tr.iter_rows(named=True):
        aid = row["activity_id"]
        if aid not in cache:
            g = con.execute(
                f"SELECT timestamp, altitude, speed, heart_rate, distance FROM samples WHERE activity_id={aid} ORDER BY timestamp"
            ).pl()
            t = g["timestamp"].dt.epoch("s").to_numpy().astype(float)
            p, gr = virtual_power(
                g["speed"].to_numpy(),
                g["altitude"].to_numpy().astype(float),
                g["distance"].to_numpy().astype(float),
                t,
                consts["mass_kg"],
            )
            cache[aid] = {
                "t": t,
                "p": p,
                "grade": gr,
                "v": np.nan_to_num(g["speed"].to_numpy().astype(float)),
            }
        c = cache[aid]
        m = (c["t"] >= row["t_entry"]) & (c["t"] <= row["t_exit"]) & (c["v"] > 0.5)
        vp_mean.append(float(c["p"][m].mean()) if m.any() else None)
        vp_med.append(float(np.median(c["p"][m])) if m.any() else None)
        grade_mean.append(float(c["grade"][m].mean()) if m.any() else None)
        speed_mean.append(float(c["v"][m].mean()) if m.any() else None)
    tr = tr.with_columns(
        pl.Series("vp_mean_w", vp_mean, dtype=pl.Float64),
        pl.Series("vp_median_w", vp_med, dtype=pl.Float64),
        pl.Series("grade_mean", grade_mean, dtype=pl.Float64),
        pl.Series("speed_mean_ms", speed_mean, dtype=pl.Float64),
    )
    # prior-night wellness: daily row for the ride date (Garmin sleep for date D = night before D)
    daily = con.execute(
        "SELECT date, sleep_seconds, sleep_score, hrv_last_night, readiness_score, resting_hr, ts_acute_load, ts_chronic_load, ts_acwr_percent FROM daily"
    ).pl()
    tr = tr.join(daily, on="date", how="left")
    t0 = tr["date"].min()
    tr = tr.with_columns(
        ((pl.col("date") - t0).dt.total_days() // 7).alias("week"),
        (pl.col("date") - t0).dt.total_days().alias("day_index"),
        (pl.col("elapsed_at_entry_s") / 3600.0).alias("elapsed_h"),
        (pl.col("wx_temp_c") - 20.0).alias("temp_c20"),
        (
            pl.col("start_local").dt.hour() + pl.col("start_local").dt.minute() / 60
        ).alias("hour_local"),
        (pl.col("dist_m") / pl.col("moving_s")).alias("moving_speed_ms"),
        (pl.col("sleep_seconds") / 3600.0).alias("sleep_h"),
        pl.col("wx_desc")
        .str.to_lowercase()
        .str.contains("rain|drizzle|shower|snow")
        .fill_null(False)
        .alias("wet_now"),
    )
    tr.write_parquet(DATASET / "traversal_features.parquet")
    log.info("traversal_features: %d rows", tr.height)
    return tr


def build_daily_load(
    con: duckdb.DuckDBPyConnection, consts: dict[str, float]
) -> pl.DataFrame:
    acts = con.execute(
        "SELECT activity_id, date, type_key, duration, activityTrainingLoad FROM activities WHERE has_fit ORDER BY date"
    ).pl()
    rows = []
    for a in acts.iter_rows(named=True):
        g = con.execute(
            f"SELECT timestamp, heart_rate, speed FROM samples WHERE activity_id={a['activity_id']} ORDER BY timestamp"
        ).pl()
        if g.height < 30:
            continue
        t = g["timestamp"].dt.epoch("s").to_numpy().astype(float)
        hr = g["heart_rate"].to_numpy().astype(float)
        fl = hr_quality_flags(hr, t, g["speed"].to_numpy().astype(float))
        hr = np.where(fl == 0, hr, np.nan)
        dt = np.r_[np.diff(t), 1.0]
        dt = np.clip(dt, 0, 30)
        rows.append(
            {
                "activity_id": a["activity_id"],
                "date": a["date"],
                "type_key": a["type_key"],
                "is_ride": a["type_key"] in RIDE_TYPES,
                "minutes": float(np.nansum(dt) / 60),
                "exp_trimp": exp_trimp(hr, dt, consts["hr_rest"], consts["hr_max"]),
                "edwards_trimp": edwards_trimp(hr, dt, consts["hr_max"]),
                "garmin_load": a["activityTrainingLoad"],
                "hr_valid_frac": float(np.mean(~np.isnan(hr))),
            }
        )
    per_act = pl.DataFrame(rows)
    per_act.write_parquet(DATASET / "activity_load.parquet")
    d0, d1 = acts["date"].min(), acts["date"].max()
    days = pl.DataFrame({"date": pl.date_range(d0, d1, "1d", eager=True)})
    agg = per_act.group_by("date").agg(
        pl.col("exp_trimp").sum(),
        pl.col("edwards_trimp").sum(),
        pl.col("garmin_load").sum(),
        pl.col("minutes").sum(),
        pl.col("is_ride").sum().alias("n_rides"),
        pl.len().alias("n_activities"),
        (pl.col("exp_trimp") * pl.col("is_ride")).sum().alias("ride_exp_trimp"),
    )
    daily = days.join(agg, on="date", how="left").fill_null(0).sort("date")
    daily.write_parquet(DATASET / "daily_load.parquet")
    log.info(
        "daily_load: %d days, mean exp-TRIMP/day %.1f",
        daily.height,
        daily["exp_trimp"].mean(),
    )
    return daily


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    consts = athlete_constants()
    log.info("constants: %s", consts)
    con = duckdb.connect(str(DATASET / "garmin.duckdb"), read_only=True)
    build_traversal_features(con, consts)
    build_daily_load(con, consts)
    con.close()
    con = duckdb.connect(str(DATASET / "garmin.duckdb"))
    for name in ("traversal_features", "daily_load", "activity_load"):
        con.execute(
            f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{(DATASET / (name + '.parquet')).as_posix()}')"
        )
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
