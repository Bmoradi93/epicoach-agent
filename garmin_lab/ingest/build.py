"""Build the analysis dataset (Parquet + DuckDB) from ``data/raw/``.

Usage::

    pixi run -e research python -m garmin_lab.ingest.build

Outputs in ``data/dataset/``::

    activities.parquet   one row per activity (list summary + FIT session + weather + gear)
    samples.parquet      FIT records for every activity (timestamp, lat/lon, alt, speed, HR, grit, flow, ...)
    laps.parquet         FIT laps
    daily.parquet        one row per calendar day (sleep, HRV, readiness, training status, daily summary, body battery)
    vo2max.parquet       VO2max history (generic + cycling)
    garmin.duckdb        views over all of the above
    inventory.md         human-readable data inventory
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

import duckdb
import polars as pl

from .client import REPO_ROOT
from .fit_parse import parse_fit

log = logging.getLogger("garmin_lab.build")


def _f2c(f: Any) -> float | None:
    """Fahrenheit -> Celsius (Garmin's weather endpoint reports F)."""
    return (f - 32.0) * 5.0 / 9.0 if isinstance(f, (int, float)) else None


def _load(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def _range_files(raw: Path, name: str) -> Iterable[Any]:
    d = raw / "range" / name
    if not d.exists():
        return []
    return (_load(p) for p in sorted(d.glob("*.json")))


# ----------------------------------------------------------------------------
# activities + samples
# ----------------------------------------------------------------------------
ACT_LIST_KEYS = [
    "activityId",
    "activityName",
    "startTimeLocal",
    "startTimeGMT",
    "distance",
    "duration",
    "elapsedDuration",
    "movingDuration",
    "elevationGain",
    "elevationLoss",
    "averageSpeed",
    "maxSpeed",
    "averageHR",
    "maxHR",
    "calories",
    "aerobicTrainingEffect",
    "anaerobicTrainingEffect",
    "activityTrainingLoad",
    "vO2MaxValue",
    "deviceId",
    "minTemperature",
    "maxTemperature",
    "avgGradeAdjustedSpeed",
    "hrTimeInZone_1",
    "hrTimeInZone_2",
    "hrTimeInZone_3",
    "hrTimeInZone_4",
    "hrTimeInZone_5",
    "startLatitude",
    "startLongitude",
    "locationName",
    "hasPolyline",
    "manualActivity",
    "timeZoneId",
]


def build_activities_and_samples(raw: Path, out: Path) -> pl.DataFrame:
    acts = _load(raw / "activities_list.json") or []
    rows: list[dict[str, Any]] = []
    sample_frames: list[pl.DataFrame] = []
    lap_rows: list[dict[str, Any]] = []
    for i, a in enumerate(acts, 1):
        aid = int(a["activityId"])
        d = raw / "activities" / str(aid)
        row: dict[str, Any] = {k: a.get(k) for k in ACT_LIST_KEYS}
        row["activity_id"] = aid
        row["type_key"] = a.get("activityType", {}).get("typeKey")
        row["parent_type_id"] = a.get("activityType", {}).get("parentTypeId")
        row["sport_type_id"] = a.get("sportTypeId")
        w = _load(d / "weather.json") or {}
        # Garmin's weather endpoint reports Fahrenheit
        row["wx_temp_c"] = _f2c(w.get("temp"))
        row["wx_apparent_temp_c"] = _f2c(w.get("apparentTemp"))
        row["wx_dewpoint_c"] = _f2c(w.get("dewPoint"))
        row["wx_humidity"] = w.get("relativeHumidity")
        row["wx_wind_speed"] = w.get("windSpeed")
        row["wx_wind_dir"] = w.get("windDirection")
        row["wx_desc"] = (
            (w.get("weatherTypeDTO") or {}).get("desc")
            if isinstance(w.get("weatherTypeDTO"), dict)
            else None
        )
        g = _load(d / "gear.json") or []
        row["gear_names"] = (
            ";".join(
                str(x.get("customMakeModel") or x.get("displayName") or "") for x in g
            )
            if g
            else None
        )
        s = _load(d / "summary.json") or {}
        sdto = s.get("summaryDTO") or {}
        row["summary_training_effect_label"] = sdto.get("trainingEffectLabel")
        row["summary_avg_temp"] = sdto.get("averageTemperature")
        row["summary_moving_duration"] = sdto.get("movingDuration")
        row["summary_avg_grade"] = sdto.get("avgGrade")
        row["summary_max_grade"] = sdto.get("maxGrade")
        # FIT
        fit_path = d / "original.zip"
        row["has_fit"] = fit_path.exists() and fit_path.stat().st_size > 0
        row["n_records"] = 0
        row["hr_coverage"] = None
        row["gps_coverage"] = None
        if row["has_fit"]:
            try:
                fit = parse_fit(fit_path)
            except Exception as exc:  # noqa: BLE001
                log.warning("FIT parse failed for %s: %s", aid, exc)
                fit = {
                    "records": [],
                    "laps": [],
                    "session": {},
                    "device": {},
                    "hr_zones": {},
                }
            recs = fit["records"]
            row["n_records"] = len(recs)
            if recs:
                df = pl.DataFrame(
                    recs,
                    schema={
                        "timestamp": pl.Datetime("us", "UTC"),
                        "position_lat": pl.Float64,
                        "position_long": pl.Float64,
                        "distance": pl.Float64,
                        "enhanced_altitude": pl.Float64,
                        "enhanced_speed": pl.Float64,
                        "heart_rate": pl.Int32,
                        "cadence": pl.Int32,
                        "temperature": pl.Float64,
                        "power": pl.Int32,
                        "grit": pl.Float64,
                        "flow": pl.Float64,
                        "grade": pl.Float64,
                    },
                    strict=False,
                )
                df = df.with_columns(pl.lit(aid).alias("activity_id")).rename(
                    {
                        "position_lat": "lat",
                        "position_long": "lon",
                        "enhanced_altitude": "altitude",
                        "enhanced_speed": "speed",
                    }
                )
                row["hr_coverage"] = float(df["heart_rate"].is_not_null().mean())
                row["gps_coverage"] = float(df["lat"].is_not_null().mean())
                sample_frames.append(df)
            ses = fit["session"]
            row["fit_sport"] = ses.get("sport")
            row["fit_sub_sport"] = ses.get("sub_sport")
            row["fit_profile_name"] = ses.get("sport_profile_name")
            row["fit_total_grit"] = ses.get("total_grit")
            row["fit_avg_flow"] = ses.get("avg_flow")
            row["fit_training_load_peak"] = ses.get("training_load_peak")
            row["fit_total_ascent"] = ses.get("total_ascent")
            row["fit_total_descent"] = ses.get("total_descent")
            row["fit_avg_temperature"] = ses.get("avg_temperature")
            dev = fit["device"]
            row["device_product"] = dev.get("garmin_product")
            row["device_sw"] = dev.get("software_version")
            hz = fit["hr_zones"]
            row["hr_zone_bounds"] = (
                json.dumps(hz.get("hr_zone_high_boundary")) if hz else None
            )
            row["fit_resting_hr"] = hz.get("resting_heart_rate") if hz else None
            for j, lap in enumerate(fit["laps"]):
                lap_rows.append({"activity_id": aid, "lap_index": j, **lap})
        rows.append(row)
        if i % 50 == 0:
            log.info("parsed %d/%d activities", i, len(acts))

    acts_df = pl.DataFrame(rows, strict=False, infer_schema_length=None)
    acts_df = acts_df.with_columns(
        pl.col("startTimeLocal")
        .str.to_datetime("%Y-%m-%d %H:%M:%S")
        .alias("start_local"),
        pl.col("startTimeGMT")
        .str.to_datetime("%Y-%m-%d %H:%M:%S")
        .dt.replace_time_zone("UTC")
        .alias("start_utc"),
    ).with_columns(pl.col("start_local").dt.date().alias("date"))
    acts_df.write_parquet(out / "activities.parquet")
    if sample_frames:
        samples = pl.concat(sample_frames, how="vertical_relaxed").sort(
            ["activity_id", "timestamp"]
        )
        samples.write_parquet(out / "samples.parquet")
        log.info("samples: %d rows", samples.height)
    if lap_rows:
        pl.DataFrame(lap_rows, strict=False, infer_schema_length=None).write_parquet(
            out / "laps.parquet"
        )
    log.info("activities: %d rows", acts_df.height)
    return acts_df


# ----------------------------------------------------------------------------
# daily
# ----------------------------------------------------------------------------
def build_daily(raw: Path, out: Path) -> pl.DataFrame:
    rows: dict[str, dict[str, Any]] = {}

    def R(day: str) -> dict[str, Any]:
        return rows.setdefault(day, {"date": day})

    # per-day REST: sleep + training status
    for d in sorted((raw / "daily").glob("*")):
        day = d.name
        sl = _load(d / "sleep.json") or {}
        dto = sl.get("dailySleepDTO") or {}
        r = R(day)
        r["sleep_seconds"] = dto.get("sleepTimeSeconds")
        r["sleep_deep_s"] = dto.get("deepSleepSeconds")
        r["sleep_light_s"] = dto.get("lightSleepSeconds")
        r["sleep_rem_s"] = dto.get("remSleepSeconds")
        r["sleep_awake_s"] = dto.get("awakeSleepSeconds")
        r["sleep_start_gmt"] = dto.get("sleepStartTimestampGMT")
        r["sleep_end_gmt"] = dto.get("sleepEndTimestampGMT")
        r["sleep_score"] = ((dto.get("sleepScores") or {}).get("overall") or {}).get(
            "value"
        )
        r["sleep_avg_overnight_hrv"] = dto.get("avgOvernightHrv")
        r["sleep_resting_hr"] = sl.get("restingHeartRate")
        r["sleep_avg_spo2"] = dto.get("averageSpO2Value")
        r["sleep_avg_respiration"] = dto.get("averageRespirationValue")
        r["sleep_body_battery_change"] = dto.get("bodyBatteryChange")
        ts = _load(d / "training_status.json") or {}
        mts = ts.get("mostRecentTrainingStatus") or {}
        latest = mts.get("latestTrainingStatusData") or {}
        if latest:
            dev = next(iter(latest.values()))
            r["ts_training_status"] = dev.get("trainingStatus")
            r["ts_feedback"] = dev.get("trainingStatusFeedbackPhrase")
            r["ts_fitness_trend"] = dev.get("fitnessTrend")
            r["ts_sport"] = dev.get("sport")
            acute = dev.get("acuteTrainingLoadDTO") or {}
            r["ts_acute_load"] = acute.get("dailyTrainingLoadAcute")
            r["ts_chronic_load"] = acute.get("dailyTrainingLoadChronic")
            r["ts_acwr_percent"] = acute.get("acwrPercent")
            r["ts_acwr_status"] = acute.get("acwrStatus")
            r["ts_min_load"] = acute.get("minTrainingLoadChronic")
            r["ts_max_load"] = acute.get("maxTrainingLoadChronic")
        vo2 = ts.get("mostRecentVO2Max") or {}
        r["vo2max_generic"] = (vo2.get("generic") or {}).get("vo2MaxPreciseValue")
        r["vo2max_cycling"] = (vo2.get("cycling") or {}).get("vo2MaxPreciseValue")
        lb = (ts.get("mostRecentTrainingLoadBalance") or {}).get(
            "metricsTrainingLoadBalanceDTOMap"
        ) or {}
        if lb:
            b = next(iter(lb.values()))
            r["load_aerobic_low"] = b.get("monthlyLoadAerobicLow")
            r["load_aerobic_high"] = b.get("monthlyLoadAerobicHigh")
            r["load_anaerobic"] = b.get("monthlyLoadAnaerobic")
            r["load_focus_feedback"] = b.get("trainingBalanceFeedbackPhrase")
        ha = ts.get("heatAltitudeAcclimationDTO") or {}
        r["heat_acclimation_pct"] = ha.get("heatAcclimationPercentage")
        r["altitude_acclimation"] = ha.get("altitudeAcclimation")

    # GraphQL: daily summary v2
    for blob in _range_files(raw, "daily_v2"):
        for x in (
            ((blob or {}).get("data") or {}).get("userDailySummaryV2Scalar") or {}
        ).get("data") or []:
            r = R(x["calendarDate"])
            hr = x.get("heartRate") or {}
            r["resting_hr"] = hr.get("restingValue")
            r["min_hr"] = hr.get("minValue")
            r["max_hr"] = hr.get("maxValue")
            st = x.get("stress") or {}
            r["stress_avg"] = st.get("avgLevel")
            r["stress_max"] = st.get("maxLevel")
            r["stress_rest_proportion"] = st.get("restProportion")
            r["stress_high_proportion"] = st.get("highStressProportion")
            bb = x.get("bodyBattery") or {}
            r["bb_high"] = bb.get("maxValue")
            r["bb_low"] = bb.get("minValue")
            r["bb_charged"] = bb.get("chargedValue")
            r["bb_drained"] = bb.get("drainedValue")
            mv = x.get("movement") or {}
            r["steps"] = (mv.get("steps") or {}).get("value")
            r["distance_m"] = (mv.get("steps") or {}).get("distanceInMeters")
            r["floors_ascended"] = (mv.get("floorsAscended") or {}).get("value")
            im = x.get("intensityMinutes") or {}
            r["moderate_min"] = im.get("moderate")
            r["vigorous_min"] = im.get("vigorous")
            rs = x.get("respiration") or {}
            r["resp_avg"] = rs.get("avgValue")
            po = x.get("pulseOx") or {}
            r["spo2_avg"] = po.get("avgValue")
            r["spo2_min"] = po.get("minValue")
            cal = x.get("calories") or {}
            r["active_kcal"] = cal.get("burnedActive")
            r["total_kcal"] = cal.get("burnedTotal")

    # GraphQL: HRV
    for blob in _range_files(raw, "hrv"):
        for x in (
            ((blob or {}).get("data") or {}).get("heartRateVariabilityScalar") or {}
        ).get("hrvSummaries") or []:
            r = R(x["calendarDate"])
            r["hrv_weekly_avg"] = x.get("weeklyAvg")
            r["hrv_last_night"] = x.get("lastNightAvg")
            r["hrv_last_night_5min_high"] = x.get("lastNight5MinHigh")
            r["hrv_status"] = x.get("status")
            bl = x.get("baseline") or {}
            r["hrv_baseline_low"] = bl.get("balancedLow")
            r["hrv_baseline_high"] = bl.get("balancedUpper")

    # GraphQL: readiness (multiple per day; keep the earliest-morning one = max score? keep first by timestamp)
    best: dict[str, dict[str, Any]] = {}
    for blob in _range_files(raw, "readiness"):
        for x in ((blob or {}).get("data") or {}).get(
            "trainingReadinessRangeScalar"
        ) or []:
            day = x["calendarDate"]
            if day not in best or (x.get("timestamp") or "") < (
                best[day].get("timestamp") or ""
            ):
                best[day] = x
    for day, x in best.items():
        r = R(day)
        r["readiness_score"] = x.get("score")
        r["readiness_level"] = x.get("level")
        r["readiness_sleep_score"] = x.get("sleepScore")
        r["readiness_recovery_time"] = x.get("recoveryTime")
        r["readiness_hrv_factor"] = x.get("hrvFactorPercent")
        r["readiness_acute_load"] = x.get("acuteLoad")
        r["readiness_stress_history"] = x.get("stressHistoryFactorPercent")

    # REST range: body battery per day
    for blob in _range_files(raw, "body_battery"):
        for x in blob or []:
            r = R(x["date"])
            r["bb_charged_rest"] = x.get("charged")
            r["bb_drained_rest"] = x.get("drained")

    if not rows:
        df = pl.DataFrame({"date": pl.Series([], dtype=pl.Date)})
    else:
        df = (
            pl.DataFrame(list(rows.values()), strict=False, infer_schema_length=None)
            .with_columns(pl.col("date").str.to_date())
            .sort("date")
        )
    df.write_parquet(out / "daily.parquet")
    log.info("daily: %d rows, %d cols", df.height, df.width)
    return df


def build_vo2max(raw: Path, out: Path) -> pl.DataFrame | None:
    rows = []
    for blob in _range_files(raw, "vo2max"):
        for x in ((blob or {}).get("data") or {}).get("vo2MaxScalar") or []:
            for kind in ("generic", "cycling"):
                g = x.get(kind) or {}
                if g.get("calendarDate"):
                    rows.append(
                        {
                            "date": g["calendarDate"],
                            "kind": kind,
                            "vo2max": g.get("vo2MaxPreciseValue"),
                            "fitness_age": g.get("fitnessAge"),
                        }
                    )
    if not rows:
        return None
    df = (
        pl.DataFrame(rows)
        .with_columns(pl.col("date").str.to_date())
        .unique()
        .sort(["kind", "date"])
    )
    df.write_parquet(out / "vo2max.parquet")
    return df


# ----------------------------------------------------------------------------
def build_duckdb(out: Path) -> None:
    db = out / "garmin.duckdb"
    if db.exists():
        db.unlink()
    con = duckdb.connect(str(db))
    # segments/traversals are produced downstream (segments.discover / segments.routes) and
    # survive a dataset rebuild; register them too so twin.features can run after a re-build.
    for name in (
        "activities",
        "samples",
        "laps",
        "daily",
        "vo2max",
        "segments",
        "traversals",
    ):
        p = out / f"{name}.parquet"
        if p.exists():
            con.execute(
                f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{p.as_posix()}')"
            )
    con.execute(
        """
        CREATE VIEW rides AS
        SELECT * FROM activities
        WHERE type_key IN ('mountain_biking','cycling','gravel_cycling','road_biking')
        """
    )
    con.close()


def write_inventory(out: Path, acts: pl.DataFrame, daily: pl.DataFrame) -> str:
    con = duckdb.connect(str(out / "garmin.duckdb"), read_only=True)
    q = lambda s: con.execute(s).fetchall()  # noqa: E731
    lines = ["# Data inventory", ""]
    lines += [
        f"- Activities: {acts.height}  ({acts['date'].min()} .. {acts['date'].max()})"
    ]
    lines += [
        "",
        "## Activities by type",
        "",
        "| type | n | hours | with FIT | median HR cov |",
        "|---|---|---|---|---|",
    ]
    for t, n, h, f, hc in q(
        "SELECT type_key, count(*), round(sum(duration)/3600,1), sum(has_fit::int), round(median(hr_coverage),2) FROM activities GROUP BY 1 ORDER BY 2 DESC"
    ):
        lines.append(f"| {t} | {n} | {h} | {f} | {hc} |")
    lines += [
        "",
        "## Rides (cycling family) by month",
        "",
        "| month | rides | hours | km | ascent m | mean avgHR | median grit |",
        "|---|---|---|---|---|---|---|",
    ]
    for m, n, h, km, asc, hr, grit in q(
        "SELECT strftime(date,'%Y-%m'), count(*), round(sum(duration)/3600,1), round(sum(distance)/1000,0), round(sum(elevationGain),0), round(avg(averageHR),0), round(median(fit_total_grit),1) FROM rides GROUP BY 1 ORDER BY 1"
    ):
        lines.append(f"| {m} | {n} | {h} | {km} | {asc} | {hr} | {grit} |")
    (n_s,) = q("SELECT count(*) FROM samples")[0]
    (n_rs,) = q("SELECT count(*) FROM samples s JOIN rides r USING(activity_id)")[0]
    lines += ["", f"- FIT samples: {n_s:,} total; {n_rs:,} in rides"]
    (hr_cov, gps_cov, temp_cov, cad_cov) = q(
        "SELECT round(avg((heart_rate IS NOT NULL)::int),3), round(avg((lat IS NOT NULL)::int),3), round(avg((temperature IS NOT NULL)::int),3), round(avg((cadence IS NOT NULL)::int),3) FROM samples s JOIN rides r USING(activity_id)"
    )[0]
    lines += [
        f"- Ride sample coverage: HR {hr_cov}, GPS {gps_cov}, temperature {temp_cov}, cadence {cad_cov}"
    ]
    lines += [
        "",
        "## Daily wellness",
        "",
        (
            f"- Days: {daily.height} ({daily['date'].min()} .. {daily['date'].max()})"
            if daily.height
            else "- Days: 0"
        ),
    ]
    for col in (
        "sleep_seconds",
        "sleep_score",
        "hrv_last_night",
        "readiness_score",
        "resting_hr",
        "ts_acute_load",
        "stress_avg",
        "bb_high",
    ):
        if col in daily.columns:
            cov = daily[col].is_not_null().mean()
            lines.append(f"- `{col}` coverage: {cov:.0%}")
    lines += ["", "## Devices", ""]
    for p, sw, n in q(
        "SELECT device_product, min(device_sw)||'..'||max(device_sw), count(*) FROM activities WHERE device_product IS NOT NULL GROUP BY 1"
    ):
        lines.append(f"- {p} (sw {sw}): {n} activities")
    (wx,) = q("SELECT round(avg((wx_temp_c IS NOT NULL)::int),2) FROM rides")[0]
    lines += ["", f"- Rides with weather record: {wx:.0%}"]
    con.close()
    text = "\n".join(lines) + "\n"
    (out / "inventory.md").write_text(text)
    return text


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    raw = REPO_ROOT / "data" / "raw"
    out = REPO_ROOT / "data" / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    acts = build_activities_and_samples(raw, out)
    daily = build_daily(raw, out)
    build_vo2max(raw, out)
    build_duckdb(out)
    print(write_inventory(out, acts, daily))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
