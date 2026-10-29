"""Parse Garmin ``original.zip`` (FIT) files into flat record/lap/session tables."""

from __future__ import annotations

import datetime as dt
import io
import logging
import zipfile
from pathlib import Path
from typing import Any

import fitdecode

log = logging.getLogger(__name__)

SEMICIRCLE = 180.0 / 2**31

RECORD_FIELDS = (
    "timestamp",
    "position_lat",
    "position_long",
    "distance",
    "enhanced_altitude",
    "altitude",
    "enhanced_speed",
    "speed",
    "heart_rate",
    "cadence",
    "temperature",
    "power",
    "grit",
    "flow",
    "grade",
)

LAP_FIELDS = (
    "timestamp",
    "start_time",
    "total_elapsed_time",
    "total_timer_time",
    "total_distance",
    "total_ascent",
    "total_descent",
    "avg_heart_rate",
    "max_heart_rate",
    "enhanced_avg_speed",
    "enhanced_max_speed",
    "total_grit",
    "avg_flow",
    "total_calories",
    "lap_trigger",
)

SESSION_FIELDS = (
    "start_time",
    "timestamp",
    "sport",
    "sub_sport",
    "sport_profile_name",
    "total_elapsed_time",
    "total_timer_time",
    "total_moving_time",
    "total_distance",
    "total_ascent",
    "total_descent",
    "avg_heart_rate",
    "max_heart_rate",
    "enhanced_avg_speed",
    "enhanced_max_speed",
    "total_grit",
    "avg_flow",
    "total_calories",
    "training_load_peak",
    "total_training_effect",
    "total_anaerobic_training_effect",
    "avg_temperature",
    "max_temperature",
)


def _val(frame: fitdecode.FitDataMessage, name: str) -> Any:
    try:
        v = frame.get_value(name, fallback=None)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(v, dt.datetime):
        return v
    if isinstance(v, (tuple, list)):
        return None
    return v


def _open_fit(path: Path) -> bytes:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".fit")]
            if not names:
                raise ValueError(f"no .fit inside {path}")
            return z.read(names[0])
    return path.read_bytes()


def parse_fit(path: Path) -> dict[str, Any]:
    """Return ``{"records": [...], "laps": [...], "session": {...}, "device": {...}, "hr_zones": {...}}``."""
    data = _open_fit(path)
    records: list[dict[str, Any]] = []
    laps: list[dict[str, Any]] = []
    session: dict[str, Any] = {}
    device: dict[str, Any] = {}
    hr_zones: dict[str, Any] = {}
    with fitdecode.FitReader(io.BytesIO(data)) as fr:
        for frame in fr:
            if frame.frame_type != fitdecode.FIT_FRAME_DATA:
                continue
            if frame.name == "record":
                row = {k: _val(frame, k) for k in RECORD_FIELDS}
                # SI units: distance m, speed m/s, altitude m; positions semicircles -> degrees
                for k in ("position_lat", "position_long"):
                    if row[k] is not None:
                        row[k] = row[k] * SEMICIRCLE
                if row["enhanced_altitude"] is None:
                    row["enhanced_altitude"] = row.pop("altitude")
                else:
                    row.pop("altitude")
                if row["enhanced_speed"] is None:
                    row["enhanced_speed"] = row.pop("speed")
                else:
                    row.pop("speed")
                records.append(row)
            elif frame.name == "lap":
                laps.append({k: _val(frame, k) for k in LAP_FIELDS})
            elif frame.name == "session" and not session:
                session = {k: _val(frame, k) for k in SESSION_FIELDS}
            elif frame.name == "device_info" and not device:
                if _val(frame, "device_index") in ("creator", 0):
                    device = {
                        "serial_number": _val(frame, "serial_number"),
                        "garmin_product": _val(frame, "garmin_product"),
                        "software_version": _val(frame, "software_version"),
                    }
            elif frame.name == "time_in_zone" and not hr_zones:
                try:
                    hr_zones = {
                        "hr_zone_high_boundary": list(
                            frame.get_value("hr_zone_high_boundary", fallback=()) or ()
                        ),
                        "time_in_hr_zone": list(
                            frame.get_value("time_in_hr_zone", fallback=()) or ()
                        ),
                        "resting_heart_rate": frame.get_value(
                            "resting_heart_rate", fallback=None
                        ),
                        "max_heart_rate": frame.get_value(
                            "max_heart_rate", fallback=None
                        ),
                        "hr_calc_type": frame.get_value("hr_calc_type", fallback=None),
                    }
                except Exception:  # noqa: BLE001
                    hr_zones = {}
    return {
        "records": records,
        "laps": laps,
        "session": session,
        "device": device,
        "hr_zones": hr_zones,
    }
