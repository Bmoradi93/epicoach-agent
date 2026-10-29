"""Declared route instruments: gate-to-gate segments defined by the athlete.

    pixi run -e research python -m garmin_lab.segments.routes

Discovered segments (``discover.py``) are keyed on named OSM trails, which
fragments long climbs on service roads. A *route instrument* is instead
declared here by two GPS gates and a corridor (an OSM way name the ride must
follow between the gates). Traversals are written in the same schema as
discovered ones and merged into ``segments.parquet`` / ``traversals.parquet``
(rows with the same ``segment_id`` are replaced), so ``twin.features`` and
everything downstream treat them identically.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

from garmin_lab.ingest.client import REPO_ROOT
from garmin_lab.segments.osm import OSM_DIR

log = logging.getLogger("garmin_lab.segments.routes")
DATASET = REPO_ROOT / "data" / "dataset"
CORRIDOR_M = 20.0
RIDE_TYPES = ("mountain_biking", "cycling", "gravel_cycling", "road_biking")


@dataclass(frozen=True)
class Route:
    segment_id: str
    area: str
    trail_name: str
    gate_in: tuple[float, float]  # lat, lon
    gate_out: tuple[float, float]
    r_in_m: float = 40.0
    r_out_m: float = 30.0
    # OSM way name(s) the path must follow between the gates
    corridor_way: str | tuple[str, ...] | None = None
    min_corridor_frac: float = 0.8
    corridor_m: float = CORRIDOR_M
    osm_highway: str = "service"

    @property
    def corridor_names(self) -> tuple[str, ...]:
        if self.corridor_way is None:
            return ()
        if isinstance(self.corridor_way, str):
            return (self.corridor_way,)
        return self.corridor_way


ROUTES: tuple[Route, ...] = (
    # --- Burnaby Mountain (active instrument; 84 climbs / 21 descents on record) ---
    Route(
        segment_id="burnaby:Trans Canada Trail:east->UniversityDr",
        area="burnaby",
        trail_name="Trans Canada Trail (east gate to University Drive)",
        gate_in=(49.2801235, -122.8966711),  # TCT below Dead Moped junction
        gate_out=(49.2803465, -122.9210474),  # TCT meets University Drive West
        corridor_way="Trans Canada Trail",
        corridor_m=25.0,  # tree cover: looser than a service road
        osm_highway="path",
    ),
    Route(
        segment_id="burnaby:Poplar->Nicoles:desc",
        area="burnaby",
        trail_name="Nicoles Trail (descent, entered via Poplar Trail)",
        # Clock starts where Poplar meets Nicoles: every historical descent passes
        # here, while only 6/25 start at the Poplar entry 200 m above.
        # 2026-10-04: gate_in moved from the OSM junction node (49.2715, -122.9092) to the
        # athlete's lap-press point, 49 m east. All 25 descents pass within 10 m of the
        # press point; the OSM node sat 36-41 m off the ridden line for 10 of them and
        # missed the 2026-10-04 instrument ride by 0.7 m.
        gate_in=(49.271601, -122.908549),
        gate_out=(49.2679190, -122.9024495),  # Nicoles Trail exit at Pipeline
        corridor_way=("Poplar Trail", "Nicoles Trail"),
        corridor_m=30.0,
        min_corridor_frac=0.7,
        osm_highway="path",
    ),
    # --- Fromme (previous instrument, kept for history) ---
    Route(
        segment_id="fromme:Mountain Highway:lot->7thSecret",
        area="fromme",
        trail_name="Mountain Highway (parking lot to 7th Secret)",
        gate_in=(49.3589, -123.0355),  # Mountain Highway parking lot, 359 m
        gate_out=(49.36390, -123.05750),  # 7th Secret trailhead, 7th switchback, 782 m
        corridor_way="Grouse Mountain Highway",
    ),
)

INSTRUMENT_CLIMB = ROUTES[0].segment_id
INSTRUMENT_DESCENT = ROUTES[1].segment_id


class _Proj:
    def __init__(self, lat0: float, lon0: float) -> None:
        self.lat0, self.lon0 = lat0, lon0
        self.kx = 111_320.0 * np.cos(np.radians(lat0))
        self.ky = 110_540.0

    def xy(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        return np.c_[
            (np.asarray(lon) - self.lon0) * self.kx,
            (np.asarray(lat) - self.lat0) * self.ky,
        ]


def _corridor_tree(
    area: str, way_names: tuple[str, ...], proj: _Proj
) -> cKDTree | None:
    path = OSM_DIR / f"{area}_ways.json"
    if not path.exists() or not way_names:
        return None
    ways = [
        w
        for w in json.loads(path.read_text())["elements"]
        if w.get("tags", {}).get("name") in way_names
    ]
    pts = np.array([(p["lat"], p["lon"]) for w in ways for p in w["geometry"]])
    return cKDTree(proj.xy(pts[:, 0], pts[:, 1])) if len(pts) else None


def _first_pass(
    P: np.ndarray, gate: np.ndarray, r: float, start: int = 0
) -> int | None:
    d = np.linalg.norm(P[start:] - gate, axis=1)
    hit = np.where(d < r)[0]
    return int(hit[0]) + start if len(hit) else None


def _skip_dwell(
    P: np.ndarray,
    t: np.ndarray,
    sp: np.ndarray,
    gate: np.ndarray,
    r: float,
    i0: int,
    i1: int,
    gap_s: float = 20.0,
    v_min: float = 0.5,
) -> int:
    """Start the clock when the rider rolls *out* of the entry circle, not when they
    arrive in it. A stop at the gate (lap press, regroup) shows up either as
    stationary samples or, with the watch's auto-pause, as a recording gap. Both are
    skipped while still inside the circle."""
    j = i0
    while j < i1 - 1 and np.linalg.norm(P[j + 1] - gate) < r:
        if t[j + 1] - t[j] > gap_s or not sp[j] > v_min:
            i0 = j + 1
        j += 1
    return i0


def traversal_row(
    route: Route, g: pl.DataFrame, i0: int, i1: int, corridor_frac: float
) -> dict[str, Any]:
    """Same fields as ``discover.extract_traversals`` for the slice [i0, i1)."""
    q = g[i0:i1]
    t = q["timestamp"].dt.epoch("s").to_numpy().astype(float)
    sp = np.nan_to_num(q["speed"].to_numpy().astype(float))
    alt = q["altitude"].to_numpy().astype(float)
    hr = q["heart_rate"].to_numpy().astype(float)
    dist = q["distance"].to_numpy().astype(float)
    dts = np.diff(t)
    moving = float(np.sum(np.where(sp[:-1] > 0.5, dts, 0.0)))
    dur = float(t[-1] - t[0])
    ok = ~np.isnan(hr)
    return {
        "activity_id": int(q["activity_id"][0]),
        "t_entry": float(t[0]),
        "t_exit": float(t[-1]),
        "duration_s": dur,
        "moving_s": min(moving, dur),
        "n_samples": int(len(q)),
        "cover": float(corridor_frac),
        "max_gap_s": float(dts.max()) if len(dts) else 0.0,
        "hr_mean": float(hr[ok].mean()) if ok.any() else None,
        "hr_max": float(np.nanmax(hr)) if ok.any() else None,
        "hr_ok_frac": float(ok.mean()),
        "has_stop": bool((dur - min(moving, dur)) > 20),
        "alt_entry": float(alt[0]),
        "alt_exit": float(alt[-1]),
        "d_alt": float(alt[-1] - alt[0]),
        "dist_m": float(dist[-1] - dist[0]),
        "elapsed_at_entry_s": float(t[0] - g["timestamp"][0].timestamp()),
        "dist_at_entry_m": float(dist[0]),
        "segment_id": route.segment_id,
    }


def find_traversals(route: Route, samples: pl.DataFrame) -> list[dict[str, Any]]:
    """Every gate_in -> gate_out pass (in that order) per ride that stays in the corridor."""
    proj = _Proj(*route.gate_in)
    gin, gout = proj.xy(*route.gate_in)[0], proj.xy(*route.gate_out)[0]
    tree = _corridor_tree(route.area, route.corridor_names, proj)
    out: list[dict[str, Any]] = []
    for _, g in samples.group_by("activity_id", maintain_order=True):
        g = g.sort("timestamp")
        P = proj.xy(g["lat"].to_numpy(), g["lon"].to_numpy())
        sp = np.nan_to_num(g["speed"].to_numpy().astype(float))
        ts = g["timestamp"].dt.epoch("s").to_numpy().astype(float)
        start = 0
        while (i0 := _first_pass(P, gin, route.r_in_m, start)) is not None:
            i1 = _first_pass(P, gout, route.r_out_m, i0 + 1)
            if i1 is None:
                break
            i0 = _skip_dwell(P, ts, sp, gin, route.r_in_m, i0, i1)
            frac = 1.0
            if tree is not None:
                d, _ = tree.query(P[i0:i1])
                frac = float((d < route.corridor_m).mean())
            if frac >= route.min_corridor_frac and i1 - i0 >= 60:
                out.append(traversal_row(route, g, i0, i1, frac))
            start = i1 + 1
    return out


def segment_row(route: Route, travs: list[dict[str, Any]]) -> dict[str, Any]:
    durs = np.array([x["duration_s"] for x in travs])
    movs = np.array([x["moving_s"] for x in travs])
    d_alt = float(np.median([x["d_alt"] for x in travs]))
    length = float(np.median([x["dist_m"] for x in travs]))
    return {
        "segment_id": route.segment_id,
        "area": route.area,
        "trail_name": route.trail_name,
        "osm_highway": route.osm_highway,
        "direction": "fwd",
        "kind": "climb" if d_alt > 15 else ("descent" if d_alt < -15 else "rolling"),
        "length_m": length,
        "d_alt_m": d_alt,
        "grade_pct": 100 * d_alt / length,
        "n_traversals": len(travs),
        "n_rides": len({x["activity_id"] for x in travs}),
        "median_duration_s": float(np.median(durs)),
        "duration_cv": float(np.std(durs) / np.mean(durs)),
        "median_moving_s": float(np.median(movs)),
        "moving_cv": float(np.std(movs) / np.mean(movs)),
        "moving_cv_robust": 1.4826
        * float(np.median(np.abs(movs - np.median(movs))) / np.median(movs)),
        "n_with_stop": int(((durs - movs) > 20).sum()),
        "median_hr": float(np.nanmedian([x["hr_mean"] or np.nan for x in travs])),
        "gate_in_lat": route.gate_in[0],
        "gate_in_lon": route.gate_in[1],
        "gate_out_lat": route.gate_out[0],
        "gate_out_lon": route.gate_out[1],
        "chain_length_m": length,
        "gate_s_lo": 0.0,
        "gate_s_hi": length,
    }


def merge(existing: pl.DataFrame, new: pl.DataFrame) -> pl.DataFrame:
    """Replace rows of ``existing`` whose segment_id appears in ``new``."""
    ids = new["segment_id"].unique()
    kept = existing.filter(~pl.col("segment_id").is_in(ids.implode()))
    return pl.concat([kept, new.select(existing.columns)], how="vertical_relaxed")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    rides = pl.read_parquet(DATASET / "activities.parquet").filter(
        pl.col("type_key").is_in(RIDE_TYPES)
    )["activityId"]
    samples = (
        pl.read_parquet(DATASET / "samples.parquet")
        .filter(pl.col("activity_id").is_in(rides.implode()))
        .drop_nulls(["lat", "lon", "altitude"])
    )
    seg_rows, trav_rows = [], []
    for route in ROUTES:
        travs = find_traversals(route, samples)
        if not travs:
            log.warning("%s: no traversals", route.segment_id)
            continue
        seg_rows.append(segment_row(route, travs))
        trav_rows.extend(travs)
        log.info(
            "%s: %d traversals, median moving %.1f min, CV %.3f, HR %.0f",
            route.segment_id,
            len(travs),
            seg_rows[-1]["median_moving_s"] / 60,
            seg_rows[-1]["moving_cv"],
            seg_rows[-1]["median_hr"],
        )
    new_segs = pl.DataFrame(seg_rows)
    new_travs = pl.DataFrame(
        trav_rows, strict=False, infer_schema_length=None
    ).with_columns(
        pl.from_epoch(pl.col("t_entry").cast(pl.Int64), time_unit="s")
        .dt.replace_time_zone("UTC")
        .alias("t_entry_utc")
    )
    segs = merge(pl.read_parquet(DATASET / "segments.parquet"), new_segs)
    travs = merge(pl.read_parquet(DATASET / "traversals.parquet"), new_travs)
    segs.write_parquet(DATASET / "segments.parquet")
    travs.write_parquet(DATASET / "traversals.parquet")
    log.info("segments: %d, traversals: %d", segs.height, travs.height)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
