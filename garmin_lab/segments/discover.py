"""Discover repeated trail segments with consensus gates and extract traversals.

Usage::

    pixi run -e research python -m garmin_lab.segments.discover

For every named trail chain and direction, the *segment* is the longest
contiguous stretch of trail covered by at least ``MIN_SHARE`` of the rides that
touch that trail (and by at least ``MIN_RIDES`` rides). Its two endpoints are
the *gates*. A *traversal* is a ride passing the entry gate and then the exit
gate with monotone progress and no long stop in between; entry/exit times are
interpolated at the gate crossings.

Outputs ``data/dataset/segments.parquet`` and ``data/dataset/traversals.parquet``
and refreshes ``garmin.duckdb`` views.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import duckdb
import numpy as np
import polars as pl

from garmin_lab.ingest.client import REPO_ROOT
from garmin_lab.segments.osm import AREAS, area_for, fetch_area
from garmin_lab.segments.trails import (
    Chain,
    Projector,
    TrailIndex,
    build_chains,
)
from garmin_lab.signals.hr_qc import hr_quality_flags

log = logging.getLogger("garmin_lab.segments")

DATASET = REPO_ROOT / "data" / "dataset"
BIN_M = 10.0
MIN_RIDES = 5
MIN_SHARE = 0.6
MIN_SEG_LEN_M = 250.0
MIN_SEG_DUR_S = 60.0
GAP_FILL_S = 30
MAX_STOP_S = 90
MIN_COVER = 0.85
RIDE_TYPES = ("mountain_biking", "cycling", "gravel_cycling", "road_biking")


@dataclass
class Run:
    activity_id: int
    chain: int
    idx: np.ndarray  # sample indices within the ride
    s: np.ndarray  # along-trail positions (may contain nan)
    forward: bool


def _fill_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = mask.copy()
    n = len(mask)
    i = 0
    while i < n:
        if not out[i]:
            j = i
            while j < n and not out[j]:
                j += 1
            if 0 < i and j < n and (j - i) <= max_gap:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    res = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            res.append((i, j))
            i = j
        else:
            i += 1
    return res


def ride_runs(aid: int, ci: np.ndarray, s: np.ndarray, n_chains: int) -> list[Run]:
    runs: list[Run] = []
    for k in np.unique(ci[ci >= 0]):
        mask = _fill_gaps(ci == k, GAP_FILL_S)
        for a, b in _runs(mask):
            if b - a < 30:
                continue
            idx = np.arange(a, b)
            sk = np.where(ci[a:b] == k, s[a:b], np.nan)
            valid = sk[~np.isnan(sk)]
            if len(valid) < 20:
                continue
            # direction from robust slope of s over valid samples
            x = np.flatnonzero(~np.isnan(sk))
            slope = np.polyfit(x, valid, 1)[0] if len(x) > 2 else 0.0
            if abs(slope) < 0.3:  # < 0.3 m/s net progress: dithering at a junction
                continue
            runs.append(Run(aid, int(k), idx, sk, forward=slope > 0))
    return runs


def consensus_gate(
    chain: Chain, cover: list[tuple[float, float]]
) -> tuple[float, float] | None:
    """Longest stretch covered by >= MIN_SHARE of covering rides and >= MIN_RIDES."""
    nb = int(np.ceil(chain.length / BIN_M)) + 1
    counts = np.zeros(nb)
    for lo, hi in cover:
        counts[int(lo // BIN_M) : int(hi // BIN_M) + 1] += 1
    need = max(MIN_RIDES, MIN_SHARE * len(cover))
    ok = counts >= need
    best = None
    for a, b in _runs(ok):
        if best is None or (b - a) > (best[1] - best[0]):
            best = (a, b)
    if best is None:
        return None
    lo, hi = best[0] * BIN_M, min(best[1] * BIN_M, chain.length)
    if hi - lo < MIN_SEG_LEN_M:
        return None
    return lo, hi


def extract_traversals(
    run: Run,
    gate: tuple[float, float],
    t: np.ndarray,
    alt: np.ndarray,
    hr: np.ndarray,
    hr_ok: np.ndarray,
    speed: np.ndarray,
    dist: np.ndarray,
    L: float,
) -> list[dict[str, Any]]:
    """Find every gate-to-gate pass inside a run (laps give several)."""
    a, b = gate
    s = run.s.copy()
    if not run.forward:
        s = L - s
        a, b = L - b, L - a
    idx = run.idx
    out = []
    i = 0
    n = len(s)

    def interp_time(k: int, target: float) -> float:
        # time at which s crosses target between k-1 and k
        s0, s1 = s[k - 1], s[k]
        t0, t1 = t[idx[k - 1]], t[idx[k]]
        if np.isnan(s0) or s1 == s0:
            return float(t1)
        f = (target - s0) / (s1 - s0)
        return float(t0 + f * (t1 - t0))

    while i < n:
        # find entry: first k with s>=a after being <a (or run starting inside but near a)
        k = i
        while k < n and (np.isnan(s[k]) or s[k] < a):
            k += 1
        if k >= n:
            break
        if k == 0 or np.isnan(s[k - 1]) or s[k - 1] >= a:
            if s[k] > a + 30:  # started mid-segment: not a full traversal
                # skip until we drop below a again
                while k < n and (np.isnan(s[k]) or s[k] >= a):
                    k += 1
                i = k
                continue
            t_in = float(t[idx[k]])
        else:
            t_in = interp_time(k, a)
        k_in = k
        # find exit
        while k < n and (np.isnan(s[k]) or s[k] < b):
            k += 1
        if k >= n:
            break
        t_out = interp_time(k, b)
        k_out = k
        seg_idx = idx[k_in:k_out]
        seg_s = s[k_in:k_out]
        if len(seg_idx) < 10:
            i = k
            continue
        dur = t_out - t_in
        inside = (~np.isnan(seg_s)) & (seg_s >= a - 15) & (seg_s <= b + 15)
        cover = inside.mean()
        dts = np.diff(t[seg_idx])
        max_gap = float(dts.max()) if len(dts) else 0.0
        mono = np.nanmean(np.diff(seg_s) >= -10) if len(seg_s) > 1 else 1.0
        if (
            cover >= MIN_COVER
            and max_gap <= MAX_STOP_S
            and mono >= 0.9
            and dur >= MIN_SEG_DUR_S
        ):
            sp = speed[seg_idx]
            moving = float(
                np.nansum(np.where(np.nan_to_num(sp) > 0.5, np.r_[dts, 1.0], 0.0))
            )
            h = hr[seg_idx].astype(float)
            ok = hr_ok[seg_idx] & ~np.isnan(h)
            out.append(
                {
                    "activity_id": run.activity_id,
                    "t_entry": t_in,
                    "t_exit": t_out,
                    "duration_s": dur,
                    "moving_s": min(moving, dur),
                    "n_samples": int(len(seg_idx)),
                    "cover": float(cover),
                    "max_gap_s": max_gap,
                    "hr_mean": float(h[ok].mean()) if ok.any() else None,
                    "hr_max": float(np.nanmax(h)) if (~np.isnan(h)).any() else None,
                    "hr_ok_frac": float(ok.mean()),
                    "has_stop": bool((dur - min(moving, dur)) > 20),
                    "alt_entry": float(alt[seg_idx[0]]),
                    "alt_exit": float(alt[seg_idx[-1]]),
                    "d_alt": float(alt[seg_idx[-1]] - alt[seg_idx[0]]),
                    "dist_m": float(dist[seg_idx[-1]] - dist[seg_idx[0]]),
                    "elapsed_at_entry_s": float(t_in - t[0]),
                    "dist_at_entry_m": float(dist[seg_idx[0]]),
                }
            )
        i = k
    return out


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    con = duckdb.connect(str(DATASET / "garmin.duckdb"), read_only=True)
    rides = con.execute(
        "SELECT activity_id, startLatitude, startLongitude, start_utc FROM activities "
        f"WHERE type_key IN {RIDE_TYPES} AND has_fit AND startLatitude IS NOT NULL ORDER BY start_utc"
    ).fetchall()

    seg_rows: list[dict[str, Any]] = []
    trav_rows: list[dict[str, Any]] = []

    for area, (s_, w_, n_, e_) in AREAS.items():
        area_rides = [r for r in rides if area_for(r[1], r[2]) == area]
        if not area_rides:
            continue
        proj = Projector((s_ + n_) / 2, (w_ + e_) / 2)
        chains = build_chains(fetch_area(area), area, proj)
        index = TrailIndex(chains)
        log.info("%s: %d rides, %d trail chains", area, len(area_rides), len(chains))

        # pass 1: snap every ride, keep per-ride arrays and runs
        ride_data: dict[int, dict[str, np.ndarray]] = {}
        all_runs: list[Run] = []
        for aid, *_ in area_rides:
            g = con.execute(
                "SELECT timestamp, lat, lon, altitude, speed, heart_rate, distance FROM samples "
                f"WHERE activity_id={aid} ORDER BY timestamp"
            ).pl()
            if g.height < 100:
                continue
            t = g["timestamp"].dt.epoch("s").to_numpy().astype(float)
            lat = g["lat"].to_numpy().astype(float)
            lon = g["lon"].to_numpy().astype(float)
            has = ~np.isnan(lat)
            ci = np.full(g.height, -1)
            s = np.full(g.height, np.nan)
            if has.sum() > 50:
                c, ss, _ = index.snap(proj.xy(lat[has], lon[has]))
                ci[has] = c
                s[has] = ss
            hr = g["heart_rate"].to_numpy().astype(float)
            sp = g["speed"].to_numpy().astype(float)
            flags = hr_quality_flags(hr, t, sp)
            ride_data[aid] = {
                "t": t,
                "alt": g["altitude"].to_numpy().astype(float),
                "hr": hr,
                "hr_ok": flags == 0,
                "speed": sp,
                "dist": g["distance"].to_numpy().astype(float),
            }
            all_runs.extend(ride_runs(aid, ci, s, len(chains)))

        # pass 2: consensus gates per (chain, direction)
        by_key: dict[tuple[int, bool], list[Run]] = {}
        for r in all_runs:
            by_key.setdefault((r.chain, r.forward), []).append(r)
        for (k, fwd), runs in by_key.items():
            chain = chains[k]
            per_ride: dict[int, list[float]] = {}
            for r in runs:
                v = r.s[~np.isnan(r.s)]
                per_ride.setdefault(r.activity_id, []).extend([v.min(), v.max()])
            if len(per_ride) < MIN_RIDES:
                continue
            cover = [(min(v), max(v)) for v in per_ride.values()]
            gate = consensus_gate(chain, cover)
            if gate is None:
                continue
            seg_id = f"{chain.chain_id}:{'fwd' if fwd else 'rev'}"
            travs = []
            for r in runs:
                rd = ride_data[r.activity_id]
                for tr in extract_traversals(
                    r,
                    gate,
                    rd["t"],
                    rd["alt"],
                    rd["hr"],
                    rd["hr_ok"],
                    rd["speed"],
                    rd["dist"],
                    chain.length,
                ):
                    tr["segment_id"] = seg_id
                    travs.append(tr)
            if len({x["activity_id"] for x in travs}) < MIN_RIDES:
                continue
            d_alt = float(np.median([x["d_alt"] for x in travs]))
            length = gate[1] - gate[0]
            ga = proj.latlon(
                chain.pts[
                    [
                        int(np.searchsorted(chain.s, gate[0])),
                        min(int(np.searchsorted(chain.s, gate[1])), len(chain.s) - 1),
                    ]
                ]
            )
            if not fwd:
                ga = ga[::-1]
            durs = np.array([x["duration_s"] for x in travs])
            movs = np.array([x["moving_s"] for x in travs])
            mad_cv = 1.4826 * float(
                np.median(np.abs(movs - np.median(movs))) / np.median(movs)
            )
            seg_rows.append(
                {
                    "segment_id": seg_id,
                    "area": area,
                    "trail_name": chain.name,
                    "osm_highway": chain.highway,
                    "direction": "fwd" if fwd else "rev",
                    "kind": (
                        "climb"
                        if d_alt > 15
                        else ("descent" if d_alt < -15 else "rolling")
                    ),
                    "length_m": length,
                    "d_alt_m": d_alt,
                    "grade_pct": 100 * d_alt / length,
                    "n_traversals": len(travs),
                    "n_rides": len({x["activity_id"] for x in travs}),
                    "median_duration_s": float(np.median(durs)),
                    "duration_cv": float(np.std(durs) / np.mean(durs)),
                    "median_moving_s": float(np.median(movs)),
                    "moving_cv": float(np.std(movs) / np.mean(movs)),
                    "moving_cv_robust": mad_cv,
                    "n_with_stop": int(((durs - movs) > 20).sum()),
                    "median_hr": float(
                        np.nanmedian([x["hr_mean"] or np.nan for x in travs])
                    ),
                    "gate_in_lat": float(ga[0, 0]),
                    "gate_in_lon": float(ga[0, 1]),
                    "gate_out_lat": float(ga[1, 0]),
                    "gate_out_lon": float(ga[1, 1]),
                    "chain_length_m": chain.length,
                    "gate_s_lo": gate[0],
                    "gate_s_hi": gate[1],
                }
            )
            trav_rows.extend(travs)
        log.info(
            "%s: %d segments so far, %d traversals", area, len(seg_rows), len(trav_rows)
        )

    segs = pl.DataFrame(seg_rows).sort("n_rides", descending=True)
    travs = pl.DataFrame(trav_rows, strict=False, infer_schema_length=None)
    travs = travs.with_columns(
        pl.from_epoch(pl.col("t_entry").cast(pl.Int64), time_unit="s")
        .dt.replace_time_zone("UTC")
        .alias("t_entry_utc"),
    )
    segs.write_parquet(DATASET / "segments.parquet")
    travs.write_parquet(DATASET / "traversals.parquet")
    con.close()
    con = duckdb.connect(str(DATASET / "garmin.duckdb"))
    for name in ("segments", "traversals"):
        con.execute(
            f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{(DATASET / (name + '.parquet')).as_posix()}')"
        )
    con.close()
    log.info("DONE: %d segments, %d traversals", segs.height, travs.height)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
