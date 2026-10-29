"""Build named trail chains from OSM ways and snap ride samples onto them.

A *chain* is one named trail: all OSM ways sharing a name within an area,
joined end-to-end where endpoints meet (<= 15 m). Each chain is densified to
5 m points with a cumulative along-trail distance ``s`` (m), so a ride sample
snapped to the chain gets a scalar position along the trail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)

SNAP_M = 20.0
DENSIFY_M = 5.0
JOIN_M = 15.0


@dataclass
class Projector:
    lat0: float
    lon0: float
    m_lat: float = 111_320.0
    m_lon: float = field(init=False)

    def __post_init__(self) -> None:
        self.m_lon = 111_320.0 * float(np.cos(np.radians(self.lat0)))

    def xy(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        return np.c_[
            (np.asarray(lon) - self.lon0) * self.m_lon,
            (np.asarray(lat) - self.lat0) * self.m_lat,
        ]

    def latlon(self, xy: np.ndarray) -> np.ndarray:
        return np.c_[
            xy[:, 1] / self.m_lat + self.lat0, xy[:, 0] / self.m_lon + self.lon0
        ]


@dataclass
class Chain:
    chain_id: str
    name: str
    area: str
    pts: np.ndarray  # (N,2) metres
    s: np.ndarray  # (N,) cumulative distance
    osm_ids: list[int]
    highway: str = "path"

    @property
    def length(self) -> float:
        return float(self.s[-1])


def _densify(P: np.ndarray) -> np.ndarray:
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        d = np.linalg.norm(b - a)
        n = max(1, int(d / DENSIFY_M))
        for t in np.linspace(0, 1, n + 1)[1:]:
            out.append(a + t * (b - a))
    return np.array(out)


def _join_polylines(polys: list[np.ndarray]) -> list[np.ndarray]:
    """Greedily join polylines whose endpoints are within JOIN_M."""
    polys = [p.copy() for p in polys]
    changed = True
    while changed and len(polys) > 1:
        changed = False
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                a, b = polys[i], polys[j]
                pairs = {
                    ("ae", "bs"): np.linalg.norm(a[-1] - b[0]),
                    ("ae", "be"): np.linalg.norm(a[-1] - b[-1]),
                    ("as", "bs"): np.linalg.norm(a[0] - b[0]),
                    ("as", "be"): np.linalg.norm(a[0] - b[-1]),
                }
                k, d = min(pairs.items(), key=lambda kv: kv[1])
                if d <= JOIN_M:
                    if k == ("ae", "bs"):
                        new = np.vstack([a, b[1:]])
                    elif k == ("ae", "be"):
                        new = np.vstack([a, b[::-1][1:]])
                    elif k == ("as", "bs"):
                        new = np.vstack([a[::-1], b[1:]])
                    else:
                        new = np.vstack([b, a[1:]])
                    polys[i] = new
                    del polys[j]
                    changed = True
                    break
            if changed:
                break
    return polys


def build_chains(ways: list[dict[str, Any]], area: str, proj: Projector) -> list[Chain]:
    by_name: dict[str, list[dict[str, Any]]] = {}
    for w in ways:
        by_name.setdefault(w["tags"]["name"], []).append(w)
    chains: list[Chain] = []
    for name, ws in by_name.items():
        polys = []
        ids = []
        hw = sorted(
            (w["tags"].get("highway", "path") for w in ws),
            key=lambda h: -sum(1 for x in ws if x["tags"].get("highway") == h),
        )[0]
        for w in ws:
            g = np.array([(p["lat"], p["lon"]) for p in w["geometry"]])
            if len(g) < 2:
                continue
            polys.append(proj.xy(g[:, 0], g[:, 1]))
            ids.append(w["id"])
        if not polys:
            continue
        joined = _join_polylines(polys)
        for k, P in enumerate(joined):
            P = _densify(P)
            if len(P) < 2:
                continue
            s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
            if s[-1] < 100:
                continue
            cid = f"{area}:{name}" + (f"#{k}" if len(joined) > 1 else "")
            chains.append(Chain(cid, name, area, P, s, ids, hw))
    return chains


class TrailIndex:
    """KD-tree over all chain points for fast snapping."""

    def __init__(self, chains: list[Chain]) -> None:
        self.chains = chains
        pts = np.vstack([c.pts for c in chains])
        self.chain_idx = np.concatenate(
            [np.full(len(c.pts), i) for i, c in enumerate(chains)]
        )
        self.s = np.concatenate([c.s for c in chains])
        self.tree = cKDTree(pts)

    def snap(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (chain index or -1, along-trail s, distance) per sample."""
        d, i = self.tree.query(xy)
        ok = d <= SNAP_M
        ci = np.where(ok, self.chain_idx[i], -1)
        s = np.where(ok, self.s[i], np.nan)
        return ci, s, d
