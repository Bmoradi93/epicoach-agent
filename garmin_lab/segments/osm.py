"""Fetch and cache named OpenStreetMap trail ways for each riding area."""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from typing import Any

from garmin_lab.ingest.client import REPO_ROOT

log = logging.getLogger(__name__)

OSM_DIR = REPO_ROOT / "data" / "osm"

# (name, south, west, north, east) - derived from ride sample bounding boxes + margin
AREAS: dict[str, tuple[float, float, float, float]] = {
    "burnaby": (49.250, -122.965, 49.300, -122.870),
    "fromme": (49.335, -123.105, 49.392, -123.025),
    "seymour": (49.305, -123.025, 49.360, -122.930),
    "squamish": (49.725, -123.135, 49.785, -123.045),
}

TRAIL_HIGHWAYS = {
    "path",
    "track",
    "cycleway",
    "footway",
    "bridleway",
    "service",
    "unclassified",
    "steps",
}

MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]


def area_for(lat: float, lon: float) -> str | None:
    for name, (s, w, n, e) in AREAS.items():
        if s <= lat <= n and w <= lon <= e:
            return name
    return None


def fetch_area(name: str, force: bool = False) -> list[dict[str, Any]]:
    """Return named trail ways (with geometry) for an area, fetching from Overpass if not cached."""
    OSM_DIR.mkdir(parents=True, exist_ok=True)
    path = OSM_DIR / f"{name}_ways.json"
    if path.exists() and not force:
        data = json.loads(path.read_text())
    else:
        s, w, n, e = AREAS[name]
        q = f'[out:json][timeout:25];way["highway"]["name"]({s},{w},{n},{e});out tags geom;'
        data = None
        for url in MIRRORS:
            try:
                req = urllib.request.Request(
                    url,
                    data=urllib.parse.urlencode({"data": q}).encode(),
                    headers={"User-Agent": "garmin-ai-agents research"},
                )
                data = json.loads(urllib.request.urlopen(req, timeout=60).read())
                break
            except Exception as exc:  # noqa: BLE001
                log.warning("Overpass %s failed for %s: %s", url, name, str(exc)[:80])
                time.sleep(3)
        if data is None:
            raise RuntimeError(f"could not fetch OSM for {name}")
        path.write_text(json.dumps(data))
        log.info("fetched %d OSM ways for %s", len(data["elements"]), name)
    return [
        w
        for w in data["elements"]
        if w.get("type") == "way"
        and "geometry" in w
        and w.get("tags", {}).get("highway") in TRAIL_HIGHWAYS
        and w.get("tags", {}).get("name")
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for a in AREAS:
        ways = fetch_area(a)
        print(
            a,
            len(ways),
            "named trail ways;",
            len({w["tags"]["name"] for w in ways}),
            "names",
        )
