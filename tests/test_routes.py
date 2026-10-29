"""Declared route instruments: gate passes, multi-name corridors, instrument wiring."""

import numpy as np
import polars as pl

from garmin_lab.segments import routes
from garmin_lab.segments.routes import (
    INSTRUMENT_CLIMB,
    INSTRUMENT_DESCENT,
    ROUTES,
    Route,
    find_traversals,
    segment_row,
)


def _ride(aid: int, lats, lons, alts, t0: int = 1_700_000_000) -> pl.DataFrame:
    n = len(lats)
    return pl.DataFrame(
        {
            "activity_id": [aid] * n,
            "timestamp": pl.from_epoch(
                pl.Series(np.arange(t0, t0 + n, dtype="int64")), time_unit="s"
            ).dt.replace_time_zone("UTC"),
            "lat": np.asarray(lats, float),
            "lon": np.asarray(lons, float),
            "altitude": np.asarray(alts, float),
            "speed": np.full(n, 2.0),
            "heart_rate": np.full(n, 150.0),
            "distance": np.cumsum(np.full(n, 2.0)),
        }
    )


def test_corridor_names_accepts_str_tuple_or_none():
    r = Route("a", "x", "t", (0, 0), (0, 1))
    assert r.corridor_names == ()
    assert Route("a", "x", "t", (0, 0), (0, 1), corridor_way="A").corridor_names == (
        "A",
    )
    assert Route(
        "a", "x", "t", (0, 0), (0, 1), corridor_way=("A", "B")
    ).corridor_names == ("A", "B")


def test_find_traversals_gate_to_gate_without_corridor():
    # straight line north, 200 samples at ~2 m each; gates at sample 20 and 180
    lats = 49.0 + np.arange(200) * 2 / 110_540.0
    lons = np.full(200, -123.0)
    alts = np.linspace(100, 160, 200)
    route = Route(
        "t:seg",
        "x",
        "t",
        gate_in=(lats[20], -123.0),
        gate_out=(lats[180], -123.0),
        r_in_m=5,
        r_out_m=5,
    )
    tr = find_traversals(route, _ride(1, lats, lons, alts))
    assert len(tr) == 1
    row = tr[0]
    assert 155 <= row["duration_s"] <= 165
    assert row["d_alt"] > 40
    assert row["segment_id"] == "t:seg"
    seg = segment_row(route, tr)
    assert seg["kind"] == "climb"
    assert seg["n_traversals"] == 1


def test_descent_kind_from_negative_alt():
    lats = 49.0 + np.arange(200) * 2 / 110_540.0
    route = Route(
        "t:down",
        "x",
        "t",
        gate_in=(lats[10], -123.0),
        gate_out=(lats[190], -123.0),
        r_in_m=5,
        r_out_m=5,
    )
    tr = find_traversals(
        route, _ride(2, lats, np.full(200, -123.0), np.linspace(300, 100, 200))
    )
    assert segment_row(route, tr)["kind"] == "descent"


def test_multi_name_corridor_tree_merges_ways(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(routes, "OSM_DIR", tmp_path)
    (tmp_path / "x_ways.json").write_text(
        json.dumps(
            {
                "elements": [
                    {"tags": {"name": "A"}, "geometry": [{"lat": 49.0, "lon": -123.0}]},
                    {
                        "tags": {"name": "B"},
                        "geometry": [{"lat": 49.001, "lon": -123.0}],
                    },
                    {
                        "tags": {"name": "C"},
                        "geometry": [{"lat": 49.002, "lon": -123.0}],
                    },
                ]
            }
        )
    )
    proj = routes._Proj(49.0, -123.0)
    assert routes._corridor_tree("x", ("A", "B"), proj).n == 2
    assert routes._corridor_tree("x", ("A",), proj).n == 1
    assert routes._corridor_tree("x", (), proj) is None


def test_instrument_constants_point_at_declared_burnaby_routes():
    ids = {r.segment_id for r in ROUTES}
    assert INSTRUMENT_CLIMB in ids and INSTRUMENT_DESCENT in ids
    assert INSTRUMENT_CLIMB.startswith("burnaby:")
    assert INSTRUMENT_DESCENT.startswith("burnaby:")
    # agents and seed import the same constants (single source of truth)
    from garmin_lab.agents import tools
    from garmin_lab.registry import seed

    assert tools.INSTRUMENT_CLIMB == INSTRUMENT_CLIMB
    assert seed.INSTRUMENT == {"climb": INSTRUMENT_CLIMB, "descent": INSTRUMENT_DESCENT}
