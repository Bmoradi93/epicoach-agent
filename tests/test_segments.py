"""Unit tests for segment discovery primitives and HR quality control."""

import numpy as np

from garmin_lab.segments.discover import (
    Run,
    consensus_gate,
    extract_traversals,
    ride_runs,
)
from garmin_lab.segments.trails import Chain
from garmin_lab.signals.hr_qc import FLATLINE, JUMP, MISSING, hr_quality_flags


def _straight_chain(length=1000.0, step=5.0):
    n = int(length / step) + 1
    pts = np.c_[np.linspace(0, length, n), np.zeros(n)]
    s = np.linspace(0, length, n)
    return Chain("t:Test", "Test", "t", pts, s, [1])


def test_consensus_gate_picks_common_stretch():
    chain = _straight_chain()
    cover = [(0, 1000)] * 3 + [
        (300, 800)
    ] * 4  # 7 rides; only 300..800 covered by >=60%
    gate = consensus_gate(chain, cover)
    assert gate is not None
    lo, hi = gate
    assert 280 <= lo <= 320 and 780 <= hi <= 820


def test_consensus_gate_requires_min_rides():
    chain = _straight_chain()
    assert consensus_gate(chain, [(0, 1000)] * 3) is None


def test_extract_single_forward_traversal():
    n = 200  # 1 Hz, 5 m/s along a 1000 m chain
    t = np.arange(n, dtype=float)
    s = 5.0 * t
    run = Run(1, 0, np.arange(n), s, forward=True)
    alt = np.linspace(100, 150, n)
    hr = np.full(n, 150.0)
    ok = np.ones(n, dtype=bool)
    speed = np.full(n, 5.0)
    dist = s.copy()
    trs = extract_traversals(run, (200.0, 800.0), t, alt, hr, ok, speed, dist, L=1000.0)
    assert len(trs) == 1
    tr = trs[0]
    assert abs(tr["duration_s"] - 120.0) < 1.5
    assert abs(tr["t_entry"] - 40.0) < 1.0 and abs(tr["t_exit"] - 160.0) < 1.0
    assert tr["d_alt"] > 0 and tr["hr_mean"] == 150.0


def test_extract_reverse_direction_uses_flipped_gates():
    n = 200
    t = np.arange(n, dtype=float)
    s = 1000.0 - 5.0 * t  # riding the chain backwards
    run = Run(1, 0, np.arange(n), s, forward=False)
    trs = extract_traversals(
        run,
        (200.0, 800.0),
        t,
        np.zeros(n),
        np.full(n, 120.0),
        np.ones(n, bool),
        np.full(n, 5.0),
        5.0 * t,
        L=1000.0,
    )
    assert len(trs) == 1 and abs(trs[0]["duration_s"] - 120.0) < 1.5


def test_extract_rejects_long_stop():
    n = 200
    t = np.arange(n, dtype=float)
    t[100:] += 200.0  # 200 s gap in the middle
    s = 5.0 * np.arange(n)
    run = Run(1, 0, np.arange(n), s, forward=True)
    trs = extract_traversals(
        run,
        (200.0, 800.0),
        t,
        np.zeros(n),
        np.full(n, 120.0),
        np.ones(n, bool),
        np.full(n, 5.0),
        s,
        L=1000.0,
    )
    assert trs == []


def test_ride_runs_direction_and_gap_fill():
    n = 300
    ci = np.zeros(n, dtype=int)
    ci[100:110] = -1  # 10 s GPS loss should be bridged
    s = 4.0 * np.arange(n)
    runs = ride_runs(7, ci, s, 1)
    assert len(runs) == 1 and runs[0].forward and runs[0].activity_id == 7


def test_hr_quality_flags():
    t = np.arange(200, dtype=float)
    hr = np.full(200, 140.0)
    hr[50] = np.nan
    hr[120] = 190.0  # +50 bpm jump
    hr[121] = 140.0
    fl = hr_quality_flags(hr, t, np.full(200, 3.0))
    assert fl[50] & MISSING
    assert fl[120] & JUMP
    assert fl[51] & 16 and fl[70] & 16 and not (fl[71] & 16)  # post-gap window is 20 s
    # a 60 s flatline while moving is flagged
    assert (fl[130:190] & FLATLINE).all()
    # but the first 59 identical samples before any flatline threshold are clean of FLATLINE
    assert not (fl[0:49] & FLATLINE).any()
