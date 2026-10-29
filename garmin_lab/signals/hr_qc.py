"""Per-sample quality flags for optical wrist heart rate.

Flags (bit mask in ``hr_qc``)::

    1  missing
    2  implausible value (<35 or >205 bpm)
    4  jump: |dHR/dt| > 8 bpm/s
    8  flatline: identical value for >= 60 s while moving (>1 m/s)
   16  post-gap: first 20 s after a gap/jump (sensor re-acquiring)

``hr_ok`` is True when ``hr_qc == 0``. No cadence is recorded on this device,
so cadence-lock cannot be detected directly; flatline + jump catch its usual
signature (sudden lock to a constant, sudden release).
"""

from __future__ import annotations

import numpy as np
import polars as pl

MISSING, IMPLAUSIBLE, JUMP, FLATLINE, POSTGAP = 1, 2, 4, 8, 16


def hr_quality_flags(
    hr: np.ndarray, t_s: np.ndarray, speed: np.ndarray | None = None
) -> np.ndarray:
    """Return an int flag array aligned with ``hr`` (NaN for missing)."""
    n = len(hr)
    flags = np.zeros(n, dtype=np.int16)
    hrf = hr.astype(float)
    miss = np.isnan(hrf)
    flags[miss] |= MISSING
    flags[(~miss) & ((hrf < 35) | (hrf > 205))] |= IMPLAUSIBLE

    if n > 1:
        dt = np.diff(t_s).astype(float)
        dt[dt <= 0] = 1.0
        rate = np.abs(np.diff(hrf)) / dt
        jump = np.zeros(n, dtype=bool)
        jump[1:] = rate > 8.0
        flags[jump] |= JUMP

        # flatline while moving
        same = np.zeros(n, dtype=bool)
        same[1:] = np.diff(hrf) == 0
        run = np.zeros(n, dtype=int)
        for i in range(1, n):
            run[i] = run[i - 1] + 1 if same[i] else 0
        moving = (
            np.ones(n, dtype=bool)
            if speed is None
            else (np.nan_to_num(speed.astype(float)) > 1.0)
        )
        # mark the whole run once it reaches 60 s of identical values
        flat_end = (run >= 60) & moving
        if flat_end.any():
            for i in np.flatnonzero(flat_end):
                flags[max(0, i - run[i]) : i + 1] |= FLATLINE

        # post-gap: 20 s after any missing or jump sample
        bad = miss | jump
        if bad.any():
            idx = np.flatnonzero(bad)
            for i in idx:
                flags[i + 1 : i + 21] |= POSTGAP
    return flags


def add_hr_qc(samples: pl.DataFrame) -> pl.DataFrame:
    """Add ``hr_qc`` and ``hr_ok`` columns to a samples frame (per activity)."""
    out = []
    for aid, g in samples.group_by("activity_id", maintain_order=True):
        g = g.sort("timestamp")
        t = g["timestamp"].dt.epoch("s").to_numpy()
        hr = g["heart_rate"].to_numpy().astype(float)
        sp = g["speed"].to_numpy().astype(float) if "speed" in g.columns else None
        fl = hr_quality_flags(hr, t, sp)
        out.append(g.with_columns(pl.Series("hr_qc", fl, dtype=pl.Int16)))
    df = pl.concat(out)
    return df.with_columns((pl.col("hr_qc") == 0).alias("hr_ok"))
