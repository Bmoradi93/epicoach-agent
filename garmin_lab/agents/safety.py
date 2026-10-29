"""Hard training-safety constraints. These are code, not judgement.

A candidate week is a list of 7 planned daily loads (exp-TRIMP) plus a maximum
heart-rate target. ``check_week`` returns the list of violated constraints;
an empty list means feasible. The LLM may explain a veto; it cannot lift one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# 2026-10-22, with the athlete's explicit consent (registry event `protocol_change`): the
# ACWR cap was raised from 1.3 to 1.5 (upper edge of the Gabbett moderate band) and the weekly
# tolerance from +-15% to +-35%, because the pre-registered `overload_130` design could never
# pass +-15% and had been vetoed in both live cycles. Unchanged: HR cap, readiness gate, rest
# day, illness rule. New: an overload week (> +15%) may not follow another overload week.
ACWR_LO, ACWR_HI = 0.8, 1.5
WEEKLY_TOL = 0.35
OVERLOAD_FRAC = 1.15  # a week above this share of target counts as an overload
MIN_REST_DAYS = 1
HR_CAP_FRAC = 0.92
MIN_READINESS = 30


@dataclass
class AthleteState:
    load_history: np.ndarray  # daily exp-TRIMP, most recent last (>= 28 days)
    hr_max: float
    readiness: float | None = None
    illness_reported: bool = False
    unavailable_days: list[int] = field(
        default_factory=list
    )  # 0..6 offsets from tomorrow
    block_weekly_target: float | None = None  # if None, use trailing 4-week mean


def acwr_after(history: np.ndarray, planned: np.ndarray) -> np.ndarray:
    """ACWR evaluated at the end of each planned day (7-day acute / 28-day chronic)."""
    full = np.concatenate([history, planned])
    out = np.empty(len(planned))
    for i in range(len(planned)):
        end = len(history) + i + 1
        acute = full[end - 7 : end].mean()
        chronic = full[end - 28 : end].mean()
        out[i] = acute / chronic if chronic > 1e-6 else (np.inf if acute > 0 else 1.0)
    return out


def check_week(
    state: AthleteState, planned_loads: list[float], hr_target_max: float | None
) -> list[str]:
    v: list[str] = []
    p = np.asarray(planned_loads, dtype=float)
    if len(p) != 7:
        return ["a plan must cover exactly 7 days"]
    if state.illness_reported:
        if p.sum() > 0:
            v.append("illness reported: only rest may be prescribed")
    if state.readiness is not None and state.readiness < MIN_READINESS and p[0] > 0:
        v.append(
            f"training readiness {state.readiness:.0f} < {MIN_READINESS}: tomorrow must be rest"
        )
    ac = acwr_after(state.load_history, p)
    if (ac > ACWR_HI).any():
        v.append(
            f"ACWR exceeds {ACWR_HI} on day(s) {[int(i) + 1 for i in np.flatnonzero(ac > ACWR_HI)]} (max {ac.max():.2f})"
        )
    if (ac < ACWR_LO).all() and p.sum() > 0:
        v.append(
            f"ACWR below {ACWR_LO} all week ({ac.max():.2f}): detraining plan is not a training plan"
        )
    target = (
        state.block_weekly_target
        if state.block_weekly_target is not None
        else float(state.load_history[-28:].sum() / 4)
    )
    if (
        target > 0
        and not (1 - WEEKLY_TOL) * target <= p.sum() <= (1 + WEEKLY_TOL) * target
    ):
        # deload weeks are permitted explicitly when the plan says so (<= 50% of target)
        if not p.sum() <= 0.5 * target:
            v.append(
                f"weekly load {p.sum():.0f} outside ±{int(WEEKLY_TOL*100)}% of target {target:.0f} and not a deload (<= {0.5*target:.0f})"
            )
    if target > 0 and p.sum() > OVERLOAD_FRAC * target:
        last_week = float(state.load_history[-7:].sum())
        if last_week > OVERLOAD_FRAC * target:
            v.append(
                f"overload week ({p.sum():.0f}) after an overload week ({last_week:.0f} > {OVERLOAD_FRAC:.2f}x target {target:.0f}): consecutive overloads are not permitted"
            )
    if (p == 0).sum() < MIN_REST_DAYS:
        v.append(f"fewer than {MIN_REST_DAYS} rest day(s)")
    if hr_target_max is not None and hr_target_max > HR_CAP_FRAC * state.hr_max:
        v.append(
            f"heart-rate target {hr_target_max:.0f} exceeds cap {HR_CAP_FRAC * state.hr_max:.0f}"
        )
    for d in state.unavailable_days:
        if 0 <= d < 7 and p[d] > 0:
            v.append(f"day {d + 1} is marked unavailable")
    return v
