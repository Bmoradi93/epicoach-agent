# Data Workflow

How data flows from your watch to the registry, and what runs when.

---

## AFTER EACH RIDE

**→ Follow `RIDE_CHECKLIST.md`.** It is the single operational document: pre-ride checks, the
4-lap protocol, the 10 post-ride steps with expected output and failure branches, diagnostics,
the Sunday refit, and the list of things that are never done. This file is the background.

### What is safe where

| If the repo is deleted… | …you still have | Because |
|---|---|---|
| Registry (hypotheses, ledger, every score) | ✅ in git | `data/registry.sqlite` tracked |
| Weekly reports, workouts, transcripts | ✅ in git | `data/cycles/` tracked |
| Per-ride, per-traversal, per-day tables | ✅ in git | `data/dataset/*.parquet` tracked (not `samples`) |
| Frozen trail geometry | ✅ in git | `data/osm/` tracked |
| Which activities exist | ✅ in git | `data/raw/activities_list.json` tracked |
| FIT files, 1 Hz samples, sleep/HRV raw | ⚠️ backup only | personal + large; re-downloadable from Garmin while it lasts |
| Twin fits, replay outputs | ⚠️ backup only | re-computable from the above (~1.5 h total) |

---

## The Golden Rule

> A prediction is always made from a fit that has **not seen the ride it predicts.**

Per-ride sync scores predictions and re-issues new ones from the **current** (stale) fit.
The weekly cycle refits the twin **after** all predictions for that week are already scored.
This is the same rolling-origin discipline used in the replay benchmark, applied live.

---

## Two Loops

### Loop 1 — Per-Ride Sync (~1 min)
Run this after every ride syncs to Garmin Connect.

Until `sync.py` exists, the steps are the commands in `RIDE_CHECKLIST.md` §C. Order matters:

| # | Step | What it does |
|---|---|---|
| 1 | **Fetch** | Pull new activities + wellness from Garmin Connect |
| 2 | **Build** | Rebuild dataset parquets (activities, samples, daily) |
| 3 | **Routes** | Detect instrument traversals by gate (laps are a cross-check) |
| 4 | **Features** | Rebuild traversal_features.parquet |
| 5 | **Score** | Score any open ledger predictions against new traversals — BEFORE any refit |
| 6 | **Re-issue** | Register new predictions for the next ride from the CURRENT fit |
| 7 | **Report** | Print: actual vs predicted, HR compliance, sleep, belief changes |

**What does NOT happen here:** twin refit. The fit stays stale until the weekly cycle.

---

### Loop 2 — Weekly Cycle (~15 min)
Run once per week, on your rest day (Sunday).

```bash
pixi run -e research python -m garmin_lab.agents.cycle
```

Steps (in order):

| # | Step | What it does |
|---|---|---|
| 1 | **Refit** | Refit HR model, skill model, FFM, recovery with all new data |
| 2 | **Hypotheses** | Hypothesizer reviews residuals, proposes/refines hypotheses |
| 3 | **Evidence** | Statistician evaluates open hypotheses against new fit |
| 4 | **Design** | Designer scores workout menu by EIG, safety-checks top plans |
| 5 | **Pre-register** | Ledger predictions for next week's fresh + loaded tests |
| 6 | **Report** | Scribe writes weekly report (data/cycles/report_YYYY-MM-DD.md) |
| 7 | **Push** | *(Phase 5, not built yet)* Write workouts to Garmin Connect |

---

## Scoring Rules

- **Climb:** matched when a TCT traversal is found in the new ride by **gate detection** (`routes.py`). Lap presses are a cross-check, not the source of truth (gates are what the 79 historical rides were measured with).
- **Descent:** same — Nicoles traversal by gates. Since 2026-10-04 the gate-in is the athlete's lap-press point at the Poplar–Nicoles junction and the clock starts when the rider rolls out of the gate circle, so gate and lap agree within ~2 s.
- **HR compliance:** logged but does not invalidate the score. A ride at 170 bpm instead of 160 is scored and flagged.
- **Sleep:** pulled from wellness for the ride date. Used to annotate the descent score for H3.
- **Prediction window:** a prediction is scored by the **first traversal whose entry timestamp is after the prediction was sealed** (`registered_at`) and before its `target_time`. Later rides in the window are data only. Compared by timestamp, not date.

---

## Lap Protocol

See `RIDE_CHECKLIST.md` §B.

---

## What the Agent Prescribes (and what it never touches)

The **instrument rides never change**: TCT climb at 160 ± 5 bpm, Nicoles descent at self-selected
effort, 4 lap presses, twice a week — day 0 (*fresh*, after the Saturday rest day) and day 4
(*loaded*). They are the measurement; changing them would break the 80-ride baseline.

What the agent designs each Sunday is **the training between the tests**: one plan from the menu
in `agents/eig.py`, expressed as daily exp-TRIMP targets.

| Pattern | Free-day load (vs your 28-day average) | Why the agent might pick it |
|---|---|---|
| `conventional` | 100%, spread over Mon/Tue/Wed/Fri | control week |
| `front_loaded` / `back_loaded` | 100%, pushed early / late | separates fatigue (τ₂) from fitness (τ₁) |
| `deload_50` / `deload_25` / `tests_only` | 50% / 25% / 0% | reveals how fast fatigue clears |
| `overload_115` / `overload_130` | 115% / 130% | reveals how fast fatigue builds (safety-capped) |
| two- and three-week versions | e.g. overload → deload | larger contrasts for τ₁ |

It picks the plan with the highest expected information gain over the open hypotheses, subject to
`agents/safety.py` (ACWR bounds, HR cap, readiness gate). The output is in
`data/cycles/report_<date>.md`: a 7-day table of load targets plus the two sealed test predictions.

Your job: ride the instrument tests exactly as always, and get the free days roughly on target
(any riding counts — load is measured from HR, not from a specific route). Saturday is always rest.

---

## What's Built vs What's Pending

| Component | Status |
|---|---|
| `garmin_lab/ingest/export.py` — fetch from Garmin | ✅ Built |
| `garmin_lab/ingest/build.py` — rebuild dataset | ✅ Built |
| `garmin_lab/segments/routes.py` — gate detection | ✅ Built |
| `garmin_lab/twin/features.py` — traversal features | ✅ Built |
| `garmin_lab/registry/store.py` — score + re-issue | ✅ Built |
| `garmin_lab/ingest/sync.py` — **per-ride sync script** | 🔲 To build |
| `garmin_lab/agents/cycle.py` — weekly cycle | ✅ Built |
| `garmin_lab/pusher/` — push workouts to watch | 🔲 Phase 5 |

---

## Current Instrument

| | Segment ID | Gates | History |
|---|---|---|---|
| **Climb** | `burnaby:Trans Canada Trail:east->UniversityDr` | (49.2801, −122.8967) → (49.2803, −122.9210) | 81 traversals (79 historical + 10-03, 10-04) |
| **Descent** | `burnaby:Poplar->Nicoles:desc` | (49.27160, −122.90855) → (49.2679, −122.9024) | 25 traversals (23 historical + 10-03, 10-04) |

Protocol: hold **160 ± 5 bpm** on the climb. Self-selected effort on the descent.

---

## Ledger State (updated after each ride)

Run this to see open predictions:

```bash
pixi run -e research python -c "
from garmin_lab.registry import Registry
reg = Registry('data/registry.sqlite')
for p in reg.ledger():
    status = 'SCORED' if p['scored_at'] else ('WITHDRAWN' if p['withdrawn_at'] else 'OPEN')
    print(f\"pred {p['id']} (H{p['hypothesis_id']}) [{status}] {p['target']} due {p['target_time'][:10]}\")
    if p['scored_at']:
        print(f\"  outcome={p['outcome']:.0f}  crps={p['crps']:.1f}\")
"
```

---

## Gotchas (learned 2026-10-03, first instrument ride)

1. **`activities_list.json` is a cache.** `export.py` only fetches activities that appear in it.
   A new ride is invisible until the list is refreshed **from the API** — do NOT rebuild it from the
   local `summary.json` stubs (they lack `startTimeLocal`; the build will crash). `sync.py` must call
   `get_activities_by_date(since, today)` and overwrite the list every run.
2. **`--since` on export also bounds the list.** `--since 2026-10-03` with a missing cache produced a
   one-activity dataset and wiped 18 months from the parquets. Always fetch the full range.
3. **`build.py` recreates `garmin.duckdb` from scratch.** It now also registers `segments` and
   `traversals` views (fixed in build.py), otherwise `twin.features` dies with
   `Table with name traversals does not exist` and `traversal_features.parquet` silently stays stale.
   **Always check the features timestamp / row count after a build.**
4. **Gate timing vs lap timing.** Gates and laps agree on the climb (±3 s) and, since 2026-10-04,
   on the descent (±2 s). Scoring uses **gates**; laps are a cross-check.
5. **Descent gate_in was wrong until 2026-10-04.** It sat on the OSM junction node, 49 m from where
   the trail actually meets. 10/25 descents passed 36–41 m from it; ride 24605778669 missed by 0.7 m.
   Moved to the athlete's lap-press point (all 25 within 10 m). Baseline median moving 337 → 316 s.
   Pred 6 was scored **as sealed** against the old baseline (option A); logged as `instrument_correction`
   in the registry events. Preds ≥ 7 use the corrected gate.
6. **Descent forecast uses the skill model** (since 2026-10-04; `agents/tools._descent_prediction`):
   segment level + random-walk skill state (one step ahead) + covariates at protocol conditions
   (dry, habitual sleep, median elapsed_h). Before: historical median with the model's noise only.
   Pred 8 withdrawn (no outcome observed) and re-issued as pred 10.
7. **Gate clock starts when the rider rolls *out* of the entry circle** (`routes._skip_dwell`):
   stationary samples and auto-pause gaps inside the circle are skipped. Clean Nicoles descents
   (`~has_stop`) went 5 → 17; the skill model now sees the instrument.
8. **`--ledger-only` is idempotent**: it will not re-issue a target that already has an open prediction.
9. **`--ledger-only` reshapes posteriors by the fit's own dims**, not the current features
   (`agents/tools.py`). The fit is stale by design; features grow daily. Fixed 2026-10-04.

## Weekly (Sunday) — refit + cycle

See `RIDE_CHECKLIST.md` §E. Only after all of the week's rides have been scored (Golden Rule).
