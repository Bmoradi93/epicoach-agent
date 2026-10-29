# Instrument Ride Checklist

The single operational document for the prospective N=1 experiment. Follow it exactly on
every instrument ride. Background and rationale live in `data_workflow.md`; this file is
only *what to do, in what order, and what the output must look like*.

All commands run from the repo root with the `research` pixi env.

---

## A. Before the ride

| Check | How | If not |
|---|---|---|
| Two predictions are **open** (one climb, one descent) with `due` in the future | `make ledger` (or the "Ledger" command in §D) | Run step 7 below *before* riding; a ride without a sealed prediction cannot be scored |
| Watch has GPS lock before starting the recording | watch | Wait; a late lock loses the warm-up but not the instrument |
| Protocol for the day is known: climb target **160 ± 5 bpm**, descent **self-selected** | `data_workflow.md` → Current Instrument | — |

Do **not** look at the open prediction numbers before riding if you can avoid it. The
test is of the twin, not of your pacing to a number.

## B. During the ride — lap protocol

Press **LAP** four times, at the same physical spots every time (±3 m is fine):

| Lap | Where | Marks |
|---|---|---|
| 1 | TCT east gate — bottom of the Trans Canada Trail climb | climb start |
| 2 | University Drive — where the gravel meets the road | climb end |
| 3 | Poplar/Nicoles junction — your habitual stopping spot | descent start |
| 4 | Pipeline — Nicoles exit | descent end |

- Stopping *at* a lap point is fine (the clock starts when you roll out of the gate circle).
- No laps = training ride (goes to the twin, never scored). Laps = test. Be deliberate.
- Stopping *between* lap 1–2 or 3–4 marks the traversal `has_stop`; it is still scored (we score
  moving time) but is excluded from the skill-model fit.
- Everything outside laps 1–2 and 3–4 is ignored for scoring.

## C. After the ride — 10 steps (~3 min)

Wait until the ride appears on Garmin Connect. Every step is idempotent and skips cached data.

### 0. Refresh the activity list from the API
```bash
pixi run -e research python -c "
import json, pathlib, datetime as dt
from dotenv import load_dotenv; load_dotenv('.env')
from garmin_lab.ingest.client import make_client
acts = make_client().get_activities_by_date('2025-04-01', dt.date.today().isoformat(), None, 'asc')
pathlib.Path('data/raw/activities_list.json').write_text(json.dumps(acts))
print(len(acts), 'activities; latest', acts[-1]['activityId'], acts[-1]['startTimeLocal'][:10])"
```
**Expect:** count up by ≥1 vs last time; `latest` is today's activity id and date.
**If not:** the watch has not synced yet. Wait, re-run. Never rebuild this file from local stubs.

### 1. Fetch new FIT + splits + wellness
```bash
pixi run -e research python -m garmin_lab.ingest.export --since 2025-04-01
```
**Expect:** last line `DONE  calls=N  {'fetched': >0, 'skipped': ..., 'failed': 0}`.
**If `failed` > 0:** re-run once (Garmin rate limit); if it persists, stop and report.
Never use a narrow `--since`.

### 2. Rebuild dataset
```bash
pixi run -e research python -m garmin_lab.ingest.build
```
**Expect:** exits 0, `data/dataset/garmin.duckdb` and `*.parquet` have today's timestamp.

### 3. Detect instrument traversals
```bash
pixi run -e research python -m garmin_lab.segments.routes
```
**Expect:** both `burnaby:` lines show `n_traversals` **up by exactly 1** vs the last ride
(check `data_workflow.md` → Current Instrument for the previous counts, then update them).
**If a count did not grow:** the gate missed. Do **not** score by hand. Run the gate
diagnostic (§D) and report: it is either a GPS problem (ride is lost for that instrument, note it in
the registry events) or an instrument problem (gate moves, like 2026-10-04; requires a decision).

### 4. Rebuild features
```bash
pixi run -e research python -m garmin_lab.twin.features
```
**Expect:** `traversal_features: N rows` with N up by the number of new traversals.

### 5. Verify
```bash
pixi run -e research python -c "
import polars as pl
f = pl.read_parquet('data/dataset/traversal_features.parquet')
for sid in ('burnaby:Trans Canada Trail:east->UniversityDr','burnaby:Poplar->Nicoles:desc'):
    t = f.filter(pl.col('segment_id')==sid).sort('date', descending=True)
    r = t.row(0, named=True)
    print(f\"{sid.split(':')[1]:20s} n={t.height:3d} latest={r['date']} moving={r['moving_s']:.0f}s hr={r['hr_mean']:.0f} has_stop={r['has_stop']} sleep={r['sleep_h']:.2f}\")"
```
**Expect:** `latest` = today for both; `moving` within ~3 s of your lap 1→2 and lap 3→4 times;
climb `hr` within 160 ± 5 if you held protocol (logged either way, does not invalidate the score).
**If `latest` is not today:** stop. Steps 2–4 did not pick up the ride; re-run from step 2 and
look for errors you filtered out.

### 6. Score open predictions
```bash
pixi run -e research python -m garmin_lab.registry.score          # add --dry-run to preview
```
**Expect:** one `scored pred` line per open prediction that now has a qualifying ride.
**Rules (enforced in `registry/score.py` + registry invariant I10):**
- a prediction is scored by the **first traversal of its segment strictly after it was sealed**
  (`registered_at`, compared by timestamp) and not after its `target_time`;
- **one traversal scores at most one prediction** (`outcome_ref`); predictions are processed in
  due-date order, so the earlier-due one claims the earlier ride;
- scored **as sealed**, always — even if the instrument or forecast method was wrong (pred 6).
- Never withdraw a prediction whose outcome has been observed. Never edit a distribution.
- A ride with no open prediction for it is **data only**, never scored retroactively.

**Only protocol rides score** (since 2026-10-22): a traversal qualifies only if the ride has a
lap press within 60 s of both the gate entry and the gate exit. Training rides up the TCT without
the 4-lap protocol are data for the twin but never claim a prediction. So: **press the laps on
test rides, don't press them on training rides.** That is the whole distinction.

### 7. Re-issue predictions for the next ride (no refit!)
```bash
pixi run -e research python -m garmin_lab.registry.seed --ledger-only
```
**Expect:** `ledger: 2 predictions registered for before <date+14d>`, one climb (normal, seconds)
and one descent (lognormal, log-seconds). Idempotent: if a target is already open it is skipped.
**If it prints a `_note` about segment sets / refit:** the stored fit is out of date with the
feature table. Do not refit now unless it is Sunday; report it.

### 8. Commit the record
```bash
git add data/ data_workflow.md && git commit -m "ride $(date +%F): scored + re-issued" && git push
```
Tracked: registry, cycles, dataset tables (not `samples`), OSM, activity list.
Not tracked: FIT files, 1 Hz samples, duckdb, twin fits, replay.

### 9. Back up the bulk data
```bash
bash scripts/backup_data.sh            # -> ~/backups/garmin-ai-agents (+ dated registry snapshot)
# BACKUP_DIR=/media/<drive> bash scripts/backup_data.sh   # when an external drive is mounted
```
**Expect:** `registry snapshots: K` with K up by 1.

### 10. Report
Paste the output of steps 5 and 6 to Fable, plus anything that did not match an **Expect** line.
Update the traversal counts in `data_workflow.md` → Current Instrument.

**Do not refit the twin.** Refits happen on Sunday (§E), after every prediction of the week is scored.

---

## D. Diagnostics

**Ledger** — open / scored / withdrawn predictions:
```bash
pixi run -e research python -c "
import json, math
from garmin_lab.registry import Registry
for p in Registry('data/registry.sqlite').ledger():
    st='SCORED' if p['scored_at'] else ('WITHDRAWN' if p['withdrawn_at'] else 'OPEN')
    d=json.loads(p['distribution']); mu=d['mu'] if d['family']=='normal' else math.exp(d['mu'])
    line=f\"pred {p['id']:2d} (H{p['hypothesis_id']}) [{st:9s}] {p['target'].split(':')[1]:20s} mu={mu:5.0f}s due {p['target_time'][:10]}\"
    if p['scored_at']: line+=f\"  -> {p['outcome']:.0f}s crps={p['crps']:.2f}\"
    print(line)"
```

**Gate diagnostic** — how close did a ride pass to each gate (use when a count did not grow):
```bash
pixi run -e research python -c "
import sys, polars as pl, numpy as np
from garmin_lab.segments.routes import ROUTES
aid = int(sys.argv[1])
s = pl.read_parquet('data/dataset/samples.parquet').filter(pl.col('activity_id')==aid)
for r in ROUTES:
    if r.area != 'burnaby': continue
    for name, g, rad in (('in', r.gate_in, r.r_in_m), ('out', r.gate_out, r.r_out_m)):
        d = np.hypot((s['lat'].to_numpy()-g[0])*111320, (s['lon'].to_numpy()-g[1])*111320*np.cos(np.radians(g[0])))
        print(f'{r.segment_id:48s} gate_{name:3s} min {np.nanmin(d):5.1f} m (radius {rad:.0f})')" <ACTIVITY_ID>
```
A gate is hit when `min` < `radius`. If a gate is missed by a few metres on a ride that clearly
followed the trail, that is an instrument question — bring it to Fable; do not hand-score.

**Lap vs gate** — cross-check the scored time against your lap presses:
```bash
pixi run -e research python -c "
import sys, polars as pl
aid=int(sys.argv[1])
l=pl.read_parquet('data/dataset/laps.parquet').filter(pl.col('activity_id')==aid).sort('lap_index')
print(l.select(['lap_index','total_timer_time','total_distance','avg_heart_rate','total_ascent','total_descent']))" <ACTIVITY_ID>
```
Lap index 1 = climb (laps 1→2), lap index 3 = descent (laps 3→4). Should agree with step 5 within ~3 s.

---

## E. Weekly (Sunday) — refit + agent cycle

Only after all of the week's rides are scored (Golden Rule: a prediction is always made from a fit
that has not seen the ride it predicts).

```bash
pixi run -e research python -m garmin_lab.twin.hr_model
pixi run -e research python -m garmin_lab.twin.skill_model
pixi run -e research python -m garmin_lab.twin.ffm
pixi run -e research python -m garmin_lab.twin.recovery
pixi run -e research python -m garmin_lab.agents.cycle      # needs GEMINI_API_KEY; writes data/cycles/report_*.md
pixi run -e research python -m garmin_lab.registry.seed --ledger-only   # re-issue from the fresh fit
git add data/ && git commit -m "weekly $(date +%F): refit + cycle" && git push
bash scripts/backup_data.sh
```

---

## F. Things that are never done

- Refit the twin on a ride day.
- Score a traversal by hand from lap times.
- Withdraw or edit a prediction whose outcome has been observed.
- Rebuild `activities_list.json` from local files.
- Use a narrow `--since` on export.
- Change a gate without logging an `instrument_correction` event and scoring open predictions as sealed.
- Score a prediction against a ride that happened before it was sealed.
- Ride an instrument test with no open prediction (sync first).
- Press the lap button at the gates on a training ride (it would make it a test).
