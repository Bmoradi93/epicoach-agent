# EpiCoach

Code, hypothesis registry, and derived dataset for **"The Athlete as a
Scientist of One: Agentic Bayesian Experimental Design for Individualized
Physiology from Consumer Wearables"** (preprint forthcoming on arXiv).

EpiCoach treats one athlete as the subject of ongoing, accountable $N{=}1$
science. A digital twin inferred from wrist heart rate, GPS, and barometric
altitude issues falsifiable, dated predictions about future rides into an
evidence-gated hypothesis registry; an LLM-agent weekly cycle (Google ADK)
proposes hypotheses and designs the next week's training as a
safety-constrained experiment. See the paper for the full method and
results.

## What's in this repository

| Path | Contents |
|---|---|
| `garmin_lab/` | The research system: ingestion/QC, segment discovery, the twin (heart-rate response, skill, fitness-fatigue), the hypothesis registry, the replay benchmark, and the weekly agent cycle |
| `garminconnect/` | Vendored, lightly modified Garmin Connect API wrapper used by ingestion |
| `garmin_agent/` | A separate, simpler interactive Google ADK chat assistant over Garmin Connect (not the paper's weekly-cycle system) |
| `garmin_mcp_server.py` | Standalone MCP server exposing the same Garmin Connect tools over stdio |
| `tests/` | Unit tests for the registry, safety constraints, segment discovery, twin, and replay harness |
| `scripts/make_figures.py` | Regenerates the paper's data figures from `data/dataset/` |
| `data/registry.sqlite` | The hypothesis registry and full prediction ledger: every hypothesis, belief, prediction, piece of evidence, and audit event |
| `data/cycles/` | Weekly agent reports, transcripts, and prescribed workouts |
| `data/osm/` | Cached OpenStreetMap trail geometry for the riding areas |
| `data/dataset/` | Derived, privacy-reviewed tables (see below) |
| `RIDE_CHECKLIST.md`, `data_workflow.md` | The operational protocol for instrument rides, sync, and weekly refits |

## Data availability and privacy

This is a single-subject ($N{=}1$) study and the subject is an author. We
release what supports reproducing the paper's analysis while withholding
what would expose the athlete's home location or raw day-to-day health
telemetry. Specifically:

**Included**, in `data/dataset/`:
- `activities.parquet` — per-ride summaries (duration, HR, elevation,
  training load, weather, device metrics). **Exact GPS start/end
  coordinates and location name have been removed** from the version
  published here.
- `segments.parquet`, `laps.parquet`, `traversals.parquet` — discovered
  trail segments and per-traversal timing/HR. Segment gate coordinates are
  included; they are points on named, public trails already identified in
  the paper, not the athlete's home.
- `activity_load.parquet`, `daily_load.parquet` — derived daily training
  load (TRIMP variants), used by the fitness-fatigue model.
- `vo2max.parquet` — daily VO2max/fitness-age estimates, used as an
  auxiliary channel in the paper.
- `inventory.md` — a plain-text summary of the above.

**Not included, anywhere in this repository**:
- Raw per-second sensor samples (`samples.parquet` in the internal dataset).
- The raw Garmin API activity-list dump, which contains the account owner's
  full name, profile image URLs, and exact per-activity GPS coordinates.
- The day-by-day personal health panel (`daily.parquet`): sleep stages,
  overnight HRV, resting heart rate, stress, body battery, steps, SpO2, and
  readiness, at daily resolution. The paper reports what this file supports
  in aggregate; the file itself is not published.

**Consequence**: `garmin_lab/agents/tools.py` (the live weekly-cycle agent)
reads `daily.parquet` and a live Garmin Connect session directly, and will
not run end-to-end against the dataset shipped here. The registry, the
replay benchmark, and the twin model fits — the three things the paper
commits to releasing — all run against what is included.

## Reproducing

```bash
# install the environment (pixi: https://pixi.sh)
pixi install

# run the test suite
pixi run -e research test-lab

# regenerate the paper's figures from data/dataset/
pixi run -e research python scripts/make_figures.py

# inspect the hypothesis registry
pixi run -e research python -c "
from garmin_lab.registry.store import Registry
r = Registry('data/registry.sqlite')
print(r.list())
"
```

Re-running ingestion (`garmin_lab.ingest.export` / `.build`) or the live
agent cycle (`garmin_lab.agents.cycle`) requires your own Garmin Connect
account and credentials in a local `.env` (see `data_workflow.md`); they
cannot be run against this repository's data alone.

## Citation

A preprint is forthcoming on arXiv; this section will be updated with the
full citation once posted.

## License

MIT, see `LICENSE`. `garminconnect/` retains its original upstream
attribution (Ron Klinkien); additions are (c) 2026 Behnam Moradi.
