"""Per-ride sync: RIDE_CHECKLIST.md §C, steps 0-9, as one command.

    pixi run -e research python -m garmin_lab.ingest.sync            # full sync
    pixi run -e research python -m garmin_lab.ingest.sync --no-git   # skip commit/push/backup

Steps: refresh activity list from the API -> export -> build -> routes -> features ->
verify -> score open predictions -> re-issue -> commit + push -> backup. Stops at the first
step whose "Expect" condition fails and says what to do. Never refits the twin.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import subprocess
import sys
from pathlib import Path

import polars as pl

from garmin_lab.ingest.client import REPO_ROOT
from garmin_lab.segments.routes import INSTRUMENT_CLIMB, INSTRUMENT_DESCENT

log = logging.getLogger("garmin_lab.ingest.sync")
DATA = REPO_ROOT / "data"
DATASET = DATA / "dataset"
SINCE = "2025-04-01"
PY = [sys.executable, "-m"]


class SyncError(RuntimeError):
    pass


def _run(mod: str, *args: str) -> str:
    r = subprocess.run(PY + [mod, *args], capture_output=True, text=True, cwd=REPO_ROOT)
    out = (r.stdout + r.stderr)
    if r.returncode != 0:
        raise SyncError(f"{mod} failed (exit {r.returncode}):\n{out[-2000:]}")
    return out


def _sh(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if r.returncode != 0:
        raise SyncError(f"{' '.join(cmd)} failed:\n{(r.stdout + r.stderr)[-1000:]}")
    return r.stdout


def _instrument_counts() -> dict[str, int]:
    if not (DATASET / "traversals.parquet").exists():
        return {}
    t = pl.read_parquet(DATASET / "traversals.parquet")
    return {
        s: int(t.filter(pl.col("segment_id") == s).height)
        for s in (INSTRUMENT_CLIMB, INSTRUMENT_DESCENT)
    }


def step0_refresh_list() -> dict:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    from garmin_lab.ingest.client import make_client

    before = (
        len(json.loads((DATA / "raw" / "activities_list.json").read_text()))
        if (DATA / "raw" / "activities_list.json").exists()
        else 0
    )
    acts = make_client().get_activities_by_date(
        SINCE, dt.date.today().isoformat(), None, "asc"
    )
    (DATA / "raw" / "activities_list.json").write_text(json.dumps(acts))
    last = acts[-1]
    print(
        f"[0] activity list: {before} -> {len(acts)}; latest {last['activityId']} "
        f"{last['startTimeLocal'][:16]} {last['activityType']['typeKey']}"
    )
    if len(acts) <= before:
        print("    (no new activities: the watch may not have synced yet; continuing)")
    return last


def step1_export() -> None:
    out = _run("garmin_lab.ingest.export", "--since", SINCE)
    done = [ln for ln in out.splitlines() if "DONE" in ln]
    print(f"[1] export: {done[-1].split('DONE', 1)[1].strip() if done else out[-200:]}")
    if "'failed': 0" not in (done[-1] if done else ""):
        raise SyncError("export reported failures; re-run once, then report")


def step2_build() -> None:
    _run("garmin_lab.ingest.build")
    age = dt.datetime.now().timestamp() - (DATASET / "activities.parquet").stat().st_mtime
    if age > 600:
        raise SyncError("build did not refresh dataset parquets")
    print("[2] build: dataset parquets refreshed")


def step3_routes(before: dict[str, int]) -> dict[str, int]:
    _run("garmin_lab.segments.routes")
    after = _instrument_counts()
    for s in after:
        d = after[s] - before.get(s, 0)
        print(f"[3] {s.split(':')[1]:20s} traversals {before.get(s, 0)} -> {after[s]} ({d:+d})")
    return after


def step4_features() -> None:
    out = _run("garmin_lab.twin.features")
    ln = [x for x in out.splitlines() if "traversal_features:" in x]
    print(f"[4] {ln[-1].split('INFO', 1)[-1].strip() if ln else 'features rebuilt'}")


def step5_verify(latest_date: str) -> None:
    f = pl.read_parquet(DATASET / "traversal_features.parquet")
    for s in (INSTRUMENT_CLIMB, INSTRUMENT_DESCENT):
        t = f.filter(pl.col("segment_id") == s).sort("date", descending=True)
        r = t.row(0, named=True)
        flag = "" if str(r["date"]) == latest_date else "   <- not today's ride"
        print(
            f"[5] {s.split(':')[1]:20s} n={t.height:3d} latest={r['date']} moving={r['moving_s']:.0f}s "
            f"hr={r['hr_mean']:.0f} has_stop={r['has_stop']} sleep={r['sleep_h']:.2f}{flag}"
        )


def step6_score() -> int:
    from garmin_lab.registry.score import load_traversals, score_open
    from garmin_lab.registry.store import Registry

    rows = score_open(Registry(DATA / "registry.sqlite"), load_traversals())
    print(f"[6] scored {len(rows)} prediction(s)")
    return len(rows)


def step7_reissue() -> None:
    out = _run("garmin_lab.registry.seed", "--ledger-only")
    ln = [x for x in out.splitlines() if "ledger:" in x]
    print(f"[7] {ln[-1].split('ledger:', 1)[1].strip() if ln else 'ledger re-issued'}")
    from garmin_lab.registry.store import Registry

    open_ = [
        p
        for p in Registry(DATA / "registry.sqlite").ledger()
        if not p["scored_at"] and not p["withdrawn_at"]
    ]
    segs = {p["target"].split("@", 1)[1] for p in open_}
    missing = {INSTRUMENT_CLIMB, INSTRUMENT_DESCENT} - segs
    if missing:
        raise SyncError(f"no open prediction for {missing}; the next test cannot be scored")
    for p in sorted(open_, key=lambda p: p["target_time"]):
        d = json.loads(p["distribution"])
        mu = d["mu"] if d["family"] == "normal" else __import__("math").exp(d["mu"])
        print(f"    open: pred {p['id']} {p['target'].split(':')[1]:20s} mu={mu:.0f}s due {p['target_time'][:10]}")


def step8_commit(ride: str) -> None:
    _sh(["git", "add", "data/"])
    st = _sh(["git", "status", "--porcelain", "data/"])
    if not st.strip():
        print("[8] git: nothing to commit")
        return
    _sh(["git", "commit", "-q", "-m", f"ride {ride}: synced, scored, re-issued"])
    _sh(["git", "push", "-q", "origin", "main"])
    print(f"[8] git: committed + pushed ({_sh(['git', 'log', '--oneline', '-1']).strip()})")


def step9_backup() -> None:
    out = _sh(["bash", "scripts/backup_data.sh"])
    print(f"[9] backup: {out.strip().splitlines()[-1]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-git", action="store_true", help="skip commit/push/backup")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    try:
        before = _instrument_counts()
        last = step0_refresh_list()
        step1_export()
        step2_build()
        step3_routes(before)
        step4_features()
        step5_verify(last["startTimeLocal"][:10])
        step6_score()
        step7_reissue()
        if not a.no_git:
            step8_commit(f"{last['startTimeLocal'][:10]} ({last['activityId']})")
            step9_backup()
    except SyncError as e:
        print(f"\nSYNC STOPPED: {e}\nDo not refit. Report this to Fable.", file=sys.stderr)
        return 1
    print("\nsync complete. Twin NOT refit (Sunday's job).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
