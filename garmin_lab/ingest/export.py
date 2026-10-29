"""Resumable bulk export of Garmin Connect history to ``data/raw/``.

Usage (from repo root)::

    pixi run -e research python -m garmin_lab.ingest.export --since 2024-09-01

Every API response is written to disk as JSON (or the original FIT zip) and
is never re-fetched if the file already exists, so the job can be interrupted
and resumed. Nothing here transforms data; see ``build.py`` for that.

Layout::

    data/raw/activities_list.json              all activity summaries in range
    data/raw/activities/<id>/summary.json      GET /activity-service/activity/<id>
    data/raw/activities/<id>/original.zip      FIT inside
    data/raw/activities/<id>/{splits,hr_zones,weather,gear}.json   (cycling only)
    data/raw/daily/<date>/{sleep,training_status}.json
    data/raw/range/<name>/<start>_<end>.json   GraphQL / REST range queries
    data/raw/profile/{user_profile,devices,gear,personal_records}.json
    data/raw/errors.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

from garminconnect import Garmin

from .client import REPO_ROOT, Paced, make_client

log = logging.getLogger("garmin_lab.export")

CYCLING_TYPES = {
    "cycling",
    "mountain_biking",
    "gravel_cycling",
    "road_biking",
    "e_bike_mountain",
    "e_bike_fitness",
    "cyclocross",
    "indoor_cycling",
    "virtual_ride",
    "bmx",
    "track_cycling",
    "downhill_biking",
    "recumbent_cycling",
}

RANGE_CHUNK_DAYS = 28


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    tmp.replace(path)


def _daterange(start: dt.date, end: dt.date) -> Iterator[dt.date]:
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def _chunks(
    start: dt.date, end: dt.date, days: int
) -> Iterator[tuple[dt.date, dt.date]]:
    a = start
    while a <= end:
        b = min(a + dt.timedelta(days=days - 1), end)
        yield a, b
        a = b + dt.timedelta(days=1)


class Exporter:
    def __init__(self, api: Garmin, data_dir: Path, pace_s: float) -> None:
        self.api = api
        self.p = Paced(api, min_interval_s=pace_s)
        self.raw = data_dir / "raw"
        self.raw.mkdir(parents=True, exist_ok=True)
        self.errors = self.raw / "errors.jsonl"
        self.stats: dict[str, int] = {"fetched": 0, "skipped": 0, "failed": 0}

    # -- generic cached fetch -------------------------------------------------
    def fetch(
        self, path: Path, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any | None:
        """Call ``fn`` and cache the JSON result at ``path``; skip if cached."""
        if path.exists():
            self.stats["skipped"] += 1
            return None
        try:
            obj = self.p.call(fn, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            self.stats["failed"] += 1
            with self.errors.open("a") as fh:
                fh.write(
                    json.dumps(
                        {
                            "path": str(path.relative_to(self.raw)),
                            "fn": getattr(fn, "__name__", str(fn)),
                            "args": [str(a) for a in args],
                            "error": f"{type(exc).__name__}: {exc}"[:500],
                            "at": dt.datetime.now().isoformat(timespec="seconds"),
                        }
                    )
                    + "\n"
                )
            log.warning("FAILED %s: %s", path.name, str(exc)[:120])
            return None
        _write_json(path, obj)
        self.stats["fetched"] += 1
        return obj

    def fetch_bytes(
        self, path: Path, fn: Callable[..., bytes], *args: Any, **kwargs: Any
    ) -> None:
        if path.exists() and path.stat().st_size > 0:
            self.stats["skipped"] += 1
            return
        try:
            data = self.p.call(fn, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            self.stats["failed"] += 1
            with self.errors.open("a") as fh:
                fh.write(
                    json.dumps(
                        {
                            "path": str(path.relative_to(self.raw)),
                            "error": str(exc)[:500],
                        }
                    )
                    + "\n"
                )
            log.warning("FAILED %s: %s", path, str(exc)[:120])
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.stats["fetched"] += 1

    # -- profile ---------------------------------------------------------------
    def export_profile(self) -> None:
        d = self.raw / "profile"
        prof = self.fetch(d / "user_profile.json", self.api.get_user_profile)
        if prof is None and (d / "user_profile.json").exists():
            prof = json.loads((d / "user_profile.json").read_text())
        self.fetch(d / "devices.json", self.api.get_devices)
        self.fetch(d / "personal_records.json", self.api.get_personal_record)
        self.fetch(
            d / "primary_training_device.json", self.api.get_primary_training_device
        )
        if prof and "id" in prof:
            self.fetch(d / "gear.json", self.api.get_gear, prof["id"])

    # -- activities -----------------------------------------------------------
    def export_activities(
        self, since: dt.date, until: dt.date, with_fit: bool = True
    ) -> list[dict[str, Any]]:
        list_path = self.raw / "activities_list.json"
        acts: list[dict[str, Any]]
        if list_path.exists():
            acts = json.loads(list_path.read_text())
            log.info("activities_list.json cached: %d activities", len(acts))
        else:
            acts = self.p.call(
                self.api.get_activities_by_date,
                since.isoformat(),
                until.isoformat(),
                None,
                "asc",
            )
            _write_json(list_path, acts)
            log.info("fetched %d activities in [%s, %s]", len(acts), since, until)

        for i, a in enumerate(acts, 1):
            aid = a["activityId"]
            typ = a.get("activityType", {}).get("typeKey", "")
            d = self.raw / "activities" / str(aid)
            self.fetch(d / "summary.json", self.api.get_activity, aid)
            if with_fit:
                self.fetch_bytes(
                    d / "original.zip",
                    self.api.download_activity,
                    aid,
                    dl_fmt=self.api.ActivityDownloadFormat.ORIGINAL,
                )
            if typ in CYCLING_TYPES:
                self.fetch(d / "splits.json", self.api.get_activity_splits, aid)
                self.fetch(
                    d / "hr_zones.json", self.api.get_activity_hr_in_timezones, aid
                )
                self.fetch(d / "weather.json", self.api.get_activity_weather, aid)
                self.fetch(d / "gear.json", self.api.get_activity_gear, aid)
            if i % 25 == 0:
                log.info("activities %d/%d  %s", i, len(acts), self.stats)
        return acts

    # -- daily single-date endpoints ------------------------------------------
    def export_daily(self, since: dt.date, until: dt.date) -> None:
        days = list(_daterange(since, until))
        for i, day in enumerate(days, 1):
            s = day.isoformat()
            d = self.raw / "daily" / s
            self.fetch(d / "sleep.json", self.api.get_sleep_data, s)
            self.fetch(d / "training_status.json", self.api.get_training_status, s)
            if i % 50 == 0:
                log.info("daily %d/%d  %s", i, len(days), self.stats)

    # -- range endpoints -------------------------------------------------------
    def _gql(self, query: str) -> Any:
        return self.api.query_garmin_graphql({"query": query})

    def export_ranges(self, since: dt.date, until: dt.date) -> None:
        dn = self.api.display_name
        gql_templates = {
            "daily_v2": 'query{{userDailySummaryV2Scalar(startDate:"{s}", endDate:"{e}")}}',
            "hrv": 'query{{heartRateVariabilityScalar(startDate:"{s}", endDate:"{e}")}}',
            "readiness": 'query{{trainingReadinessRangeScalar(startDate:"{s}", endDate:"{e}")}}',
            "vo2max": 'query{{vo2MaxScalar(startDate:"{s}", endDate:"{e}")}}',
            "sleep_summaries": 'query{{sleepSummariesScalar(startDate:"{s}", endDate:"{e}")}}',
            "training_status_weekly": (
                'query{{trainingStatusWeeklyScalar(startDate:"{s}", endDate:"{e}", displayName:"'
                + dn
                + '")}}'
            ),
            "endurance_weekly": 'query{{enduranceScoreScalar(startDate:"{s}", endDate:"{e}", aggregation:"weekly")}}',
            "weight": 'query{{weightScalar(startDate:"{s}", endDate:"{e}")}}',
        }
        for a, b in _chunks(since, until, RANGE_CHUNK_DAYS):
            s, e = a.isoformat(), b.isoformat()
            for name, tpl in gql_templates.items():
                self.fetch(
                    self.raw / "range" / name / f"{s}_{e}.json",
                    self._gql,
                    tpl.format(s=s, e=e),
                )
            # REST range endpoints
            self.fetch(
                self.raw / "range" / "body_battery" / f"{s}_{e}.json",
                self.api.get_body_battery,
                s,
                e,
            )
            self.fetch(
                self.raw / "range" / "daily_steps" / f"{s}_{e}.json",
                self.api.get_daily_steps,
                s,
                e,
            )
            self.fetch(
                self.raw / "range" / "hill_score" / f"{s}_{e}.json",
                self.api.get_hill_score,
                s,
                e,
            )
        # race predictions: <= 1 year per call
        for a, b in _chunks(since, until, 364):
            s, e = a.isoformat(), b.isoformat()
            self.fetch(
                self.raw / "range" / "race_predictions" / f"{s}_{e}.json",
                self.api.get_race_predictions,
                s,
                e,
                "daily",
            )
        self.fetch(
            self.raw / "range" / "body_composition" / f"{since}_{until}.json",
            self.api.get_body_composition,
            since.isoformat(),
            until.isoformat(),
        )
        log.info("ranges done  %s", self.stats)


# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--since", required=True, help="YYYY-MM-DD")
    ap.add_argument(
        "--until",
        default=dt.date.today().isoformat(),
        help="YYYY-MM-DD (default today)",
    )
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    ap.add_argument(
        "--pace", type=float, default=0.6, help="min seconds between API calls"
    )
    ap.add_argument("--skip-fit", action="store_true")
    ap.add_argument("--skip-daily", action="store_true")
    ap.add_argument("--skip-ranges", action="store_true")
    ap.add_argument(
        "--only", choices=["profile", "activities", "daily", "ranges"], default=None
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    since = dt.date.fromisoformat(args.since)
    until = dt.date.fromisoformat(args.until)
    data_dir = Path(args.data_dir)

    api = make_client()
    ex = Exporter(api, data_dir, args.pace)
    log.info("exporting %s..%s into %s", since, until, data_dir)

    steps = [args.only] if args.only else ["profile", "activities", "daily", "ranges"]
    if "profile" in steps:
        ex.export_profile()
    if "activities" in steps:
        ex.export_activities(since, until, with_fit=not args.skip_fit)
    if "daily" in steps and not args.skip_daily:
        ex.export_daily(since, until)
    if "ranges" in steps and not args.skip_ranges:
        ex.export_ranges(since, until)
    log.info("DONE  calls=%d  %s", ex.p.n_calls, ex.stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
