"""PHIA-style ReAct forecasting baseline for the replay benchmark.

A single LLM agent with a SQL tool over the traversal-feature table, restricted
to rows with ``day_index <= origin`` (the cutoff is enforced in the tool, not
the prompt). It is asked to return a Gaussian predictive distribution for each
future traversal. Responses are cached on disk per (target, origin) so a rerun
makes no API calls.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re

import duckdb
import numpy as np
import polars as pl
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types

from garmin_lab.ingest.client import REPO_ROOT
from garmin_lab.replay.forecasters import Forecaster, _y

log = logging.getLogger("garmin_lab.react")
load_dotenv(REPO_ROOT / ".env")
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
CACHE = REPO_ROOT / "data" / "replay" / "react_cache"
MODEL = "gemini-3.8-flash"

_CUTOFF: dict[str, int] = {"day": 0}
_HIST: dict[str, pl.DataFrame] = {}

COLS = [
    "segment_id",
    "kind",
    "day_index",
    "week",
    "moving_s",
    "hr_mean",
    "vp_mean_w",
    "temp_c20",
    "elapsed_h",
    "sleep_h",
    "hrv_last_night",
    "readiness_score",
    "wet_now",
    "hour_local",
    "dist_m",
]


def query_history(sql: str) -> dict:
    """Run a read-only DuckDB SQL query against table `hist` (traversal features up to the forecast
    origin). Columns: segment_id, kind, day_index, week, moving_s, hr_mean, vp_mean_w, temp_c20,
    elapsed_h, sleep_h, hrv_last_night, readiness_score, wet_now, hour_local, dist_m. Max 60 rows returned.
    """
    if re.search(
        r"\b(insert|update|delete|drop|create|attach|copy|pragma)\b", sql, re.I
    ):
        return {"error": "read-only"}
    con = duckdb.connect()
    con.register("hist", _HIST["df"].to_arrow())
    try:
        df = con.execute(sql).pl().head(60)
        return {"rows": json.loads(df.write_json()), "n": df.height}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)[:300]}
    finally:
        con.close()


INSTR = """You are a sports-data analyst agent. You have one tool, query_history(sql), over the table `hist`
holding every repeated-segment traversal of ONE mountain biker up to today (day_index = {origin}).
Your task: give a Gaussian predictive distribution (mu, sigma) for each FUTURE traversal listed below.

Target: {target_desc}

Future traversals (predict each; conditions are known):
{future}

Work like a data scientist: query the segment's history (recent values, trend, spread), consider the
conditions given, and reason explicitly. Use at most 6 queries. Then answer with ONLY a JSON list, one
object per future traversal in the given order: [{{"idx": 0, "mu": ..., "sigma": ...}}, ...].
sigma must be a positive number reflecting your genuine uncertainty."""

TARGET_DESC = {
    "log_moving_s": "natural log of moving time in seconds on the segment (e.g. ln(300 s) = 5.70). A typical segment-level spread is 0.05-0.15.",
    "hr_mean": "mean heart rate (bpm) on the climb, given the realised virtual power (W), temperature (deg C above 20) and elapsed hours into the ride.",
}


def _future_text(future: pl.DataFrame, target: str) -> str:
    lines = []
    for i, r in enumerate(future.iter_rows(named=True)):
        cond = f"segment={r['segment_id']}, day_index={r['day_index']}"
        if target == "hr_mean":
            cond += f", vp_mean_w={r['vp_mean_w']:.0f}, temp_c20={r['temp_c20']:.1f}, elapsed_h={r['elapsed_h']:.2f}"
        else:
            cond += f", dist_m={r['dist_m']:.0f}, hr_mean={r['hr_mean']:.0f}, temp_c20={r['temp_c20']:.1f}, elapsed_h={r['elapsed_h']:.2f}"
        lines.append(f"  idx {i}: {cond}")
    return "\n".join(lines)


async def _ask(prompt: str) -> str:
    agent = LlmAgent(
        name="ReActForecaster",
        model=MODEL,
        instruction="Follow the user's task exactly.",
        tools=[FunctionTool(query_history)],
        generate_content_config=types.GenerateContentConfig(temperature=0.0),
    )
    runner = InMemoryRunner(agent=agent, app_name="react")
    session = await runner.session_service.create_session(app_name="react", user_id="u")
    text = ""
    async for ev in runner.run_async(
        user_id="u",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
    ):
        if ev.content and ev.content.parts:
            for p in ev.content.parts:
                if p.text:
                    text = p.text
    return text


def _parse(text: str, n: int) -> list[tuple[float, float]] | None:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
        out = {
            int(a["idx"]): (float(a["mu"]), max(float(a["sigma"]), 1e-3)) for a in arr
        }
        return [out[i] for i in range(n)] if all(i in out for i in range(n)) else None
    except Exception:  # noqa: BLE001
        return None


def react(target: str, floor: float) -> Forecaster:
    CACHE.mkdir(parents=True, exist_ok=True)

    def fn(hist: pl.DataFrame, future: pl.DataFrame, origin_day: int):
        _HIST["df"] = hist.select([c for c in COLS if c in hist.columns])
        _CUTOFF["day"] = origin_day
        key = hashlib.sha1(
            f"{target}|{origin_day}|{future['segment_id'].to_list()}|{future['day_index'].to_list()}".encode()
        ).hexdigest()[:16]
        path = CACHE / f"{target}_{origin_day}_{key}.json"
        if path.exists():
            text = json.loads(path.read_text())["text"]
        else:
            prompt = INSTR.format(
                origin=origin_day,
                target_desc=TARGET_DESC[target],
                future=_future_text(future, target),
            )
            text = asyncio.run(_ask(prompt))
            path.write_text(json.dumps({"prompt": prompt, "text": text}))
        parsed = _parse(text, future.height)
        if parsed is None:
            log.warning(
                "ReAct @%d: unparseable answer; falling back to history mean",
                origin_day,
            )
            y = _y(hist, target)
            return np.full(future.height, y.mean()), np.full(
                future.height, max(y.std(), floor) * 1.5
            )
        mu = np.array([p[0] for p in parsed])
        sd = np.maximum(np.array([p[1] for p in parsed]), floor)
        return mu, sd

    return Forecaster("react_agent", target, fn, family="agent")
