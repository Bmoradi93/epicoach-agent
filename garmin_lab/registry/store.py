"""SQLite-backed hypothesis registry.

Invariants enforced by the store (each has a unit test in ``tests/test_registry.py``):

 I1  ``logit_belief`` changes only through :meth:`Registry.record_evidence`.
 I2  ``record_evidence`` requires a ``fit_id`` present in ``fits`` whose
     ``data_cutoff`` is not later than the evidence's ``as_of`` date.
 I3  ``predict`` refuses a prediction whose registration time is after the
     target ride's start (the ledger is pre-registration or nothing).
 I4  State transitions: ``supported`` needs belief >= 0.8, ``refuted`` needs
     belief <= 0.2, ``undetermined`` needs the test budget exhausted with
     |cumulative log BF| < log(3).
 I5  ``refine`` / ``merge`` require a parent that has a verdict or evidence.
 I6  Every call is logged in ``events`` with its ``actor`` (LLM vs tool).
I10 An observation (``outcome_ref``) scores at most one prediction.
 I7  ``register`` refuses a spec whose canonical key matches any live (non-retired)
     hypothesis; use ``refine`` to build on an existing one instead.
 I8  Verdicts (``supported``, ``refuted``, ``undetermined``) are not permanent:
     ``record_evidence`` re-evaluates the state after every update so beliefs
     can fall back below the threshold if new data disagrees.

Beliefs are stored as log-odds; ``belief()`` returns a probability.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Iterable

STATES = ("proposed", "under_test", "supported", "refuted", "undetermined", "retired")
LOG3 = math.log(3.0)

SCHEMA = """
CREATE TABLE IF NOT EXISTS hypotheses (
  id INTEGER PRIMARY KEY,
  text TEXT NOT NULL,
  formal_spec TEXT NOT NULL,
  state TEXT NOT NULL,
  logit_belief REAL NOT NULL,
  test_budget INTEGER NOT NULL DEFAULT 6,
  n_tests INTEGER NOT NULL DEFAULT 0,
  cum_log_bf REAL NOT NULL DEFAULT 0.0,
  created_by TEXT NOT NULL,
  created_at TEXT NOT NULL,
  parent_ids TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS fits (
  fit_id TEXT PRIMARY KEY,
  model_spec_hash TEXT NOT NULL,
  data_cutoff TEXT NOT NULL,
  posterior_path TEXT,
  loo REAL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY,
  hypothesis_id INTEGER NOT NULL REFERENCES hypotheses(id),
  target TEXT NOT NULL,
  target_time TEXT NOT NULL,
  distribution TEXT NOT NULL,
  registered_at TEXT NOT NULL,
  scored_at TEXT,
  outcome REAL,
  crps REAL,
  logscore REAL,
  brier REAL,
  withdrawn_at TEXT,
  withdrawn_reason TEXT
);
CREATE TABLE IF NOT EXISTS evidence (
  id INTEGER PRIMARY KEY,
  hypothesis_id INTEGER NOT NULL REFERENCES hypotheses(id),
  fit_id TEXT NOT NULL REFERENCES fits(fit_id),
  log_bf REAL NOT NULL,
  method TEXT NOT NULL,
  as_of TEXT NOT NULL,
  recorded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY,
  hypothesis_id INTEGER,
  kind TEXT NOT NULL,
  payload TEXT NOT NULL,
  actor TEXT NOT NULL,
  at TEXT NOT NULL
);
"""


class RegistryError(Exception):
    """Raised when a call would violate a registry invariant."""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _iso(x: str | dt.datetime | dt.date) -> str:
    if isinstance(x, dt.datetime):
        if x.tzinfo is None:
            x = x.replace(tzinfo=dt.timezone.utc)
        return x.isoformat(timespec="seconds")
    if isinstance(x, dt.date):
        return x.isoformat()
    return str(x)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Registry:
    def __init__(self, path: str | Path = ":memory:", clock=None) -> None:
        self.con = sqlite3.connect(str(path))
        self.con.row_factory = sqlite3.Row
        self.con.executescript(SCHEMA)
        self._migrate()
        self._clock = clock or _now

    def _migrate(self) -> None:
        """Add columns introduced after a registry file was created (additive only)."""
        have = {r["name"] for r in self.con.execute("PRAGMA table_info(predictions)")}
        for col in ("withdrawn_at", "withdrawn_reason", "outcome_ref"):
            if col not in have:
                self.con.execute(f"ALTER TABLE predictions ADD COLUMN {col} TEXT")
        self.con.commit()

    # ------------------------------------------------------------------ util
    def _log(
        self, kind: str, actor: str, hid: int | None, payload: dict[str, Any]
    ) -> None:
        self.con.execute(
            "INSERT INTO events(hypothesis_id, kind, payload, actor, at) VALUES (?,?,?,?,?)",
            (hid, kind, json.dumps(payload, default=str), actor, self._clock()),
        )
        self.con.commit()

    def _get(self, hid: int) -> sqlite3.Row:
        row = self.con.execute("SELECT * FROM hypotheses WHERE id=?", (hid,)).fetchone()
        if row is None:
            raise RegistryError(f"no hypothesis {hid}")
        return row

    def belief(self, hid: int) -> float:
        return _sigmoid(self._get(hid)["logit_belief"])

    # ------------------------------------------------------------- fit store
    def register_fit(
        self,
        fit_id: str,
        model_spec_hash: str,
        data_cutoff: str | dt.date,
        posterior_path: str | None = None,
        loo: float | None = None,
        actor: str = "statistician",
    ) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO fits(fit_id, model_spec_hash, data_cutoff, posterior_path, loo, created_at) VALUES (?,?,?,?,?,?)",
            (
                fit_id,
                model_spec_hash,
                _iso(data_cutoff),
                posterior_path,
                loo,
                self._clock(),
            ),
        )
        self._log(
            "register_fit",
            actor,
            None,
            {"fit_id": fit_id, "data_cutoff": _iso(data_cutoff)},
        )

    # --------------------------------------------------------------- 7 tools
    def _spec_key(self, spec: dict[str, Any] | str) -> str:
        """Canonical key for a formal spec: sorted JSON of base+operation+args."""
        if isinstance(spec, str):
            try:
                spec = json.loads(spec)
            except json.JSONDecodeError:
                return spec
        return json.dumps(
            {k: spec[k] for k in ("base", "operation", "args") if k in spec},
            sort_keys=True,
        )

    def _live_spec_keys(self) -> set[str]:
        """Canonical keys of all non-retired hypotheses."""
        rows = self.con.execute(
            "SELECT formal_spec FROM hypotheses WHERE state != 'retired'"
        ).fetchall()
        return {self._spec_key(r["formal_spec"]) for r in rows}

    def register(
        self,
        text: str,
        formal_spec: dict[str, Any] | str,
        prior_prob: float = 0.5,
        actor: str = "hypothesizer",
        parent_ids: Iterable[int] = (),
        test_budget: int = 6,
    ) -> int:
        if not 0.05 <= prior_prob <= 0.95:
            raise RegistryError(
                "prior must be in [0.05, 0.95]; the LLM may not pre-decide a hypothesis"
            )
        spec = (
            formal_spec
            if isinstance(formal_spec, str)
            else json.dumps(formal_spec, sort_keys=True)
        )
        # I7: refuse a spec whose canonical key matches any live hypothesis
        key = self._spec_key(spec)
        if key in self._live_spec_keys():
            raise RegistryError(
                f"I7: a live hypothesis with spec key {key!r} already exists; "
                "use refine() to build on it instead of registering a duplicate"
            )
        cur = self.con.execute(
            "INSERT INTO hypotheses(text, formal_spec, state, logit_belief, test_budget, created_by, created_at, parent_ids) VALUES (?,?,?,?,?,?,?,?)",
            (
                text,
                spec,
                "proposed",
                _logit(prior_prob),
                test_budget,
                actor,
                self._clock(),
                json.dumps(list(parent_ids)),
            ),
        )
        hid = int(cur.lastrowid)
        self._log("register", actor, hid, {"text": text, "prior": prior_prob})
        return hid

    def predict(
        self,
        hid: int,
        target: str,
        target_time: str | dt.datetime,
        distribution: dict[str, Any],
        actor: str = "statistician",
    ) -> int:
        h = self._get(hid)
        if h["state"] in ("retired", "refuted", "supported", "undetermined"):
            raise RegistryError(f"hypothesis {hid} is closed ({h['state']})")
        now = self._clock()
        tt = _iso(target_time)
        if now >= tt:
            raise RegistryError(
                f"prediction registered at {now} is not before target time {tt}"
            )
        cur = self.con.execute(
            "INSERT INTO predictions(hypothesis_id, target, target_time, distribution, registered_at) VALUES (?,?,?,?,?)",
            (hid, target, tt, json.dumps(distribution), now),
        )
        if h["state"] == "proposed":
            self.con.execute(
                "UPDATE hypotheses SET state='under_test' WHERE id=?", (hid,)
            )
        self._log(
            "predict",
            actor,
            hid,
            {"target": target, "target_time": tt, "distribution": distribution},
        )
        return int(cur.lastrowid)

    def score_prediction(
        self,
        pid: int,
        outcome: float,
        crps: float | None = None,
        logscore: float | None = None,
        brier: float | None = None,
        actor: str = "statistician",
        outcome_ref: str | None = None,
    ) -> None:
        """I10: an observation scores at most one prediction. ``outcome_ref`` identifies the
        observation (e.g. ``"<activity_id>@<segment_id>"``); a second prediction may not claim it."""
        if outcome_ref is not None:
            prior = self.con.execute(
                "SELECT id FROM predictions WHERE outcome_ref=? AND scored_at IS NOT NULL",
                (outcome_ref,),
            ).fetchone()
            if prior is not None:
                raise RegistryError(
                    f"I10: observation {outcome_ref} already scored prediction {prior['id']}"
                )
        self.con.execute(
            "UPDATE predictions SET outcome=?, crps=?, logscore=?, brier=?, scored_at=?, outcome_ref=? WHERE id=?",
            (outcome, crps, logscore, brier, self._clock(), outcome_ref, pid),
        )
        self._log(
            "score_prediction",
            actor,
            None,
            {"prediction_id": pid, "outcome": outcome, "crps": crps, "brier": brier, "outcome_ref": outcome_ref},
        )

    def withdraw_prediction(
        self, pid: int, reason: str, actor: str = "athlete"
    ) -> None:
        """I9: a pre-registered prediction is never deleted. If its target can no longer be
        observed (e.g. the instrument changed before the ride), it is marked withdrawn with a
        reason and stays in the ledger as an audit record. Scored predictions cannot be withdrawn.
        """
        row = self.con.execute(
            "SELECT scored_at, withdrawn_at FROM predictions WHERE id=?", (pid,)
        ).fetchone()
        if row is None:
            raise RegistryError(f"prediction {pid} does not exist")
        if row["scored_at"] is not None:
            raise RegistryError(f"prediction {pid} is already scored")
        if row["withdrawn_at"] is not None:
            raise RegistryError(f"prediction {pid} is already withdrawn")
        self.con.execute(
            "UPDATE predictions SET withdrawn_at=?, withdrawn_reason=? WHERE id=?",
            (self._clock(), reason, pid),
        )
        self._log(
            "withdraw_prediction", actor, None, {"prediction_id": pid, "reason": reason}
        )

    def record_evidence(
        self,
        hid: int,
        log_bf: float,
        fit_id: str,
        method: str,
        as_of: str | dt.date,
        actor: str = "statistician",
    ) -> float:
        """The ONLY path that moves a belief. Returns the new belief probability."""
        if actor != "statistician":
            raise RegistryError("only the statistician tool may record evidence")
        fit = self.con.execute(
            "SELECT * FROM fits WHERE fit_id=?", (fit_id,)
        ).fetchone()
        if fit is None:
            raise RegistryError(f"unknown fit_id {fit_id}")
        if fit["data_cutoff"] > _iso(as_of):
            raise RegistryError(
                f"fit {fit_id} used data up to {fit['data_cutoff']}, after evidence date {as_of}"
            )
        h = self._get(hid)
        if h["state"] in ("retired",):
            raise RegistryError(f"hypothesis {hid} is retired")
        if not math.isfinite(log_bf):
            raise RegistryError("log_bf must be finite")
        log_bf = max(
            min(log_bf, 10.0), -10.0
        )  # cap a single piece of evidence at BF 22000
        self.con.execute(
            "INSERT INTO evidence(hypothesis_id, fit_id, log_bf, method, as_of, recorded_at) VALUES (?,?,?,?,?,?)",
            (hid, fit_id, log_bf, method, _iso(as_of), self._clock()),
        )
        self.con.execute(
            "UPDATE hypotheses SET logit_belief = logit_belief + ?, cum_log_bf = cum_log_bf + ?, n_tests = n_tests + 1 WHERE id=?",
            (log_bf, log_bf, hid),
        )
        self._log(
            "record_evidence",
            actor,
            hid,
            {"log_bf": log_bf, "fit_id": fit_id, "method": method},
        )
        belief = self.belief(hid)
        self._auto_transition(hid)
        return belief

    def _auto_transition(self, hid: int) -> None:
        """I8: re-evaluate state after every evidence update so verdicts can be
        reversed if new data disagrees. Retired hypotheses are never touched."""
        h = self._get(hid)
        if h["state"] == "retired":
            return
        p = _sigmoid(h["logit_belief"])
        n, budget = h["n_tests"], h["test_budget"]
        cur = h["state"]
        # Verdict thresholds take priority; then check if a prior verdict should
        # be reopened because belief has drifted back into the ambiguous zone.
        if p >= 0.8:
            new = "supported"
        elif p <= 0.2:
            new = "refuted"
        elif cur in ("supported", "refuted"):
            # belief moved back into the ambiguous zone — reopen
            new = "under_test" if n > 0 else "proposed"
        elif n >= budget and abs(h["cum_log_bf"]) < LOG3:
            new = "undetermined"
        elif cur == "undetermined":
            # budget still exhausted but belief moved — reopen
            new = "under_test"
        else:
            return  # no change needed
        if new != cur:
            self.con.execute("UPDATE hypotheses SET state=? WHERE id=?", (new, hid))
            self._log(
                "auto_transition",
                "statistician",
                hid,
                {"from": cur, "to": new, "belief": round(p, 3)},
            )

    def transition(self, hid: int, new_state: str, actor: str = "statistician") -> None:
        if new_state not in STATES:
            raise RegistryError(f"unknown state {new_state}")
        h = self._get(hid)
        p = _sigmoid(h["logit_belief"])
        if new_state == "supported" and p < 0.8:
            raise RegistryError(f"supported needs belief >= 0.8 (have {p:.2f})")
        if new_state == "refuted" and p > 0.2:
            raise RegistryError(f"refuted needs belief <= 0.2 (have {p:.2f})")
        if new_state == "undetermined":
            if h["n_tests"] < h["test_budget"]:
                raise RegistryError(
                    f"undetermined needs budget exhausted ({h['n_tests']}/{h['test_budget']} tests)"
                )
            if abs(h["cum_log_bf"]) >= LOG3:
                raise RegistryError("undetermined needs |cumulative log BF| < log 3")
        if new_state == "under_test" and h["state"] != "proposed":
            raise RegistryError("under_test is entered via predict()")
        self.con.execute("UPDATE hypotheses SET state=? WHERE id=?", (new_state, hid))
        self._log("transition", actor, hid, {"from": h["state"], "to": new_state})

    def _can_spawn(self, hid: int) -> bool:
        h = self._get(hid)
        has_ev = (
            self.con.execute(
                "SELECT count(*) FROM evidence WHERE hypothesis_id=?", (hid,)
            ).fetchone()[0]
            > 0
        )
        return h["state"] in ("supported", "refuted", "undetermined") or has_ev

    def refine(
        self,
        parent_id: int,
        text: str,
        formal_spec: dict[str, Any] | str,
        prior_prob: float = 0.5,
        actor: str = "hypothesizer",
    ) -> int:
        if not self._can_spawn(parent_id):
            raise RegistryError(
                f"parent {parent_id} has neither verdict nor evidence; cannot refine"
            )
        hid = self.register(
            text, formal_spec, prior_prob, actor=actor, parent_ids=[parent_id]
        )
        self._log("refine", actor, hid, {"parent": parent_id})
        return hid

    def merge(
        self,
        parent_ids: Iterable[int],
        text: str,
        formal_spec: dict[str, Any] | str,
        prior_prob: float = 0.5,
        actor: str = "hypothesizer",
    ) -> int:
        parents = list(parent_ids)
        if len(parents) < 2:
            raise RegistryError("merge needs >= 2 parents")
        for p in parents:
            if not self._can_spawn(p):
                raise RegistryError(
                    f"parent {p} has neither verdict nor evidence; cannot merge"
                )
        hid = self.register(
            text, formal_spec, prior_prob, actor=actor, parent_ids=parents
        )
        for p in parents:
            h = self._get(p)
            if h["state"] in ("proposed", "under_test"):
                self.con.execute(
                    "UPDATE hypotheses SET state='retired' WHERE id=?", (p,)
                )
        self._log("merge", actor, hid, {"parents": parents})
        return hid

    def retire(self, hid: int, reason: str, actor: str = "hypothesizer") -> None:
        self._get(hid)
        self.con.execute("UPDATE hypotheses SET state='retired' WHERE id=?", (hid,))
        self._log("retire", actor, hid, {"reason": reason})

    def list(self, state: str | None = None) -> list[dict[str, Any]]:
        q = (
            "SELECT * FROM hypotheses"
            + (" WHERE state=?" if state else "")
            + " ORDER BY id"
        )
        rows = self.con.execute(q, (state,) if state else ()).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["belief"] = _sigmoid(d.pop("logit_belief"))
            d["parent_ids"] = json.loads(d["parent_ids"])
            d["formal_spec"] = (
                json.loads(d["formal_spec"])
                if d["formal_spec"].startswith("{")
                else d["formal_spec"]
            )
            out.append(d)
        return out

    def get(self, hid: int) -> dict[str, Any]:
        d = [x for x in self.list() if x["id"] == hid][0]
        d["evidence"] = [
            dict(r)
            for r in self.con.execute(
                "SELECT * FROM evidence WHERE hypothesis_id=? ORDER BY id", (hid,)
            )
        ]
        d["predictions"] = [
            dict(r)
            for r in self.con.execute(
                "SELECT * FROM predictions WHERE hypothesis_id=? ORDER BY id", (hid,)
            )
        ]
        d["events"] = [
            dict(r)
            for r in self.con.execute(
                "SELECT * FROM events WHERE hypothesis_id=? ORDER BY id", (hid,)
            )
        ]
        return d

    def ledger(self) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self.con.execute("SELECT * FROM predictions ORDER BY target_time")
        ]
