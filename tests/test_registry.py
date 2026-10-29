"""One test per registry invariant (I1-I6) plus lifecycle happy paths."""

import datetime as dt
import math
import sqlite3

import pytest

from garmin_lab.registry import Registry, RegistryError


class Clock:
    def __init__(self, t: dt.datetime) -> None:
        self.t = t

    def __call__(self) -> str:
        return self.t.isoformat(timespec="seconds")

    def advance(self, **kw) -> None:
        self.t += dt.timedelta(**kw)


@pytest.fixture
def reg():
    clock = Clock(dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc))
    r = Registry(":memory:", clock=clock)
    r.register_fit("fit-1", "hash-a", data_cutoff="2026-01-01")
    r._clock_obj = clock  # type: ignore[attr-defined]
    return r


def test_register_sets_prior_and_state(reg):
    hid = reg.register(
        "HR drifts above 28C",
        {"param": "gamma_dT", "op": ">", "value": 0},
        prior_prob=0.6,
    )
    h = reg.get(hid)
    assert h["state"] == "proposed" and abs(h["belief"] - 0.6) < 1e-9
    assert h["events"][0]["actor"] == "hypothesizer"


def test_register_rejects_extreme_prior(reg):
    with pytest.raises(RegistryError):
        reg.register("x", "spec-1", prior_prob=0.99)


def test_I1_belief_only_moves_via_record_evidence(reg):
    hid = reg.register("x", "spec-2", 0.5)
    # there is no public setter; even a direct SQL write is what we're guarding against,
    # so verify the API surface: transition/predict/retire leave belief unchanged
    reg.predict(
        hid,
        "seg:Expresso",
        reg._clock_obj.t + dt.timedelta(days=1),
        {"mu": 400, "sigma": 20},
    )
    assert abs(reg.belief(hid) - 0.5) < 1e-9
    reg.record_evidence(hid, math.log(3), "fit-1", "loo", as_of="2026-01-02")
    assert abs(reg.belief(hid) - 0.75) < 1e-9  # odds 1 -> 3


def test_I2_evidence_requires_known_fit_before_as_of(reg):
    hid = reg.register("x", "spec-3", 0.5)
    with pytest.raises(RegistryError):
        reg.record_evidence(hid, 1.0, "fit-does-not-exist", "loo", as_of="2026-01-02")
    reg.register_fit("fit-future", "h", data_cutoff="2026-03-01")
    with pytest.raises(RegistryError):
        reg.record_evidence(
            hid, 1.0, "fit-future", "loo", as_of="2026-02-01"
        )  # fit saw later data


def test_I2_only_statistician_records_evidence(reg):
    hid = reg.register("x", "spec-4", 0.5)
    with pytest.raises(RegistryError):
        reg.record_evidence(
            hid, 2.0, "fit-1", "loo", as_of="2026-01-02", actor="hypothesizer"
        )


def test_I3_prediction_must_precede_target(reg):
    hid = reg.register("x", "spec-5", 0.5)
    past = reg._clock_obj.t - dt.timedelta(hours=1)
    with pytest.raises(RegistryError):
        reg.predict(hid, "seg:Expresso", past, {"mu": 400, "sigma": 20})
    pid = reg.predict(
        hid,
        "seg:Expresso",
        reg._clock_obj.t + dt.timedelta(hours=1),
        {"mu": 400, "sigma": 20},
    )
    assert reg.get(hid)["state"] == "under_test"
    reg.score_prediction(pid, outcome=410.0, crps=8.0)
    assert reg.ledger()[0]["crps"] == 8.0


def test_I4_transitions_are_gated(reg):
    hid = reg.register("x", "spec-6", 0.5)
    with pytest.raises(RegistryError):
        reg.transition(hid, "supported")
    with pytest.raises(RegistryError):
        reg.transition(hid, "refuted")
    with pytest.raises(RegistryError):
        reg.transition(hid, "undetermined")  # budget not exhausted
    reg.record_evidence(hid, 2.0, "fit-1", "loo", as_of="2026-01-02")  # p ~ 0.88
    reg.transition(hid, "supported")
    assert reg.get(hid)["state"] == "supported"


def test_I4_undetermined_requires_exhausted_budget_and_flat_evidence(reg):
    hid = reg.register("tau1<42", "spec-7", 0.5, test_budget=3)
    for bf in (0.3, -0.2, 0.1):
        reg.record_evidence(hid, bf, "fit-1", "bf", as_of="2026-01-02")
    reg.transition(hid, "undetermined")
    assert reg.get(hid)["state"] == "undetermined"
    hid2 = reg.register("y", "spec-8", 0.5, test_budget=1)
    reg.record_evidence(hid2, 1.5, "fit-1", "bf", as_of="2026-01-02")
    with pytest.raises(RegistryError):
        reg.transition(hid2, "undetermined")  # evidence is not flat


def test_I5_refine_and_merge_need_tested_parents(reg):
    a = reg.register("a", "spec-9", 0.5)
    b = reg.register("b", "spec-10", 0.5)
    with pytest.raises(RegistryError):
        reg.refine(a, "a'", "spec2")
    with pytest.raises(RegistryError):
        reg.merge([a, b], "ab", "spec3")
    reg.record_evidence(a, 0.5, "fit-1", "bf", as_of="2026-01-02")
    reg.record_evidence(b, -0.5, "fit-1", "bf", as_of="2026-01-02")
    c = reg.refine(a, "a'", "spec2")
    assert reg.get(c)["parent_ids"] == [a]
    m = reg.merge([a, b], "ab", "spec3")
    assert sorted(reg.get(m)["parent_ids"]) == [a, b]
    assert reg.get(a)["state"] == "retired"  # unresolved parents retire on merge


def test_I6_every_call_is_logged_with_actor(reg):
    hid = reg.register("x", "spec-11", 0.5, actor="hypothesizer")
    reg.record_evidence(hid, 0.2, "fit-1", "bf", as_of="2026-01-02")
    kinds = [(e["kind"], e["actor"]) for e in reg.get(hid)["events"]]
    assert ("register", "hypothesizer") in kinds and (
        "record_evidence",
        "statistician",
    ) in kinds


def test_evidence_is_capped(reg):
    hid = reg.register("x", "spec-12", 0.5)
    reg.record_evidence(hid, 1e6, "fit-1", "bf", as_of="2026-01-02")
    assert reg.belief(hid) < 1.0 and reg.get(hid)["evidence"][0]["log_bf"] == 10.0


def test_persists_to_disk(tmp_path):
    p = tmp_path / "reg.sqlite"
    r = Registry(p)
    r.register_fit("f", "h", "2026-01-01")
    hid = r.register("x", "spec-13", 0.5)
    del r
    assert (
        sqlite3.connect(p).execute("SELECT count(*) FROM hypotheses").fetchone()[0] == 1
    )
    assert Registry(p).belief(hid) == 0.5


# ------------------------------------------------------------------ I7 & I8
SPEC_A = {
    "base": "hr_model",
    "operation": "directional_claim",
    "args": {"param": "gamma_d", "op": ">", "value": 0},
}
SPEC_B = {
    "base": "hr_model",
    "operation": "add_covariate",
    "args": {"covariate": "sleep_h"},
}


def test_i7_refuses_duplicate_live_spec(reg):
    reg.register("first", SPEC_A, 0.5)
    with pytest.raises(RegistryError, match="I7"):
        reg.register("second, same spec", SPEC_A, 0.5)


def test_i7_allows_same_spec_after_retire(reg):
    hid = reg.register("first", SPEC_A, 0.5)
    reg.retire(hid, "superseded")
    hid2 = reg.register("second", SPEC_A, 0.5)  # must not raise
    assert hid2 != hid


def test_i7_allows_different_spec(reg):
    reg.register("first", SPEC_A, 0.5)
    hid2 = reg.register("second", SPEC_B, 0.5)
    assert hid2 > 0


def test_i8_auto_promotes_to_supported(reg):
    hid = reg.register("x", SPEC_A, 0.5)
    reg.record_evidence(hid, 10.0, "fit-1", "bf", as_of="2026-01-02")
    assert reg.get(hid)["state"] == "supported"


def test_i8_auto_demotes_from_supported(reg):
    hid = reg.register("x", SPEC_A, 0.5)
    # moderate evidence: belief reaches ~0.88 (supported)
    reg.record_evidence(hid, 2.1, "fit-1", "bf", as_of="2026-01-02")
    assert reg.get(hid)["state"] == "supported"
    # counter-evidence: belief falls back below 0.8 -> reopened
    reg.record_evidence(hid, -1.8, "fit-1", "bf2", as_of="2026-01-03")
    h = reg.get(hid)
    assert h["state"] == "under_test"
    assert h["belief"] < 0.8


def test_i8_auto_refutes(reg):
    hid = reg.register("x", SPEC_A, 0.5)
    reg.record_evidence(hid, -10.0, "fit-1", "bf", as_of="2026-01-02")
    assert reg.get(hid)["state"] == "refuted"


def test_i8_auto_reopens_refuted(reg):
    hid = reg.register("x", SPEC_A, 0.5)
    # moderate evidence against: belief ~0.12 (refuted)
    reg.record_evidence(hid, -2.1, "fit-1", "bf", as_of="2026-01-02")
    assert reg.get(hid)["state"] == "refuted"
    # counter-evidence: belief rises above 0.2 -> reopened
    reg.record_evidence(hid, 1.8, "fit-1", "bf2", as_of="2026-01-03")
    h = reg.get(hid)
    assert h["state"] == "under_test"
    assert h["belief"] > 0.2


def test_i8_retired_never_touched(reg):
    hid = reg.register("x", SPEC_A, 0.5)
    reg.retire(hid, "done")
    # record_evidence on a retired hypothesis should still raise (I2 path)
    with pytest.raises(RegistryError):
        reg.record_evidence(hid, 1.0, "fit-1", "bf", as_of="2026-01-02")


def test_I9_withdraw_prediction_is_append_only(reg):
    hid = reg.register("x", "spec-9", 0.5)
    pid = reg.predict(
        hid,
        "moving_s@old-instrument",
        reg._clock_obj.t + dt.timedelta(days=3),
        {"mu": 2900, "sigma": 240},
    )
    reg.withdraw_prediction(pid, "instrument changed before the ride")
    row = reg.ledger()[0]
    assert row["withdrawn_at"] is not None
    assert "instrument changed" in row["withdrawn_reason"]
    assert row["scored_at"] is None  # still unscored, still present
    # cannot withdraw twice, cannot withdraw a scored prediction, cannot withdraw a ghost
    with pytest.raises(RegistryError):
        reg.withdraw_prediction(pid, "again")
    pid2 = reg.predict(
        hid, "seg:B", reg._clock_obj.t + dt.timedelta(days=3), {"mu": 1, "sigma": 1}
    )
    reg.score_prediction(pid2, outcome=1.0)
    with pytest.raises(RegistryError):
        reg.withdraw_prediction(pid2, "too late")
    with pytest.raises(RegistryError):
        reg.withdraw_prediction(999, "ghost")
    kinds = (
        [e["kind"] for e in reg.events()]
        if hasattr(reg, "events")
        else [r["kind"] for r in reg.con.execute("SELECT kind FROM events")]
    )
    assert "withdraw_prediction" in kinds


def test_migration_adds_withdrawn_columns(tmp_path):
    """A registry file created before I9 gains the new columns on open."""
    path = tmp_path / "old.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE hypotheses (id INTEGER PRIMARY KEY);
        CREATE TABLE predictions (
          id INTEGER PRIMARY KEY, hypothesis_id INTEGER NOT NULL, target TEXT NOT NULL,
          target_time TEXT NOT NULL, distribution TEXT NOT NULL, registered_at TEXT NOT NULL,
          scored_at TEXT, outcome REAL, crps REAL, logscore REAL, brier REAL);
        """
    )
    con.commit()
    con.close()
    reg = Registry(path)
    cols = {r["name"] for r in reg.con.execute("PRAGMA table_info(predictions)")}
    assert {"withdrawn_at", "withdrawn_reason"} <= cols
