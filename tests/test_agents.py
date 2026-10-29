"""Tests for the deterministic agent layer: template validation, safety gate, EIG scorer."""

import numpy as np
import pytest

from garmin_lab.agents.safety import AthleteState, acwr_after, check_week
from garmin_lab.agents.templates import TemplateError, validate_spec


def test_validate_spec_accepts_canonical_and_aliased():
    a = validate_spec(
        {
            "base": "hr_model",
            "operation": "directional_claim",
            "args": {"param": "gamma_d", "op": ">", "value": 0},
        }
    )
    b = validate_spec({"model": "hr_model", "op": ">", "param": "gamma_d", "value": 0})
    assert a.key() == b.key()


def test_validate_spec_rejects_unknowns():
    with pytest.raises(TemplateError):
        validate_spec(
            {
                "base": "hr_model",
                "operation": "add_covariate",
                "args": {"covariate": "moon_phase"},
            }
        )
    with pytest.raises(TemplateError):
        validate_spec(
            {
                "base": "nope",
                "operation": "add_covariate",
                "args": {"covariate": "sleep_h"},
            }
        )
    with pytest.raises(TemplateError):
        validate_spec(
            {
                "base": "hr_model",
                "operation": "directional_claim",
                "args": {"param": "beta_sleep_h", "op": ">", "value": 0},
            }
        )
    with pytest.raises(TemplateError):  # random walk on tau is unidentifiable by rule
        validate_spec(
            {"base": "ffm", "operation": "add_random_walk", "args": {"param": "tau1"}}
        )


def _state(acute=70.0, chronic=70.0, **kw):
    hist = np.concatenate([np.full(21, chronic), np.full(7, acute)])
    return AthleteState(load_history=hist, hr_max=200.0, readiness=80.0, **kw)


def test_acwr_after_steady_state_is_one():
    st = _state()
    ac = acwr_after(st.load_history, np.full(7, 70.0))
    assert np.allclose(ac, 1.0)


def test_safety_vetoes_spike_and_accepts_steady():
    st = _state()
    assert check_week(st, [70] * 6 + [0], 160.0) == []
    v = check_week(st, [200, 200, 200, 200, 200, 200, 0], 160.0)
    assert any("ACWR" in x for x in v) and any("weekly load" in x for x in v)


def test_safety_allows_single_overload_but_not_consecutive():
    st = _state()
    target = st.load_history[-28:].sum() / 4
    over = [0.0] + [1.30 * target / 5] * 5 + [0.0]  # 130% week, 2 rest days
    assert check_week(st, over, 160.0) == []
    st2 = _state()
    st2.load_history = np.concatenate([st2.load_history[:-7], np.asarray(over)])
    v = check_week(st2, over, 160.0)
    assert any("consecutive overloads" in x for x in v)


def test_safety_requires_rest_day_and_caps_hr():
    st = _state()
    assert any("rest day" in x for x in check_week(st, [70] * 7, 160.0))
    assert any("heart-rate target" in x for x in check_week(st, [70] * 6 + [0], 190.0))


def test_safety_readiness_and_illness_gates():
    st = _state(illness_reported=True)
    assert any("illness" in x for x in check_week(st, [70] * 6 + [0], 160.0))
    st = _state()
    st.readiness = 20.0
    assert any("readiness" in x for x in check_week(st, [70] * 6 + [0], 160.0))
    assert not any(
        "readiness" in x for x in check_week(st, [0, 70, 70, 70, 70, 70, 0], 160.0)
    )


def test_safety_allows_explicit_deload():
    st = _state()
    assert (
        check_week(st, [35, 0, 35, 0, 35, 0, 0], 160.0) == []
    )  # 105 = 21% of 490: deload, ACWR low but not all week


# ------------------------------------------------------------------ compile
def _frame(n=40, seed=0):
    import polars as pl

    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "hour_local": rng.uniform(6, 20, n),
            "sleep_h": np.where(rng.uniform(size=n) < 0.2, np.nan, rng.normal(7, 1, n)),
            "temp_c20": rng.normal(0, 5, n),
            "elapsed_h": rng.uniform(0, 3, n),
            "wet_now": rng.integers(0, 2, n).astype(float),
            "day_index": np.arange(n) * 7,
        }
    )


def test_compile_structural_terms_standardised_and_reusable():
    from garmin_lab.agents.compile import build_terms

    f = _frame()
    specs = [
        validate_spec(
            {
                "base": "hr_model",
                "operation": "add_covariate",
                "args": {"covariate": "hour_local"},
            }
        ),
        validate_spec(
            {
                "base": "hr_model",
                "operation": "add_interaction",
                "args": {"covariate_a": "sleep_h", "covariate_b": "wet_now"},
            }
        ),
        validate_spec(
            {
                "base": "hr_model",
                "operation": "add_changepoint",
                "args": {"day_index": 140},
            }
        ),
        validate_spec(
            {
                "base": "hr_model",
                "operation": "split_by_condition",
                "args": {"param": "gamma_d", "covariate": "temp_c20", "threshold": 5},
            }
        ),
        validate_spec(
            {
                "base": "hr_model",
                "operation": "directional_claim",
                "args": {"param": "gamma_d", "op": ">", "value": 0},
            }
        ),
    ]
    tb = build_terms(specs, "hr_model", f)
    X = tb.design(f)
    assert X.shape == (40, 4)  # the directional claim adds no column
    assert np.allclose(X[:, [0, 2, 3]].std(0), 1, atol=1e-9)
    assert np.isfinite(
        X
    ).all()  # missing sleep -> fit mean -> z = 0 (so column 1 has sd < 1)
    assert X[:, 1].std() < 1.0
    X2 = tb.design(f.head(5))  # same transformation on a new frame
    assert np.allclose(X2, X[:5])
    assert tb.extra(f)[1] == 3.0


def test_compile_rejects_no_contrast_and_unknown_param_split():
    from garmin_lab.agents.compile import build_terms

    f = _frame()
    cp = validate_spec(
        {
            "base": "hr_model",
            "operation": "add_changepoint",
            "args": {"day_index": 10_000},
        }
    )
    assert (
        build_terms([cp], "hr_model", f).terms == []
    )  # all zeros: not identifiable yet
    sp = validate_spec(
        {
            "base": "hr_model",
            "operation": "split_by_condition",
            "args": {"param": "sigma", "covariate": "temp_c20", "threshold": 5},
        }
    )
    assert build_terms([sp], "hr_model", f).terms == []


def test_savage_dickey_at_zero_direction():
    from garmin_lab.agents.compile import savage_dickey_at_zero

    rng = np.random.default_rng(1)
    null = savage_dickey_at_zero(rng.normal(0.0, 0.5, 4000), prior_sd=3.0)
    effect = savage_dickey_at_zero(rng.normal(4.0, 0.5, 4000), prior_sd=3.0)
    assert null["log_bf"] < 0 < effect["log_bf"]
    assert 0.8 < null["shrinkage"] < 0.9
    flat = savage_dickey_at_zero(rng.normal(0.0, 2.9, 4000), prior_sd=3.0)
    assert flat["shrinkage"] < 0.10  # would be gated
