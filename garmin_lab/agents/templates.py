"""Constrained model-template vocabulary.

The Modeler LLM may only express a hypothesis as a *spec* built from these
operations on a known base model. Specs are validated here; nothing from the
LLM becomes code. Unknown variables, parameters or operations are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

BASE_MODELS: dict[str, dict[str, Any]] = {
    "hr_model": {
        "outcome": "hr_mean (climb traversals)",
        "params": [
            "beta_vp50",
            "gamma_T",
            "gamma_d",
            "gamma_dT",
            "sigma_u",
            "sigma_ride",
            "sigma",
            "c_week",
        ],
        "in_model": ["vp_mean_w", "temp_c20", "elapsed_h", "temp_c20*elapsed_h"],
        "covariates": [
            "vp_mean_w",
            "temp_c20",
            "elapsed_h",
            "sleep_h",
            "hrv_last_night",
            "readiness_score",
            "wet_now",
            "hour_local",
            "wx_humidity",
        ],
    },
    "skill_model": {
        "outcome": "log moving_s (descent traversals)",
        "in_model": ["wet_now", "sleep_h", "elapsed_h"],
        "params": [
            "delta_wet",
            "delta_sleep",
            "delta_elapsed",
            "sigma_S",
            "sigma",
            "S_week",
            "lambda_seg",
        ],
        "covariates": [
            "sleep_h",
            "wet_now",
            "elapsed_h",
            "hrv_last_night",
            "readiness_score",
            "hour_local",
            "temp_c20",
        ],
    },
    "ffm": {
        "outcome": "log moving_speed_ms at matched HR (climb traversals)",
        "in_model": ["hr_mean", "temp_c20", "elapsed_h", "season"],
        "params": [
            "tau1",
            "tau2",
            "k1",
            "k2_over_k1",
            "b_hr10",
            "b_T",
            "b_elapsed",
            "b_season_sin",
            "b_season_cos",
            "sigma",
        ],
        "covariates": [
            "temp_c20",
            "elapsed_h",
            "sleep_h",
            "hrv_last_night",
            "readiness_score",
        ],
    },
}

OPERATIONS = {
    "add_covariate": {
        "args": ["covariate"],
        "doc": "add a linear covariate with a N(0, 0.05*scale) prior",
    },
    "add_interaction": {
        "args": ["covariate_a", "covariate_b"],
        "doc": "add a product term of two covariates",
    },
    "add_changepoint": {
        "args": ["day_index"],
        "doc": "allow a step change in the intercept at a given day",
    },
    "add_random_walk": {
        "args": ["param"],
        "doc": "let a fixed parameter drift as a weekly Gaussian random walk",
    },
    "split_by_condition": {
        "args": ["param", "covariate", "threshold"],
        "doc": "separate values of a parameter above/below a covariate threshold",
    },
    "directional_claim": {
        "args": ["param", "op", "value"],
        "doc": "no model change; test P(param op value) by Savage-Dickey",
    },
}


class TemplateError(ValueError):
    pass


@dataclass
class ModelSpec:
    base: str
    operation: str
    args: dict[str, Any]
    notes: str = ""
    warnings: list[str] = field(default_factory=list)

    def key(self) -> str:
        return f"{self.base}:{self.operation}:" + ",".join(
            f"{k}={v}" for k, v in sorted(self.args.items())
        )


_ALIASES = {
    "model": "base",
    "base_model": "base",
    "op": "operation",
    "operation_type": "operation",
    "op_type": "operation",
}
_ARG_KEYS = {
    "covariate",
    "covariate_a",
    "covariate_b",
    "param",
    "parameter",
    "value",
    "threshold",
    "day_index",
    "direction",
    "comparison",
}


def _normalise(raw: dict[str, Any]) -> dict[str, Any]:
    """Accept common LLM variants: aliased keys, args at top level, '>'/'<' given as op."""
    out: dict[str, Any] = {"args": dict(raw.get("args") or {})}
    for k, v in raw.items():
        if k == "args":
            continue
        kk = _ALIASES.get(k, k)
        if kk in ("base", "operation", "notes"):
            out[kk] = v
        elif kk in _ARG_KEYS:
            out["args"][
                {"parameter": "param", "direction": "op", "comparison": "op"}.get(
                    kk, kk
                )
            ] = v
    # an operation given as a comparator means a directional claim
    if out.get("operation") in (">", "<"):
        out["args"]["op"] = out["operation"]
        out["operation"] = "directional_claim"
    if out.get("operation") == "directional_claim" and "op" not in out["args"]:
        for k in ("direction", "comparison", "op_type"):
            if k in raw:
                out["args"]["op"] = raw[k]
    return out


def validate_spec(raw: dict[str, Any]) -> ModelSpec:
    raw = _normalise(raw)
    base = raw.get("base")
    if base not in BASE_MODELS:
        raise TemplateError(
            f"unknown base model {base!r}; allowed: {sorted(BASE_MODELS)}"
        )
    op = raw.get("operation")
    if op not in OPERATIONS:
        raise TemplateError(f"unknown operation {op!r}; allowed: {sorted(OPERATIONS)}")
    args = raw.get("args") or {}
    need = OPERATIONS[op]["args"]
    missing = [a for a in need if a not in args]
    if missing:
        raise TemplateError(f"operation {op} needs args {need}; missing {missing}")
    bm = BASE_MODELS[base]
    warnings: list[str] = []
    for k, v in args.items():
        if k.startswith("covariate") and v not in bm["covariates"]:
            raise TemplateError(
                f"covariate {v!r} not available for {base}; allowed: {bm['covariates']}"
            )
        if k == "param" and v not in bm["params"]:
            raise TemplateError(
                f"parameter {v!r} not in {base}; allowed: {bm['params']}"
            )
        if k == "op" and v not in (">", "<"):
            raise TemplateError("op must be '>' or '<'")
    if op == "add_interaction" and args["covariate_a"] == args["covariate_b"]:
        raise TemplateError("interaction of a covariate with itself")
    # identifiability red flags
    if (
        base == "skill_model"
        and args.get("covariate") in ("hrv_last_night", "readiness_score")
        and "sleep" in str(raw.get("notes", "")).lower()
    ):
        warnings.append(
            "HRV/readiness are themselves functions of sleep; collinear with delta_sleep"
        )
    if (
        base == "ffm"
        and op in ("add_random_walk",)
        and args.get("param") in ("tau1", "tau2")
    ):
        raise TemplateError(
            "time constants are already weakly identified; a random walk on them is not identifiable"
        )
    if op == "directional_claim" and args.get("param") in (
        "sigma",
        "sigma_u",
        "sigma_ride",
        "sigma_S",
        "nu",
    ):
        warnings.append(
            "claim about a nuisance scale; is this physiologically meaningful?"
        )
    return ModelSpec(
        base=base,
        operation=op,
        args=dict(args),
        notes=str(raw.get("notes", "")),
        warnings=warnings,
    )


def vocabulary_text() -> str:
    lines = ["BASE MODELS:"]
    for k, v in BASE_MODELS.items():
        lines.append(
            f"  {k}: outcome={v['outcome']}; params={v['params']}; "
            f"currently_in_model={v['in_model']}; available_covariates={v['covariates']}"
        )
    lines.append(
        "'available_covariates' are columns that MAY be added; only 'currently_in_model' terms are in the base model."
    )
    lines.append("OPERATIONS:")
    for k, v in OPERATIONS.items():
        lines.append(f"  {k}(args={v['args']}): {v['doc']}")
    lines.append("SPEC FORMAT (JSON, exactly these keys):")
    lines.append(
        '  {"base": "hr_model", "operation": "directional_claim", "args": {"param": "gamma_T", "op": ">", "value": 0}, "notes": "..."}'
    )
    lines.append(
        '  {"base": "hr_model", "operation": "add_covariate", "args": {"covariate": "hour_local"}, "notes": "..."}'
    )
    lines.append(
        '  {"base": "skill_model", "operation": "add_interaction", "args": {"covariate_a": "sleep_h", "covariate_b": "wet_now"}}'
    )
    lines.append(
        "Only parameters already in a base model can carry a directional_claim; a NEW covariate's effect is"
    )
    lines.append(
        "tested by add_covariate (the statistician refits), not by claiming a parameter that does not exist yet."
    )
    return "\n".join(lines)
