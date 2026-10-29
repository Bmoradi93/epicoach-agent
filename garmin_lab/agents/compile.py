"""Compile validated model specs into design columns for the twin models.

A *structural* spec (``add_covariate``, ``add_interaction``, ``add_changepoint``,
``split_by_condition``) becomes one extra linear term ``x_j * theta_j`` with
prior ``theta_j ~ N(0, PRIOR_SD[base])``. The term is built from the fit
frame's statistics (mean / sd of the covariate) and can then be evaluated on
any frame with the same columns, so the forecaster applies exactly the
transformation the fit used. Nothing from the LLM becomes code: a spec that
cannot be expressed as such a term is reported as not compilable.

Evidence for a structural term is Savage-Dickey at ``theta_j = 0``:
``log BF10 = log p(0 | prior) - log p(0 | posterior)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import polars as pl

from garmin_lab.agents.templates import ModelSpec

STRUCTURAL_OPS = frozenset(
    {"add_covariate", "add_interaction", "add_changepoint", "split_by_condition"}
)

# prior sd of one extra term, in outcome units per 1 sd of the (standardised) column
PRIOR_SD = {"hr_model": 3.0, "skill_model": 0.03, "ffm": 0.02}

# the design column that each base-model parameter multiplies (for split_by_condition)
_PARAM_COLUMN: dict[tuple[str, str], Callable[[pl.DataFrame], np.ndarray]] = {
    ("hr_model", "beta_vp50"): lambda f: (_col(f, "vp_mean_w") - 140.0) / 50.0,
    ("hr_model", "gamma_T"): lambda f: _col(f, "temp_c20"),
    ("hr_model", "gamma_d"): lambda f: _col(f, "elapsed_h"),
    ("hr_model", "gamma_dT"): lambda f: _col(f, "temp_c20") * _col(f, "elapsed_h"),
    ("skill_model", "delta_wet"): lambda f: _col(f, "wet_now"),
    ("skill_model", "delta_sleep"): lambda f: np.nan_to_num(_col(f, "sleep_h") - 7.0),
    ("skill_model", "delta_elapsed"): lambda f: _col(f, "elapsed_h"),
    ("ffm", "b_hr10"): lambda f: (_col(f, "hr_mean") - 145.0) / 10.0,
    ("ffm", "b_T"): lambda f: _col(f, "temp_c20"),
    ("ffm", "b_elapsed"): lambda f: _col(f, "elapsed_h"),
}


def _col(f: pl.DataFrame, name: str) -> np.ndarray:
    return f[name].cast(pl.Float64).to_numpy().astype(float)


@dataclass(frozen=True)
class Term:
    """One compiled column: ``fn(frame) -> raw column``; standardised with fit-frame stats."""

    name: str
    fn: Callable[[pl.DataFrame], np.ndarray]
    mean: float
    sd: float

    def __call__(self, f: pl.DataFrame) -> np.ndarray:
        x = np.nan_to_num(self.fn(f), nan=self.mean)  # missing -> fit mean -> z = 0
        return (x - self.mean) / self.sd


@dataclass
class TermBuilder:
    """Design matrix for a list of structural specs on one base model."""

    base: str
    terms: list[Term] = field(default_factory=list)

    @property
    def prior_sd(self) -> float:
        return PRIOR_SD[self.base]

    def design(self, f: pl.DataFrame) -> np.ndarray:
        return (
            np.column_stack([t(f) for t in self.terms])
            if self.terms
            else np.zeros((f.height, 0))
        )

    def extra(self, f: pl.DataFrame) -> tuple[np.ndarray, float] | None:
        return (self.design(f), self.prior_sd) if self.terms else None


def is_structural(spec: ModelSpec | dict[str, Any]) -> bool:
    op = spec.operation if isinstance(spec, ModelSpec) else spec.get("operation")
    return op in STRUCTURAL_OPS


def _raw_fn(spec: ModelSpec) -> Callable[[pl.DataFrame], np.ndarray] | None:
    a = spec.args
    if spec.operation == "add_covariate":
        return lambda f, c=a["covariate"]: _col(f, c)
    if spec.operation == "add_interaction":
        return lambda f, c1=a["covariate_a"], c2=a["covariate_b"]: _col(f, c1) * _col(
            f, c2
        )
    if spec.operation == "add_changepoint":
        return lambda f, d=int(a["day_index"]): (_col(f, "day_index") >= d).astype(
            float
        )
    if spec.operation == "split_by_condition":
        base_col = _PARAM_COLUMN.get((spec.base, a["param"]))
        if base_col is None:
            return None
        thr = float(a["threshold"])
        return lambda f, c=a["covariate"], g=base_col: g(f) * (_col(f, c) > thr).astype(
            float
        )
    return None


def compile_term(spec: ModelSpec, fit_frame: pl.DataFrame) -> Term | None:
    """Build a standardised term from a spec, or None if the spec is not compilable
    (non-structural, unsupported parameter, or constant column in the fit data)."""
    fn = _raw_fn(spec)
    if fn is None:
        return None
    x = fn(fit_frame)
    finite = x[np.isfinite(x)]
    if len(finite) < 5:
        return None
    mean, sd = float(finite.mean()), float(finite.std())
    if sd < 1e-9:
        return None  # no contrast in the data yet: not identifiable
    return Term(name=spec.key(), fn=fn, mean=mean, sd=sd)


def build_terms(
    specs: list[ModelSpec], base: str, fit_frame: pl.DataFrame
) -> TermBuilder:
    terms = [compile_term(s, fit_frame) for s in specs if s.base == base]
    return TermBuilder(base=base, terms=[t for t in terms if t is not None])


def savage_dickey_at_zero(theta_post: np.ndarray, prior_sd: float) -> dict[str, float]:
    """log BF10 for theta != 0 with a N(0, prior_sd) prior, Gaussian approximation to the posterior
    (adequate for a linear coefficient). Also returns the shrinkage used by the identifiability gate.
    """
    m, s = float(theta_post.mean()), float(max(theta_post.std(), 1e-9))
    log_post0 = -0.5 * (m / s) ** 2 - np.log(s * np.sqrt(2 * np.pi))
    log_prior0 = -np.log(prior_sd * np.sqrt(2 * np.pi))
    return {
        "log_bf": float(log_prior0 - log_post0),
        "shrinkage": float(1.0 - s / prior_sd),
        "post_mean": m,
        "post_sd": s,
    }
