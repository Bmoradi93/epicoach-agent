"""Expected information gain of candidate training weeks about the fitness-fatigue parameters.

A *design* is a 7-day load schedule (exp-TRIMP per day) followed by a test
traversal of a known climb segment at a fixed heart rate on day 8 (optionally
also a mid-week test). The outcome is the log speed on the test climb(s).

    EIG(d) = E_{theta, y|theta,d}[ log p(y|theta,d) - log p(y|d) ]

estimated by nested Monte Carlo over posterior samples (Rainforth et al.
2018), with inner samples shared across designs. Because tau1, tau2 and k only
affect the outcome through the latent performance p_d, EIG about theta equals
the information the design carries about the *difference* between parameter
hypotheses in their predicted p on test days: designs that make different
(tau1, tau2) predict different speeds are informative.

The utility traded against EIG is the expected performance change over a
14-day horizon under the same posterior.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np

from garmin_lab.twin.ffm import TWIN_DIR, FFMData, _filter, load_ffm_data

LOAD_SCALE = 100.0


@dataclass
class Design:
    name: str
    loads: list[float]  # daily exp-TRIMP values (7, 14 or 21 days)
    test_days: list[int] = field(
        default_factory=lambda: [7]
    )  # day offsets (0 = first day) of test climbs
    hr_target_max: float | None = 165.0
    archetype: str = ""
    test_repeats: int = (
        1  # repeated efforts of the test climb per test day (noise / sqrt(n))
    )

    @property
    def weeks(self) -> int:
        return len(self.loads) // 7

    def first_week(self) -> list[float]:
        return list(self.loads[:7])

    def describe(self) -> str:
        return f"{self.name} ({self.weeks} wk): loads={[int(x) for x in self.loads]} tests on day(s) {[d + 1 for d in self.test_days]}"


# The instrument protocol: two fixed test rides every week, same climb, 160 +- 5 bpm.
# day 0 = FRESH test (first day after the rest day), day 4 = LOADED test (after the week's training).
# Roll-in + ~17 min at 160 bpm on the Trans Canada Trail + Nicoles descent + cool-down.
# Historical short rides containing this climb: median 58 exp-TRIMP (n=19) at ~150 bpm.
INSTRUMENT_TRIMP = 60.0
TEST_DAYS = (0, 4)
REST_DAY = 6


def _week(w: float, pattern: str) -> list[float]:
    """Seven daily loads: the two instrument rides are fixed; the remaining training (w - 2*I,
    scaled by the pattern) is distributed over the free days. Day 6 is always a rest day so the
    next week's first test is fresh."""
    free = max(w - 2 * INSTRUMENT_TRIMP, 0.0)
    pats = {  # (scale of free load, shares over free days 1,2,3,5)
        "conv": (1.0, [0.3, 0.3, 0.2, 0.2]),
        "conv_2x": (1.0, [0.5, 0, 0.5, 0]),
        "front": (1.0, [0.6, 0.4, 0, 0]),
        "back": (1.0, [0, 0, 0.4, 0.6]),
        "deload50": (0.5, [0.3, 0.3, 0.2, 0.2]),
        "deload25": (0.25, [0.5, 0, 0.5, 0]),
        "tests_only": (0.0, [0, 0, 0, 0]),
        "over115": (1.15, [0.3, 0.3, 0.2, 0.2]),
        "over130": (1.30, [0.3, 0.3, 0.2, 0.2]),
    }
    scale, shares = pats[pattern]
    out = [0.0] * 7
    for d in TEST_DAYS:
        out[d] = INSTRUMENT_TRIMP
    for d, sh in zip((1, 2, 3, 5), shares):
        out[d] = free * scale * sh
    return out


def _tests(weeks: int) -> list[int]:
    return [7 * k + d for k in range(weeks) for d in TEST_DAYS]


def default_menu(weekly_target: float) -> list[Design]:
    """Candidate plans. Every week contains the two protocol test rides (fresh / loaded); plans
    differ only in the training between them. One-, two- and three-week designs; the agent commits
    to the first week and re-plans, but scores the full contrast."""
    w = weekly_target

    def one(name: str, pat: str, arch: str) -> Design:
        return Design(name, _week(w, pat), _tests(1), archetype=arch)

    def two(name: str, p1: str, p2: str, arch: str) -> Design:
        return Design(name, _week(w, p1) + _week(w, p2), _tests(2), archetype=arch)

    menu = [
        one("conventional", "conv", "conventional"),
        one("conv_2_sessions", "conv_2x", "conventional"),
        one("front_loaded", "front", "block"),
        one("back_loaded", "back", "block"),
        one("deload_50", "deload50", "deload"),
        one("deload_25", "deload25", "deload"),
        one("tests_only", "tests_only", "deload"),
        one("overload_115", "over115", "overload"),
        one("overload_130", "over130", "overload"),
        two("conv_conv", "conv", "conv", "conventional"),
        two("deload_then_conv", "deload50", "conv", "deload"),
        two("tests_only_then_conv", "tests_only", "conv", "deload"),
        two("overload_then_deload", "over115", "deload50", "overload"),
        two("over130_then_tests_only", "over130", "tests_only", "overload"),
        two("conv_then_over130", "conv", "over130", "overload"),
        Design(
            "block3_then_taper",
            _week(w, "over115") + _week(w, "over115") + _week(w, "deload50"),
            _tests(3),
            archetype="block",
        ),
        Design("conv3", _week(w, "conv") * 3, _tests(3), archetype="conventional"),
        Design(
            "over130_tests_only_conv",
            _week(w, "over130") + _week(w, "tests_only") + _week(w, "conv"),
            _tests(3),
            archetype="overload",
        ),
    ]
    return menu


class EIGScorer:
    def __init__(
        self,
        idata: az.InferenceData | None = None,
        data: FFMData | None = None,
        n_outer: int = 400,
        n_inner: int = 400,
        seed: int = 0,
        test_sigma: float | None = None,
    ):
        """``test_sigma``: log-speed noise of the designated test segment at fixed HR (default: model sigma)."""
        self.d = data or load_ffm_data()
        self.idata = idata or az.from_netcdf(TWIN_DIR / "ffm_full.nc")
        post = self.idata.posterior
        flat = lambda v: post[v].values.reshape(-1)  # noqa: E731
        n = len(flat("tau1"))
        rng = np.random.default_rng(seed)
        self.idx_outer = rng.choice(n, size=min(n_outer, n), replace=False)
        self.idx_inner = rng.choice(n, size=min(n_inner, n), replace=False)
        self.tau1 = flat("tau1")
        self.tau2 = flat("tau2")
        self.k1 = flat("k1")
        self.k2 = flat("k2")
        self.sigma = (
            flat("sigma")
            if test_sigma is None
            else np.full_like(flat("sigma"), test_sigma)
        )
        self.nu = flat("nu") + 2
        w = jnp.asarray(self.d.w)
        # terminal filter states for every posterior sample (vectorised over samples)
        self.F_end = np.asarray(
            jax.vmap(lambda t: _filter(w, jnp.exp(-1 / t))[-1])(jnp.asarray(self.tau1))
        )
        self.G_end = np.asarray(
            jax.vmap(lambda t: _filter(w, jnp.exp(-1 / t))[-1])(jnp.asarray(self.tau2))
        )
        # p is centred in the model; the centring constant per sample
        self.p_mean = (
            post["p_day"].values.reshape(n, -1).mean(1) * 0.0
        )  # p_day already centred -> 0 offset; keep for clarity
        self.p_last = post["p_day"].values.reshape(n, -1)[:, -1]
        self.rng = rng
        # common random numbers across designs: fixed Gaussian noise per (outer sample, test slot)
        self._eps = rng.standard_normal(size=(len(self.idx_outer), 8))

    # ------------------------------------------------------------------
    def _p_future(self, idx: np.ndarray, loads: np.ndarray, horizon: int) -> np.ndarray:
        """Latent performance on each of the next ``horizon`` days for posterior samples idx.
        Returns shape (len(idx), horizon). Loads beyond the 7 planned days are zero."""
        t1, t2, k1, k2 = self.tau1[idx], self.tau2[idx], self.k1[idx], self.k2[idx]
        F, G = self.F_end[idx].copy(), self.G_end[idx].copy()
        # reconstruct the centring offset so that p_future is comparable to p_last
        offset = self.p_last[idx] - (k1 * F - k2 * G)
        out = np.empty((len(idx), horizon))
        sched = np.zeros(horizon)
        n = min(len(loads), horizon)
        sched[:n] = loads[:n] / LOAD_SCALE
        for h in range(horizon):
            F = np.exp(-1 / t1) * F + sched[h]
            G = np.exp(-1 / t2) * G + sched[h]
            out[:, h] = k1 * F - k2 * G + offset
        return out

    def eig(self, design: Design) -> float:
        """Nested-MC EIG (nats) of the test outcomes about theta = (tau1, tau2, k1, k2)."""
        loads = np.asarray(design.loads, dtype=float)
        H = max(design.test_days) + 1
        p_out = self._p_future(self.idx_outer, loads, H)[:, design.test_days]  # (No, T)
        p_in = self._p_future(self.idx_inner, loads, H)[:, design.test_days]  # (Ni, T)
        sig_o = self.sigma[self.idx_outer][:, None] / np.sqrt(design.test_repeats)
        sig_i = self.sigma[self.idx_inner][None, :, None] / np.sqrt(design.test_repeats)
        # simulate outcomes y = p + eps for outer samples (common random numbers across designs)
        y = p_out + self._eps[:, : p_out.shape[1]] * sig_o
        # log p(y | theta_outer) : Gaussian approx with sigma_outer
        lp_self = -0.5 * np.sum(
            ((y - p_out) / sig_o) ** 2 + np.log(2 * np.pi * sig_o**2), axis=1
        )
        # log p(y | d) = log mean_inner p(y | theta_inner)
        diff = (y[:, None, :] - p_in[None, :, :]) / sig_i  # (No, Ni, T)
        lp_inner = -0.5 * np.sum(
            diff**2 + np.log(2 * np.pi * sig_i**2), axis=2
        )  # (No, Ni)
        lp_marg = np.logaddexp.reduce(lp_inner, axis=1) - np.log(lp_inner.shape[1])
        return float(np.mean(lp_self - lp_marg))

    def expected_perf_change(self, design: Design, horizon: int | None = None) -> float:
        """Expected change in latent log-speed performance from today to the end of the design (+7 d)."""
        loads = np.asarray(design.loads, dtype=float)
        horizon = horizon or (len(loads) + 7)
        p = self._p_future(self.idx_outer, loads, horizon)
        return float(np.mean(p[:, -1] - self.p_last[self.idx_outer]))

    def score_menu(
        self, menu: list[Design], lam: float = 0.5, perf_scale: float = 0.02
    ) -> list[dict]:
        rows = []
        for d in menu:
            e = self.eig(d)
            dp = self.expected_perf_change(d)
            rows.append(
                {
                    "name": d.name,
                    "archetype": d.archetype,
                    "weeks": d.weeks,
                    "eig_nats": e,
                    "eig_per_week": e / d.weeks,
                    "exp_perf_change": dp,
                    "utility": lam * e / d.weeks + (1 - lam) * dp / perf_scale,
                    "loads": d.loads,
                    "first_week": d.first_week(),
                    "test_days": d.test_days,
                    "test_repeats": d.test_repeats,
                    "hr_target_max": d.hr_target_max,
                }
            )
        rows.sort(key=lambda r: -r["utility"])
        return rows
