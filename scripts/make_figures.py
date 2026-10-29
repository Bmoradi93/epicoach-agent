"""Regenerate the paper's data figures from the current dataset, twin fits and registry.

    pixi run -e research python scripts/make_figures.py

Writes paper_template/figures/fig_instrument_map.pdf, fig_prospective.pdf, fig_posteriors.pdf
(+ .png previews). Other figures (replay CRPS forest, fixed-HR argument, registry trajectory)
come from the replay outputs and are unchanged by the prospective phase.
"""

from __future__ import annotations

import datetime as dt
import json
import math

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import lognorm, norm

from garmin_lab.ingest.client import REPO_ROOT
from garmin_lab.registry.score import load_traversals
from garmin_lab.registry.store import Registry
from garmin_lab.segments.osm import fetch_area
from garmin_lab.segments.routes import INSTRUMENT_CLIMB, INSTRUMENT_DESCENT, ROUTES

FIG = REPO_ROOT / "paper_template" / "figures"
DATASET = REPO_ROOT / "data" / "dataset"
TWIN = REPO_ROOT / "data" / "twin"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 7, "pdf.fonttype": 42})

BLUE, RED, GREY, INK = "#2b6cb0", "#c53030", "#a0aec0", "#1a202c"


def _save(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    print("wrote", name)


def instrument_map() -> None:
    routes = {r.segment_id: r for r in ROUTES}
    tr = pl.read_parquet(DATASET / "traversals.parquet")
    smp = pl.read_parquet(DATASET / "samples.parquet").select(["activity_id", "timestamp", "lat", "lon"])
    ways = fetch_area("burnaby")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.6, 1]})
    for ax, sid, col, title in (
        (axes[0], INSTRUMENT_CLIMB, BLUE, "Trans Canada Trail climb"),
        (axes[1], INSTRUMENT_DESCENT, RED, "Nicoles descent via Poplar"),
    ):
        r = routes[sid]
        t = tr.filter(pl.col("segment_id") == sid)
        lons, lats = [], []
        for row in t.to_dicts():
            s = smp.filter(pl.col("activity_id") == row["activity_id"])
            e = s["timestamp"].dt.epoch("s").to_numpy()
            m = (e >= row["t_entry"]) & (e <= row["t_exit"])
            ax.plot(s["lon"].to_numpy()[m], s["lat"].to_numpy()[m], "-", color=col, lw=0.4, alpha=0.35, zorder=2)
            lons += s["lon"].to_numpy()[m].tolist(); lats += s["lat"].to_numpy()[m].tolist()
        pad = 0.0012
        x0, x1, y0, y1 = min(lons) - pad, max(lons) + pad, min(lats) - pad, max(lats) + pad
        for w in ways:
            if w.get("tags", {}).get("name") in r.corridor_names:
                g = [p for p in w["geometry"] if x0 <= p["lon"] <= x1 and y0 <= p["lat"] <= y1]
                ax.plot([p["lon"] for p in g], [p["lat"] for p in g], ".", color=GREY, ms=1.5, zorder=1)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.plot(r.gate_in[1], r.gate_in[0], "o", mfc="white", mec=INK, ms=7, zorder=5)
        ax.plot(r.gate_out[1], r.gate_out[0], "s", mfc="white", mec=INK, ms=7, zorder=5)
        ax.set_title(f"{title}  (n = {t.height})")
        ax.set_aspect(1 / math.cos(math.radians(r.gate_in[0])))
        ax.tick_params(labelsize=6)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))
    _save(fig, "fig_instrument_map")


def prospective() -> None:
    reg = Registry(REPO_ROOT / "data" / "registry.sqlite")
    f = load_traversals()
    start = dt.date(2026, 10, 1)
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.6), sharex=True)
    panels = (
        (axes[0], INSTRUMENT_CLIMB, "Climb moving time at 160 ± 5 bpm (s)", BLUE, "normal"),
        (axes[1], INSTRUMENT_DESCENT, "Descent moving time (s)", RED, "lognormal"),
    )
    events = {
        dt.date(2026, 10, 4): "gate + forecast\ncorrection",
        dt.date(2026, 10, 11): "refit +\ncycle 1",
        dt.date(2026, 10, 17): "refit +\ncycle 2",
    }
    for ax, sid, ylabel, col, fam in panels:
        rides = f.filter((pl.col("segment_id") == sid) & (pl.col("date") >= start)).sort("date")
        prot = rides.filter(pl.col("protocol"))
        trn = rides.filter(~pl.col("protocol"))
        ax.plot(prot["date"].to_list(), prot["moving_s"].to_list(), "o", color=col, ms=5, zorder=5, label="instrument ride (laps)")
        if trn.height:
            ax.plot(trn["date"].to_list(), trn["moving_s"].to_list(), "x", color=GREY, ms=5, zorder=4, label="training ride through gates")
        for p in reg.ledger():
            if p["target"].split("@", 1)[1] != sid:
                continue
            d = json.loads(p["distribution"])
            t0 = dt.datetime.fromisoformat(p["registered_at"]).date()
            t1 = dt.datetime.fromisoformat(p["target_time"]).date()
            if d["family"] == "normal":
                mu, lo, hi = d["mu"], norm.ppf(0.05, d["mu"], d["sigma"]), norm.ppf(0.95, d["mu"], d["sigma"])
            else:
                mu = math.exp(d["mu"])
                lo, hi = lognorm.ppf(0.05, d["sigma"], scale=mu), lognorm.ppf(0.95, d["sigma"], scale=mu)
            if p["withdrawn_at"]:
                ax.hlines(mu, t0, t1, colors=GREY, linestyles=":", lw=1, zorder=2)
                continue
            if p["scored_at"]:
                aid = int(p["outcome_ref"].split("@")[0])
                t1 = f.filter((pl.col("activity_id") == aid) & (pl.col("segment_id") == sid))["date"][0]
            ax.fill_between([t0, t1], [lo, lo], [hi, hi], color=col, alpha=0.10, lw=0, zorder=1)
            ax.hlines(mu, t0, t1, colors=col, lw=1.2, zorder=3)
            if p["scored_at"]:
                ax.vlines(t1, mu, p["outcome"], colors=INK, lw=0.6, zorder=4)
                ax.plot(t1, p["outcome"], "o", mfc="none", mec=INK, ms=5, mew=0.6, zorder=6)
        for d_ in events:
            ax.axvline(d_, color=GREY, lw=0.6, ls="--", zorder=0)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    axes[0].set_ylim(880, 1180)
    axes[1].set_ylim(220, 380)
    for d_, lab in events.items():
        axes[0].text(d_, 1175, lab, fontsize=6, color=GREY, ha="center", va="top")
    axes[0].legend(loc="lower left", frameon=False)
    axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=3))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[1].set_xlim(start, dt.date(2026, 10, 24))
    fig.align_ylabels(axes)
    _save(fig, "fig_prospective")


def posteriors() -> None:
    post = az.from_netcdf(TWIN / "ffm_full.nc").posterior
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.0))
    specs = (
        ("tau1", r"$\tau_1$ fitness (d)", (math.log(42), 0.3), 42),
        ("tau2", r"$\tau_2$ fatigue (d)", (math.log(7), 0.4), 7),
        ("k2_over_k1", r"$k_2/k_1$", (math.log(2), 0.3), None),
    )
    for ax, (k, lab, (pm, ps), ref) in zip(axes, specs):
        v = post[k].values.ravel()
        x = np.linspace(max(v.min() * 0.5, 1e-3), np.percentile(v, 99.5) * 1.3, 400)
        ax.plot(x, lognorm.pdf(x, ps, scale=math.exp(pm)), color=GREY, lw=1, label="prior")
        ax.hist(v, bins=60, density=True, color=BLUE, alpha=0.6, label="posterior")
        if ref is not None:
            ax.axvline(ref, color=INK, lw=0.8, ls="--")
        ax.set_title(lab)
        ax.set_yticks([])
        ax.tick_params(labelsize=6)
    axes[0].legend(frameon=False)
    _save(fig, "fig_posteriors")


if __name__ == "__main__":
    instrument_map()
    prospective()
    posteriors()
