#!/usr/bin/env python3
"""Create clean slide figures for Roman event 0_2_2705.

Writes separate high-resolution figures for the single-lens fit and its weak
planet-signal candidate.  The input fit is the persisted refined PSPL result
from the Roman simulation analysis.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, MultipleLocator


EVENT = os.environ.get("JAC_EVENT", "0_2_2705")
ROOT = Path(__file__).resolve().parents[1]
INPUT = Path(os.environ.get(
    "JAC_JSON",
    str(ROOT.parent / "roman_simu" / "anomaly_finder_result" / "planet_signal_data" / "planet_signal_result_605_2705.json"),
))
OUTDIR = Path(__file__).resolve().parent


def save(fig: plt.Figure, stem: str) -> None:
    """Save a slide-ready raster and a lossless vector version."""
    if "--signal-only" in sys.argv and stem == "single_lens_fit":
        return
    # 4000 px-wide PNG: native 4K slide resolution without the memory cost of
    # an oversized raster canvas.  The accompanying PDF remains vector.
    for suffix, kwargs in ((".png", {"dpi": 300}), (".pdf", {})):
        path = OUTDIR / f"{EVENT}_{stem}{suffix}"
        fig.savefig(path, facecolor="white", **kwargs)
        print(f"wrote {path}")


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=4, width=0.8, color="#4B5563")


def main() -> None:
    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    series, fit = payload["series"], payload["fit"]
    candidate = payload["candidates"][0]

    time = np.asarray(series["time"], dtype=float)
    flux = np.asarray(series["flux"], dtype=float) * 1e9
    ferr = np.asarray(series["ferr"], dtype=float) * 1e9
    residual_sigma = np.asarray(series["residual"], dtype=float) / np.asarray(series["ferr"], dtype=float)
    signal = np.asarray(series["signal_mask"], dtype=bool)
    t0 = float(fit["t0"])
    t_anom = float(candidate["peak_time"])
    t_start, t_end = float(candidate["t_start"]), float(candidate["t_end"])

    model = payload["plot"]["model_curve"]
    t_model = np.asarray(model["time"], dtype=float)
    f_model = np.asarray(model["flux"], dtype=float) * 1e9

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 17,
        "axes.labelsize": 18,
        "axes.titlesize": 23,
        "xtick.labelsize": 21,
        "ytick.labelsize": 21,
        "axes.linewidth": 0.9,
    })

    # 1) Whole light curve: only the measured points and the PSPL curve.
    fig, ax = plt.subplots(figsize=(12.5, 6.5), layout="constrained")
    x = time - t0
    xm = t_model - t0
    ax.plot(xm, f_model, color="#111111", lw=4.0, zorder=5)
    ax.errorbar(x, flux, yerr=ferr, fmt=".", ms=14.0, color="C0", ecolor="C0",
                elinewidth=0.55, alpha=0.95, rasterized=True, zorder=3)
    if EVENT == "0_161_832":
        x_extent = max(abs(float(np.min(x))), abs(float(np.max(x))))
        x_limits = (-x_extent, x_extent)
    elif EVENT == "0_952_1403":
        x_limits = (-50.0, 50.0)
    elif os.environ.get("JAC_FULL_SEASON", "").lower() in {"1", "true", "yes"}:
        x_min, x_max = float(np.min(x)), float(np.max(x))
        x_pad = 0.015 * (x_max - x_min)
        x_limits = (x_min - x_pad, x_max + x_pad)
    else:
        x_extent = max(4.0, abs(t_anom - t0) + 2.0)
        x_limits = (-x_extent, x_extent)
    ylo = min(float(np.min(flux - ferr)), float(np.min(f_model)))
    yhi = max(float(np.max(flux + ferr)), float(np.max(f_model)))
    ypad = 0.05 * (yhi - ylo)
    ax.set(xlim=x_limits, ylim=(ylo - ypad, yhi + ypad))
    ax.xaxis.set_major_locator(MultipleLocator(2))
    if (
        EVENT in {"0_161_832", "0_952_1403"}
        or os.environ.get("JAC_FULL_SEASON", "").lower() in {"1", "true", "yes"}
    ):
        ax.xaxis.set_major_locator(MaxNLocator(nbins=9))
    style_axis(ax)
    save(fig, "single_lens_fit")
    plt.close(fig)

    # 2) Candidate zoom: the same minimal PSPL/data-only treatment.
    zoom_half_width = 0.19  # days, about 4.6 hours either side of the candidate
    keep = np.abs(time - t_anom) <= zoom_half_width
    keep_model = np.abs(t_model - t_anom) <= zoom_half_width
    zoom_time = time - t0
    model_zoom_time = t_model - t0
    fig, ax_zoom = plt.subplots(figsize=(12.5, 6.5), layout="constrained")
    ax_zoom.spines[["top", "right"]].set_visible(False)
    ax_zoom.errorbar(zoom_time[keep], flux[keep], yerr=ferr[keep], fmt="o", ms=10.0,
                     color="C0", ecolor="C0", elinewidth=.65, capsize=0, rasterized=True)
    ax_zoom.plot(model_zoom_time[keep_model], f_model[keep_model], color="#111111", lw=4.0)
    ax_zoom.set(xlim=(t_anom - t0 - zoom_half_width, t_anom - t0 + zoom_half_width))
    style_axis(ax_zoom)
    save(fig, "weak_planet_signal")
    plt.close(fig)


if __name__ == "__main__":
    main()
