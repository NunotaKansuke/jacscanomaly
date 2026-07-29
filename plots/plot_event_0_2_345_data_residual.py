#!/usr/bin/env python3
"""Standalone data-only and PSPL-residual figures for Roman event 0_2_345."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "roman_simu/anomaly_finder_result/planet_signal_data/planet_signal_result_602_345.json"
OUTDIR = Path(__file__).resolve().parent


def base_axis(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=4, width=0.8, color="#4B5563")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=9))


def save(fig, stem: str) -> None:
    fig.savefig(OUTDIR / f"0_2_345_{stem}.png", dpi=300, facecolor="white")
    fig.savefig(OUTDIR / f"0_2_345_{stem}.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    d = json.loads(DATA.read_text())
    s = d["series"]
    t0 = float(d["fit"]["t0"])
    t = np.asarray(s["time"], dtype=float) - t0
    flux = np.asarray(s["flux"], dtype=float) * 1e9
    ferr = np.asarray(s["ferr"], dtype=float) * 1e9
    model = np.asarray(s["model_flux"], dtype=float) * 1e9
    residual = flux - model

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 17,
                         "xtick.labelsize": 21, "ytick.labelsize": 21})
    xpad = 0.015 * (float(t.max()) - float(t.min()))
    xlim = (float(t.min()) - xpad, float(t.max()) + xpad)

    # Data only: no fitted curve, annotations, title, labels, or shading.
    fig, ax = plt.subplots(figsize=(10.5, 7.5), layout="constrained")
    ax.errorbar(t, flux, yerr=ferr, fmt=".", ms=14.0, color="C0", ecolor="C0",
                elinewidth=0.55, alpha=0.95, rasterized=True)
    ypad = 0.05 * (float(flux.max()) - float(flux.min()))
    ax.set(xlim=xlim, ylim=(float(flux.min()) - ypad, float(flux.max()) + ypad))
    base_axis(ax)
    save(fig, "data_only")

    # Residual: data minus the refined PSPL model at each observation time.
    fig, ax = plt.subplots(figsize=(10.5, 7.5), layout="constrained")
    ax.axhline(0.0, color="#111111", lw=2.0, zorder=1)
    ax.errorbar(t, residual, yerr=ferr, fmt=".", ms=14.0, color="C0", ecolor="C0",
                elinewidth=0.55, alpha=0.95, rasterized=True, zorder=2)
    rpad = 0.08 * (float(residual.max()) - float(residual.min()))
    ax.set(xlim=xlim, ylim=(float(residual.min()) - rpad, float(residual.max()) + rpad))
    base_axis(ax)
    save(fig, "residual")


if __name__ == "__main__":
    main()
