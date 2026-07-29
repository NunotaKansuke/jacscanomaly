#!/usr/bin/env python3
"""Render the simple PSPL/data-only zoom for Roman 0_2_2705."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EVENT = os.environ.get("JAC_EVENT", "0_2_2705")
OUT = Path(__file__).resolve().parent / f"{EVENT}_weak_planet_signal"
DATA = Path(os.environ.get(
    "JAC_JSON",
    str(Path(__file__).resolve().parents[2] / "roman_simu/anomaly_finder_result/planet_signal_data/planet_signal_result_605_2705.json"),
))


def main() -> None:
    d = json.loads(DATA.read_text())
    s, cand = d["series"], d["candidates"][0]
    t0 = float(d["fit"]["t0"])
    t = np.asarray(s["time"]); f = np.asarray(s["flux"]) * 1e9; e = np.asarray(s["ferr"]) * 1e9
    center, half = cand["peak_time"], 0.19
    keep = np.abs(t - center) <= half
    model = d["plot"]["model_curve"]
    mt, mf = np.asarray(model["time"]), np.asarray(model["flux"]) * 1e9
    mk = np.abs(mt - center) <= half
    h, mh = t - t0, mt - t0

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 17,
                         "axes.labelsize": 18, "xtick.labelsize": 21, "ytick.labelsize": 21})
    fig, ax = plt.subplots(figsize=(10.5, 7.5), layout="constrained")
    ax.spines[["top", "right"]].set_visible(False)
    ax.errorbar(h[keep], f[keep], yerr=e[keep], fmt="o", ms=10.0,
                color="C0", ecolor="C0", elinewidth=.65,
                capsize=0, rasterized=True)
    ax.plot(mh[mk], mf[mk], color="#111111", lw=4.0)
    ax.set(xlim=(center - t0 - half, center - t0 + half))
    fig.savefig(OUT.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), facecolor="white")


if __name__ == "__main__":
    main()
