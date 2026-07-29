#!/usr/bin/env python3
"""Residual zoom for event 0_128_2203 with jacscanomaly feature markers."""

from pathlib import Path
import json

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jacscanomaly import Finder, FinderConfig, PlanetSignalConfig, PlanetSignalExtractor
from jacscanomaly.singlelens_model import A_pspl_func


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "roman_simu/anomaly_finder_result/planet_signal_data/planet_signal_result_121_2203.json"
OUTDIR = Path(__file__).resolve().parent


def main() -> None:
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    series = payload["series"]
    time = np.asarray(series["time"], dtype=float)
    flux = np.asarray(series["flux"], dtype=float) * 1.0e9
    ferr = np.asarray(series["ferr"], dtype=float) * 1.0e9

    input_fit = payload["fit"]
    params = np.asarray([input_fit["t0"], input_fit["tE"], input_fit["u0"]], dtype=float)
    fs = float(input_fit["fs"]) * 1.0e9
    fb = float(input_fit["fb"]) * 1.0e9
    model = fs * np.asarray(A_pspl_func(jnp.asarray(params), jnp.asarray(time))) + fb
    residual = flux - model

    extractor = PlanetSignalExtractor(
        Finder(FinderConfig(fitter_kind="pspl", single_fit_backend="cpp", grid_backend="cpp")),
        PlanetSignalConfig(baseline_mode="beam_interval", seed_min_dchi2=100.0, max_iter=3),
    )
    signal = extractor.run(time, flux, ferr, x0=params, refit=False)
    features = signal.measure_features()

    t0 = float(input_fit["t0"])
    x = time - t0
    if features.features:
        center_time = float(features.strongest.time)
        max_duration = max(float(f.timescale) for f in features.features)
        zoom_half = max(12.0, 0.6 * max_duration)
    else:
        center_time = float(payload["candidates"][0]["peak_time"])
        zoom_half = 4.0
    center_x = center_time - t0
    xlim = (center_x - zoom_half, center_x + zoom_half)
    keep = (x >= xlim[0]) & (x <= xlim[1])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 20,
        "xtick.labelsize": 25,
        "ytick.labelsize": 25,
        "axes.linewidth": 0.9,
    })
    fig, ax = plt.subplots(figsize=(12.5, 6.5), layout="constrained")
    for feature in features.features:
        ax.axvspan(
            float(feature.t_start) - t0,
            float(feature.t_end) - t0,
            color="#D62728", alpha=0.16, zorder=0,
        )
        ax.axvline(float(feature.time) - t0, color="#D62728", lw=3.0, zorder=2)
    ax.axhline(0.0, color="#111111", lw=3.5, zorder=1)
    ax.errorbar(
        x[keep], residual[keep], yerr=ferr[keep], fmt="o", ms=10.0,
        color="C0", ecolor="C0", elinewidth=0.65,
        capsize=0, rasterized=True, zorder=3,
    )
    rlo = min(0.0, float(np.min(residual[keep] - ferr[keep])))
    rhi = max(0.0, float(np.max(residual[keep] + ferr[keep])))
    rpad = 0.08 * (rhi - rlo)
    ax.set(xlim=xlim, ylim=(rlo - rpad, rhi + rpad))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=4, width=0.8)
    fig.savefig(OUTDIR / "0_128_2203_input_pspl_residual_zoom.png", dpi=300, facecolor="white")
    fig.savefig(OUTDIR / "0_128_2203_input_pspl_residual_zoom.pdf", facecolor="white")
    plt.close(fig)
    print("features", features.feature_dicts())


if __name__ == "__main__":
    main()
