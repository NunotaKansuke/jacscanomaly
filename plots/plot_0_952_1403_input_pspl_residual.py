#!/usr/bin/env python3
"""Zoomed residuals of the 0_952_1403 input PSPL model."""

from pathlib import Path
import json

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jacscanomaly import Finder, FinderConfig, PlanetSignalConfig, PlanetSignalExtractor
from jacscanomaly.singlelens_model import A_pspl_func


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "roman_simu/anomaly_finder_result/planet_signal_data/planet_signal_result_1339_1403.json"
OUTDIR = Path(__file__).resolve().parent


def main() -> None:
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    series = payload["series"]
    time = np.asarray(series["time"], dtype=float)
    flux = np.asarray(series["flux"], dtype=float) * 1.0e9
    ferr = np.asarray(series["ferr"], dtype=float) * 1.0e9

    # Residual relative to the PSPL supplied to the anomaly scan (the red
    # Input PSPL in the comparison light-curve figure).
    input_fit = payload["fit"]
    params = np.asarray(
        [input_fit["t0"], input_fit["tE"], input_fit["u0"]], dtype=float
    )
    fs = float(input_fit["fs"]) * 1.0e9
    fb = float(input_fit["fb"]) * 1.0e9
    A = np.asarray(A_pspl_func(jnp.asarray(params), jnp.asarray(time)))
    residual = flux - (fs * A + fb)

    # Measure the local feature width with the current jacscanomaly
    # implementation, instead of using the much wider candidate mask.
    extractor = PlanetSignalExtractor(
        Finder(FinderConfig(fitter_kind="pspl", single_fit_backend="cpp", grid_backend="cpp")),
        PlanetSignalConfig(baseline_mode="beam_interval", seed_min_dchi2=100.0, max_iter=3),
    )
    signal = extractor.run(time, flux, ferr, x0=params, refit=False)
    features = signal.measure_features()

    # Keep the horizontal coordinate identical to the refit light-curve
    # comparison figure (time relative to the all-point refitted t0).
    refit_path = OUTDIR / "0_952_1403_pspl_refit_params.json"
    refit_t0 = float(json.loads(refit_path.read_text())["t0"]) if refit_path.exists() else float(input_fit["t0"])
    candidate = payload["candidates"][0]
    # There are two local positive features in this anomaly; retain both
    # rather than selecting only the one nearest the saved candidate peak.
    anomaly_features = tuple(features.features)
    zoom_half = 10.0
    x = time - refit_t0
    xlim = (-zoom_half, zoom_half)
    keep = (x >= xlim[0]) & (x <= xlim[1])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 20,
        "xtick.labelsize": 25,
        "ytick.labelsize": 25,
        "axes.linewidth": 0.9,
    })
    fig, ax = plt.subplots(figsize=(12.5, 6.5), layout="constrained")
    # jacscanomaly feature intervals and their anomaly-time markers.
    for feature in anomaly_features:
        ax.axvspan(
            float(feature.t_start) - refit_t0,
            float(feature.t_end) - refit_t0,
            color="#D62728", alpha=0.16, zorder=0,
        )
        ax.axvline(
            float(feature.time) - refit_t0,
            color="#D62728", lw=3.0, zorder=2,
        )
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
    fig.savefig(OUTDIR / "0_952_1403_input_pspl_residual_zoom.png", dpi=300, facecolor="white")
    fig.savefig(OUTDIR / "0_952_1403_input_pspl_residual_zoom.pdf", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
