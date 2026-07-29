#!/usr/bin/env python3
"""Full-season 0_952_1403 plots with jacscanomaly signal points removed."""

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
FIGSIZE = (8.0, 6.0)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTDIR / f"0_952_1403_{stem}.png", dpi=300, facecolor="white")
    fig.savefig(OUTDIR / f"0_952_1403_{stem}.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    series = payload["series"]
    time = np.asarray(series["time"], dtype=float)
    flux = np.asarray(series["flux"], dtype=float) * 1.0e9
    ferr = np.asarray(series["ferr"], dtype=float) * 1.0e9

    input_fit = payload["fit"]
    x0 = np.asarray([input_fit["t0"], input_fit["tE"], input_fit["u0"]], dtype=float)

    # Use the current jacscanomaly signal extractor to determine the points
    # excluded from the baseline fit.
    finder = Finder(FinderConfig(fitter_kind="pspl", single_fit_backend="cpp", grid_backend="cpp"))
    extractor = PlanetSignalExtractor(
        finder,
        PlanetSignalConfig(baseline_mode="beam_interval", seed_min_dchi2=100.0, max_iter=3),
    )
    signal = extractor.run(time, flux, ferr, x0=x0, refit=False)
    keep = ~np.asarray(signal.signal_mask, dtype=bool)

    # Refit the PSPL using only the non-anomalous measurements.
    refit = finder.fit_single_lens(time[keep], flux[keep], ferr[keep], x0=x0)
    refit_params = np.asarray(refit.params, dtype=float)
    refit_t0 = float(refit_params[0])
    refit_fs = float(refit.fs)
    refit_fb = float(refit.fb)

    input_params = x0
    input_fs = float(input_fit["fs"]) * 1.0e9
    input_fb = float(input_fit["fb"]) * 1.0e9
    xlim = (-50.0, 50.0)
    x_model = np.linspace(xlim[0], xlim[1], 3000)
    t_model = refit_t0 + x_model
    f_input = input_fs * np.asarray(A_pspl_func(jnp.asarray(input_params), jnp.asarray(t_model))) + input_fb
    f_refit = refit_fs * np.asarray(A_pspl_func(jnp.asarray(refit_params), jnp.asarray(t_model))) + refit_fb
    x_data = time[keep] - refit_t0

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 20,
        "xtick.labelsize": 25,
        "ytick.labelsize": 25,
        "axes.linewidth": 0.9,
    })

    ylo = float(np.min(flux - ferr))
    yhi = float(np.max(flux + ferr))
    ylo = min(ylo, float(np.min(f_input)), float(np.min(f_refit)))
    yhi = max(yhi, float(np.max(f_input)), float(np.max(f_refit)))
    ypad = 0.05 * (yhi - ylo)

    # Removed data with the original/input PSPL.
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    ax.plot(x_model, f_input, color="#D62728", lw=5.0, zorder=4)
    ax.errorbar(
        x_data, flux[keep], yerr=ferr[keep], fmt=".", ms=14.0,
        color="C0", ecolor="C0", elinewidth=0.55,
        alpha=0.95, rasterized=True, zorder=3,
    )
    ax.set(xlim=xlim, ylim=(ylo - ypad, yhi + ypad))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=4, width=0.8)
    save(fig, "single_lens_fit_anomaly_removed_input_pspl")

    # Removed data with the PSPL refit after masking the jacscanomaly signal.
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    ax.plot(x_model, f_refit, color="#111111", lw=5.0, zorder=4)
    ax.errorbar(
        x_data, flux[keep], yerr=ferr[keep], fmt=".", ms=14.0,
        color="C0", ecolor="C0", elinewidth=0.55,
        alpha=0.95, rasterized=True, zorder=3,
    )
    ax.set(xlim=xlim, ylim=(ylo - ypad, yhi + ypad))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=4, width=0.8)
    save(fig, "single_lens_fit_anomaly_removed_refit_pspl")

    # Keep every measurement visible: normal points remain C0 and the points
    # excluded by jacscanomaly are highlighted in orange.
    x_all = time - refit_t0
    anomaly = ~keep

    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    ax.plot(x_model, f_input, color="#D62728", lw=5.0, zorder=7)
    ax.errorbar(
        x_all[keep], flux[keep], yerr=ferr[keep], fmt=".", ms=14.0,
        color="C0", ecolor="C0", elinewidth=0.55,
        alpha=0.95, rasterized=True, zorder=3,
    )
    ax.errorbar(
        x_all[anomaly], flux[anomaly], yerr=ferr[anomaly], fmt=".", ms=14.0,
        color="#F28E2B", ecolor="#F28E2B", elinewidth=0.55,
        alpha=0.98, rasterized=True, zorder=5,
    )
    ax.set(xlim=xlim, ylim=(ylo - ypad, yhi + ypad))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=4, width=0.8)
    save(fig, "single_lens_fit_anomaly_mask_input_pspl")

    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    ax.plot(x_model, f_refit, color="#111111", lw=5.0, zorder=6)
    ax.errorbar(
        x_all[keep], flux[keep], yerr=ferr[keep], fmt=".", ms=14.0,
        color="C0", ecolor="C0", elinewidth=0.55,
        alpha=0.95, rasterized=True, zorder=3,
    )
    ax.errorbar(
        x_all[anomaly], flux[anomaly], yerr=ferr[anomaly], fmt=".", ms=14.0,
        color="#F28E2B", ecolor="#F28E2B", elinewidth=0.55,
        alpha=0.98, rasterized=True, zorder=5,
    )
    ax.set(xlim=xlim, ylim=(ylo - ypad, yhi + ypad))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=4, width=0.8)
    save(fig, "single_lens_fit_anomaly_mask_refit_pspl")

    (OUTDIR / "0_952_1403_anomaly_removed_fit.json").write_text(
        json.dumps({
            "n_total": int(time.size),
            "n_removed": int(np.sum(~keep)),
            "n_kept": int(np.sum(keep)),
            "refit_t0": float(refit_params[0]),
            "refit_tE": float(refit_params[1]),
            "refit_u0": float(refit_params[2]),
            "refit_Fs_scaled_1e9": refit_fs,
            "refit_Fb_scaled_1e9": refit_fb,
            "refit_chi2": float(refit.chi2),
            "refit_chi2_dof": float(refit.chi2_dof),
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print("removed", int(np.sum(~keep)), "kept", int(np.sum(keep)), "refit", refit.params)


if __name__ == "__main__":
    main()
