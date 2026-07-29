#!/usr/bin/env python3
"""Refit PSPLs to the full saved light curves for two Roman events."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jacscanomaly import Finder, FinderConfig
from jacscanomaly.singlelens_model import A_pspl_func


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = Path(__file__).resolve().parent
FIGSIZE = (8.0, 6.0)
EVENTS = {
    "0_161_832": ROOT.parent / "roman_simu/anomaly_finder_result/planet_signal_data/planet_signal_result_1654_832.json",
    "0_952_1403": ROOT.parent / "roman_simu/anomaly_finder_result/planet_signal_data/planet_signal_result_1339_1403.json",
}


def save(fig: plt.Figure, event: str, stem: str) -> None:
    fig.savefig(OUTDIR / f"{event}_{stem}.png", dpi=300, facecolor="white")
    fig.savefig(OUTDIR / f"{event}_{stem}.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 20,
        "xtick.labelsize": 25,
        "ytick.labelsize": 25,
        "axes.linewidth": 0.9,
    })
    config = FinderConfig(
        fitter_kind="pspl",
        single_fit_backend="cpp",
        grid_backend="cpp",
    )

    for event, path in EVENTS.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        series = payload["series"]
        time = np.asarray(series["time"], dtype=float)
        flux = np.asarray(series["flux"], dtype=float) * 1.0e9
        ferr = np.asarray(series["ferr"], dtype=float) * 1.0e9
        signal_mask = np.asarray(series.get("signal_mask", np.zeros_like(time, dtype=bool)), dtype=bool)
        old_fit = payload["fit"]
        x0 = np.asarray([old_fit["t0"], old_fit["tE"], old_fit["u0"]], dtype=float)

        # Refit the PSPL to every saved point in the event light curve.
        fit = Finder(config).fit_single_lens(time, flux, ferr, x0=x0)
        params = np.asarray(fit.params, dtype=float)
        fs = float(fit.fs)
        fb = float(fit.fb)
        t0, tE, u0 = params[:3]

        x = time - t0

        # Full event view.  Keep the event-specific x ranges requested for the
        # corresponding light-curve figures.
        if event == "0_161_832":
            extent = max(abs(float(np.min(x))), abs(float(np.max(x))))
            xlim = (-extent, extent)
        else:
            xlim = (-50.0, 50.0)
        # Sample the model directly over the displayed x range so the black
        # curve reaches both plot edges, even when the saved data do not.
        xm = np.linspace(xlim[0], xlim[1], 2500)
        t_model = t0 + xm
        A_model = np.asarray(A_pspl_func(jnp.asarray(params), jnp.asarray(t_model)))
        f_model = fs * A_model + fb
        input_model = None
        if event == "0_952_1403":
            # The PSPL supplied to the anomaly scan (before the all-point
            # refit) is stored in the JSON ``fit`` block.
            input_fit = payload["fit"]
            input_params = np.asarray(
                [input_fit["t0"], input_fit["tE"], input_fit["u0"]], dtype=float
            )
            input_fs = float(input_fit["fs"]) * 1.0e9
            input_fb = float(input_fit["fb"]) * 1.0e9
            A_input = np.asarray(A_pspl_func(jnp.asarray(input_params), jnp.asarray(t_model)))
            input_model = input_fs * A_input + input_fb

        fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
        ax.plot(xm, f_model, color="#111111", lw=4.0, zorder=5)
        ax.errorbar(
            x, flux, yerr=ferr, fmt=".", ms=14.0,
            color="C0", ecolor="C0", elinewidth=0.55,
            alpha=0.95, rasterized=True, zorder=3,
        )
        ylo = min(float(np.min(flux - ferr)), float(np.min(f_model)))
        yhi = max(float(np.max(flux + ferr)), float(np.max(f_model)))
        if input_model is not None:
            ylo = min(ylo, float(np.min(input_model)))
            yhi = max(yhi, float(np.max(input_model)))
        ypad = 0.05 * (yhi - ylo)
        ax.set(xlim=xlim, ylim=(ylo - ypad, yhi + ypad))
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out", length=4, width=0.8)
        save(fig, event, "single_lens_fit_refit")
        # A companion version with only the candidate-anomaly points in orange.
        ax.errorbar(
            x[signal_mask], flux[signal_mask], yerr=ferr[signal_mask], fmt=".", ms=14.0,
            color="#F28E2B", ecolor="#F28E2B", elinewidth=0.55,
            alpha=0.98, rasterized=True, zorder=4,
        )
        save(fig, event, "single_lens_fit_refit_anomaly_orange")

        if input_model is not None:
            # Separate comparison figures: thick red = input PSPL,
            # solid black = all-point refit PSPL.
            fig_in, ax_in = plt.subplots(figsize=FIGSIZE, layout="constrained")
            ax_in.plot(xm, input_model, color="#D62728", lw=5.0, zorder=7)
            ax_in.plot(xm, f_model, color="#111111", lw=4.0, zorder=5)
            ax_in.errorbar(
                x, flux, yerr=ferr, fmt=".", ms=14.0,
                color="C0", ecolor="C0", elinewidth=0.55,
                alpha=0.95, rasterized=True, zorder=3,
            )
            ax_in.set(xlim=xlim, ylim=(ylo - ypad, yhi + ypad))
            ax_in.spines[["top", "right"]].set_visible(False)
            ax_in.tick_params(direction="out", length=4, width=0.8)
            save(fig_in, event, "single_lens_fit_refit_with_input_pspl")
            ax_in.errorbar(
                x[signal_mask], flux[signal_mask], yerr=ferr[signal_mask], fmt=".", ms=14.0,
                color="#F28E2B", ecolor="#F28E2B", elinewidth=0.55,
                alpha=0.98, rasterized=True, zorder=6,
            )
            save(fig_in, event, "single_lens_fit_refit_with_input_pspl_anomaly_orange")

        # Candidate-centered view using the same refitted PSPL curve.
        candidate = payload["candidates"][0]
        t_anom = float(candidate["peak_time"])
        zoom_half = 0.19
        keep = np.abs(time - t_anom) <= zoom_half
        zoom_xlim = (t_anom - t0 - zoom_half, t_anom - t0 + zoom_half)
        # Again sample exactly across the zoom limits rather than clipping a
        # coarser full-range model grid.
        x_zoom_model = np.linspace(zoom_xlim[0], zoom_xlim[1], 1200)
        t_zoom_model = t0 + x_zoom_model
        A_zoom_model = np.asarray(A_pspl_func(jnp.asarray(params), jnp.asarray(t_zoom_model)))
        f_zoom_model = fs * A_zoom_model + fb
        f_zoom_input = None
        if input_model is not None:
            A_zoom_input = np.asarray(A_pspl_func(jnp.asarray(input_params), jnp.asarray(t_zoom_model)))
            f_zoom_input = input_fs * A_zoom_input + input_fb
        fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
        ax.errorbar(
            time[keep] - t0, flux[keep], yerr=ferr[keep], fmt="o", ms=10.0,
            color="C0", ecolor="C0", elinewidth=0.65,
            capsize=0, rasterized=True, zorder=3,
        )
        ax.plot(x_zoom_model, f_zoom_model, color="#111111", lw=4.0, zorder=2)
        ax.set(xlim=zoom_xlim)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out", length=4, width=0.8)
        save(fig, event, "weak_planet_signal_refit")
        ax.errorbar(
            time[keep & signal_mask] - t0, flux[keep & signal_mask],
            yerr=ferr[keep & signal_mask], fmt="o", ms=10.0,
            color="#F28E2B", ecolor="#F28E2B", elinewidth=0.65,
            capsize=0, rasterized=True, zorder=4,
        )
        save(fig, event, "weak_planet_signal_refit_anomaly_orange")

        if f_zoom_input is not None:
            fig_in, ax_in = plt.subplots(figsize=FIGSIZE, layout="constrained")
            ax_in.plot(x_zoom_model, f_zoom_input, color="#D62728", lw=5.0, zorder=7)
            ax_in.plot(x_zoom_model, f_zoom_model, color="#111111", lw=4.0, zorder=2)
            ax_in.errorbar(
                time[keep] - t0, flux[keep], yerr=ferr[keep], fmt="o", ms=10.0,
                color="C0", ecolor="C0", elinewidth=0.65,
                capsize=0, rasterized=True, zorder=3,
            )
            ax_in.set(xlim=zoom_xlim)
            ax_in.spines[["top", "right"]].set_visible(False)
            ax_in.tick_params(direction="out", length=4, width=0.8)
            save(fig_in, event, "weak_planet_signal_refit_with_input_pspl")
            ax_in.errorbar(
                time[keep & signal_mask] - t0, flux[keep & signal_mask],
                yerr=ferr[keep & signal_mask], fmt="o", ms=10.0,
                color="#F28E2B", ecolor="#F28E2B", elinewidth=0.65,
                capsize=0, rasterized=True, zorder=4,
            )
            save(fig_in, event, "weak_planet_signal_refit_with_input_pspl_anomaly_orange")

        params_out = {
            "event": event,
            "n_points": int(time.size),
            "t0": float(t0),
            "tE": float(tE),
            "u0": float(u0),
            "Fs_scaled_1e9": fs,
            "Fb_scaled_1e9": fb,
            "chi2": float(fit.chi2),
            "chi2_dof": float(fit.chi2_dof),
        }
        (OUTDIR / f"{event}_pspl_refit_params.json").write_text(
            json.dumps(params_out, indent=2) + "\n",
            encoding="utf-8",
        )
        print(event, params_out)


if __name__ == "__main__":
    main()
