#!/usr/bin/env python3
"""Wide residual view with the scan-template and flat fits for 0_2_345."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from matplotlib.ticker import MaxNLocator
from jacscanomaly.anomaly_models import get_anom_plot_model_masked


ROOT = Path(__file__).resolve().parents[1]
SIGNAL_JSON = ROOT.parent / "roman_simu/anomaly_finder_result/planet_signal_data/planet_signal_result_602_345.json"
SCAN_DAT = ROOT.parent / "roman_simu/anomaly_finder_result/result_0000-2370.dat"
OUTDIR = Path(__file__).resolve().parent


def weighted_linear_fit(design: np.ndarray, y: np.ndarray, err: np.ndarray) -> np.ndarray:
    w = 1.0 / np.maximum(err, 1e-30) ** 2
    return np.linalg.solve((design.T * w) @ design, (design.T * w) @ y)


def main() -> None:
    d = json.loads(SIGNAL_JSON.read_text())
    s = d["series"]
    t = np.asarray(s["time"], dtype=float)
    residual = np.asarray(s["residual"], dtype=float) * 1e9
    ferr = np.asarray(s["ferr"], dtype=float) * 1e9

    # The scan's anomaly-template location and effective width are the same
    # values used by the original template search for this event.
    scan_row = next(line.split() for line in SCAN_DAT.read_text().splitlines()
                    if line.strip().startswith("602 "))
    t_anom, teff = float(scan_row[5]), float(scan_row[6])
    mask = np.abs(t - t_anom) <= 3.0 * teff

    # Flat residual model: weighted mean in the template evaluation window.
    w = 1.0 / np.maximum(ferr[mask], 1e-30) ** 2
    flat_level = float(np.sum(w * residual[mask]) / np.sum(w))

    # Two standard anomaly templates; retain the lower-chi2 one, as in the
    # library's masked anomaly model.
    q = 1.0 + ((t[mask] - t_anom) / teff) ** 2
    a0 = 1.0 / np.sqrt(q)
    a1 = (q + 2.0) / np.sqrt(q * (q + 4.0))
    beta0 = weighted_linear_fit(np.column_stack((a0, np.ones_like(a0))), residual[mask], ferr[mask])
    beta1 = weighted_linear_fit(np.column_stack((a1, np.ones_like(a1))), residual[mask], ferr[mask])
    chi0 = np.sum(((residual[mask] - (beta0[0] * a0 + beta0[1])) / ferr[mask]) ** 2)
    chi1 = np.sum(((residual[mask] - (beta1[0] * a1 + beta1[1])) / ferr[mask]) ** 2)
    use_a1 = chi1 < chi0
    beta = beta1 if use_a1 else beta0
    template_kind = a1 if use_a1 else a0

    # Show only a little more than the template support, centered in the
    # original PSPL time coordinate (t - t0), rather than recentering on the
    # anomaly itself.
    t0_pspl = float(d["fit"]["t0"])
    template_half_width = 3.0 * teff
    half_width = template_half_width + 0.04
    visible = np.abs(t - t_anom) <= half_width
    tx = np.linspace(t_anom - 3.0 * teff, t_anom + 3.0 * teff, 500)
    qq = 1.0 + ((tx - t_anom) / teff) ** 2
    aa = (qq + 2.0) / np.sqrt(qq * (qq + 4.0)) if use_a1 else 1.0 / np.sqrt(qq)
    template_line = beta[0] * aa + beta[1]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
                         "xtick.labelsize": 24, "ytick.labelsize": 24})
    fig, ax = plt.subplots(figsize=(14.0, 4.0), layout="constrained")
    ax.spines[["top", "right"]].set_visible(False)
    ax.errorbar(t[visible] - t0_pspl, residual[visible], yerr=ferr[visible], fmt=".", ms=26.0,
                color="C0", ecolor="C0", elinewidth=0.45, alpha=0.9,
                rasterized=True, zorder=2)
    model_x = tx - t0_pspl
    ax.plot([model_x[0], model_x[-1]], [flat_level, flat_level],
            color="black", lw=9.0, zorder=4)
    ax.plot(model_x, template_line, color="red", lw=9.0, zorder=5)
    ymin = float(np.min(residual[visible] - ferr[visible]))
    ymax = float(np.max(residual[visible] + ferr[visible]))
    pad = 0.08 * max(ymax - ymin, 1e-6)
    x_center = t_anom - t0_pspl
    ax.set(xlim=(x_center - half_width, x_center + half_width), ylim=(ymin - pad, ymax + pad))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=9))
    fig.savefig(OUTDIR / "0_2_345_residual_template_flat.png", dpi=300, facecolor="white")
    fig.savefig(OUTDIR / "0_2_345_residual_template_flat.pdf", facecolor="white")
    plt.close(fig)

    # A quiet comparison region around the original PSPL t0, well separated
    # from the anomaly.  Only the local flat residual model is shown here.
    quiet_visible = np.abs(t - t0_pspl) <= half_width
    wq = 1.0 / np.maximum(ferr[quiet_visible], 1e-30) ** 2
    quiet_flat = float(np.sum(wq * residual[quiet_visible]) / np.sum(wq))
    fig, ax = plt.subplots(figsize=(14.0, 4.0), layout="constrained")
    ax.spines[["top", "right"]].set_visible(False)
    ax.errorbar(t[quiet_visible] - t0_pspl, residual[quiet_visible], yerr=ferr[quiet_visible],
                fmt=".", ms=26.0, color="C0", ecolor="C0", elinewidth=0.55,
                alpha=0.9, rasterized=True, zorder=2)
    ax.plot([-half_width, half_width], [quiet_flat, quiet_flat],
            color="black", lw=9.0, zorder=4)
    qmin = float(np.min(residual[quiet_visible] - ferr[quiet_visible]))
    qmax = float(np.max(residual[quiet_visible] + ferr[quiet_visible]))
    qpad = 0.08 * max(qmax - qmin, 1e-6)
    ax.set(xlim=(-half_width, half_width), ylim=(qmin - qpad, qmax + qpad))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=9))
    fig.savefig(OUTDIR / "0_2_345_residual_quiet_flat.png", dpi=300, facecolor="white")
    fig.savefig(OUTDIR / "0_2_345_residual_quiet_flat.pdf", facecolor="white")
    plt.close(fig)

    # Forced anomaly-template comparison in the same quiet interval, using
    # jacscanomaly's actual masked template (not a Gaussian surrogate).
    # Pick a quiet segment away from the real anomaly where the forced
    # jacscanomaly fit gives a positive bump-like excursion.
    forced_center = 8546.36
    template_mask = np.abs(t - forced_center) <= 3.0 * teff
    template_xmin = forced_center - t0_pspl - 3.0 * teff
    template_xmax = forced_center - t0_pspl + 3.0 * teff
    xq = t[template_mask] - t0_pspl
    yq = residual[template_mask]
    eq = ferr[template_mask]
    wq = 1.0 / np.maximum(eq, 1e-30) ** 2
    template_flat = float(np.sum(wq * yq) / np.sum(wq))
    t_plot = np.linspace(t0_pspl + template_xmin, t0_pspl + template_xmax, 500)
    template_j, _ = get_anom_plot_model_masked(
        jnp.asarray(t_plot), jnp.asarray(forced_center), jnp.asarray(teff),
        jnp.asarray(t), jnp.asarray(residual), 1.0 / jnp.asarray(ferr) ** 2,
        jnp.asarray(template_mask),
    )
    template_line = np.asarray(template_j)
    fig, ax = plt.subplots(figsize=(14.0, 4.0), layout="constrained")
    ax.spines[["top", "right"]].set_visible(False)
    ax.errorbar(xq, yq, yerr=eq, fmt=".", ms=26.0, color="C0", ecolor="C0",
                elinewidth=0.55, alpha=0.9, rasterized=True, zorder=2)
    ax.plot([template_xmin, template_xmax], [template_flat, template_flat],
            color="black", lw=9.0, zorder=4)
    ax.plot(np.linspace(template_xmin, template_xmax, 500), template_line,
            color="red", lw=9.0, zorder=5)
    bmin = float(min(np.min(yq - eq), np.min(template_line), template_flat))
    bmax = float(max(np.max(yq + eq), np.max(template_line), template_flat))
    bpad = 0.08 * max(bmax - bmin, 1e-6)
    x_margin = 0.02
    ax.set(xlim=(template_xmin - x_margin, template_xmax + x_margin),
           ylim=(bmin - bpad, bmax + bpad))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=9))
    fig.savefig(OUTDIR / "0_2_345_residual_quiet_forced_bump.png", dpi=300, facecolor="white")
    fig.savefig(OUTDIR / "0_2_345_residual_quiet_forced_bump.pdf", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
