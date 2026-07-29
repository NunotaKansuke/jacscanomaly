#!/usr/bin/env python3
"""Run jacscanomaly's local-template grid and plot the t0--teff Δχ² map."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle

from jacscanomaly import Finder, FinderConfig


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "roman_simu/anomaly_finder_result/planet_signal_data/planet_signal_result_602_345.json"
OUTDIR = Path(__file__).resolve().parent
STEM = "0_2_345_t0_teff_dchi2_map"
ZOOM_STEM = "0_2_345_t0_teff_dchi2_map_zoom"
COMPOSITE_STEM = "0_2_345_t0_teff_dchi2_map_with_zoom"


def main() -> None:
    d = json.loads(DATA.read_text())
    series = d["series"]
    time = np.asarray(series["time"], dtype=float)
    flux = np.asarray(series["flux"], dtype=float) * 1.0e9
    ferr = np.asarray(series["ferr"], dtype=float) * 1.0e9
    fit = d["fit"]
    x0 = np.asarray([fit["t0"], fit["tE"], fit["u0"]], dtype=float)

    # These are the production jacscanomaly settings used for the event scan.
    # Keeping the PSPL geometry fixed makes the map a direct scan of the same
    # residual series used in the light-curve figures.
    config = FinderConfig(
        fitter_kind="pspl",
        single_fit_backend="cpp",
        grid_backend="cpp",
        gap=100.0,
        teff_init=0.03,
        common_ratio=4.0 / 3.0,
        teff_grid_n=24,
        dt0_coeff=0.17,
        teff_coeff=3.0,
        min_pts_in_window=4,
    )
    result = Finder(config).run(
        time,
        flux,
        ferr,
        x0=x0,
        refit=False,
        verbose=False,
    )
    metrics = np.asarray(result.grid_metrics_all, dtype=float)
    if metrics.ndim != 2 or metrics.shape[1] < 3 or metrics.size == 0:
        raise RuntimeError("jacscanomaly returned no grid metrics")

    # Columns are [t0, teff, dchi2, ...].  Use the grid's anomaly-center time
    # directly as t_c on the horizontal axis.
    x = metrics[:, 0]
    teff = metrics[:, 1]
    dchi2 = np.maximum(metrics[:, 2], 0.0)
    best_idx = int(np.argmax(dchi2))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 20,
        "xtick.labelsize": 25,
        "ytick.labelsize": 25,
    })
    fig, ax = plt.subplots(figsize=(15.0, 7.0), layout="constrained")

    # A mild power stretch keeps the weak part of the scan visible while
    # preserving the strong anomaly peak.  The plotted values remain Δχ².
    vmax = float(np.max(dchi2))
    norm = PowerNorm(gamma=0.38, vmin=0.0, vmax=vmax)
    points = ax.scatter(
        x,
        teff,
        c=dchi2,
        cmap="viridis",
        norm=norm,
        s=20.0,
        marker="s",
        linewidths=0.0,
        rasterized=True,
    )
    ax.scatter(
        [x[best_idx]],
        [teff[best_idx]],
        s=105.0,
        facecolors="none",
        edgecolors="black",
        linewidths=1.8,
        zorder=4,
    )

    ax.set_yscale("log")
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(float(np.min(teff)) * 0.9, float(np.max(teff)) * 1.1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=5, width=0.9)

    fig.savefig(OUTDIR / f"{STEM}.png", dpi=300, facecolor="white")
    fig.savefig(OUTDIR / f"{STEM}.pdf", facecolor="white")
    np.savez_compressed(
        OUTDIR / f"{STEM}.npz",
        grid_metrics=metrics,
        pspl_fit=np.asarray(x0, dtype=float),
        best_grid=metrics[best_idx],
    )
    plt.close(fig)

    # A large, signal-focused view.  This uses exactly the same grid values;
    # only the displayed t0 and teff range is cropped around the best point.
    zoom_x_half = 0.80
    zoom_teff_min = 0.025
    zoom_teff_max = 0.30
    full_teff_max = 1.0
    zoom_mask = (
        (np.abs(x - x[best_idx]) <= zoom_x_half)
        & (teff >= zoom_teff_min)
        & (teff <= zoom_teff_max)
    )
    if int(np.sum(zoom_mask)) == 0:
        raise RuntimeError("no grid points in the zoom range")

    xz = x[zoom_mask]
    tz = teff[zoom_mask]
    dz = dchi2[zoom_mask]
    zoom_vmax = float(np.max(dz))
    zoom_norm = PowerNorm(gamma=0.38, vmin=0.0, vmax=zoom_vmax)

    fig, ax = plt.subplots(figsize=(15.0, 6.5), layout="constrained")
    zoom_points = ax.scatter(
        xz,
        tz,
        c=dz,
        cmap="viridis",
        norm=zoom_norm,
        s=58.0,
        marker="s",
        linewidths=0.0,
        rasterized=True,
    )
    ax.scatter(
        [x[best_idx]],
        [teff[best_idx]],
        s=260.0,
        facecolors="none",
        edgecolors="black",
        linewidths=2.2,
        zorder=4,
    )
    ax.set_yscale("log")
    ax.set_xlim(float(x[best_idx] - zoom_x_half), float(x[best_idx] + zoom_x_half))
    ax.set_ylim(zoom_teff_min, zoom_teff_max)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=6, width=1.0)
    fig.savefig(OUTDIR / f"{ZOOM_STEM}.png", dpi=300, facecolor="white")
    fig.savefig(OUTDIR / f"{ZOOM_STEM}.pdf", facecolor="white")
    plt.close(fig)

    # Composite figure: the complete scan at left, with the displayed zoom
    # range boxed, and a second panel showing that same range enlarged.
    fig = plt.figure(figsize=(16.0, 7.0), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=(1.65, 1.0))
    ax_full = fig.add_subplot(gs[0, 0])
    ax_detail = fig.add_subplot(gs[0, 1])

    ax_full.scatter(
        x,
        teff,
        c=dchi2,
        cmap="viridis",
        norm=norm,
        s=17.0,
        marker="s",
        linewidths=0.0,
        rasterized=True,
    )
    ax_full.add_patch(
        Rectangle(
            (x[best_idx] - zoom_x_half, zoom_teff_min),
            2.0 * zoom_x_half,
            zoom_teff_max - zoom_teff_min,
            fill=False,
            edgecolor="black",
            linewidth=2.4,
            zorder=5,
        )
    )
    ax_full.set_yscale("log")
    ax_full.set_xlim(float(np.min(x)), float(np.max(x)))
    ax_full.set_ylim(float(np.min(teff)) * 0.9, float(np.max(teff)) * 1.1)
    ax_full.spines[["top", "right"]].set_visible(False)
    ax_full.tick_params(direction="out", length=6, width=1.0)

    ax_detail.scatter(
        xz,
        tz,
        c=dz,
        cmap="viridis",
        norm=norm,
        s=58.0,
        marker="s",
        linewidths=0.0,
        rasterized=True,
    )
    ax_detail.scatter(
        [x[best_idx]],
        [teff[best_idx]],
        s=260.0,
        facecolors="none",
        edgecolors="black",
        linewidths=2.2,
        zorder=4,
    )
    ax_detail.set_yscale("log")
    ax_detail.set_xlim(float(x[best_idx] - zoom_x_half), float(x[best_idx] + zoom_x_half))
    ax_detail.set_ylim(zoom_teff_min, zoom_teff_max)
    ax_detail.spines[["top", "right"]].set_visible(False)
    ax_detail.tick_params(direction="out", length=6, width=1.0)

    fig.savefig(OUTDIR / f"{COMPOSITE_STEM}.png", dpi=300, facecolor="white")
    fig.savefig(OUTDIR / f"{COMPOSITE_STEM}.pdf", facecolor="white")
    plt.close(fig)

    # Separate dense-cell versions.  The scan has a geometric teff grid and
    # a different t0 spacing on each row; drawing each row as cells (rather
    # than small point markers) removes the apparent vertical gaps.
    unique_teff = np.unique(teff)
    log_teff = np.log(unique_teff)
    log_edges = np.empty(log_teff.size + 1, dtype=float)
    log_edges[1:-1] = 0.5 * (log_teff[:-1] + log_teff[1:])
    log_edges[0] = log_teff[0] - 0.5 * (log_teff[1] - log_teff[0])
    log_edges[-1] = log_teff[-1] + 0.5 * (log_teff[-1] - log_teff[-2])
    teff_edges = np.exp(log_edges)

    def x_cell_edges(xrow: np.ndarray) -> np.ndarray:
        xrow = np.asarray(xrow, dtype=float)
        if xrow.size == 1:
            return np.asarray([xrow[0] - 0.01, xrow[0] + 0.01])
        mid = 0.5 * (xrow[:-1] + xrow[1:])
        return np.concatenate(([xrow[0] - (mid[0] - xrow[0])], mid,
                               [xrow[-1] + (xrow[-1] - mid[-1])]))

    def draw_dense_rows(ax, keep: np.ndarray, *, add_box: bool = False):
        mesh = None
        for i, tv in enumerate(unique_teff):
            row = keep & (teff == tv)
            if not np.any(row):
                continue
            order = np.argsort(x[row])
            xr = x[row][order]
            zr = dchi2[row][order]
            # Clip cell centers to the selected window before making edges.
            xe = x_cell_edges(xr)
            mesh = ax.pcolormesh(
                xe,
                np.asarray([teff_edges[i], teff_edges[i + 1]]),
                zr[None, :],
                cmap="viridis",
                norm=norm,
                shading="flat",
                rasterized=True,
            )
        if mesh is None:
            raise RuntimeError("no rows available for dense map")
        if add_box:
            ax.add_patch(Rectangle(
                (x[best_idx] - zoom_x_half, zoom_teff_min),
                2.0 * zoom_x_half,
                zoom_teff_max - zoom_teff_min,
                fill=False,
                edgecolor="black",
                linewidth=2.4,
                zorder=5,
            ))
        return mesh

    # Full event, separate file.
    fig, ax = plt.subplots(figsize=(15.0, 7.0), layout="constrained")
    dense_full = draw_dense_rows(
        ax,
        teff <= full_teff_max,
        add_box=True,
    )
    ax.set_yscale("log")
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(float(np.min(teff)) * 0.9, full_teff_max)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=6, width=1.0)
    fig.savefig(OUTDIR / f"{STEM}.png", dpi=300, facecolor="white")
    fig.savefig(OUTDIR / f"{STEM}.pdf", facecolor="white")
    plt.close(fig)

    # Signal-focused zoom, separate file.
    zoom_keep = (
        (np.abs(x - x[best_idx]) <= zoom_x_half)
        & (teff >= zoom_teff_min)
        & (teff <= zoom_teff_max)
    )
    fig, ax = plt.subplots(figsize=(15.0, 6.5), layout="constrained")
    dense_zoom = draw_dense_rows(ax, zoom_keep)
    ax.scatter(
        [x[best_idx]],
        [teff[best_idx]],
        s=260.0,
        facecolors="none",
        edgecolors="black",
        linewidths=2.2,
        zorder=4,
    )
    ax.set_yscale("log")
    ax.set_xlim(float(x[best_idx] - zoom_x_half), float(x[best_idx] + zoom_x_half))
    ax.set_ylim(zoom_teff_min, zoom_teff_max)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=6, width=1.0)
    fig.savefig(OUTDIR / f"{ZOOM_STEM}.png", dpi=300, facecolor="white")
    fig.savefig(OUTDIR / f"{ZOOM_STEM}.pdf", facecolor="white")
    plt.close(fig)

    print(
        f"grid_points={metrics.shape[0]} "
        f"best_t0={metrics[best_idx, 0]:.9f} "
        f"best_teff={metrics[best_idx, 1]:.9g} "
        f"best_dchi2={metrics[best_idx, 2]:.6f}"
    )


if __name__ == "__main__":
    main()
