#!/usr/bin/env python3
"""Scatter plots of the jacscanomaly t_c--Δχ² grid and overlap peaks."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from jacscanomaly import Finder, FinderConfig


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "roman_simu/anomaly_finder_result/planet_signal_data/planet_signal_result_602_345.json"
OUTDIR = Path(__file__).resolve().parent


def run_scan():
    d = json.loads(DATA.read_text())
    s = d["series"]
    time = np.asarray(s["time"], dtype=float)
    flux = np.asarray(s["flux"], dtype=float) * 1.0e9
    ferr = np.asarray(s["ferr"], dtype=float) * 1.0e9
    fit = d["fit"]
    x0 = np.asarray([fit["t0"], fit["tE"], fit["u0"]], dtype=float)
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
    raw = np.asarray(result.grid_metrics_all[:, [0, 2]], dtype=float)
    overlap = np.asarray(result.clusters_all[:, [0, 2]], dtype=float)
    return raw, overlap


def make_plot(points: np.ndarray, y_limits: tuple[float, float], stem: str, marker_size: float, alpha: float) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 20,
        "xtick.labelsize": 25,
        "ytick.labelsize": 25,
    })
    fig, ax = plt.subplots(figsize=(12.5, 5.8), layout="constrained")
    positive = np.maximum(points[:, 1], np.finfo(float).tiny)
    log_values = np.log10(positive)
    lo = float(np.min(log_values))
    hi = float(np.max(log_values))
    scale = (log_values - lo) / max(hi - lo, 1.0e-12)
    sizes = marker_size * (0.65 + 1.35 * scale)
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=sizes,
        color="C0",
        alpha=alpha,
        linewidths=0.0,
        rasterized=True,
    )
    ax.set_yscale("log")
    ax.set_xlim(float(np.min(points[:, 0])), float(np.max(points[:, 0])))
    ax.set_ylim(*y_limits)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=6, width=1.0)
    fig.savefig(OUTDIR / f"{stem}.png", dpi=300, facecolor="white")
    fig.savefig(OUTDIR / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def make_overlay(raw: np.ndarray, overlap: np.ndarray, y_limits: tuple[float, float]) -> None:
    """Draw the complete grid with the overlap-thinned peaks overlaid in red."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 20,
        "xtick.labelsize": 25,
        "ytick.labelsize": 25,
    })
    fig, ax = plt.subplots(figsize=(12.5, 5.8), layout="constrained")
    ax.scatter(
        raw[:, 0],
        raw[:, 1],
        s=5.0,
        color="C0",
        alpha=0.20,
        linewidths=0.0,
        rasterized=True,
        zorder=1,
    )
    positive = np.maximum(overlap[:, 1], np.finfo(float).tiny)
    log_values = np.log10(positive)
    lo = float(np.min(log_values))
    hi = float(np.max(log_values))
    scale = (log_values - lo) / max(hi - lo, 1.0e-12)
    sizes = 72.0 * (0.65 + 1.35 * scale)
    ax.scatter(
        overlap[:, 0],
        overlap[:, 1],
        s=sizes,
        color="#D62728",
        alpha=0.98,
        linewidths=0.0,
        zorder=3,
    )
    ax.set_yscale("log")
    ax.set_xlim(float(np.min(raw[:, 0])), float(np.max(raw[:, 0])))
    ax.set_ylim(*y_limits)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=6, width=1.0)
    fig.savefig(OUTDIR / "0_2_345_tc_dchi2_scatter_overlay.png", dpi=300, facecolor="white")
    fig.savefig(OUTDIR / "0_2_345_tc_dchi2_scatter_overlay.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    raw, overlap = run_scan()
    positive = raw[:, 1][raw[:, 1] > 0.0]
    raw_limits = (max(float(np.min(positive)) * 0.8, 1.0e-5), float(np.max(positive)) * 1.40)
    overlap_positive = overlap[:, 1][overlap[:, 1] > 0.0]
    overlap_limits = (
        max(float(np.min(overlap_positive)) * 0.8, 1.0e-2),
        float(np.max(overlap_positive)) * 1.35,
    )
    make_plot(raw, overlap_limits, "0_2_345_tc_dchi2_scatter_raw", marker_size=9.0, alpha=0.40)
    make_plot(overlap, overlap_limits, "0_2_345_tc_dchi2_scatter_overlap", marker_size=48.0, alpha=0.95)
    make_overlay(raw, overlap, overlap_limits)
    np.savez_compressed(
        OUTDIR / "0_2_345_tc_dchi2_scatter_data.npz",
        raw_grid=raw,
        overlap_peaks=overlap,
    )
    print(f"raw_points={raw.shape[0]} overlap_points={overlap.shape[0]}")


if __name__ == "__main__":
    main()
