from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class CandidateQuality:
    """
    Per-candidate quality diagnostics derived from per-point chi-square improvement.

    Attributes
    ----------
    n_window : int
        Number of points inside the local chi2 evaluation window.
    n_contrib : int
        Number of points with improvement above the configured threshold.
    n_eff : float
        Effective number of contributing points (participation-ratio style).
    peak_frac : float
        Fraction of total positive improvement carried by the strongest point.
    rho1 : float
        Lag-1 autocorrelation of signed per-point improvements within the window.
    longest_run : int
        Longest consecutive run length among above-threshold contributing points.
    """
    n_window: int
    n_contrib: int
    n_eff: float
    peak_frac: float
    rho1: float
    longest_run: int


@dataclass(frozen=True)
class BestCandidate:
    """
    Best anomaly candidate selected from all extracted clusters.

    Attributes
    ----------
    t0 : float
        Candidate center time.
    teff : float
        Candidate effective timescale.
    dchi2 : float
        Improvement in chi-square: chi2_null - chi2_anom (larger is better).
    med_others : float
        Median dchi2 among the bulk of the other candidates
        (excluding the best, with the upper tail trimmed when configured).
    std_others : float
        Standard deviation of dchi2 among the bulk of the other candidates
        (excluding the best, with the upper tail trimmed when configured).
    score : float
        Standardized score of the best candidate.
        Computed as ``(dchi2_best - med_others) / std_others``.
        (may be NaN/inf depending on the number of candidates / std_others).
    quality : CandidateQuality
        Per-point support and temporal diagnostics for this candidate.
    """
    t0: float
    teff: float
    dchi2: float
    med_others: float
    std_others: float
    score: float
    quality: CandidateQuality


@dataclass(frozen=True)
class SeasonSummary:
    """
    Summary of the anomaly scan for a single season.

    Attributes
    ----------
    season_idx : int
        0-based season index.
    t_start, t_end : float
        Time range of the season.
    n_grid : int
        Number of grid points evaluated in this season.
    clusters : np.ndarray
        Extracted clusters for this season, shape (K, 3) with rows [t0, teff, dchi2].
    grid_metrics : np.ndarray
        Raw per-grid diagnostics, shape (N, 9), columns:
        [t0, teff, dchi2, n_window, n_contrib, n_eff, peak_frac, rho1, longest_run].
    """
    season_idx: int
    t_start: float
    t_end: float
    n_grid: int
    clusters: np.ndarray  # shape (K,3): [t0, teff, dchi2]
    grid_metrics: np.ndarray  # shape (N,9): [t0,teff,dchi2,n_window,n_contrib,n_eff,peak_frac,rho1,longest_run]


@dataclass(frozen=True)
class AnomalyResult:
    """
    Output of :meth:`scanomaly.finder.Finder.run`.

    This object is designed to be convenient for plotting and downstream analysis.
    Arrays are stored on CPU as NumPy arrays.

    Attributes
    ----------
    time, flux, ferr : np.ndarray
        Input light curve arrays (1D).
    fit : SingleLensFitResult
        PSPL fitting result (contains params, fs, fb, chi2, model_flux, residual, etc.).
    residual : np.ndarray
        Flux residuals on CPU: flux - model_flux.
    model_flux : np.ndarray
        PSPL model flux on CPU.
    chi2_dof : float
        Reduced chi-square of the PSPL fit.
    seasons : list[SeasonSummary]
        Per-season summaries including clusters.
    clusters_all : np.ndarray
        Flattened clusters across all seasons, shape (N, 3) with rows [t0, teff, dchi2].
    grid_metrics_all : np.ndarray
        Flattened per-grid diagnostics, shape (M, 9), columns:
        [t0, teff, dchi2, n_window, n_contrib, n_eff, peak_frac, rho1, longest_run].
    best : BestCandidate | None
        Best candidate over all clusters, or None if no candidate exists.
    """
    # input (CPU numpy arrays for fast plotting)
    time: np.ndarray
    flux: np.ndarray
    ferr: np.ndarray

    # PSPL fit
    fit: SingleLensFitResult
    residual: np.ndarray
    model_flux: np.ndarray
    chi2_dof: float

    # grid/clusters
    seasons: List[SeasonSummary]
    clusters_all: np.ndarray  # shape (N,3)
    grid_metrics_all: np.ndarray  # shape (M,9)

    # best candidate
    best: Optional[BestCandidate]

    def summary_dict(self) -> Dict[str, Any]:
        """
        Return a compact summary dictionary suitable for logging/serialization.
        """
        out: Dict[str, Any] = {
            "n_points": int(self.time.size),
            "n_seasons": int(len(self.seasons)),
            "n_clusters": int(self.clusters_all.shape[0]),
            "n_grid_total": int(sum(s.n_grid for s in self.seasons)),
            "chi2_dof": float(self.chi2_dof),
            "has_best": bool(self.best is not None),
        }
        if self.best is None:
            return out

        b = self.best
        q = b.quality
        out.update(
            {
                "best_t0": float(b.t0),
                "best_teff": float(b.teff),
                "best_dchi2": float(b.dchi2),
                "best_score": float(b.score),
                "best_n_window": int(q.n_window),
                "best_n_contrib": int(q.n_contrib),
                "best_n_eff": float(q.n_eff),
                "best_peak_frac": float(q.peak_frac),
                "best_rho1": float(q.rho1),
                "best_longest_run": int(q.longest_run),
            }
        )
        return out

    def summary_text(self) -> str:
        """
        Return a CLI-friendly multi-line summary.
        """
        d = self.summary_dict()
        lines = [
            "=== jacscanomaly summary ===",
            f"points      : {d['n_points']}",
            f"seasons     : {d['n_seasons']}",
            f"grid total  : {d['n_grid_total']}",
            f"clusters    : {d['n_clusters']}",
            f"chi2 / dof  : {d['chi2_dof']:.3f}",
        ]

        if not d["has_best"]:
            lines.append("best        : None")
            return "\n".join(lines)

        lines.extend(
            [
                "",
                "=== best candidate ===",
                f"t0          : {d['best_t0']:.6f}",
                f"teff        : {d['best_teff']:.6g}",
                f"dchi2       : {d['best_dchi2']:.6g}",
                f"score       : {d['best_score']:.3f}",
                "",
                "=== quality ===",
                f"n_window    : {d['best_n_window']}",
                f"n_contrib   : {d['best_n_contrib']}",
                f"n_eff       : {d['best_n_eff']:.3f}",
                f"peak_frac   : {d['best_peak_frac']:.3f}",
                f"rho1        : {d['best_rho1']:.3f}",
                f"longest_run : {d['best_longest_run']}",
            ]
        )
        return "\n".join(lines)

    def print_summary(self) -> None:
        """
        Print a CLI-friendly summary.
        """
        print(self.summary_text())

    def summary_table(self):
        """
        Return a notebook-friendly single-row table.

        Returns
        -------
        pandas.DataFrame | list[dict]
            DataFrame when pandas is available, otherwise a single-item list.
        """
        row = self.summary_dict()
        try:
            import pandas as pd
            return pd.DataFrame([row])
        except Exception:
            return [row]

    def __str__(self) -> str:
        return self.summary_text()
