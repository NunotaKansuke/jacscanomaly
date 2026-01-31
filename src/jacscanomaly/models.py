from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

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
        Median dchi2 among all other candidates (excluding the best).
    std_others : float
        Standard deviation of dchi2 among all other candidates (excluding the best).
    score : float
        Standardized score of the best candidate:
            (dchi2_best - med_others) / std_others
        (may be NaN/inf depending on the number of candidates / std_others).
    """
    t0: float
    teff: float
    dchi2: float
    med_others: float
    std_others: float
    score: float


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
    """
    season_idx: int
    t_start: float
    t_end: float
    n_grid: int
    clusters: np.ndarray  # shape (K,3): [t0, teff, dchi2]


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
    fit : PSPLFitResult
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
    best : BestCandidate | None
        Best candidate over all clusters, or None if no candidate exists.
    """
    # input (CPU numpy arrays for fast plotting)
    time: np.ndarray
    flux: np.ndarray
    ferr: np.ndarray

    # PSPL fit
    fit: PSPLFitResult
    residual: np.ndarray
    model_flux: np.ndarray
    chi2_dof: float

    # grid/clusters
    seasons: List[SeasonSummary]
    clusters_all: np.ndarray  # shape (N,3)

    # best candidate
    best: Optional[BestCandidate]
