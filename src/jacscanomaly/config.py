from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FinderConfig:
    """
    Configuration for :class:`scanomaly.finder.Finder`.

    This dataclass contains *only* hyperparameters that control the behavior of the
    anomaly search. It is intentionally dependency-free (no NumPy/JAX imports) and
    frozen for reproducibility.

    Parameter groups
    ----------------
    1) Season splitting:
       Split the time series into seasons based on large time gaps.

    2) Grid construction:
       Build a (t0, teff) grid per season.

    3) Grid scan:
       Evaluate delta-chi2 on the grid within a local window.

    4) Cluster extraction:
       Group overlapping candidates and pick the best per cluster.
    """

    # ==================================================
    # 1) Season splitting
    # ==================================================
    gap: float = 100.0
    """Time gap threshold for splitting seasons. A new season starts when dt > gap."""

    # ==================================================
    # 2) Grid construction (t0, teff)
    # ==================================================
    teff_init: float = 0.03
    """Initial teff value for the grid (first element of the geometric series)."""

    common_ratio: float = 4.0 / 3.0
    """Common ratio for the geometric series of teff values."""

    teff_grid_n: int = 5
    """Number of teff values in the grid."""

    dt0_coeff: float = 0.17
    """
    Grid spacing coefficient for t0:
        dt0 = dt0_coeff * teff
    """

    # ==================================================
    # 3) Grid scan (local evaluation window)
    # ==================================================
    sigma: float = 3.0
    """
    Threshold parameter used in counting per-point chi2 improvement.
    (Kept for compatibility with your original `n_out` logic.)
    """

    teff_coeff: float = 3.0
    """
    Window half-width multiplier in units of teff:
        window = [t0 - teff_coeff*teff, t0 + teff_coeff*teff]
    """

    min_pts_in_window: int = 4
    """Minimum number of data points required inside the window to evaluate a grid point."""

    # ==================================================
    # 4) Cluster extraction
    # ==================================================
    overlap_sigma: float = 3.0
    """
    Overlap threshold multiplier used to group nearby grid points into clusters:
        |t0_i - t0_j| < overlap_sigma * (teff_i + teff_j)
    """

    min_cluster_points: int = 3
    """
    Stop extracting clusters when the number of remaining grid points becomes
    smaller than this value.
    """
