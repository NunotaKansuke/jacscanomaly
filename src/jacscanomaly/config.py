from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal


@dataclass(frozen=True)
class FinderConfig:
    """
    Configuration object for :class:`scanomaly.finder.Finder`.

    This dataclass collects **all hyperparameters controlling the anomaly-search
    pipeline**, excluding any numerical or model-dependent quantities.
    It is intentionally:

    - *Dependency-free* (no NumPy/JAX imports)
    - *Frozen* (immutable) for reproducibility
    - *Explicitly structured* according to pipeline stages

    The parameters are grouped according to the internal workflow of
    :class:`scanomaly.finder.Finder`:

    1. Season splitting
    2. Grid construction in (t0, teff)
    3. Grid scan and local evaluation
    4. Cluster extraction and selection

    Notes
    -----
    Parameters related to the **single-lens fitting model**
    (e.g. PSPL vs FSPL, parallax options, sky coordinates)
    are also placed here, so that a single configuration object fully
    defines the behavior of :class:`Finder`.
    """

    # ==================================================
    # 0) Single-lens fitter selection
    # ==================================================
    fitter_kind: Literal[
        "pspl",
        "fspl",
        "pspl_parallax",
        "fspl_parallax",
    ] = "pspl"
    """
    Choice of single-lens model used for the initial fit.

    Options
    -------
    - ``"pspl"`` :
        Point-Source Point-Lens (standard Paczyński curve).
    - ``"fspl"`` :
        Finite-Source Point-Lens (log-rho parameterization).
    - ``"pspl_parallax"`` :
        PSPL with annual parallax.
    - ``"fspl_parallax"`` :
        FSPL with annual parallax.
    """

    ra_deg: Optional[float] = None
    """Right ascension of the source (degrees). Required for parallax models."""

    dec_deg: Optional[float] = None
    """Declination of the source (degrees). Required for parallax models."""

    tref: Optional[float] = None
    """
    Reference time for annual parallax.

    If ``None``, the median observation time is used.
    """

    # ==================================================
    # 1) Season splitting
    # ==================================================
    gap: float = 100.0
    """
    Time gap threshold for season splitting.

    A new observing season is started whenever the time difference
    between consecutive data points exceeds this value.
    """

    # ==================================================
    # 2) Grid construction (t0, teff)
    # ==================================================
    teff_init: float = 0.03
    """
    Smallest effective timescale used in the grid.

    This is the first element of the geometric series defining the
    teff grid.
    """

    common_ratio: float = 4.0 / 3.0
    """
    Common ratio of the geometric progression used to generate teff values.
    """

    teff_grid_n: int = 20
    """
    Number of teff values in the grid.
    """

    dt0_coeff: float = 0.17
    """
    Grid spacing coefficient for the event time t0.

    The spacing is defined as::

        dt0 = dt0_coeff * teff
    """

    # ==================================================
    # 3) Grid scan (local evaluation window)
    # ==================================================
    sigma: float = 3.0
    """
    Threshold parameter used in per-point chi-square improvement tests.

    This parameter is kept mainly for compatibility with the original
    ``n_out``-based logic used in early versions of the algorithm.
    """

    teff_coeff: float = 3.0
    """
    Half-width of the local evaluation window in units of teff.

    For a grid point (t0, teff), the evaluation window is::

        [t0 - teff_coeff * teff, t0 + teff_coeff * teff]
    """

    min_pts_in_window: int = 4
    """
    Minimum number of data points required inside the local window
    to evaluate a grid point.
    """

    # ==================================================
    # 4) Cluster extraction
    # ==================================================
    overlap_sigma: float = 3.0
    """
    Overlap threshold used to group nearby grid points into clusters.

    Two grid points i and j are considered overlapping if::

        |t0_i - t0_j| < overlap_sigma * (teff_i + teff_j)
    """

    min_cluster_points: int = 3
    """
    Stop extracting clusters once the number of remaining grid points
    falls below this value.
    """
