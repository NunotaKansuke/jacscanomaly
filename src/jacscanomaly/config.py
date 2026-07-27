from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

from .criteria import CandidateCriteria


class _DisplayFloat(float):
    """Float default with a concise repr for generated signatures."""

    def __new__(cls, value: float, repr_text: str):
        obj = super().__new__(cls, value)
        obj._repr_text = repr_text
        return obj

    def __repr__(self) -> str:
        return self._repr_text


_COMMON_RATIO_DEFAULT = _DisplayFloat(4.0 / 3.0, "4.0 / 3.0")


@dataclass(frozen=True)
class FinderConfig:
    """
    Configuration object for :class:`jacscanomaly.finder.Finder`.

    This dataclass collects **all hyperparameters controlling the anomaly-search
    pipeline**, excluding any numerical or model-dependent quantities.
    It is intentionally:

    - *Dependency-free* (no NumPy/JAX imports)
    - *Frozen* (immutable) for reproducibility
    - *Explicitly structured* according to pipeline stages

    The parameters are grouped according to the internal workflow of
    :class:`jacscanomaly.finder.Finder`:

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
        "fspl_vbm_fd",
        "pspl_parallax",
        "fspl_parallax",
        "pspl_space_parallax",
        "fspl_space_parallax",
        "fspl_space_parallax_gulls_vbm_fd",
        "bic_single_lens",
    ] = "pspl"
    """
    Choice of single-lens model used for the initial fit.

    Options
    -------
    - ``"pspl"`` :
        Point-Source Point-Lens (standard Paczyński curve).
    - ``"fspl"`` :
        Finite-Source Point-Lens (log-rho parameterization).
    - ``"fspl_vbm_fd"`` :
        Finite-Source Point-Lens using VBMicrolensing ESPL magnification and
        finite-difference SciPy least squares.
    - ``"pspl_parallax"`` :
        PSPL with annual parallax.
    - ``"fspl_parallax"`` :
        FSPL with annual parallax.
    - ``"pspl_space_parallax"`` :
        PSPL with annual parallax plus a spacecraft ephemeris.
    - ``"fspl_space_parallax"`` :
        FSPL with annual parallax plus a spacecraft ephemeris.
    - ``"fspl_space_parallax_gulls_vbm_fd"`` :
        GULLS-convention FSPL space-parallax fitter using VBMicrolensing ESPL
        magnification and finite-difference SciPy least squares.
    - ``"bic_single_lens"`` :
        Select the lowest-BIC fit among PSPL and finite-difference FSPL, with
        an optional GULLS FSPL space-parallax trial.
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

    satellite_ephemeris_path: Optional[str] = None
    """
    Path to the spacecraft/observer ephemeris table used by space-parallax
    models.

    Required for ``"pspl_space_parallax"``, ``"fspl_space_parallax"``,
    ``"fspl_space_parallax_gulls_vbm_fd"``, and ``"bic_single_lens"`` when
    ``bic_include_space_parallax`` is enabled. Expected columns are
    ``JD RA_deg Dec_deg distance_AU``. The interpretation of the position is
    controlled by ``space_parallax_convention``.
    """

    space_parallax_convention: Literal["vbm", "gulls"] = "vbm"
    """
    Convention used for ``"*_space_parallax"`` models.

    - ``"vbm"`` interprets ``satellite_ephemeris_path`` as a geocentric
      VBMicrolensing/RTModel satellite table and adds it to the annual Earth
      parallax displacement.
    - ``"gulls"`` interprets ``satellite_ephemeris_path`` as a heliocentric
      GULLS observer ephemeris and uses the GULLS reference-frame subtraction
      ``x(t) - x(tref) - v(tref) * (t - tref)``.
    """

    parallax_use_HJD: bool = True
    """
    If True, input times for parallax models are treated as HJD-style times.
    If False, input times are treated as JD-style times and the observer
    light-travel correction is included in the source trajectory, matching
    VBMicrolensing's ``t_in_HJD=0`` convention.
    """

    max_piE: float = 1.0
    """Symmetric bound applied to fitted ``piEN`` and ``piEE`` when supported."""

    piE_prior_weight: float = 0.0
    """Weight for the optional linear ``|piE|`` penalty in finite-difference fits."""

    piE_prior_eps: float = 1.0e-3
    """Small numerical floor used by the finite-difference ``|piE|`` penalty."""

    bic_include_space_parallax: bool = False
    """If True, ``bic_single_lens`` also tries the GULLS FSPL space-parallax model."""

    # ==================================================
    # 0b) Automatic single-lens initialization
    # ==================================================
    auto_init_teff_min: float = 1.0
    """Smallest teff used when estimating single-lens initial values."""

    auto_init_teff_max: float = 1000.0
    """Largest teff used when estimating single-lens initial values."""

    auto_init_teff_grid_n: int = 25
    """Number of teff grid points used for automatic initialization."""

    auto_init_dt0_coeff: float = 0.25
    """t0 grid spacing coefficient used for automatic initialization."""

    auto_init_max_clusters: int = 1
    """Maximum number of scan clusters used as t0/teff seeds."""

    auto_init_min_n_eff: float = 2.0
    """
    Minimum effective number of contributing points required for automatic
    single-lens initial grid clusters.

    This suppresses initial guesses driven by one unrealistically high-weight
    data point.
    """

    auto_init_u0_min: float = 1e-4
    """Smallest allowed u0 seed after converting from teff/tE."""

    auto_init_u0_max: float = 1.0
    """Largest allowed u0 seed after converting from teff/tE."""

    auto_init_tE_min: float = 1.0
    """Smallest tE seed used in the log grid."""

    auto_init_tE_max: float = 1000.0
    """Largest tE seed used in the log grid."""

    auto_init_tE_grid_n: int = 4
    """Number of tE seeds used in the log grid."""

    auto_init_logrho: float = -7.0
    """Initial logrho used for FSPL models when x0 is omitted."""

    pspl_fit_u0_min: float = 1e-4
    """Smallest allowed absolute u0 for the C++ PSPL fitter."""

    pspl_fit_min_t0_support_points: int = 3
    """Minimum number of data points required near the fitted t0."""

    pspl_fit_t0_support_tE_coeff: float = 3.0
    """Require t0 support points within +/- coeff * tE for C++ PSPL fits."""

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

    common_ratio: float = _COMMON_RATIO_DEFAULT
    """
    Common ratio of the geometric progression used to generate teff values.
    """

    teff_grid_n: int = 24
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

    This threshold is used to count strongly contributing points in the
    per-candidate quality diagnostics.
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

    best_score_teff_ratio: float = 2.0
    """
    Maximum timescale ratio used for best-score background clusters.

    The score compares a candidate only with clusters from the same season
    whose ``teff`` differs by at most this factor. If too few such clusters
    exist, the nearest timescales from the same season are added.
    """

    best_score_min_reference_clusters: int = 8
    """
    Preferred minimum number of same-season background clusters.

    When the local ``teff`` band contains fewer clusters, the nearest
    same-season timescales are added up to this count. A score still requires
    at least two usable background clusters.
    """

    best_score_upper_clip_sigma: float = 5.0
    """
    One-sided robust clipping threshold for strong secondary candidates.

    Background clusters above ``median + value * robust_scale`` are excluded
    iteratively. The center and scale are estimated with the median and MAD,
    so strong secondary anomalies do not inflate the score normalization.
    Set to ``inf`` to disable upper clipping.
    """

    best_score_clip_maxiters: int = 3
    """Maximum number of one-sided robust clipping iterations."""

    candidate_criteria: Optional[CandidateCriteria] = None
    """
    Optional criteria applied to raw cluster peaks before best-candidate
    selection. The criteria do not alter cluster extraction or the score
    background. If ``None``, no additional selection is applied.
    """

    # ==================================================
    # 5) Grid execution mode
    # ==================================================
    
    grid_backend: Literal["jax", "cpp"] = "cpp"
    """
    Grid evaluation backend.

    - ``"cpp"`` uses the C++ for-loop backend for low-memory survey scans.
    - ``"jax"`` uses the JAX vectorized/chunked implementation.
    """

    single_fit_backend: Literal["jax", "cpp", "vbm_cpp"] = "cpp"
    """
    Single-lens fit backend.

    ``"cpp"`` is implemented for ``fitter_kind="pspl"``. ``"vbm_cpp"`` is
    implemented for ``fitter_kind="fspl_parallax"`` and uses the native
    VBMicrolensing finite-source magnification with the C++ LM solver.
    Other combinations use the JAX fitters.
    """

    grid_chunked: bool = False
    """
    Force chunked execution of the grid scan.
    
    Instead of evaluating the entire (t0, teff) grid in a single ``vmap``,
    the grid is split into smaller chunks and processed sequentially.
    
    This reduces JAX compilation size and peak memory usage at the cost
    of a small runtime overhead.
    """
    
    grid_chunk_auto: bool = False
    """
    Automatically switch to chunked execution for large grids.
    
    If enabled, the runner uses chunked evaluation only when the total
    number of grid points exceeds ``grid_chunk_threshold``. Smaller grids
    continue to use the standard fully-vectorized execution.
    """
    
    grid_chunk_size: int = 4096
    """
    Number of grid points evaluated in each chunk when chunked execution
    is enabled.
    
    Larger values improve runtime performance but increase compilation
    size and memory usage.
    """
    
    grid_chunk_threshold: int = 100_000
    """
    Minimum number of grid points required to activate automatic chunking
    when ``grid_chunk_auto`` is enabled.
    """
