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
        PSPL with annual or space parallax, selected by ``parallax_geometry``.
    - ``"fspl_parallax"`` :
        FSPL with annual or space parallax, selected by
        ``parallax_geometry``.
    """

    ra_deg: Optional[float] = None
    """Right ascension of the source (degrees). Required for parallax models."""

    dec_deg: Optional[float] = None
    """Declination of the source (degrees). Required for parallax models."""

    tref: Optional[float] = None
    """
    Reference time for annual parallax.

    If ``None``, the fitter resolves an event-centred reference time near the
    observed brightening peak (or the supplied nonlinear seed's ``t0``).
    """

    satellite_ephemeris_path: Optional[str] = None
    """
    Path to the spacecraft/observer ephemeris table used by space-parallax
    models.

    Required when ``parallax_geometry="space"`` is selected for a parallax
    fitter. Expected columns are
    ``JD RA_deg Dec_deg distance_AU``. It is Earth-relative in the default
    ``earth_geocentric_offset`` convention.
    """

    parallax_geometry: Literal["auto", "none", "annual", "space", "both"] = "auto"
    """
    Observer geometry used by parallax fitters and the physical-effect
    detector.

    ``"auto"`` chooses ``"space"`` when a satellite ephemeris is available,
    otherwise ``"annual"`` when sky coordinates are available.  This is a
    geometry decision, not a model-selection score: annual and space
    parallax are not compared against each other.  ``"both"`` is only for an
    explicit mixed diagnostic; a parallax baseline must resolve to exactly one
    of ``"annual"`` or ``"space"``.
    """

    max_piE: float = 1.0
    """Symmetric bound applied to fitted ``piEN`` and ``piEE`` when supported."""

    piE_prior_weight: float = 0.0
    """Weight for an optional parallax prior used by downstream diagnostics."""

    piE_prior_eps: float = 1.0e-3
    """Small numerical floor used by the finite-difference ``|piE|`` penalty."""

    # ==================================================
    # 0b) Automatic single-lens initialization
    # ==================================================
    auto_init_teff_min: float = 0.03
    """Smallest teff used by legacy initialization and the PSPL flat fallback."""

    auto_init_teff_max: float = 100.0
    """Largest teff used by legacy initialization and the PSPL flat fallback."""

    auto_init_teff_grid_n: int = 24
    """Number of logarithmic teff templates used by non-PSPL initialization."""

    auto_init_dt0_coeff: float = 0.25
    """Legacy t0 grid spacing coefficient used for non-PSPL initialization."""

    auto_init_max_clusters: int = 1
    """Maximum number of scan clusters used as t0/teff seeds."""

    auto_init_min_n_eff: float = 2.0
    """
    Minimum effective number of contributing points required by the legacy
    non-PSPL initial grid search.

    This suppresses initial guesses driven by one unrealistically high-weight
    data point.
    """

    auto_init_u0_min: float = 1e-4
    """Smallest u0 template used by the PSPL FFT initial-value search."""

    auto_init_u0_max: float = 1.0
    """Largest u0 template used by the PSPL FFT initial-value search."""

    auto_init_u0_grid_n: int = 8
    """Number of source-plane u0 rows evaluated together for each PSPL tE."""

    auto_init_fft_grid_dt: Optional[float] = 0.02
    """Regular FFT time spacing used for PSPL initialization."""

    auto_init_fft_max_grid_points: int = 500_000
    """Maximum regular FFT grid length used for PSPL initialization."""

    auto_init_fft_top_k: int = 4
    """Number of ranked PSPL FFT seeds passed to the fitter."""

    auto_init_max_flux_cancellation_ratio: float = 50.0
    """
    Maximum source/blend cancellation ratio allowed for an automatic PSPL seed.

    The FFT initializer profiles ``fs`` and ``fb`` independently.  A trial
    whose two terms are much larger than the observed baseline can therefore
    obtain a large apparent improvement while placing ``t0`` in an unobserved
    gap.  Such a seed is not useful to the continuous fitter.
    """

    auto_init_nearest_support_tE_coeff: float = 1.0
    """
    Require one observation within this many ``tE`` of an automatic ``t0``.

    This is a local-support guard in addition to the broader
    ``auto_init_t0_support_tE_coeff`` window.  It keeps a one-sided event near a
    season boundary usable while rejecting a model supported only by a distant
    wing across a large gap.
    """

    auto_init_fft_workers: int = -1
    """SciPy worker count for batched PSPL FFTs; -1 uses all available CPUs."""

    auto_init_fft_tE_grid_n: int = 24
    """Number of logarithmic outer tE scales used by the PSPL FFT search."""

    auto_init_tE_min: float = 1.0
    """Smallest tE scale used by PSPL FFT and legacy non-PSPL initialization."""

    auto_init_tE_max: float = 1000.0
    """Largest tE scale used by PSPL FFT and legacy non-PSPL initialization."""

    auto_init_tE_grid_n: int = 4
    """Number of legacy tE seeds used by non-PSPL initialization."""

    auto_init_logrho: float = -7.0
    """Initial logrho used for FSPL models when x0 is omitted."""

    auto_init_fspl_template_top_k: int = 4
    """Number of ranked FSPL template seeds passed to the fitter."""

    auto_init_min_t0_support_points: int = 3
    """Minimum number of observations near an automatic PSPL seed."""

    auto_init_t0_support_tE_coeff: float = 3.0
    """Support half-width for automatic PSPL seeds, in units of tE."""

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
    Maximum timescale ratio used for score background clusters.

    The score compares a candidate with clusters from all observing seasons
    whose ``teff`` differs by at most this factor. If too few such clusters
    exist, the nearest timescales from all seasons are added.
    """

    best_score_min_reference_clusters: int = 8
    """
    Preferred minimum number of all-season background clusters.

    When the local ``teff`` band contains fewer clusters, the nearest
    all-season timescales are added up to this count. A score still requires
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
    
    grid_backend: Literal["jax", "cpp", "fft"] = "cpp"
    """
    Grid evaluation backend.

    - ``"cpp"`` uses the C++ for-loop backend for low-memory survey scans.
    - ``"jax"`` uses the JAX vectorized/chunked implementation.
    - ``"fft"`` uses an oversampled regular grid and FFT correlations, then
      exactly re-evaluates extracted representatives on the original data.
    """

    fft_oversample: int = 4
    """Number of FFT calculation-grid cells per ``t0`` grid interval."""

    fft_max_grid_points: int = 1_000_000
    """Maximum regular calculation-grid length for one FFT timescale."""

    fft_singular_rtol: float = 1.0e-12
    """Relative threshold used to reject nearly constant FFT templates."""

    # ==================================================
    # 0c) Unified continuous fitter
    # ==================================================
    fitter_maxiter: int = 1000
    """Maximum number of SciPy LM function evaluations per fit."""

    fitter_tol: float = 1.0e-6
    """Common SciPy LM convergence tolerance."""

    magnification_tol: float = 1.0e-4
    """Tolerance passed to the compiled finite-source magnification kernel."""

    magnification_reltol: float = 1.0e-4
    """Relative tolerance passed to the compiled finite-source kernel."""

    parallax_observer_convention: Literal[
        "earth_geocentric_offset", "heliocentric_observer", "gulls"
    ] = "earth_geocentric_offset"
    """Canonical observer convention for native parallax fitting."""

    parallax_time_scale: Literal["jd", "hjd"] = "jd"
    """Explicit scale for times passed to the native parallax fitter."""

    parallax_time_offset: float = 0.0
    """Explicit additive offset used to normalize relative input times."""

    parallax_extrapolation: Literal["reject", "linear"] = "reject"
    """Whether native ephemerides may be linearly extrapolated."""

    parallax_earth_ephemeris: object = None
    """Optional validated ``parallax_backend.Ephemeris`` for annual parallax."""

    parallax_observer_ephemeris: object = None
    """Optional complete observer ephemeris for heliocentric/GULLS mode."""

    parallax_reference_ephemeris: object = None
    """Optional explicit reference ephemeris for heliocentric/GULLS mode."""

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
