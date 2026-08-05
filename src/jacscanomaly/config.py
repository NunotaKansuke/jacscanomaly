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

    Required for ``"pspl_space_parallax"``, ``"fspl_space_parallax"``, and
    ``"bic_single_lens"`` when
    ``bic_include_space_parallax`` is enabled. Expected columns are
    ``JD RA_deg Dec_deg distance_AU``. It is Earth-relative in the default
    ``earth_geocentric_offset`` convention.
    """

    parallax_geometry: Literal["auto", "none", "annual", "space", "both"] = "auto"
    """
    Observer geometry used by the physical-effect detector.

    ``"auto"`` chooses ``"space"`` when a satellite ephemeris is available,
    otherwise ``"annual"`` when sky coordinates are available.  This is a
    geometry decision, not a model-selection score: annual and space
    parallax are not compared against each other.  ``"both"`` is retained for
    explicit mixed/legacy analyses.
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

    pspl_fit_u0_min: float = 1e-4
    """Smallest allowed absolute u0 for the C++ PSPL fitter."""

    pspl_fit_min_t0_support_points: int = 3
    """Minimum number of data points required near the fitted t0."""

    pspl_fit_t0_support_tE_coeff: float = 3.0
    """Require t0 support points within +/- coeff * tE for C++ PSPL fits."""

    pspl_fit_nonnegative_fluxes: bool = False
    """Constrain C++ PSPL source and blend fluxes to be nonnegative."""

    pspl_fit_nonnegative_on_cancellation: bool = False
    """Use nonnegative fluxes only when the free solution strongly cancels."""

    pspl_fit_max_flux_cancellation_ratio: float = 50.0
    """Maximum source/blend cancellation before the nonnegative safeguard."""

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

    single_fit_backend: Literal["jax", "cpp", "vbm_cpp"] = "cpp"
    """
    Single-lens fit backend.

    ``"cpp"`` is implemented for ``fitter_kind="pspl"``. ``"vbm_cpp"`` is
    implemented for ``fitter_kind="fspl_parallax"`` and uses the native
    VBMicrolensing finite-source magnification with the C++ LM solver.
    Other combinations use the JAX fitters.
    """

    vbm_cpp_piE_seed_values: tuple[float, ...] = (0.0,)
    """Per-component piE values used for automatic VBM-C++ multistart fits.

    When ``fitter_kind="fspl_parallax"`` and
    ``single_fit_backend="vbm_cpp"``, jacscanomaly combines each automatic
    ``(t0, tE, u0)`` seed with this Cartesian piE grid before C++ LM fitting.
    The default is a single safe zero-parallax start. Supply, for example,
    ``(-0.5, 0.0, 0.5)`` to opt into a Cartesian parallax multistart.
    """

    vbm_cpp_logrho_seed_values: tuple[float, ...] = (-3.0,)
    """log-rho values combined with automatic VBM-C++ parallax seeds."""

    vbm_cpp_maxiter: int = 200
    """Maximum C++ LM iterations per VBM automatic-start trial."""

    vbm_cpp_damping_parameter: float = 1.0e-4
    """Initial LM damping parameter for the native VBM C++ backend."""

    vbm_cpp_tol: float = 1.0e-5
    """C++ LM convergence tolerance for the native VBM backend."""

    # ==================================================
    # 0c) Native parallax backend
    # ==================================================
    parallax_fit_backend: Literal["native_cpp"] = "native_cpp"
    """Backend used by the effect-aware parallax fallback."""

    parallax_optimizer: Literal["scipy_trf", "native_lm_polish"] = "scipy_trf"
    """Primary optimizer for native parallax fitting."""

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
