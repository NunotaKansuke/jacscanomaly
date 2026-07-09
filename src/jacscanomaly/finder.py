from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional
import logging

import numpy as np
import jax
import jax.numpy as jnp

from .config import FinderConfig
from .criteria import CandidateCriteria
from .singlelens_fit import (
    SingleLensFitResult,
    PSPLFitter,
    CPPPSPLFitter,
    FSPLFitter,
    VBMFiniteDiffFSPLFitter,
    PSPLParallaxFitter,
    FSPLParallaxFitter,
    PSPLSpaceParallaxFitter,
    FSPLSpaceParallaxFitter,
    VBMFiniteDiffGullsFSPLSpaceParallaxFitter,
    BICSingleLensFitter,
    evaluate_single_lens_fixed,
)
from .singlelens_model import (
    A_pspl_func,
    A_fspl_logrho_func,
    A_pspl_parallax_func,
    A_fspl_parallax_logrho_func,
    A_pspl_space_parallax_func,
    A_fspl_space_parallax_logrho_func,
)
from .plot import AnomalyPlotter
from .seasons import SeasonSplitter
from .extract import ResultExtractor
from .runner import SeasonGridRunner
from .models import AnomalyResult, BestCandidate, CandidateQuality
from .template_free import TemplateFreeScanner, TemplateFreeSearchConfig, TemplateFreeSearchResult

logger = logging.getLogger(__name__)


@dataclass
class Finder:
    """
    Main entry point of **jacscanomaly**.

    `Finder` orchestrates the full anomaly-search pipeline:

    1. Fit a single-lens microlensing model to the full light curve
       (PSPL / FSPL / ± annual parallax).
    2. Split the residual light curve into observing seasons.
    3. Perform grid scans on residuals within each season.
    4. Extract and merge statistically significant clusters.
    5. Select the best anomaly candidate, if any.

    The choice of single-lens model is controlled by :class:`FinderConfig`
    (via ``fitter_kind``), or by explicitly injecting a fitter instance.

    Parameters
    ----------
    config : FinderConfig, optional
        Configuration object controlling fitting, season splitting,
        grid scanning, and candidate selection.
    fitter : optional
        A single-lens fitter instance. If ``None``, a default fitter
        is constructed from ``config.fitter_kind``.
        Any object implementing::

            fit(time, flux, ferr, x0) -> SingleLensFitResult

        is acceptable.
    plotter : AnomalyPlotter, optional
        Plotting helper used by the ``plot_*`` convenience methods.

    Notes
    -----
    * The dimensionality of the initial parameter vector ``x0`` depends
      on the selected fitter:

      ============================  ============================================
      Model                         x0 parameters
      ============================  ============================================
      PSPL                          (t0, tE, u0)
      FSPL                          (t0, tE, u0, logrho)
      PSPL + parallax               (t0, tE, u0, piEN, piEE)
      FSPL + parallax               (t0, tE, u0, logrho, piEN, piEE)
      PSPL + space parallax         (t0, tE, u0, piEN, piEE)
      FSPL + space parallax         (t0, tE, u0, logrho, piEN, piEE)
      ============================  ============================================

    * For parallax models, ``ra_deg`` and ``dec_deg`` must be provided
      in :class:`FinderConfig`. If ``tref`` is not specified, the median
      observation time is used.
    """

    config: FinderConfig = field(default_factory=FinderConfig)
    fitter: Optional[object] = None
    plotter: Optional[AnomalyPlotter] = None

    def __post_init__(self) -> None:
        if self.plotter is None:
            self.plotter = AnomalyPlotter()

        splitter = SeasonSplitter(gap=self.config.gap)
        extractor = ResultExtractor(
            sigma_overlap=self.config.overlap_sigma,
            min_points=self.config.min_cluster_points,
        )
        self.runner = SeasonGridRunner(
            splitter=splitter,
            extractor=extractor,
            config=self.config,
        )

        self._last_result: Optional[AnomalyResult] = None
        self._last_template_free_result: Optional[TemplateFreeSearchResult] = None

    def _ensure_fitter(self, t_ref) -> None:
        """
        Instantiate the default single-lens fitter from the current configuration.
    
        Notes
        -----
        - If `config.fitter_kind` selects a parallax model, `ra_deg` and `dec_deg`
          must be provided. If `tref` is not set, it defaults to `median(time)`.
        """
        if self.fitter is not None:
            return
    
        k = self.config.fitter_kind
    
        # -----------------------------
        # 1) Validate model selection
        # -----------------------------
        valid = {
            "pspl",
            "fspl",
            "fspl_vbm_fd",
            "pspl_parallax",
            "fspl_parallax",
            "pspl_space_parallax",
            "fspl_space_parallax",
            "fspl_space_parallax_gulls_vbm_fd",
            "bic_single_lens",
        }
        if k not in valid:
            raise ValueError(
                f"Unknown fitter_kind '{k}'. "
                f"Valid options are: {sorted(valid)}"
            )
    
        # -----------------------------
        # 2) Validate model requirements
        # -----------------------------
        needs_sky = k in {
            "pspl_parallax",
            "fspl_parallax",
            "pspl_space_parallax",
            "fspl_space_parallax",
            "fspl_space_parallax_gulls_vbm_fd",
        } or (k == "bic_single_lens" and self.config.bic_include_space_parallax)
        if needs_sky:
            if self.config.ra_deg is None or self.config.dec_deg is None:
                raise ValueError(
                    f"{k} requires ra_deg and dec_deg in FinderConfig "
                    "(sky coordinates are required for parallax)."
                )
        needs_satellite = k in {
            "pspl_space_parallax",
            "fspl_space_parallax",
            "fspl_space_parallax_gulls_vbm_fd",
        } or (k == "bic_single_lens" and self.config.bic_include_space_parallax)
        if needs_satellite and self.config.satellite_ephemeris_path is None:
            raise ValueError(
                f"{k} requires satellite_ephemeris_path in FinderConfig."
            )
    
        # -----------------------------
        # 3) Build fitter
        # -----------------------------
        if k == "pspl":
            if self.config.single_fit_backend == "cpp":
                self.fitter = CPPPSPLFitter(
                    u0_min=float(self.config.pspl_fit_u0_min),
                    min_t0_support_points=int(self.config.pspl_fit_min_t0_support_points),
                    t0_support_tE_coeff=float(self.config.pspl_fit_t0_support_tE_coeff),
                )
            else:
                self.fitter = PSPLFitter()
            return
    
        if k == "fspl":
            self.fitter = FSPLFitter()
            return

        if k == "fspl_vbm_fd":
            self.fitter = VBMFiniteDiffFSPLFitter()
            return
    
        # Parallax variants
        tref = self.config.tref
        if tref is None:
            tref = t_ref
    
        if k == "pspl_parallax":
            self.fitter = PSPLParallaxFitter(
                RA=self.config.ra_deg,
                Dec=self.config.dec_deg,
                tref=tref,
                use_HJD=self.config.parallax_use_HJD,
            )
            return

        if k == "pspl_space_parallax":
            self.fitter = PSPLSpaceParallaxFitter(
                RA=self.config.ra_deg,
                Dec=self.config.dec_deg,
                tref=tref,
                satellite_ephemeris_path=self.config.satellite_ephemeris_path,
                use_HJD=self.config.parallax_use_HJD,
                convention=self.config.space_parallax_convention,
            )
            return

        if k == "fspl_space_parallax":
            self.fitter = FSPLSpaceParallaxFitter(
                RA=self.config.ra_deg,
                Dec=self.config.dec_deg,
                tref=tref,
                satellite_ephemeris_path=self.config.satellite_ephemeris_path,
                use_HJD=self.config.parallax_use_HJD,
                convention=self.config.space_parallax_convention,
            )
            return

        if k == "fspl_space_parallax_gulls_vbm_fd":
            self.fitter = VBMFiniteDiffGullsFSPLSpaceParallaxFitter(
                RA=self.config.ra_deg,
                Dec=self.config.dec_deg,
                tref=tref,
                satellite_ephemeris_path=self.config.satellite_ephemeris_path,
                max_piE=float(self.config.max_piE),
                piE_prior_weight=float(self.config.piE_prior_weight),
                piE_prior_eps=float(self.config.piE_prior_eps),
            )
            return

        if k == "bic_single_lens":
            self.fitter = BICSingleLensFitter(
                RA=self.config.ra_deg,
                Dec=self.config.dec_deg,
                tref=tref,
                satellite_ephemeris_path=self.config.satellite_ephemeris_path,
                max_piE=float(self.config.max_piE),
                piE_prior_weight=float(self.config.piE_prior_weight),
                piE_prior_eps=float(self.config.piE_prior_eps),
                include_space_parallax=bool(self.config.bic_include_space_parallax),
            )
            return
    
        # k == "fspl_parallax"
        self.fitter = FSPLParallaxFitter(
            RA=self.config.ra_deg,
            Dec=self.config.dec_deg,
            tref=tref,
            use_HJD=self.config.parallax_use_HJD,
        )


    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------
    def fit_single_lens(
        self,
        time,
        flux,
        ferr,
        x0=None,
    ) -> SingleLensFitResult:
        """
        Run only the single-lens fit selected by the current configuration.

        Parameters
        ----------
        time, flux, ferr : array-like
            One-dimensional light-curve arrays.
        x0 : array-like, optional
            Initial guess for the nonlinear model parameters.
            If omitted, initial values are estimated from a scan of the light curve.

        Returns
        -------
        SingleLensFitResult
            Result of the single-lens fit.
        """
        time_j, flux_j, ferr_j, x0_j, time_np, _, _ = self._to_arrays(time, flux, ferr, x0)
        self._ensure_fitter(float(np.median(time_np)))
        if x0_j is None:
            return self._fit_from_auto_initial_guesses(time_j, flux_j, ferr_j, time_np)
        return self.fitter.fit(time_j, flux_j, ferr_j, x0_j)

    def run(
        self,
        time,
        flux,
        ferr,
        x0=None,
        *,
        refit: bool = True,
        verbose: bool = True,
        log: Optional[logging.Logger] = None,
    ) -> AnomalyResult:
        """
        Run the full anomaly-search pipeline.

        Parameters
        ----------
        time, flux, ferr : array-like
            One-dimensional light-curve arrays.
        x0 : array-like, optional
            Initial guess for the single-lens model parameters. If omitted, the
            finder estimates multiple initial values and uses the best fit.
        refit : bool, optional
            If True, optimize the single-lens nonlinear parameters starting from
            ``x0``. If False, require ``x0`` and use it as fixed nonlinear
            parameters; only the linear flux parameters are solved.
        verbose : bool, optional
            If True, print progress messages.
        log : logging.Logger, optional
            Logger used for progress reporting. When omitted, module-level
            logging is used. ``verbose=False`` suppresses progress messages.

        Returns
        -------
        AnomalyResult
            Object containing the single-lens fit, residuals,
            per-season cluster summaries, and the best anomaly candidate.

        Raises
        ------
        ValueError
            If the input arrays are not finite one-dimensional arrays of equal
            length, ``ferr`` is non-positive, or ``refit=False`` is used
            without ``x0``.

        Notes
        -----
        ``refit=False`` fixes only nonlinear model parameters. The linear
        source and blend fluxes are still solved for the supplied data before
        residuals are scanned. See :doc:`workflows` for when to use this mode.
        """
        time_j, flux_j, ferr_j, x0_j, time_np, flux_np, ferr_np = self._to_arrays(
            time, flux, ferr, x0
        )

        self._ensure_fitter(float(np.median(time_np)))

        if not refit:
            fit = self._fixed_single_lens_from_x0(time_j, flux_j, ferr_j, x0_j)
        elif x0_j is None:
            if verbose:
                (logger if log is None else log).info("Estimating single-lens initial values.")
            fit = self._fit_from_auto_initial_guesses(time_j, flux_j, ferr_j, time_np)
        else:
            fit = self.fitter.fit(time_j, flux_j, ferr_j, x0_j)
        residual_j = fit.residual
        model_flux_j = fit.model_flux

        residual_np, model_flux_np, chi2_dof = jax.device_get(
            (residual_j, model_flux_j, fit.chi2_dof)
        )
        residual_np = np.asarray(residual_np, dtype=float)
        model_flux_np = np.asarray(model_flux_np, dtype=float)
        chi2_dof = float(chi2_dof)

        seasons, clusters_all, grid_metrics_all = self.runner.run(
            time_j=time_j,
            residual_j=residual_j,
            ferr_j=ferr_j,
            time_np=time_np,
            verbose=verbose,
            log=log,
        )

        best_obj = self._pick_best_candidate(clusters_all, grid_metrics_all)

        result = AnomalyResult(
            time=time_np,
            flux=flux_np,
            ferr=ferr_np,
            fit=fit,
            residual=residual_np,
            model_flux=model_flux_np,
            chi2_dof=chi2_dof,
            seasons=seasons,
            clusters_all=clusters_all,
            grid_metrics_all=grid_metrics_all,
            best=best_obj,
        )

        self._last_result = result
        return result

    def run_template_free(
        self,
        time,
        flux,
        ferr,
        x0=None,
        *,
        fit: Optional[SingleLensFitResult] = None,
        config: Optional[TemplateFreeSearchConfig] = None,
    ) -> TemplateFreeSearchResult:
        """
        Run a template-free anomaly search on single-lens residuals.

        This leaves the existing bell-template anomaly pipeline untouched.

        Parameters
        ----------
        time, flux, ferr : array-like
            One-dimensional light-curve arrays. ``ferr`` is used to normalize
            the residuals even when ``fit`` is supplied.
        x0 : array-like, optional
            Nonlinear initial values used only when ``fit`` is omitted.
        fit : SingleLensFitResult, optional
            Existing baseline fit whose residuals should be searched. Passing
            this skips all single-lens fitting.
        config : TemplateFreeSearchConfig, optional
            Template-free search settings. If omitted, the finder season gap is
            reused and all other settings use their defaults.

        Returns
        -------
        TemplateFreeSearchResult
            Residual z-scores and ranked zero-crossing candidates.

        Notes
        -----
        To scan residuals without constructing a ``Finder`` or a
        ``SingleLensFitResult``, call :class:`TemplateFreeScanner` directly.
        """
        time_j, flux_j, ferr_j, x0_j, time_np, _, ferr_np = self._to_arrays(
            time, flux, ferr, x0
        )

        if fit is None:
            self._ensure_fitter(float(np.median(time_np)))
            if x0_j is None:
                fit = self._fit_from_auto_initial_guesses(time_j, flux_j, ferr_j, time_np)
            else:
                fit = self.fitter.fit(time_j, flux_j, ferr_j, x0_j)

        residual_np = np.asarray(jax.device_get(fit.residual), dtype=float)
        scanner_config = TemplateFreeSearchConfig(gap=self.config.gap) if config is None else config
        result = TemplateFreeScanner(scanner_config).run(time_np, residual_np, ferr_np)
        self._last_template_free_result = result
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _to_arrays(self, time, flux, ferr, x0):
        """
        Validate inputs and convert them to both NumPy and JAX arrays.
        """
        time_np = np.asarray(time, dtype=float)
        flux_np = np.asarray(flux, dtype=float)
        ferr_np = np.asarray(ferr, dtype=float)

        if time_np.ndim != 1 or flux_np.ndim != 1 or ferr_np.ndim != 1:
            raise ValueError("time/flux/ferr must be 1D arrays.")
        if not (len(time_np) == len(flux_np) == len(ferr_np)):
            raise ValueError("time/flux/ferr must have the same length.")
        if np.any(~np.isfinite(time_np)) or np.any(~np.isfinite(flux_np)) or np.any(~np.isfinite(ferr_np)):
            raise ValueError("time/flux/ferr must be finite.")
        if np.any(ferr_np <= 0):
            raise ValueError("ferr must be positive.")

        time_j = jnp.asarray(time_np)
        flux_j = jnp.asarray(flux_np)
        ferr_j = jnp.asarray(ferr_np)
        x0_j = None if x0 is None else jnp.asarray(x0, dtype=time_j.dtype)

        return time_j, flux_j, ferr_j, x0_j, time_np, flux_np, ferr_np

    def _fit_from_auto_initial_guesses(
        self,
        time_j: jnp.ndarray,
        flux_j: jnp.ndarray,
        ferr_j: jnp.ndarray,
        time_np: np.ndarray,
    ) -> SingleLensFitResult:
        guesses = self._estimate_single_lens_initial_guesses(
            time_j=time_j,
            flux_j=flux_j,
            ferr_j=ferr_j,
            time_np=time_np,
        )

        best_fit = None
        best_chi2 = np.inf
        errors = []

        for x0 in guesses:
            try:
                fit = self.fitter.fit(time_j, flux_j, ferr_j, jnp.asarray(x0, dtype=time_j.dtype))
                chi2 = float(jax.device_get(fit.chi2))
            except Exception as exc:
                errors.append(exc)
                continue

            if np.isfinite(chi2) and chi2 < best_chi2:
                best_chi2 = chi2
                best_fit = fit

        if best_fit is None:
            msg = "All automatic single-lens initial guesses failed."
            if errors:
                msg += f" First error: {errors[0]}"
            raise RuntimeError(msg)

        return best_fit

    def _fixed_single_lens_from_x0(
        self,
        time_j: jnp.ndarray,
        flux_j: jnp.ndarray,
        ferr_j: jnp.ndarray,
        x0_j: Optional[jnp.ndarray],
    ) -> SingleLensFitResult:
        if x0_j is None:
            raise ValueError("Finder.run(refit=False) requires x0.")

        k = self.config.fitter_kind
        P = getattr(self.fitter, "_P", None)

        if k == "pspl":
            return evaluate_single_lens_fixed(
                time=time_j,
                flux=flux_j,
                ferr=ferr_j,
                x0=x0_j,
                build_A=A_pspl_func,
                dof=3,
                param_names=("t0", "tE", "u0"),
                min_points=4,
            )

        if k == "fspl":
            def q_to_params(q):
                t0, tE, u0, logrho = q
                return jnp.array([t0, tE, u0, jnp.exp(logrho)])

            return evaluate_single_lens_fixed(
                time=time_j,
                flux=flux_j,
                ferr=ferr_j,
                x0=x0_j,
                build_A=A_fspl_logrho_func,
                dof=4,
                param_names=("t0", "tE", "u0", "rho"),
                x_to_params=q_to_params,
                min_points=4,
                store_raw_params=True,
            )

        if k == "fspl_vbm_fd":
            return self._fixed_single_lens_from_numpy_model(
                time_j=time_j,
                flux_j=flux_j,
                ferr_j=ferr_j,
                x0_j=x0_j,
                dof=4,
                param_names=("t0", "tE", "u0", "rho"),
                parallax_projector=None,
            )

        if k == "pspl_parallax":
            return evaluate_single_lens_fixed(
                time=time_j,
                flux=flux_j,
                ferr=ferr_j,
                x0=x0_j,
                build_A=lambda p, t: A_pspl_parallax_func(p, t, P),
                dof=5,
                param_names=("t0", "tE", "u0", "piEN", "piEE"),
                min_points=6,
                parallax_projector=P,
            )

        if k == "fspl_parallax":
            def q_to_params(q):
                t0, tE, u0, logrho, piEN, piEE = q
                return jnp.array([t0, tE, u0, jnp.exp(logrho), piEN, piEE])

            return evaluate_single_lens_fixed(
                time=time_j,
                flux=flux_j,
                ferr=ferr_j,
                x0=x0_j,
                build_A=lambda q, t: A_fspl_parallax_logrho_func(q, t, P),
                dof=6,
                param_names=("t0", "tE", "u0", "rho", "piEN", "piEE"),
                x_to_params=q_to_params,
                min_points=7,
                store_raw_params=True,
                parallax_projector=P,
            )

        if k == "pspl_space_parallax":
            return evaluate_single_lens_fixed(
                time=time_j,
                flux=flux_j,
                ferr=ferr_j,
                x0=x0_j,
                build_A=lambda p, t: A_pspl_space_parallax_func(p, t, P),
                dof=5,
                param_names=("t0", "tE", "u0", "piEN", "piEE"),
                min_points=6,
                parallax_projector=P,
            )

        if k == "fspl_space_parallax":
            def q_to_params(q):
                t0, tE, u0, logrho, piEN, piEE = q
                return jnp.array([t0, tE, u0, jnp.exp(logrho), piEN, piEE])

            return evaluate_single_lens_fixed(
                time=time_j,
                flux=flux_j,
                ferr=ferr_j,
                x0=x0_j,
                build_A=lambda q, t: A_fspl_space_parallax_logrho_func(q, t, P),
                dof=6,
                param_names=("t0", "tE", "u0", "rho", "piEN", "piEE"),
                x_to_params=q_to_params,
                min_points=7,
                store_raw_params=True,
                parallax_projector=P,
            )

        if k == "fspl_space_parallax_gulls_vbm_fd":
            return self._fixed_single_lens_from_numpy_model(
                time_j=time_j,
                flux_j=flux_j,
                ferr_j=ferr_j,
                x0_j=x0_j,
                dof=6,
                param_names=("t0", "tE", "u0", "rho", "piEN", "piEE"),
                parallax_projector=P,
            )

        raise ValueError(f"Unknown fitter_kind '{k}'.")

    def _fixed_single_lens_from_numpy_model(
        self,
        *,
        time_j: jnp.ndarray,
        flux_j: jnp.ndarray,
        ferr_j: jnp.ndarray,
        x0_j: jnp.ndarray,
        dof: int,
        param_names: tuple[str, ...],
        parallax_projector,
    ) -> SingleLensFitResult:
        if not hasattr(self.fitter, "_model_and_residual"):
            raise TypeError(
                f"{type(self.fitter).__name__} does not support fixed-parameter evaluation."
            )

        time_np = np.asarray(jax.device_get(time_j), dtype=float)
        flux_np = np.asarray(jax.device_get(flux_j), dtype=float)
        ferr_np = np.maximum(np.asarray(jax.device_get(ferr_j), dtype=float), 1e-12)
        q = np.asarray(jax.device_get(x0_j), dtype=float)
        model, residual, chi2, fs, fb = self.fitter._model_and_residual(q, time_np, flux_np, ferr_np)
        rho = float(np.exp(np.clip(q[3], -50.0, 10.0)))
        params = np.asarray([q[0], q[1], q[2], rho, *q[4:]], dtype=float)

        return SingleLensFitResult(
            time=time_np,
            flux=flux_np,
            ferr=ferr_np,
            params=jnp.asarray(params),
            param_names=param_names,
            chi2=jnp.asarray(chi2),
            chi2_dof=jnp.asarray(chi2 / max(int(time_np.size) - dof, 1)),
            fs=jnp.asarray(fs),
            fb=jnp.asarray(fb),
            model_flux=jnp.asarray(model),
            residual=jnp.asarray(residual),
            raw_params=jnp.asarray(q),
            parallax_projector=parallax_projector,
        )

    def _estimate_single_lens_initial_guesses(
        self,
        *,
        time_j: jnp.ndarray,
        flux_j: jnp.ndarray,
        ferr_j: jnp.ndarray,
        time_np: np.ndarray,
    ) -> np.ndarray:
        cfg = self.config
        if cfg.auto_init_teff_min <= 0 or cfg.auto_init_teff_max <= 0:
            raise ValueError("auto_init_teff_min and auto_init_teff_max must be positive.")
        if cfg.auto_init_teff_max < cfg.auto_init_teff_min:
            raise ValueError("auto_init_teff_max must be >= auto_init_teff_min.")
        if cfg.auto_init_u0_min <= 0 or cfg.auto_init_u0_max <= 0:
            raise ValueError("auto_init_u0_min and auto_init_u0_max must be positive.")
        if cfg.auto_init_u0_max < cfg.auto_init_u0_min:
            raise ValueError("auto_init_u0_max must be >= auto_init_u0_min.")

        n_teff = max(1, int(cfg.auto_init_teff_grid_n))
        ratio = 1.0
        if n_teff > 1 and cfg.auto_init_teff_max > cfg.auto_init_teff_min:
            ratio = float((cfg.auto_init_teff_max / cfg.auto_init_teff_min) ** (1.0 / (n_teff - 1)))

        init_config = replace(
            cfg,
            teff_init=float(cfg.auto_init_teff_min),
            common_ratio=ratio,
            teff_grid_n=n_teff,
            dt0_coeff=float(cfg.auto_init_dt0_coeff),
        )
        init_runner = SeasonGridRunner(
            splitter=SeasonSplitter(gap=cfg.gap),
            extractor=ResultExtractor(sigma_overlap=cfg.overlap_sigma, min_points=1),
            config=init_config,
        )

        _, clusters, grid_metrics = init_runner.run(
            time_j=time_j,
            residual_j=flux_j,
            ferr_j=ferr_j,
            time_np=time_np,
            verbose=False,
        )

        clusters = np.asarray(clusters, dtype=float)
        if clusters.size:
            clusters = clusters[np.isfinite(clusters).all(axis=1)]
            clusters = clusters[clusters[:, 2] > 0]
            grid_metrics = np.asarray(grid_metrics, dtype=float)
            if grid_metrics.size:
                qualities = np.asarray(
                    [
                        self._grid_quality_for_cluster(float(row[0]), float(row[1]), grid_metrics)
                        for row in clusters
                    ],
                    dtype=float,
                )
                pass_eff = qualities[:, 0] >= float(cfg.auto_init_min_n_eff)
                if np.any(pass_eff):
                    clusters = clusters[pass_eff]
                    qualities = qualities[pass_eff]
                    order = np.argsort(clusters[:, 2])[::-1]
                else:
                    order = np.lexsort((-clusters[:, 2], -qualities[:, 0]))
            else:
                order = np.argsort(clusters[:, 2])[::-1]
            clusters = clusters[order[: max(1, int(cfg.auto_init_max_clusters))]]

        if clusters.size == 0:
            flux_np = np.asarray(jax.device_get(flux_j), dtype=float)
            i_peak = int(np.nanargmax(flux_np))
            span = float(np.nanmax(time_np) - np.nanmin(time_np))
            teff = min(max(0.1 * span, float(cfg.auto_init_teff_min)), float(cfg.auto_init_teff_max))
            clusters = np.asarray([[float(time_np[i_peak]), teff, 0.0]], dtype=float)

        if cfg.auto_init_tE_min <= 0 or cfg.auto_init_tE_max <= 0:
            raise ValueError("auto_init_tE_min and auto_init_tE_max must be positive.")
        if cfg.auto_init_tE_max < cfg.auto_init_tE_min:
            raise ValueError("auto_init_tE_max must be >= auto_init_tE_min.")

        n_tE = max(1, int(cfg.auto_init_tE_grid_n))
        if n_tE == 1:
            tE_grid = np.asarray([float(cfg.auto_init_tE_max)], dtype=float)
        else:
            tE_grid = np.exp(
                np.linspace(
                    np.log(float(cfg.auto_init_tE_min)),
                    np.log(float(cfg.auto_init_tE_max)),
                    n_tE,
                )
            )

        guesses = []
        for t0, teff, _ in clusters:
            teff = float(teff)
            for tE in tE_grid:
                u0 = teff / float(tE)
                if not (float(cfg.auto_init_u0_min) <= u0 <= float(cfg.auto_init_u0_max)):
                    continue
                guesses.append(self._build_initial_vector(float(t0), float(tE), float(u0)))

        if not guesses:
            t0 = float(clusters[0, 0])
            teff = float(clusters[0, 1])
            tE = min(max(teff / 0.1, float(cfg.auto_init_tE_min)), float(cfg.auto_init_tE_max))
            u0 = min(max(teff / tE, float(cfg.auto_init_u0_min)), float(cfg.auto_init_u0_max))
            guesses.append(self._build_initial_vector(t0, tE, u0))

        return np.asarray(guesses, dtype=float)

    @staticmethod
    def _grid_quality_for_cluster(t0: float, teff: float, metrics: np.ndarray) -> tuple[float, float]:
        if metrics.size == 0:
            return 0.0, 0.0
        i = int(np.argmin(np.abs(metrics[:, 0] - t0) + np.abs(metrics[:, 1] - teff)))
        return float(metrics[i, 5]), float(metrics[i, 6])

    def _build_initial_vector(self, t0: float, tE: float, u0: float) -> np.ndarray:
        k = self.config.fitter_kind
        if k == "pspl":
            return np.asarray([t0, tE, u0], dtype=float)
        if k in {"fspl", "fspl_vbm_fd"}:
            return np.asarray([t0, tE, u0, float(self.config.auto_init_logrho)], dtype=float)
        if k in {"pspl_parallax", "pspl_space_parallax"}:
            return np.asarray([t0, tE, u0, 0.0, 0.0], dtype=float)
        if k in {"fspl_parallax", "fspl_space_parallax", "fspl_space_parallax_gulls_vbm_fd"}:
            return np.asarray([t0, tE, u0, float(self.config.auto_init_logrho), 0.0, 0.0], dtype=float)
        raise ValueError(f"Unknown fitter_kind '{k}'.")

    def _pick_best_candidate(
        self,
        clusters_all: np.ndarray,
        grid_metrics_all: np.ndarray,
    ) -> Optional[BestCandidate]:
        """
        Select the strongest anomaly candidate from all extracted clusters.
        """
        if clusters_all is None or clusters_all.size == 0:
            return None

        clusters_use = np.asarray(clusters_all, dtype=float)
        clusters_use = clusters_use[np.isfinite(clusters_use).all(axis=1)]
        if clusters_use.size == 0:
            return None

        max_ind = int(np.argmax(clusters_use[:, 2]))
        best = clusters_use[max_ind]
        others = np.delete(clusters_use, max_ind, axis=0)

        if others.shape[0] >= 2:
            other_dchi2 = others[:, 2]

            # Estimate the background spread from the bulk of the distribution.
            # A few large-dchi2 secondary peaks can otherwise inflate std and
            # suppress the best-candidate score.
            trim_percentile = float(self.config.best_score_trim_percentile)
            if not (0.0 < trim_percentile <= 100.0):
                raise ValueError(
                    "best_score_trim_percentile must satisfy 0 < value <= 100."
                )

            bulk_dchi2 = other_dchi2
            if trim_percentile < 100.0:
                cutoff = float(np.percentile(other_dchi2, trim_percentile))
                trimmed_dchi2 = other_dchi2[other_dchi2 <= cutoff]
                if trimmed_dchi2.shape[0] >= 2:
                    bulk_dchi2 = trimmed_dchi2

            if bulk_dchi2.shape[0] < 2:
                bulk_dchi2 = other_dchi2

            med = float(np.median(bulk_dchi2))
            std = float(np.std(bulk_dchi2))
            score = (best[2] - med) / std if std > 0 else float("nan")
        else:
            med = std = score = float("nan")

        quality = self._quality_for_point(float(best[0]), float(best[1]), grid_metrics_all)

        return BestCandidate(
            t0=float(best[0]),
            teff=float(best[1]),
            dchi2=float(best[2]),
            med_others=med,
            std_others=std,
            score=float(score),
            quality=quality,
        )

    @staticmethod
    def _quality_for_point(t0: float, teff: float, grid_metrics_all: np.ndarray) -> CandidateQuality:
        if grid_metrics_all is None or grid_metrics_all.size == 0:
            return CandidateQuality(
                n_window=0,
                n_contrib=0,
                n_eff=0.0,
                peak_frac=0.0,
                rho1=0.0,
                longest_run=0,
            )

        dist = np.abs(grid_metrics_all[:, 0] - t0) + np.abs(grid_metrics_all[:, 1] - teff)
        i = int(np.argmin(dist))
        row = grid_metrics_all[i]
        return CandidateQuality(
            n_window=int(round(float(row[3]))),
            n_contrib=int(round(float(row[4]))),
            n_eff=float(row[5]),
            peak_frac=float(row[6]),
            rho1=float(row[7]),
            longest_run=int(round(float(row[8]))),
        )


    # ----------------------------
    # Plot sugar APIs
    # ----------------------------
    def _require_result(self) -> AnomalyResult:
        if self._last_result is None:
            raise RuntimeError("Finder.run() has not been called yet.")
        return self._last_result

    def plot_lc(self, **kwargs):
        """
        Plot light curve with single lens model using the last result.
        """
        result = self._require_result()
        return self.plotter.plot_lc(result, **kwargs)

    def plot_residual(self, **kwargs):
        """
        Plot residuals using the last result.
        """
        result = self._require_result()
        return self.plotter.plot_residual(result, **kwargs)

    def plot_anomaly_window(self, **kwargs):
        """
        Plot residuals around the best anomaly window.
        """
        result = self._require_result()
        return self.plotter.plot_anomaly_window(result, **kwargs)

    def plot_result(self, **kwargs):
        """
        Full 3-panel diagnostic plot.
        """
        result = self._require_result()
        return self.plotter.plot_result(result, **kwargs)

    def plot_template_free(self, **kwargs):
        """
        Plot the last template-free anomaly search result.
        """
        if self._last_template_free_result is None:
            raise RuntimeError("Finder.run_template_free() has not been called yet.")
        return self._last_template_free_result.plot(**kwargs)
