from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, Optional, Sequence
import logging

import numpy as np
import jax
import jax.numpy as jnp

from .config import FinderConfig
from .singlelens_fit import (
    SingleLensFitResult,
    PSPLFitter,
    CPPPSPLFitter,
    FSPLFitter,
    VBMFiniteDiffFSPLFitter,
    BICSingleLensFitter,
    evaluate_single_lens_fixed,
)
from .singlelens_model import (
    A_pspl_func,
    A_fspl_logrho_func,
)
from .parallax_backend import NativeParallaxFitter
from .plot import AnomalyPlotter
from .seasons import SeasonSplitter
from .extract import ResultExtractor
from .runner import SeasonGridRunner
from .models import AnomalyResult, BestCandidate, CandidateQuality, SeasonSummary
from .template_free import TemplateFreeScanner, TemplateFreeSearchConfig, TemplateFreeSearchResult
from .pspl_fft import PSPLFFTScanner
from .effect_detection import (
    detect_fspl_from_pspl_fit,
    detect_parallax_from_pspl_fit,
    detect_physical_effects as _detect_physical_effects,
)
from .effect_routing import RoutingThresholds, route_candidates
from .singlelens_fallback import (
    FallbackConfig,
    FallbackResult,
    make_effect_fitter,
    run_robust_fallback,
    run_staged_joint_fallback,
)
from .exact_probe import run_exact_probe
from .effect_aware import EffectAwareFinderResult, match_planet_candidates
from .parallax_backend import native_parallax_effect_score
from .contamination import protected_support_mask

if TYPE_CHECKING:
    from .planet_signal import PlanetSignalConfig


def _vbm_coordinate_string(ra_deg: float, dec_deg: float) -> str:
    """Convert decimal degrees to VBM's ``HH:MM:SS +/-DD:MM:SS`` input."""
    ra_hours = (float(ra_deg) / 15.0) % 24.0
    rah = int(ra_hours)
    ram_float = (ra_hours - rah) * 60.0
    ram = int(ram_float)
    ras = (ram_float - ram) * 60.0
    sign = "+" if float(dec_deg) >= 0.0 else "-"
    dec_abs = abs(float(dec_deg))
    ded = int(dec_abs)
    dem_float = (dec_abs - ded) * 60.0
    dem = int(dem_float)
    des = (dem_float - dem) * 60.0
    return f"{rah:02d}:{ram:02d}:{ras:08.5f} {sign}{ded:02d}:{dem:02d}:{des:07.4f}"

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
            self.fitter = make_effect_fitter(
                self.config, "annual_parallax", float(tref)
            ).fitter
            return

        if k == "pspl_space_parallax":
            self.fitter = make_effect_fitter(
                self.config, "space_parallax", float(tref)
            ).fitter
            return

        if k == "fspl_space_parallax":
            self.fitter = make_effect_fitter(
                self.config, "fspl_space_parallax", float(tref)
            ).fitter
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
                observer_convention=str(self.config.parallax_observer_convention),
                time_scale=str(self.config.parallax_time_scale),
                time_offset=float(self.config.parallax_time_offset),
                ephemeris_extrapolation=str(self.config.parallax_extrapolation),
            )
            return
    
        # k == "fspl_parallax"
        self.fitter = make_effect_fitter(
            self.config, "fspl_parallax", float(tref)
        ).fitter


    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------
    def fit_single_lens(
        self,
        time,
        flux,
        ferr,
        x0=None,
        *,
        data_kind: Literal["flux", "mag"] = "flux",
    ) -> SingleLensFitResult:
        """
        Run only the single-lens fit selected by the current configuration.

        Parameters
        ----------
        time, flux, ferr : array-like
            One-dimensional light-curve arrays. With ``data_kind="mag"``,
            ``flux`` and ``ferr`` are interpreted as magnitude and magnitude
            error, respectively, and are converted to relative flux internally.
        x0 : array-like, optional
            Initial guess for the nonlinear model parameters.
            If omitted, initial values are estimated from a scan of the light curve.
        data_kind : {"flux", "mag"}, optional
            Input photometry representation. The default ``"flux"`` preserves
            the existing API. Magnitude inputs are converted to relative flux.

        Returns
        -------
        SingleLensFitResult
            Result of the single-lens fit.
        """
        time_j, flux_j, ferr_j, x0_j, time_np, _, _ = self._to_arrays(
            time, flux, ferr, x0, data_kind=data_kind
        )
        self._ensure_fitter(float(np.median(time_np)))
        if x0_j is None:
            return self._fit_from_auto_initial_guesses(time_j, flux_j, ferr_j, time_np)
        return self.fitter.fit(time_j, flux_j, ferr_j, x0_j)

    def detect_effects(
        self,
        fit: Optional[SingleLensFitResult] = None,
        *,
        parallax_projector=None,
        space_parallax_projector=None,
        route: bool = True,
        routing_thresholds: Optional[RoutingThresholds] = None,
        include_fspl: bool = True,
        execute_exact_probe: bool = True,
        planet_mask=None,
        **fspl_kwargs,
    ):
        """Run the physical detector in shadow mode on a PSPL fit.

        The detector never calls a nonlinear FSPL/parallax optimizer.  Pass an
        existing PSPL ``fit`` (or use the last :meth:`run` result), and pass an
        observer projector when annual/space parallax should be tested.  With
        ``route=True`` the returned candidates also carry a three-stage routing
        decision; the raw detector diagnostics remain unchanged.
        """
        if fit is None:
            if self._last_result is None or self._last_result.fit is None:
                raise ValueError("detect_effects requires a PSPL fit or a previous Finder.run result.")
            fit = self._last_result.fit
        if parallax_projector is None and self.config.fitter_kind == "pspl_parallax":
            parallax_projector = getattr(self.fitter, "_P", None)
        if space_parallax_projector is None and self.config.fitter_kind == "pspl_space_parallax":
            space_parallax_projector = getattr(self.fitter, "_P", None)
        if parallax_projector is None and self.config.ra_deg is not None and self.config.dec_deg is not None:
            from .trajectory import make_parallax_projector, make_space_parallax_projector
            try:
                parallax_projector = make_parallax_projector(
                    self.config.ra_deg,
                    self.config.dec_deg,
                    float(self.config.tref if self.config.tref is not None else np.median(fit.time)),
                    use_HJD=self.config.parallax_time_scale == "hjd",
                )
            except Exception as exc:
                raise ValueError(
                    "annual-parallax detector geometry could not be constructed"
                ) from exc
        if space_parallax_projector is None and self.config.satellite_ephemeris_path is not None and self.config.ra_deg is not None and self.config.dec_deg is not None:
            from .trajectory import make_space_parallax_projector
            try:
                space_parallax_projector = make_space_parallax_projector(
                    self.config.ra_deg,
                    self.config.dec_deg,
                    float(self.config.tref if self.config.tref is not None else np.median(fit.time)),
                    self.config.satellite_ephemeris_path,
                    use_HJD=self.config.parallax_time_scale == "hjd",
                    convention="gulls" if self.config.parallax_observer_convention == "gulls" else "vbm",
                )
            except Exception as exc:
                raise ValueError(
                    "space-parallax detector geometry could not be constructed"
                ) from exc
        candidates = _detect_physical_effects(
            fit,
            parallax_projector=parallax_projector,
            space_parallax_projector=space_parallax_projector,
            include_fspl=include_fspl,
            planet_mask=planet_mask,
            **fspl_kwargs,
        )
        if not route:
            return candidates
        thresholds = RoutingThresholds() if routing_thresholds is None else routing_thresholds
        routed = route_candidates(candidates, thresholds)
        if not execute_exact_probe:
            return routed
        probed = []
        for candidate in routed:
            if candidate.decision != "exact_probe":
                probed.append(candidate)
                continue
            projector = space_parallax_projector if candidate.effect == "space_parallax" else parallax_projector
            try:
                probe = run_exact_probe(fit, projector, candidate, **fspl_kwargs)
                promoted = probe.promoted_candidate
                probed.append(promoted)
            except Exception as exc:
                probed.append(
                    candidate.with_decision(
                        "exact_probe",
                        ("exact_probe_failed", type(exc).__name__),
                    )
                )
        return tuple(probed)

    def robust_fallback(
        self,
        time,
        flux,
        ferr,
        *,
        fit: Optional[SingleLensFitResult] = None,
        base_seed=None,
        candidates=(),
        effect: str = "mixed",
        config: FallbackConfig = FallbackConfig(),
        protected_mask=None,
        known_anomaly_mask=None,
    ) -> FallbackResult:
        """Run detector-seeded contamination-aware refitting.

        This is an optional path and does not alter :meth:`run`.  The fitter is
        selected from the routed physical effect, while segmentation alternates
        with that model using inflated errors for contiguous anomaly states.
        """
        time_np = np.asarray(time, dtype=float)
        flux_np = np.asarray(flux, dtype=float)
        ferr_np = np.asarray(ferr, dtype=float)
        if fit is None and self._last_result is not None:
            fit = self._last_result.fit
        if base_seed is None:
            if fit is None:
                raise ValueError("robust_fallback requires fit or base_seed.")
            base_seed = np.asarray(fit.params, dtype=float).copy()
            names = tuple(getattr(fit, "param_names", ()))
            if "rho" in names:
                rho_index = names.index("rho")
                base_seed[rho_index] = np.log(
                    max(abs(float(base_seed[rho_index])), 1.0e-12)
                )
        self._ensure_fitter(float(np.median(time_np)))
        candidate_tuple = tuple(candidates)
        resolved_effect = self._resolve_fallback_effect(effect, candidate_tuple)
        if resolved_effect is None:
            raise ValueError(
                "robust_fallback could not resolve an effect-specific fitter; "
                "provide candidates or an explicit effect."
            )
        spec = None
        try:
            spec = make_effect_fitter(
                self.config, resolved_effect, float(np.median(time_np))
            )
        except ValueError:
            # A joint space-parallax backend can be unavailable when the
            # observer ephemeris does not cover this event. Mixed fallback
            # must still run its independently viable FSPL/annual stages.
            if not (
                effect == "mixed"
                and resolved_effect in {"fspl_parallax", "fspl_space_parallax"}
            ):
                raise
        if (
            spec is not None
            and base_seed is not None
            and np.asarray(base_seed).size not in (3, spec.parameter_dimension)
        ):
            raise ValueError(
                f"Base seed dimension {np.asarray(base_seed).size} does not match "
                f"{resolved_effect} fitter dimension {spec.parameter_dimension}."
            )
        config = FallbackConfig(
            tE_factors=config.tE_factors,
            u0_sign_flip=config.u0_sign_flip,
            parallax_radii=config.parallax_radii,
            parallax_angle_steps=config.parallax_angle_steps,
            max_seeds=config.max_seeds,
            contamination=config.contamination,
            max_point_parameter_change=config.max_point_parameter_change,
            max_basin_distance=config.max_basin_distance,
            parameter_dimension=(
                None if spec is None else spec.parameter_dimension
            ),
            default_logrho=config.default_logrho,
            u0_factors=config.u0_factors,
            rho_over_u0=config.rho_over_u0,
            t_star_factors=config.t_star_factors,
            t0_offsets=config.t0_offsets,
            max_piE=(
                config.max_piE
                if config.max_piE is not None
                else float(self.config.max_piE)
            ),
            min_bic_improvement=config.min_bic_improvement,
        )
        projector = None if spec is None else getattr(spec.fitter, "_P", None)

        def physical_effect_score(value, score_effect: Optional[str] = None) -> float:
            def candidate_score(candidate, detected=None) -> float:
                measured = candidate if detected is None else detected
                if (
                    candidate.effect == "fspl"
                    and candidate.morphology in {
                        "fspl_even_peak",
                        "fspl_flattened_peak",
                    }
                ):
                    # The compact peak is the finite-source signal itself.
                    # Removing it makes the baseline look artificially clean
                    # and can reject a fit that eliminates the full FSPL
                    # topology by orders of magnitude.
                    return max(float(measured.score), 0.0)
                return max(
                    float(measured.score_without_compact_blocks),
                    0.0,
                )

            def relevant(candidate) -> bool:
                if score_effect is None or score_effect == "mixed":
                    return True
                if candidate.effect == "fspl":
                    return "fspl" in score_effect
                return candidate.effect in score_effect or (
                    "parallax" in score_effect and "parallax" in candidate.effect
                )

            if value is fit:
                return float(
                    sum(
                        candidate_score(candidate)
                        for candidate in candidate_tuple
                        if relevant(candidate)
                    )
                )
            scores: list[float] = []
            native_parallax_scored = False
            for candidate in candidate_tuple:
                if not relevant(candidate):
                    continue
                if candidate.effect == "fspl":
                    detected = detect_fspl_from_pspl_fit(value)
                    scores.append(candidate_score(candidate, detected))
                elif candidate.effect in {"annual_parallax", "space_parallax"}:
                    native_evaluator = getattr(value, "parallax_projector", None)
                    if native_evaluator is not None and hasattr(
                        native_evaluator, "jacobian"
                    ):
                        if not native_parallax_scored:
                            scores.append(
                                native_parallax_effect_score(
                                    value,
                                    exclude_mask=candidate.compact_block_mask,
                                )
                            )
                            native_parallax_scored = True
                        continue
                    if projector is None:
                        # A single-effect FSPL stage has no parallax
                        # trajectory. Its acceptance must be based on the
                        # FSPL detector only; the joint native stage evaluates
                        # the parallax component.
                        continue
                    detected = detect_parallax_from_pspl_fit(
                        value,
                        projector,
                        space=candidate.effect == "space_parallax",
                    )
                    scores.append(
                        max(float(detected.score_without_compact_blocks), 0.0)
                    )
                else:
                    continue
            if not scores:
                raise ValueError("No residual physical-effect detector is available.")
            return float(np.sum(scores))

        effect_score_fn = physical_effect_score if candidate_tuple else None
        if resolved_effect in {"fspl_parallax", "fspl_space_parallax"} and effect == "mixed":
            return run_staged_joint_fallback(
                self.config,
                time_np,
                flux_np,
                ferr_np,
                base_seed,
                candidates=candidate_tuple,
                effect=resolved_effect,
                fallback_config=config,
                protected_mask=protected_mask,
                known_anomaly_mask=known_anomaly_mask,
                baseline_fit=fit,
                effect_score_fn=effect_score_fn,
            )
        if spec is None:  # defensive: only mixed staged fallback may omit it
            raise RuntimeError(
                f"No fitter is available for resolved effect {resolved_effect}."
            )
        return run_robust_fallback(
            spec.fitter,
            time_np,
            flux_np,
            ferr_np,
            base_seed,
            candidates=candidate_tuple,
            effect=resolved_effect,
            config=config,
            protected_mask=protected_mask,
            known_anomaly_mask=known_anomaly_mask,
            baseline_fit=fit,
            effect_score_fn=effect_score_fn,
            model_spec=spec,
        )

    @staticmethod
    def _resolve_fallback_effect(effect: str, candidates) -> Optional[str]:
        if effect != "mixed":
            return str(effect)
        effects = {str(candidate.effect) for candidate in candidates}
        if "fspl" in effects and "space_parallax" in effects:
            return "fspl_space_parallax"
        if "fspl" in effects and "annual_parallax" in effects:
            return "fspl_parallax"
        if "fspl" in effects:
            return "fspl"
        if "space_parallax" in effects:
            return "space_parallax"
        if "annual_parallax" in effects:
            return "annual_parallax"
        return None

    def run(
        self,
        time,
        flux,
        ferr,
        x0=None,
        *,
        data_kind: Literal["flux", "mag"] = "flux",
        refit: bool = True,
        verbose: bool = True,
        log: Optional[logging.Logger] = None,
    ) -> AnomalyResult:
        """
        Run the full anomaly-search pipeline.

        Parameters
        ----------
        time, flux, ferr : array-like
            One-dimensional light-curve arrays. With ``data_kind="mag"``,
            pass ``mag`` and ``magerr`` in these positions; they are converted
            to relative flux and flux error before fitting and scanning.
        x0 : array-like, optional
            Initial guess for the single-lens model parameters. If omitted, the
            finder estimates multiple initial values and uses the best fit.
        data_kind : {"flux", "mag"}, optional
            Input photometry representation. ``"mag"`` converts magnitudes to
            a numerically stable relative-flux scale internally.
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
            time, flux, ferr, x0, data_kind=data_kind
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

        best_obj = self._pick_best_candidate(
            clusters_all,
            grid_metrics_all,
            seasons=seasons,
        )

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

    def run_effect_aware(
        self,
        time,
        flux,
        ferr,
        x0=None,
        *,
        data_kind: Literal["flux", "mag"] = "flux",
        run_planet_before: bool = True,
        run_planet_after: bool = True,
        planet_config: Optional["PlanetSignalConfig"] = None,
        planet_fast_mode: bool = True,
        post_physical_max_refits: int = 3,
        routing_thresholds: Optional[RoutingThresholds] = None,
        fallback_config: FallbackConfig = FallbackConfig(),
        verbose: bool = False,
    ) -> EffectAwareFinderResult:
        """Run the explicit planet-before/physical-fallback/planet-after flow.

        ``Finder.run`` remains unchanged.  This method keeps the initial planet
        extraction even when a later robust fallback fails or is rejected.

        By default, routine effect-aware runs use a one-iteration, one-branch
        planet pass.  Set ``planet_fast_mode=False`` to use the supplied (or
        default) full configuration from the outset.  A caller that needs a
        full beam search after routing can pass ``planet_fast_mode=False``.
        """
        time_j, flux_j, ferr_j, x0_j, time_np, flux_np, ferr_np = self._to_arrays(
            time, flux, ferr, x0, data_kind=data_kind
        )
        self._ensure_fitter(float(np.median(time_np)))
        if x0_j is None:
            initial_fit = self._fit_from_auto_initial_guesses(time_j, flux_j, ferr_j, time_np)
        else:
            initial_fit = self.fitter.fit(time_j, flux_j, ferr_j, x0_j)

        planet_before = None
        reason_codes: list[str] = []
        if run_planet_before:
            from .planet_signal import PlanetSignalConfig, PlanetSignalExtractor
            resolved_planet_config = PlanetSignalConfig() if planet_config is None else planet_config
            if planet_fast_mode:
                resolved_planet_config = replace(
                    resolved_planet_config,
                    baseline_mode="beam_interval",
                    beam_max_iter=1,
                    beam_width=1,
                    beam_candidates_per_iter=1,
                    beam_probe_only=True,
                )
            planet_before = PlanetSignalExtractor(self, resolved_planet_config).run(
                time_np, flux_np, ferr_np, initial_fit=initial_fit, refit=False, verbose=verbose
            )
            reason_codes.append("planet_scan_before_completed")

        # Detector routing is intentionally kept separate from the fallback.
        # The fallback factory is the only place that selects the native
        # parallax evaluator, so a skip path remains cheap.
        thresholds = RoutingThresholds() if routing_thresholds is None else routing_thresholds
        # This entry point owns an exact-probe executor, so the router should
        # never mark it unavailable and leave a boundary candidate stranded.
        thresholds = replace(thresholds, exact_probe_available=True)
        preliminary_effects = self.detect_effects(
            fit=initial_fit,
            route=True,
            routing_thresholds=thresholds,
            execute_exact_probe=True,
        )
        preliminary_effects = tuple(preliminary_effects)
        preliminary_fallback_candidates = tuple(
            candidate
            for candidate in preliminary_effects
            if getattr(candidate, "decision", "skip") == "fallback"
        )
        mask_protection = np.zeros(time_np.size, dtype=bool)
        for candidate in preliminary_fallback_candidates:
            if candidate.seed_parameters is None:
                continue
            mask_protection |= protected_support_mask(
                time_np,
                candidate.effect,
                candidate.seed_parameters,
            )
        # A cheap first pass keeps ordinary events fast.  Preserve the full
        # beam search for an event that already looks planetary, or for one
        # that the physical-effect router judges worthy of fallback.  Doing
        # this before fallback also gives the contamination mask its best
        # available planet intervals.
        if (
            run_planet_before
            and planet_fast_mode
            and (
                bool(
                    planet_before is not None
                    and planet_before.initial_seed is not None
                    and np.isfinite(planet_before.initial_seed.dchi2)
                    and planet_before.initial_seed.dchi2
                    >= float(
                        (PlanetSignalConfig() if planet_config is None else planet_config).seed_min_dchi2
                    )
                )
                or bool(preliminary_fallback_candidates)
            )
        ):
            from .planet_signal import PlanetSignalConfig, PlanetSignalExtractor
            full_planet_config = PlanetSignalConfig() if planet_config is None else planet_config
            planet_before = PlanetSignalExtractor(self, full_planet_config).run(
                time_np,
                flux_np,
                ferr_np,
                initial_fit=initial_fit,
                initial_seed=planet_before.initial_seed,
                refit=False,
                verbose=verbose,
                mask_protection=mask_protection,
            )
            reason_codes.append("planet_scan_before_escalated")

        # The post-planet PSPL is the actual baseline for physical-effect
        # diagnosis. The preliminary detector above is only a cheap escalation
        # hint; it must not select or seed the expensive fallback.
        baseline_fit = (
            initial_fit
            if planet_before is None
            else planet_before.refined_fit
        )
        planet_mask = (
            None
            if planet_before is None
            else np.asarray(planet_before.signal_mask, dtype=bool)
        )
        effects = tuple(
            self.detect_effects(
                fit=baseline_fit,
                route=True,
                routing_thresholds=thresholds,
                execute_exact_probe=True,
                planet_mask=planet_mask,
            )
        )
        fallback_candidates = tuple(
            candidate
            for candidate in effects
            if getattr(candidate, "decision", "skip") == "fallback"
        )
        reason_codes.append("post_planet_effect_diagnostics_completed")
        fallback_result = None
        selected_fit = baseline_fit
        routing_decision = effects
        if fallback_candidates:
            known_anomaly_mask = np.zeros(time_np.size, dtype=bool)
            if planet_before is not None and planet_before.candidates:
                cadence = (
                    float(np.median(np.diff(np.sort(time_np))))
                    if time_np.size > 1
                    else 0.0
                )
                for candidate in planet_before.candidates:
                    start = float(candidate.t_start)
                    end = float(candidate.t_end)
                    padding = max(
                        2.0 * cadence,
                        0.25 * max(end - start, 0.0),
                    )
                    known_anomaly_mask |= (
                        (time_np >= start - padding)
                        & (time_np <= end + padding)
                    )
            # The planet mask and physical-effect support are complementary.
            # A time-binned low-frequency residual must remain visible to the
            # FSPL/parallax fitter; otherwise the fallback segmenter can label
            # the very signal it is meant to fit as generic contamination.
            smooth_effect_support = None
            if planet_before is not None and planet_before.candidates:
                from .residual_decomposition import decompose_binned_residual

                cadence = (
                    float(np.median(np.diff(np.sort(time_np))))
                    if time_np.size > 1
                    else 0.0
                )
                candidate_scales = [
                    max(
                        0.5
                        * max(
                            float(candidate.t_end)
                            - float(candidate.t_start),
                            0.0,
                        ),
                        cadence,
                    )
                    for candidate in planet_before.candidates
                ]
                characteristic_scale = float(
                    np.median(candidate_scales)
                )
                if (
                    planet_before.initial_seed is not None
                    and np.isfinite(planet_before.initial_seed.teff)
                    and abs(float(planet_before.initial_seed.teff)) > 0.0
                ):
                    characteristic_scale = abs(
                        float(planet_before.initial_seed.teff)
                    )
                initial_z = (
                    np.asarray(baseline_fit.residual, dtype=float)
                    / np.maximum(ferr_np, 1.0e-12)
                )
                decomposition = decompose_binned_residual(
                    time_np,
                    initial_z,
                    characteristic_scale=max(
                        characteristic_scale,
                        cadence,
                        1.0e-12,
                    ),
                )
                # Threshold the binned trend so low-level noise does not
                # become protected support. Localized planet samples are
                # explicitly removed from this support.
                smooth_effect_support = (
                    np.abs(decomposition.smooth_z) >= 1.5
                ).astype(float)
                # Preserve the effect detector's geometric support as well.
                # The binned support augments it around coherent inner-event
                # residuals; it does not replace the parallax wings or FSPL
                # shoulders selected by the diagnostic.
                for candidate in fallback_candidates:
                    if candidate.seed_parameters is None:
                        continue
                    smooth_effect_support = np.maximum(
                        smooth_effect_support,
                        protected_support_mask(
                            time_np,
                            candidate.effect,
                            candidate.seed_parameters,
                        ).astype(float),
                    )
                smooth_effect_support[known_anomaly_mask] = 0.0
                if not np.any(smooth_effect_support > 0.0):
                    smooth_effect_support = None
            try:
                fallback_result = self.robust_fallback(
                    time_np, flux_np, ferr_np, fit=baseline_fit,
                    candidates=fallback_candidates, effect="mixed",
                    config=fallback_config,
                    protected_mask=smooth_effect_support,
                    known_anomaly_mask=known_anomaly_mask,
                )
                if fallback_result.success:
                    selected_fit = fallback_result.fit
                    reason_codes.append("fallback_accepted")
                else:
                    reason_codes.append("fallback_rejected")
            except Exception as exc:
                reason_codes.append("fallback_failed")
                fallback_result = None
                logger.debug("effect-aware fallback failed", exc_info=True)
        else:
            reason_codes.append("fallback_skipped")

        planet_after = None
        accepted = bool(fallback_result is not None and fallback_result.success)
        if accepted and run_planet_after:
            planet_after = self.refine_planet_after_physical(
                time_np,
                flux_np,
                ferr_np,
                selected_fit=selected_fit,
                effect=str(fallback_result.effect),
                planet_config=planet_config,
                max_refits=post_physical_max_refits,
                verbose=verbose,
            )
            reason_codes.append("planet_after_fixed_family_warm_start")
            reason_codes.append("planet_scan_after_completed")
        elif not accepted:
            reason_codes.append("planet_after_not_needed")

        matches = match_planet_candidates(
            planet_before, planet_after if accepted else planet_before,
            season_gap=float(self.config.gap),
        )
        final_source = planet_after if accepted and planet_after is not None else planet_before
        final_candidates = tuple(() if final_source is None else final_source.candidates)
        diagnostics = {
            "n_effect_candidates": len(effects),
            "n_fallback_candidates": len(fallback_candidates),
            "fallback_accepted": bool(accepted),
            "before_candidate_count": 0 if planet_before is None else len(planet_before.candidates),
            "after_candidate_count": 0 if planet_after is None else len(planet_after.candidates),
            "candidate_categories": {category: sum(match.category == category for match in matches) for category in {match.category for match in matches}},
            "routing_thresholds": thresholds,
        }
        return EffectAwareFinderResult(
            initial_fit=initial_fit,
            selected_fit=selected_fit,
            effect_candidates=effects,
            routing_decision=routing_decision,
            fallback_result=fallback_result,
            planet_before=planet_before,
            planet_after=planet_after,
            candidate_matches=matches,
            final_candidates=final_candidates,
            reason_codes=tuple(reason_codes),
            diagnostics=diagnostics,
        )

    @staticmethod
    def _physical_model_kind(effect: str) -> str:
        return {
            "fspl": "fspl",
            "annual_parallax": "pspl_parallax",
            "space_parallax": "pspl_space_parallax",
            "fspl_parallax": "fspl_parallax",
            "fspl_space_parallax": "fspl_space_parallax",
        }.get(str(effect), str(effect))

    def refine_planet_after_physical(
        self,
        time,
        flux,
        ferr,
        *,
        selected_fit: SingleLensFitResult,
        effect: str,
        planet_config: Optional["PlanetSignalConfig"] = None,
        max_refits: int = 3,
        verbose: bool = False,
    ):
        """Re-detect a planet and locally polish one accepted model family.

        The fallback's multistart search is not repeated.  A fresh
        effect-specific fitter starts from the accepted solution, the regular
        planet scanner proposes a localized mask, and one continuation fit is
        allowed per pass.  No pre-fallback PSPL window is forced into the
        post-physical result.
        """
        from .planet_signal import PlanetSignalConfig, PlanetSignalExtractor

        max_refits = max(0, int(max_refits))
        resolved = PlanetSignalConfig() if planet_config is None else planet_config
        post_config = replace(
            resolved,
            baseline_mode="beam_interval",
            beam_max_iter=max_refits,
            beam_width=1,
            beam_candidates_per_iter=1,
            beam_probe_only=False,
            flat_baseline_on_masked_peak=False,
        )
        time_np = np.asarray(time, dtype=float)
        spec = make_effect_fitter(
            self.config,
            str(effect),
            float(np.median(time_np)),
        )
        if not hasattr(selected_fit, "model_kind"):
            object.__setattr__(
                selected_fit,
                "model_kind",
                self._physical_model_kind(str(effect)),
            )
        return PlanetSignalExtractor(self, post_config).run(
            time_np,
            np.asarray(flux, dtype=float),
            np.asarray(ferr, dtype=float),
            initial_fit=selected_fit,
            refit=False,
            verbose=verbose,
            prior_signal_windows=(),
            freeze_baseline=False,
            baseline_refit_fitter=spec.fitter,
        )

    def evaluate_saved_physical_solution(
        self,
        time,
        flux,
        ferr,
        *,
        effect: str,
        params,
        fs: Optional[float] = None,
        fb: Optional[float] = None,
    ) -> SingleLensFitResult:
        """Reconstruct an accepted physical model without optimizing it."""
        time_np = np.asarray(time, dtype=float)
        spec = make_effect_fitter(
            self.config,
            str(effect),
            float(np.median(time_np)),
        )
        if not hasattr(spec.fitter, "evaluate_fixed"):
            raise TypeError(
                f"The {effect} fitter cannot evaluate saved parameters."
            )
        # ``params`` is the public/physical representation emitted by a fit
        # (physical ``tE`` and ``rho``).  The native evaluator's seed
        # contract is also physical ``tE`` but logarithmic finite-source
        # radius.  Both spellings have existed in the backends, so normalize
        # the canonical rho slot before handing it to ``_raw_seed``.  Without
        # this conversion a saved rho=0.02 was interpreted as log(rho)=0.02,
        # producing rho=exp(0.02)=1.02 and a catastrophically bad fixed
        # evaluation in post-physical refinement.
        seed = np.asarray(params, dtype=float).reshape(-1).copy()
        raw_names = tuple(spec.raw_parameter_names)
        physical_names = tuple(spec.parameter_names)
        for raw_name in ("logrho", "log_rho"):
            if raw_name in raw_names:
                raw_index = raw_names.index(raw_name)
                if "rho" not in physical_names:
                    raise ValueError("finite-source evaluator is missing a public rho parameter")
                physical_index = physical_names.index("rho")
                seed[raw_index] = np.log(max(abs(float(seed[physical_index])), 1.0e-12))
                break
        fit = spec.fitter.evaluate_fixed(
            time_np,
            np.asarray(flux, dtype=float),
            np.asarray(ferr, dtype=float),
            seed,
        )
        if fs is not None and fb is not None:
            fitted_fs = float(np.asarray(fit.fs))
            fitted_fb = float(np.asarray(fit.fb))
            if not np.isfinite(fitted_fs) or abs(fitted_fs) <= 1.0e-30:
                raise ValueError("Cannot reconstruct magnification from zero fs.")
            magnification = (
                np.asarray(fit.model_flux, dtype=float) - fitted_fb
            ) / fitted_fs
            model_flux = float(fs) * magnification + float(fb)
            residual = np.asarray(flux, dtype=float) - model_flux
            ferr_np = np.maximum(np.asarray(ferr, dtype=float), 1.0e-12)
            chi2 = float(np.sum((residual / ferr_np) ** 2))
            fit = replace(
                fit,
                fs=float(fs),
                fb=float(fb),
                model_flux=model_flux,
                residual=residual,
                chi2=chi2,
                chi2_dof=chi2 / max(
                    time_np.size - int(spec.parameter_dimension),
                    1,
                ),
            )
        object.__setattr__(
            fit,
            "model_kind",
            self._physical_model_kind(str(effect)),
        )
        return fit

    def run_template_free(
        self,
        time,
        flux,
        ferr,
        x0=None,
        *,
        data_kind: Literal["flux", "mag"] = "flux",
        fit: Optional[SingleLensFitResult] = None,
        config: Optional[TemplateFreeSearchConfig] = None,
    ) -> TemplateFreeSearchResult:
        """
        Run a template-free anomaly search on single-lens residuals.

        This leaves the existing bell-template anomaly pipeline untouched.

        Parameters
        ----------
        time, flux, ferr : array-like
            One-dimensional light-curve arrays. With ``data_kind="mag"``,
            ``flux`` and ``ferr`` are interpreted as magnitude and magnitude
            error. Flux errors are used to normalize residuals even when
            ``fit`` is supplied.
        x0 : array-like, optional
            Nonlinear initial values used only when ``fit`` is omitted.
        data_kind : {"flux", "mag"}, optional
            Input photometry representation.
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
            time, flux, ferr, x0, data_kind=data_kind
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
    def _to_arrays(self, time, flux, ferr, x0, *, data_kind: Literal["flux", "mag"] = "flux"):
        """
        Validate inputs and convert them to flux-space NumPy and JAX arrays.

        ``data_kind="mag"`` treats ``flux`` and ``ferr`` as magnitude and
        magnitude error. The conversion uses a median magnitude zero point so
        that the relative flux is of order unity, avoiding numerical underflow
        for the large magnitude zero points commonly used in catalogs.
        """
        time_np = np.asarray(time, dtype=float)
        flux_np = np.asarray(flux, dtype=float)
        ferr_np = np.asarray(ferr, dtype=float)

        if data_kind not in {"flux", "mag"}:
            raise ValueError("data_kind must be either 'flux' or 'mag'.")
        if time_np.ndim != 1 or flux_np.ndim != 1 or ferr_np.ndim != 1:
            raise ValueError("time/flux/ferr must be 1D arrays.")
        if not (len(time_np) == len(flux_np) == len(ferr_np)):
            raise ValueError("time/flux/ferr must have the same length.")
        if np.any(~np.isfinite(time_np)) or np.any(~np.isfinite(flux_np)) or np.any(~np.isfinite(ferr_np)):
            raise ValueError("time/flux/ferr must be finite.")
        if np.any(ferr_np <= 0):
            raise ValueError("ferr must be positive.")

        if data_kind == "mag":
            mag0 = float(np.median(flux_np))
            exponent = -0.4 * np.log(10.0) * (flux_np - mag0)
            float_info = np.finfo(np.float32)
            exponent = np.clip(exponent, np.log(float_info.tiny), np.log(float_info.max))
            flux_np = np.exp(exponent)
            ferr_np = np.maximum(
                (np.log(10.0) / 2.5) * flux_np * ferr_np,
                float_info.tiny,
            )

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

        starts = [np.asarray(x0, dtype=float) for x0 in guesses]

        for x0 in starts:
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
        if isinstance(self.fitter, NativeParallaxFitter):
            return self.fitter.evaluate_fixed(time_j, flux_j, ferr_j, x0_j)

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
        if cfg.fitter_kind == "pspl":
            return self._estimate_pspl_fft_initial_guesses(
                time_j=time_j,
                flux_j=flux_j,
                ferr_j=ferr_j,
                time_np=time_np,
            )

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

    def _estimate_pspl_fft_initial_guesses(
        self,
        *,
        time_j: jnp.ndarray,
        flux_j: jnp.ndarray,
        ferr_j: jnp.ndarray,
        time_np: np.ndarray,
    ) -> np.ndarray:
        """Estimate PSPL starts with a profiled ``(u0, teff, t0)`` FFT bank."""
        cfg = self.config
        if cfg.auto_init_teff_min <= 0 or cfg.auto_init_teff_max <= 0:
            raise ValueError("auto_init_teff_min and auto_init_teff_max must be positive.")
        if cfg.auto_init_teff_max < cfg.auto_init_teff_min:
            raise ValueError("auto_init_teff_max must be >= auto_init_teff_min.")
        if cfg.auto_init_u0_min <= 0 or cfg.auto_init_u0_max <= 0:
            raise ValueError("auto_init_u0_min and auto_init_u0_max must be positive.")
        if cfg.auto_init_u0_max < cfg.auto_init_u0_min:
            raise ValueError("auto_init_u0_max must be >= auto_init_u0_min.")

        n_teff = int(cfg.auto_init_teff_grid_n)
        n_u0 = int(cfg.auto_init_u0_grid_n)
        top_k = int(cfg.auto_init_fft_top_k)
        if n_teff < 1 or n_u0 < 1:
            raise ValueError("PSPL FFT template-grid sizes must be at least one.")
        if top_k < 1:
            raise ValueError("auto_init_fft_top_k must be at least one.")

        teff_grid = np.geomspace(
            float(cfg.auto_init_teff_min),
            float(cfg.auto_init_teff_max),
            n_teff,
        )
        u0_grid = np.geomspace(
            float(cfg.auto_init_u0_min),
            float(cfg.auto_init_u0_max),
            n_u0,
        )
        scanner = PSPLFFTScanner(
            grid_dt=cfg.auto_init_fft_grid_dt,
            max_grid_points=int(cfg.auto_init_fft_max_grid_points),
        )
        search = scanner.search(
            time_np,
            np.asarray(jax.device_get(flux_j), dtype=float),
            np.asarray(jax.device_get(ferr_j), dtype=float),
            u0_grid=u0_grid,
            teff_grid=teff_grid,
            top_k=top_k,
        )

        if search.candidates:
            return np.asarray(
                [
                    self._build_initial_vector(candidate.t0, candidate.tE, candidate.u0)
                    for candidate in search.candidates
                ],
                dtype=float,
            )

        # Keep the automatic workflow recoverable for flat or numerically
        # singular light curves. The subsequent fitter will decide whether
        # this conservative seed is usable.
        flux_np = np.asarray(jax.device_get(flux_j), dtype=float)
        i_peak = int(np.nanargmax(flux_np))
        t0 = float(time_np[i_peak])
        teff = float(np.sqrt(float(cfg.auto_init_teff_min) * float(cfg.auto_init_teff_max)))
        u0 = float(np.sqrt(float(cfg.auto_init_u0_min) * float(cfg.auto_init_u0_max)))
        return np.asarray([self._build_initial_vector(t0, teff / u0, u0)], dtype=float)

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
        if k in {"fspl_parallax", "fspl_space_parallax"}:
            return np.asarray([t0, tE, u0, float(self.config.auto_init_logrho), 0.0, 0.0], dtype=float)
        raise ValueError(f"Unknown fitter_kind '{k}'.")

    def _pick_best_candidate(
        self,
        clusters_all: np.ndarray,
        grid_metrics_all: np.ndarray,
        *,
        seasons: Optional[Sequence[SeasonSummary]] = None,
    ) -> Optional[BestCandidate]:
        """
        Select the strongest accepted candidate and score it against raw clusters.

        Candidate-quality criteria are intentionally applied only after raw
        cluster extraction. The score background therefore does not change
        when selection thresholds are adjusted.
        """
        if clusters_all is None or clusters_all.size == 0:
            return None

        raw_clusters = np.asarray(clusters_all, dtype=float)
        raw_clusters = raw_clusters[np.isfinite(raw_clusters).all(axis=1)]
        if raw_clusters.size == 0:
            return None

        candidates = self._accepted_candidates(raw_clusters, grid_metrics_all)
        if candidates.size == 0:
            return None

        max_ind = int(np.argmax(candidates[:, 2]))
        best = candidates[max_ind]
        references = self._score_reference_clusters(
            best,
            raw_clusters,
            seasons=seasons,
        )
        bulk_dchi2 = self._upper_clip_score_background(references[:, 2])

        if bulk_dchi2.shape[0] >= 2:
            med = float(np.median(bulk_dchi2))
            std = self._robust_scale(bulk_dchi2)
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
            n_score_reference=int(bulk_dchi2.shape[0]),
        )

    def _accepted_candidates(
        self,
        raw_clusters: np.ndarray,
        grid_metrics_all: np.ndarray,
    ) -> np.ndarray:
        criteria = self.config.candidate_criteria
        if criteria is None:
            return raw_clusters

        accepted = []
        for cluster in raw_clusters:
            quality = self._quality_for_point(
                float(cluster[0]),
                float(cluster[1]),
                grid_metrics_all,
            )
            if criteria.accepts(dchi2=float(cluster[2]), quality=quality):
                accepted.append(cluster)
        if not accepted:
            return np.zeros((0, 3), dtype=float)
        return np.asarray(accepted, dtype=float)

    def _score_reference_clusters(
        self,
        best: np.ndarray,
        raw_clusters: np.ndarray,
        *,
        seasons: Optional[Sequence[SeasonSummary]] = None,
    ) -> np.ndarray:
        references = np.asarray(raw_clusters, dtype=float)

        if seasons is not None:
            for season in seasons:
                if float(season.t_start) <= float(best[0]) <= float(season.t_end):
                    references = references[
                        (references[:, 0] >= float(season.t_start))
                        & (references[:, 0] <= float(season.t_end))
                    ]
                    break

        same = (
            np.isclose(references[:, 0], best[0], rtol=0.0, atol=1e-9)
            & np.isclose(references[:, 1], best[1], rtol=1e-12, atol=1e-12)
            & np.isclose(references[:, 2], best[2], rtol=1e-12, atol=1e-12)
        )
        same_idx = np.flatnonzero(same)
        if same_idx.size:
            references = np.delete(references, int(same_idx[0]), axis=0)

        references = references[
            np.isfinite(references).all(axis=1)
            & (references[:, 1] > 0.0)
        ]
        if references.size == 0:
            return np.zeros((0, 3), dtype=float)

        ratio = float(self.config.best_score_teff_ratio)
        if not np.isfinite(ratio) or ratio < 1.0:
            raise ValueError("best_score_teff_ratio must be finite and >= 1.")
        log_distance = np.abs(np.log(references[:, 1] / float(best[1])))
        local = references[log_distance <= np.log(ratio) + 1e-12]

        min_reference = int(self.config.best_score_min_reference_clusters)
        if min_reference < 2:
            raise ValueError("best_score_min_reference_clusters must be >= 2.")
        if local.shape[0] >= min_reference or local.shape[0] == references.shape[0]:
            return local

        nearest = np.argsort(log_distance, kind="stable")
        return references[nearest[: min(min_reference, references.shape[0])]]

    def _upper_clip_score_background(self, values: np.ndarray) -> np.ndarray:
        bulk = np.asarray(values, dtype=float)
        bulk = bulk[np.isfinite(bulk)]
        clip_sigma = float(self.config.best_score_upper_clip_sigma)
        maxiters = int(self.config.best_score_clip_maxiters)
        if clip_sigma <= 0.0:
            raise ValueError("best_score_upper_clip_sigma must be > 0.")
        if maxiters < 0:
            raise ValueError("best_score_clip_maxiters must be >= 0.")
        if not np.isfinite(clip_sigma) or maxiters == 0:
            return bulk

        for _ in range(maxiters):
            if bulk.shape[0] < 3:
                break
            med = float(np.median(bulk))
            scale = self._robust_scale(bulk)
            if not np.isfinite(scale) or scale <= 0.0:
                break
            keep = bulk <= med + clip_sigma * scale
            if np.all(keep) or np.count_nonzero(keep) < 2:
                break
            bulk = bulk[keep]
        return bulk

    @staticmethod
    def _robust_scale(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.shape[0] < 2:
            return float("nan")

        med = float(np.median(values))
        scale = 1.482602218505602 * float(np.median(np.abs(values - med)))
        if scale > 0.0:
            return scale

        q25, q75 = np.percentile(values, [25.0, 75.0])
        scale = float(q75 - q25) / 1.3489795003921634
        if scale > 0.0:
            return scale
        return float(np.std(values))

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
