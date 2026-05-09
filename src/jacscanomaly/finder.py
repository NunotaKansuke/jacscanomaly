from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional
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
    PSPLParallaxFitter,
    FSPLParallaxFitter,
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
    Main entry point of **scanomaly**.

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

      =======================  ===============================
      Model                    x0 parameters
      =======================  ===============================
      PSPL                     (t0, tE, u0)
      FSPL                     (t0, tE, u0, logrho)
      PSPL + parallax          (t0, tE, u0, piEN, piEE)
      FSPL + parallax          (t0, tE, u0, logrho, piEN, piEE)
      =======================  ===============================

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
        valid = {"pspl", "fspl", "pspl_parallax", "fspl_parallax"}
        if k not in valid:
            raise ValueError(
                f"Unknown fitter_kind '{k}'. "
                f"Valid options are: {sorted(valid)}"
            )
    
        # -----------------------------
        # 2) Validate model requirements
        # -----------------------------
        needs_parallax = k.endswith("_parallax")
        if needs_parallax:
            if self.config.ra_deg is None or self.config.dec_deg is None:
                raise ValueError(
                    f"{k} requires ra_deg and dec_deg in FinderConfig "
                    "(sky coordinates are required for annual parallax)."
                )
    
        # -----------------------------
        # 3) Build fitter
        # -----------------------------
        if k == "pspl":
            if self.config.single_fit_backend == "cpp":
                self.fitter = CPPPSPLFitter()
            else:
                self.fitter = PSPLFitter()
            return
    
        if k == "fspl":
            self.fitter = FSPLFitter()
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
            )
            return
    
        # k == "fspl_parallax"
        self.fitter = FSPLParallaxFitter(
            RA=self.config.ra_deg,
            Dec=self.config.dec_deg,
            tref=tref,
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
        verbose : bool, optional
            If True, print progress messages.
        log : logging.Logger, optional
            Logger used for detailed progress reporting.

        Returns
        -------
        AnomalyResult
            Object containing the single-lens fit, residuals,
            per-season cluster summaries, and the best anomaly candidate.
        """
        time_j, flux_j, ferr_j, x0_j, time_np, flux_np, ferr_np = self._to_arrays(
            time, flux, ferr, x0
        )

        self._ensure_fitter(float(np.median(time_np)))

        if x0_j is None:
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

        _, clusters, _ = init_runner.run(
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

    def _build_initial_vector(self, t0: float, tE: float, u0: float) -> np.ndarray:
        k = self.config.fitter_kind
        if k == "pspl":
            return np.asarray([t0, tE, u0], dtype=float)
        if k == "fspl":
            return np.asarray([t0, tE, u0, float(self.config.auto_init_logrho)], dtype=float)
        if k == "pspl_parallax":
            return np.asarray([t0, tE, u0, 0.0, 0.0], dtype=float)
        if k == "fspl_parallax":
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

        max_ind = int(np.argmax(clusters_all[:, 2]))
        best = clusters_all[max_ind]
        others = np.delete(clusters_all, max_ind, axis=0)

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
            score = (best[2] - med) / std if std > 0 else float("inf")
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
