from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import logging

import numpy as np
import jax
import jax.numpy as jnp

from .config import FinderConfig
from .singlelens_fit import (
    SingleLensFitResult,
    PSPLFitter,
    FSPLFitter,
    PSPLParallaxFitter,
    FSPLParallaxFitter,
)
from .plot import AnomalyPlotter
from .seasons import SeasonSplitter
from .extract import ResultExtractor
from .runner import SeasonGridRunner
from .models import AnomalyResult, BestCandidate


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

    def _ensure_fitter(self, t0) -> None:
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
            self.fitter = PSPLFitter()
            return
    
        if k == "fspl":
            self.fitter = FSPLFitter()
            return
    
        # Parallax variants
        tref = self.config.tref
        if tref is None:
            tref = t0
    
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
        x0,
    ) -> SingleLensFitResult:
        """
        Run only the single-lens fit selected by the current configuration.

        Parameters
        ----------
        time, flux, ferr : array-like
            One-dimensional light-curve arrays.
        x0 : array-like
            Initial guess for the nonlinear model parameters.
            Its length must match the selected fitter.

        Returns
        -------
        SingleLensFitResult
            Result of the single-lens fit.
        """
        time_j, flux_j, ferr_j, x0_j, time_np, _, _ = self._to_arrays(time, flux, ferr, x0)
        self._ensure_fitter(x0[0])
        return self.fitter.fit(time_j, flux_j, ferr_j, x0_j)

    def run(
        self,
        time,
        flux,
        ferr,
        x0,
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
        x0 : array-like
            Initial guess for the single-lens model parameters.
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

        self._ensure_fitter(x0[0])

        fit: SingleLensFitResult = self.fitter.fit(time_j, flux_j, ferr_j, x0_j)
        residual_j = fit.residual
        model_flux_j = fit.model_flux

        residual_np, model_flux_np, chi2_dof = jax.device_get(
            (residual_j, model_flux_j, fit.chi2_dof)
        )
        residual_np = np.asarray(residual_np, dtype=float)
        model_flux_np = np.asarray(model_flux_np, dtype=float)
        chi2_dof = float(chi2_dof)

        seasons, clusters_all = self.runner.run(
            time_j=time_j,
            residual_j=residual_j,
            ferr_j=ferr_j,
            time_np=time_np,
            verbose=verbose,
            log=log,
        )

        best_obj = self._pick_best_candidate(clusters_all)

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
            best=best_obj,
        )

        self._last_result = result
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
        x0_j = jnp.asarray(x0, dtype=time_j.dtype)

        return time_j, flux_j, ferr_j, x0_j, time_np, flux_np, ferr_np

    def _pick_best_candidate(
        self,
        clusters_all: np.ndarray,
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
            med = float(np.median(others[:, 2]))
            std = float(np.std(others[:, 2]))
            score = (best[2] - med) / std if std > 0 else float("inf")
        else:
            med = std = score = float("nan")

        return BestCandidate(
            t0=float(best[0]),
            teff=float(best[1]),
            dchi2=float(best[2]),
            med_others=med,
            std_others=std,
            score=float(score),
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
