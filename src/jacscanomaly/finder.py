# scanomaly/finder.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp

from .config import FinderConfig
from .pspl import PSPLFitter
from .plot import AnomalyPlotter
from .seasons import SeasonSplitter
from .extract import ResultExtractor
from .runner import SeasonGridRunner
from .models import AnomalyResult, BestCandidate


@dataclass
class Finder:
    """
    Main entry point of scanomaly.

    Finder performs:
      1) PSPL fit on (time, flux, ferr)
      2) season splitting
      3) grid scan on PSPL residuals
      4) cluster extraction
      5) selection of the best anomaly candidate

    Users typically call :meth:`run` and then pass the returned
    :class:`scanomaly.models.AnomalyResult` to :class:`scanomaly.plot.AnomalyPlotter`.
    """

    # NOTE: use default_factory for dataclass fields (avoid shared mutable defaults)
    config: FinderConfig = field(default_factory=FinderConfig)

    # Allow dependency injection, but create defaults if None
    fitter: Optional[PSPLFitter] = None
    plotter: Optional[AnomalyPlotter] = None

    def __post_init__(self) -> None:
        if self.fitter is None:
            self.fitter = PSPLFitter()
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
        
        _last_result: Optional[AnomalyResult] = field(default=None, init=False)

    # ----------------------------
    # Public APIs
    # ----------------------------
    def fit_pspl(self, time, flux, ferr, p0):
        """
        Convenience method: run PSPL fit only.

        Returns
        -------
        PSPLFitResult
            The PSPL fitting result (JAX arrays inside).
        """
        time_j, flux_j, ferr_j, p0_j, _time_np, _flux_np, _ferr_np = self._to_arrays(time, flux, ferr, p0)
        return self.fitter.fit(time_j, flux_j, ferr_j, p0_j)

    def run(
        self,time,flux,ferr,p0,*,
        verbose: bool = True,log: Optional[logging.Logger] = None,) -> AnomalyResult:
        """
        Run the full anomaly finding pipeline.

        Parameters
        ----------
        time, flux, ferr : array-like
            1D arrays. Stored in the output as NumPy arrays on CPU for fast plotting.
        p0 : array-like
            Initial PSPL parameters (t0, tE, u0).

        Returns
        -------
        AnomalyResult
            Includes PSPL fit, residuals, per-season cluster summaries,
            flattened clusters, and the best candidate (if any).
        """
        time_j, flux_j, ferr_j, p0_j, time_np, flux_np, ferr_np = self._to_arrays(time, flux, ferr, p0)

        # 1) PSPL fit (JAX)
        fit = self.fitter.fit(time_j, flux_j, ferr_j, p0_j)
        residual_j = fit.residual
        model_flux_j = fit.model_flux

        # bring to CPU for plotting/analysis
        residual_np, model_flux_np, chi2_dof = jax.device_get((residual_j, model_flux_j, fit.chi2_dof))
        residual_np = np.asarray(residual_np, dtype=float)
        model_flux_np = np.asarray(model_flux_np, dtype=float)
        chi2_dof = float(chi2_dof)

        # 2-4) season loop & grid scan & extraction
        seasons, clusters_all = self.runner.run(
            time_j=time_j,
            residual_j=residual_j,
            ferr_j=ferr_j,
            time_np=time_np,
            verbose=verbose,
            log=log,
            )

        # best candidate selection
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

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _to_arrays(self, time, flux, ferr, p0):
        """Convert inputs into both NumPy (CPU) and JAX arrays."""
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
        p0_j = jnp.asarray(p0, dtype=time_j.dtype)

        return time_j, flux_j, ferr_j, p0_j, time_np, flux_np, ferr_np

    def _pick_best_candidate(self, clusters_all: np.ndarray) -> Optional[BestCandidate]:
        """
        Pick the single best candidate from flattened clusters and compute a standardized score.
        """
        if clusters_all is None or clusters_all.size == 0 or clusters_all.shape[0] < 1:
            return None

        # clusters_all rows: [t0, teff, dchi2]
        max_ind = int(np.argmax(clusters_all[:, 2]))
        best = clusters_all[max_ind]
        others = np.delete(clusters_all, max_ind, axis=0)

        if others.shape[0] >= 2:
            med = float(np.median(others[:, 2]))
            std = float(np.std(others[:, 2]))
            score = float((best[2] - med) / std) if std > 0 else float("inf")
        else:
            med, std, score = float("nan"), float("nan"), float("nan")

        return BestCandidate(
            t0=float(best[0]),
            teff=float(best[1]),
            dchi2=float(best[2]),
            med_others=med,
            std_others=std,
            score=score,
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
        Plot light curve with PSPL model using the last result.
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