from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence
import logging

import numpy as np
import jax
import jax.numpy as jnp

from .config import FinderConfig
from .singlelens_fit import SingleLensFitResult
from .fitters import (
    PSPLFitter,
    FSPLFitter,
)
from .plot import AnomalyPlotter
from .seasons import SeasonSplitter
from .extract import ResultExtractor
from .runner import SeasonGridRunner, _cpp_grid
from .models import (
    AnomalyResult,
    BestCandidate,
    CandidateQuality,
    ScoredCandidate,
    SeasonSummary,
)
from .parallax_backend import parallax_fitter_from_config
from .signal_scale import (
    measure_observed_magnification_scale,
    measure_observed_signal_scale,
)


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


class _GridQualityIndex:
    """Fast exact lookup for the sorted teff/t0 grid emitted by the runner.

    Extracted clusters use coordinates from this grid. The index stores only
    small slice bounds; it does not copy the million-row metric array. Queries
    that are not exact grid points, or whose grid ordering is unusual, return
    None so callers can preserve the historical nearest-point fallback.
    """

    def __init__(self, metrics: np.ndarray):
        self._metrics = np.asarray(metrics, dtype=float) if metrics is not None else None
        self._groups = {}

        if (
            self._metrics is None
            or self._metrics.ndim != 2
            or self._metrics.shape[0] == 0
            or self._metrics.shape[1] < 2
        ):
            return

        coordinates = self._metrics[:, :2]
        if not np.isfinite(coordinates).all():
            return

        changes = np.flatnonzero(self._metrics[1:, 1] != self._metrics[:-1, 1]) + 1
        bounds = np.concatenate(([0], changes, [self._metrics.shape[0]]))
        segments_by_teff = {}
        invalid_teff = set()

        for lo, hi in zip(bounds[:-1], bounds[1:]):
            teff = float(self._metrics[lo, 1])
            t0_values = self._metrics[lo:hi, 0]
            if t0_values.size > 1 and np.any(t0_values[1:] < t0_values[:-1]):
                invalid_teff.add(teff)
                continue
            segments_by_teff.setdefault(teff, []).append(
                (int(lo), int(hi), float(t0_values[0]), float(t0_values[-1]))
            )

        for teff, segments in segments_by_teff.items():
            if teff in invalid_teff:
                self._groups[teff] = None
                continue
            segments.sort(key=lambda segment: segment[2])
            starts = np.fromiter((segment[2] for segment in segments), dtype=float)
            ends = np.fromiter((segment[3] for segment in segments), dtype=float)
            # Overlapping same-teff slices make the historical first-match
            # behavior ambiguous across seasons; keep those queries on fallback.
            if starts.size > 1 and np.any(starts[1:] <= np.maximum.accumulate(ends)[:-1]):
                self._groups[teff] = None
                continue
            self._groups[teff] = (starts, ends, segments)

    def row(self, t0: float, teff: float) -> Optional[np.ndarray]:
        """Return the exact grid row, or None when legacy lookup is required."""
        group = self._groups.get(float(teff))
        if group is None:
            return None

        starts, ends, segments = group
        segment_idx = int(np.searchsorted(starts, float(t0), side="right")) - 1
        if segment_idx < 0 or float(t0) > float(ends[segment_idx]):
            return None

        lo, hi, _start, _end = segments[segment_idx]
        t0_values = self._metrics[lo:hi, 0]
        offset = int(np.searchsorted(t0_values, float(t0), side="left"))
        if offset >= t0_values.size or float(t0_values[offset]) != float(t0):
            return None
        return self._metrics[lo + offset]


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
      in :class:`FinderConfig`. If ``tref`` is not specified, an
      event-centred time near the observed brightening peak is used.
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

    def _ensure_fitter(self, t_ref) -> None:
        """
        Instantiate the default single-lens fitter from the current configuration.
    
        Notes
        -----
        - If `config.fitter_kind` selects a parallax model, `ra_deg` and `dec_deg`
          must be provided. If `tref` is not set, the caller supplies an
          event-centred reference time near the observed brightening peak.
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
        needs_sky = k in {"pspl_parallax", "fspl_parallax"}
        if needs_sky:
            if self.config.ra_deg is None or self.config.dec_deg is None:
                raise ValueError(
                    f"{k} requires ra_deg and dec_deg in FinderConfig "
                    "(sky coordinates are required for parallax)."
                )
        geometry = None
        if needs_sky:
            geometry = self._resolve_model_parallax_geometry()
        needs_satellite = geometry == "space"
        if needs_satellite and self.config.satellite_ephemeris_path is None:
            raise ValueError(
                f"{k} with parallax_geometry='space' requires "
                "satellite_ephemeris_path in FinderConfig."
            )
    
        # -----------------------------
        # 3) Build fitter
        # -----------------------------
        if k == "pspl":
            self.fitter = PSPLFitter(
                maxiter=int(self.config.fitter_maxiter),
                tol=float(self.config.fitter_tol),
            )
            return

        if k == "fspl":
            self.fitter = FSPLFitter(
                maxiter=int(self.config.fitter_maxiter),
                tol=float(self.config.fitter_tol),
                magnification_tol=float(self.config.magnification_tol),
                magnification_reltol=float(self.config.magnification_reltol),
            )
            return
    
        # Parallax variants
        tref = self.config.tref
        if tref is None:
            tref = t_ref
    
        self.fitter = parallax_fitter_from_config(
            self.config,
            finite_source=(k != "pspl_parallax"),
            geometry="space" if geometry == "space" else "annual",
            tref=float(tref),
        )

    def _resolve_model_parallax_geometry(self) -> str:
        """Resolve the one geometry used by a parallax baseline fitter."""
        geometry = str(getattr(self.config, "parallax_geometry", "auto")).lower()
        if geometry == "auto":
            geometry = (
                "space"
                if self.config.satellite_ephemeris_path is not None
                else "annual"
            )
        if geometry not in {"annual", "space"}:
            raise ValueError(
                "parallax baseline requires parallax_geometry='annual' or 'space'."
            )
        return geometry

    def _resolve_parallax_tref(
        self,
        time_np: np.ndarray,
        flux_np: np.ndarray,
        *,
        x0=None,
        fit: Optional[SingleLensFitResult] = None,
        for_parallax: bool = False,
    ) -> float:
        """Resolve the fixed parallax reference epoch for one light curve.

        ``tref`` is a coordinate/reference choice for the observer
        displacement, not another nonlinear fit parameter.  A reference near
        the event centre keeps the parallax components from being needlessly
        correlated with ``t0`` and ``tE``.  Explicit configuration wins;
        otherwise use a supplied nonlinear seed or completed fit, then locate
        the broad brightening centre in the original flux.  The latter is only
        a locator for ``tref``; the parallax fit still scores the original
        flux directly.
        """
        configured = getattr(self.config, "tref", None)
        if configured is not None:
            return float(configured)

        kind = str(getattr(self.config, "fitter_kind", "pspl"))
        if not for_parallax and kind not in {"pspl_parallax", "fspl_parallax"}:
            return float(np.median(np.asarray(time_np, dtype=float)))

        for source in (
            getattr(fit, "params", None) if fit is not None else None,
            x0,
        ):
            if source is None:
                continue
            values = np.asarray(source, dtype=float).reshape(-1)
            if values.size >= 1 and np.isfinite(values[0]):
                return float(values[0])

        time_values = np.asarray(time_np, dtype=float).reshape(-1)
        flux_values = np.asarray(flux_np, dtype=float).reshape(-1)
        valid = np.isfinite(time_values) & np.isfinite(flux_values)
        if not np.any(valid):
            raise ValueError("Cannot resolve parallax tref without finite data.")
        finite_time = time_values[valid]
        finite_flux = flux_values[valid]
        peak_index = int(np.nanargmax(finite_flux))
        peak_time = float(finite_time[peak_index])

        # Use the broad flux envelope to avoid making tref follow one noisy
        # point or a compact anomaly.  If the envelope is not measurable, the
        # observed peak is still a valid conservative locator.
        span = float(np.ptp(finite_time)) if finite_time.size > 1 else 0.0
        measured = measure_observed_magnification_scale(
            finite_time,
            finite_flux,
            center=peak_time,
            search_half_width=max(1.0, min(300.0, 0.25 * span))
            if span > 0.0
            else 1.0,
            n_bins=256,
            source="parallax_tref_peak",
        )
        if measured.valid and np.isfinite(measured.t_center):
            return float(measured.t_center)
        return peak_time


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
        time_j, flux_j, ferr_j, x0_j, time_np, flux_np, _ = self._to_arrays(
            time, flux, ferr, x0, data_kind=data_kind
        )
        self._ensure_fitter(
            self._resolve_parallax_tref(time_np, flux_np, x0=x0_j)
        )
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

        self._ensure_fitter(
            self._resolve_parallax_tref(time_np, flux_np, x0=x0_j)
        )

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

        scored_candidates = self._score_candidates(
            clusters_all,
            grid_metrics_all,
            seasons=seasons,
        )
        best_obj = self._best_from_scored_candidates(scored_candidates)

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
            scored_candidates=scored_candidates,
            observed_signal_scale=measure_observed_signal_scale(
                time_np,
                residual_np / np.maximum(ferr_np, 1.0e-12),
                source="finder_residual",
            ),
        )

        self._last_result = result
        return result

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
        """Automatic fit: the fitter owns seed generation and multistart."""
        return self.fitter.fit(time_j, flux_j, ferr_j)

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
        if hasattr(self.fitter, "evaluate_fixed"):
            return self.fitter.evaluate_fixed(time_j, flux_j, ferr_j, x0_j)

        raise ValueError(f"Unknown fitter_kind '{k}'.")

    def _score_candidates(
        self,
        clusters_all: np.ndarray,
        grid_metrics_all: np.ndarray,
        *,
        seasons: Optional[Sequence[SeasonSummary]] = None,
    ) -> list[ScoredCandidate]:
        """Score every finite extracted cluster against an all-season background.

        ``seasons`` remains in the private signature for compatibility with
        callers that already pass the runner's season summaries. It is
        intentionally not used for normalization: the score background spans
        all seasons. The ``teff`` locality and robust upper clipping are still
        applied independently for each candidate.

        The returned list is sorted by descending finite score. Non-finite
        scores are retained at the end so that small-reference diagnostics are
        not silently discarded.
        """
        del seasons
        if clusters_all is None or clusters_all.size == 0:
            return []

        raw_clusters = np.asarray(clusters_all, dtype=float)
        raw_clusters = raw_clusters[np.isfinite(raw_clusters).all(axis=1)]
        if raw_clusters.size == 0:
            return []

        quality_index = _GridQualityIndex(grid_metrics_all)
        background_stats = self._indexed_score_backgrounds(raw_clusters)
        scored = [
            self._score_candidate(
                candidate,
                raw_clusters,
                grid_metrics_all,
                quality_index=quality_index,
                background_stats=background_stats[index],
            )
            for index, candidate in enumerate(raw_clusters)
        ]
        return sorted(
            scored,
            key=lambda candidate: (
                bool(np.isfinite(candidate.score)),
                float(candidate.score) if np.isfinite(candidate.score) else -np.inf,
                float(candidate.dchi2),
            ),
            reverse=True,
        )

    def _indexed_score_backgrounds(self, raw_clusters: np.ndarray) -> list:
        """Reuse sorted teff-local backgrounds and leave-one-out order statistics.

        The native path needs unique candidate epochs so that the historical
        first-isclose exclusion is exactly the candidate's own row. Ambiguous,
        sparse and degenerate cases continue through the scalar reference.
        """
        count = len(raw_clusters)
        result = [None] * count
        if (
            _cpp_grid is None
            or not hasattr(_cpp_grid, "score_backgrounds")
            or count < 2
            or np.any(raw_clusters[:, 1] <= 0.0)
            or np.any(np.diff(np.sort(raw_clusters[:, 0])) <= 1e-9)
        ):
            return result
        ratio = float(self.config.best_score_teff_ratio)
        minimum = int(self.config.best_score_min_reference_clusters)
        sigma = float(self.config.best_score_upper_clip_sigma)
        maxiters = int(self.config.best_score_clip_maxiters)
        if not np.isfinite(ratio) or ratio < 1.0 or minimum < 2 or not sigma > 0.0 or maxiters < 0:
            return result
        for teff in np.unique(raw_clusters[:, 1]):
            candidates = np.flatnonzero(raw_clusters[:, 1] == teff)
            local = np.flatnonzero(
                np.abs(np.log(raw_clusters[:, 1] / float(teff))) <= np.log(ratio) + 1e-12
            )
            if local.size - 1 < minimum and local.size != count:
                continue
            excluded = np.searchsorted(local, candidates).astype(np.int64)
            statistics = _cpp_grid.score_backgrounds(
                raw_clusters[local, 2], excluded, clip_sigma=sigma, maxiters=maxiters
            )
            for index, row in zip(candidates, statistics):
                if row[3]:
                    result[int(index)] = (float(row[0]), float(row[1]), int(row[2]))
        return result

    def _score_candidate(
        self,
        candidate: np.ndarray,
        raw_clusters: np.ndarray,
        grid_metrics_all: np.ndarray,
        *,
        quality_index: Optional[_GridQualityIndex] = None,
        background_stats: Optional[tuple[float, float, int]] = None,
    ) -> ScoredCandidate:
        if background_stats is None:
            references = self._score_reference_clusters(candidate, raw_clusters)
            bulk_dchi2 = self._upper_clip_score_background(references[:, 2])
            n_score_reference = int(bulk_dchi2.shape[0])
            if n_score_reference >= 2:
                med = float(np.median(bulk_dchi2))
                std = self._robust_scale(bulk_dchi2)
            else:
                med = std = float("nan")
        else:
            med, std, n_score_reference = background_stats
        score = (candidate[2] - med) / std if std > 0 else float("nan")

        quality_row = (
            None
            if quality_index is None
            else quality_index.row(float(candidate[0]), float(candidate[1]))
        )
        quality = (
            self._quality_from_row(quality_row)
            if quality_row is not None
            else self._quality_for_point(
                float(candidate[0]),
                float(candidate[1]),
                grid_metrics_all,
            )
        )
        return ScoredCandidate(
            t0=float(candidate[0]),
            teff=float(candidate[1]),
            dchi2=float(candidate[2]),
            med_others=med,
            std_others=std,
            score=float(score),
            quality=quality,
            n_score_reference=n_score_reference,
        )

    def _best_from_scored_candidates(
        self,
        scored_candidates: Sequence[ScoredCandidate],
    ) -> Optional[BestCandidate]:
        """Return the maximum-``dchi2`` accepted scored candidate."""
        if not scored_candidates:
            return None

        criteria = self.config.candidate_criteria
        eligible = [
            candidate
            for candidate in scored_candidates
            if criteria is None
            or criteria.accepts(
                dchi2=float(candidate.dchi2),
                quality=candidate.quality,
            )
        ]
        if not eligible:
            return None

        selected = max(eligible, key=lambda candidate: float(candidate.dchi2))
        return BestCandidate(
            t0=selected.t0,
            teff=selected.teff,
            dchi2=selected.dchi2,
            med_others=selected.med_others,
            std_others=selected.std_others,
            score=selected.score,
            quality=selected.quality,
            n_score_reference=selected.n_score_reference,
        )

    def _accepted_candidates(
        self,
        raw_clusters: np.ndarray,
        grid_metrics_all: np.ndarray,
    ) -> np.ndarray:
        criteria = self.config.candidate_criteria
        if criteria is None:
            return raw_clusters

        quality_index = _GridQualityIndex(grid_metrics_all)
        accepted = []
        for cluster in raw_clusters:
            quality_row = quality_index.row(float(cluster[0]), float(cluster[1]))
            quality = (
                self._quality_from_row(quality_row)
                if quality_row is not None
                else self._quality_for_point(
                    float(cluster[0]),
                    float(cluster[1]),
                    grid_metrics_all,
                )
            )
            if criteria.accepts(dchi2=float(cluster[2]), quality=quality):
                accepted.append(cluster)
        if not accepted:
            return np.zeros((0, 3), dtype=float)
        return np.asarray(accepted, dtype=float)

    def _score_reference_clusters(
        self,
        candidate: np.ndarray,
        raw_clusters: np.ndarray,
        *,
        seasons: Optional[Sequence[SeasonSummary]] = None,
    ) -> np.ndarray:
        """Return all-season, comparable-timescale background clusters.

        ``seasons`` is retained for compatibility with the former
        same-season implementation, but is intentionally ignored. The score
        is normalized from the complete event-wide cluster population.
        """
        del seasons
        references = np.asarray(raw_clusters, dtype=float)

        same = (
            np.isclose(references[:, 0], candidate[0], rtol=0.0, atol=1e-9)
            & np.isclose(references[:, 1], candidate[1], rtol=1e-12, atol=1e-12)
            & np.isclose(references[:, 2], candidate[2], rtol=1e-12, atol=1e-12)
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
        log_distance = np.abs(np.log(references[:, 1] / float(candidate[1])))
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
    def _quality_from_row(row: np.ndarray) -> CandidateQuality:
        return CandidateQuality(
            n_window=int(round(float(row[3]))),
            n_contrib=int(round(float(row[4]))),
            n_eff=float(row[5]),
            peak_frac=float(row[6]),
            rho1=float(row[7]),
            longest_run=int(round(float(row[8]))),
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
        return Finder._quality_from_row(grid_metrics_all[i])


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
