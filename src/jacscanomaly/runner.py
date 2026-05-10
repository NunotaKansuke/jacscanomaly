# scanomaly/runner.py
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import List, Tuple, Optional
import time

import logging
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, lax, device_get, block_until_ready

from .config import FinderConfig
from .models import SeasonSummary
from .seasons import SeasonSplitter
from .extract import ResultExtractor
from .anomaly_models import get_chi2_anom_masked, get_chi2_flat_masked

try:
    from . import _cpp_grid
except ImportError:  # pragma: no cover - optional compiled backend
    _cpp_grid = None

logger = logging.getLogger(__name__)


class GridRunner:
    """
    Low-level JAX grid runner for local template scan.

    This runner evaluates, for each grid point (t0_ref, teff_ref), the chi-square
    improvement of an anomaly template model relative to a null (flat) model
    within a local time window.

    Definitions
    ----------
    - Window mask:
        mask = |t - t0_ref| < teff_coeff * teff_ref
    - Chi-square improvement:
        delta_chi2 = chi2_flat - chi2_anom
      (positive values indicate the anomaly template fits better than the null model)

    Notes
    -----
    - The actual chi2 computation is delegated to:
        get_chi2_anom_masked(t0, teff, time, flux, w, mask) -> (chi2_total, chi2s_per_point)
        get_chi2_flat_masked(flux, w, mask)                 -> (chi2_total, chi2s_per_point)

    - Outside the evaluation window, we ignore points by zeroing their weights
      via mask (w * mask).
    """

    @staticmethod
    @partial(jit, static_argnames=("sigma", "teff_coeff", "min_pts"))
    def _grid_point(
        t0_ref: jnp.ndarray,
        teff_ref: jnp.ndarray,
        time: jnp.ndarray,
        flux: jnp.ndarray,
        w: jnp.ndarray,
        *,
        sigma: float = 3.0,
        teff_coeff: float = 3.0,
        min_pts: int = 4,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        mask = (time > (t0_ref - teff_coeff * teff_ref)) & (time < (t0_ref + teff_coeff * teff_ref))
        n_pts = jnp.sum(mask.astype(jnp.int32))

        def valid(_):
            chi2_anom, chi2s_anom = get_chi2_anom_masked(t0_ref, teff_ref, time, flux, w, mask)
            chi2_flat, chi2s_flat = get_chi2_flat_masked(flux, w, mask)

            dchi2 = chi2_flat - chi2_anom

            wm = w * mask.astype(w.dtype)
            eps = jnp.asarray(1e-30, wm.dtype)
            norm_flat = chi2s_flat / jnp.maximum(wm, eps)
            norm_anom = chi2s_anom / jnp.maximum(wm, eps)
            diff = norm_flat - norm_anom

            improvement = jnp.where(mask, jnp.maximum(diff, 0.0), 0.0)
            contrib = (improvement > (sigma ** 2)) & mask

            n_window = n_pts.astype(jnp.int32)
            n_contrib = jnp.sum(contrib).astype(jnp.int32)

            sum_u = jnp.sum(improvement)
            sum_u2 = jnp.sum(improvement * improvement)
            n_eff = jnp.where(sum_u2 > 0.0, (sum_u * sum_u) / sum_u2, 0.0).astype(jnp.float32)
            peak_frac = jnp.where(sum_u > 0.0, jnp.max(improvement) / sum_u, 0.0).astype(jnp.float32)

            n_pair = jnp.sum(mask[:-1] & mask[1:])
            mean = jnp.where(n_window > 0, jnp.sum(diff * mask.astype(diff.dtype)) / n_window, 0.0)
            centered = diff - mean
            var = jnp.where(
                n_window > 0,
                jnp.sum((centered * centered) * mask.astype(diff.dtype)) / n_window,
                0.0,
            )
            cov1 = jnp.where(
                n_pair > 0,
                jnp.sum((centered[:-1] * centered[1:]) * (mask[:-1] & mask[1:]).astype(diff.dtype)) / n_pair,
                0.0,
            )
            rho1 = jnp.where((n_pair > 0) & (var > 0.0), cov1 / var, 0.0).astype(jnp.float32)

            def run_body(carry, is_on):
                cur, maxv = carry
                cur = jnp.where(is_on, cur + 1, 0)
                maxv = jnp.maximum(maxv, cur)
                return (cur, maxv), None

            (_, longest_run), _ = lax.scan(
                run_body, (jnp.asarray(0, jnp.int32), jnp.asarray(0, jnp.int32)), contrib
            )

            return (
                dchi2.astype(jnp.float32),
                n_window,
                n_contrib,
                n_eff,
                peak_frac,
                rho1,
                longest_run.astype(jnp.int32),
            )

        def invalid(_):
            return (
                jnp.asarray(0.0, jnp.float32),
                jnp.asarray(0, jnp.int32),
                jnp.asarray(0, jnp.int32),
                jnp.asarray(0.0, jnp.float32),
                jnp.asarray(0.0, jnp.float32),
                jnp.asarray(0.0, jnp.float32),
                jnp.asarray(0, jnp.int32),
            )

        return lax.cond(n_pts >= min_pts, valid, invalid, operand=None)

    @staticmethod
    @partial(jit, static_argnames=("sigma", "teff_coeff", "min_pts"))
    def run(
        time: jnp.ndarray,
        flux: jnp.ndarray,
        w: jnp.ndarray,
        t0_flat: jnp.ndarray,
        teff_flat: jnp.ndarray,
        *,
        sigma: float = 3.0,
        teff_coeff: float = 3.0,
        min_pts: int = 4,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        dchi2, n_window, n_contrib, n_eff, peak_frac, rho1, longest_run = vmap(
            lambda t0_ref, teff_ref: GridRunner._grid_point(
                t0_ref, teff_ref, time, flux, w,
                sigma=sigma, teff_coeff=teff_coeff, min_pts=min_pts
            )
        )(t0_flat, teff_flat)

        return t0_flat, teff_flat, dchi2, n_window, n_contrib, n_eff, peak_frac, rho1, longest_run

    @staticmethod
    @partial(jit, static_argnames=("sigma", "teff_coeff", "min_pts"))
    def _run_vmap(
        time: jnp.ndarray,
        flux: jnp.ndarray,
        w: jnp.ndarray,
        t0_flat: jnp.ndarray,
        teff_flat: jnp.ndarray,
        *,
        sigma: float,
        teff_coeff: float,
        min_pts: int,
    ):
        return vmap(
            lambda t0_ref, teff_ref: GridRunner._grid_point(
                t0_ref, teff_ref, time, flux, w,
                sigma=sigma, teff_coeff=teff_coeff, min_pts=min_pts
            )
        )(t0_flat, teff_flat)

    @staticmethod
    @partial(jit, static_argnames=("sigma", "teff_coeff", "min_pts", "chunk_size"))
    def run_chunked(
        time: jnp.ndarray,
        flux: jnp.ndarray,
        w: jnp.ndarray,
        t0_flat: jnp.ndarray,
        teff_flat: jnp.ndarray,
        *,
        sigma: float = 3.0,
        teff_coeff: float = 3.0,
        min_pts: int = 4,
        chunk_size: int = 4096,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        n = t0_flat.shape[0]
        n_chunks = (n + chunk_size - 1) // chunk_size
        n_pad = n_chunks * chunk_size
        pad = n_pad - n

        t0p = jnp.pad(t0_flat, (0, pad))
        tep = jnp.pad(teff_flat, (0, pad))

        dchi2_buf = jnp.zeros((n_pad,), dtype=jnp.float32)
        nwin_buf = jnp.zeros((n_pad,), dtype=jnp.int32)
        ncontrib_buf = jnp.zeros((n_pad,), dtype=jnp.int32)
        neff_buf = jnp.zeros((n_pad,), dtype=jnp.float32)
        peak_buf = jnp.zeros((n_pad,), dtype=jnp.float32)
        rho1_buf = jnp.zeros((n_pad,), dtype=jnp.float32)
        longest_buf = jnp.zeros((n_pad,), dtype=jnp.int32)

        def body(i, carry):
            dchi2_buf, nwin_buf, ncontrib_buf, neff_buf, peak_buf, rho1_buf, longest_buf = carry
            s = i * chunk_size

            t0c = lax.dynamic_slice(t0p, (s,), (chunk_size,))
            tec = lax.dynamic_slice(tep, (s,), (chunk_size,))

            dchi2_c, nwin_c, ncontrib_c, neff_c, peak_c, rho1_c, longest_c = GridRunner._run_vmap(
                time, flux, w, t0c, tec,
                sigma=sigma, teff_coeff=teff_coeff, min_pts=min_pts
            )

            dchi2_buf = lax.dynamic_update_slice(dchi2_buf, dchi2_c.astype(jnp.float32), (s,))
            nwin_buf = lax.dynamic_update_slice(nwin_buf, nwin_c.astype(jnp.int32), (s,))
            ncontrib_buf = lax.dynamic_update_slice(ncontrib_buf, ncontrib_c.astype(jnp.int32), (s,))
            neff_buf = lax.dynamic_update_slice(neff_buf, neff_c.astype(jnp.float32), (s,))
            peak_buf = lax.dynamic_update_slice(peak_buf, peak_c.astype(jnp.float32), (s,))
            rho1_buf = lax.dynamic_update_slice(rho1_buf, rho1_c.astype(jnp.float32), (s,))
            longest_buf = lax.dynamic_update_slice(longest_buf, longest_c.astype(jnp.int32), (s,))
            return (dchi2_buf, nwin_buf, ncontrib_buf, neff_buf, peak_buf, rho1_buf, longest_buf)

        dchi2_buf, nwin_buf, ncontrib_buf, neff_buf, peak_buf, rho1_buf, longest_buf = lax.fori_loop(
            0, n_chunks, body, (dchi2_buf, nwin_buf, ncontrib_buf, neff_buf, peak_buf, rho1_buf, longest_buf)
        )
        return (
            t0_flat,
            teff_flat,
            dchi2_buf[:n],
            nwin_buf[:n],
            ncontrib_buf[:n],
            neff_buf[:n],
            peak_buf[:n],
            rho1_buf[:n],
            longest_buf[:n],
        )

    @staticmethod
    def run_auto(
        time: jnp.ndarray,
        flux: jnp.ndarray,
        w: jnp.ndarray,
        t0_flat: jnp.ndarray,
        teff_flat: jnp.ndarray,
        *,
        sigma: float = 3.0,
        teff_coeff: float = 3.0,
        min_pts: int = 4,
        chunk_size: int = 4096,
        threshold: int = 200_000,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if int(t0_flat.shape[0]) >= int(threshold):
            return GridRunner.run_chunked(
                time, flux, w, t0_flat, teff_flat,
                sigma=sigma, teff_coeff=teff_coeff, min_pts=min_pts,
                chunk_size=chunk_size,
            )
        return GridRunner.run(
            time, flux, w, t0_flat, teff_flat,
            sigma=sigma, teff_coeff=teff_coeff, min_pts=min_pts,
        )


@dataclass
class SeasonGridRunner:
    splitter: SeasonSplitter
    extractor: ResultExtractor
    config: FinderConfig

    def run(
        self,
        *,
        time_j: jnp.ndarray,
        residual_j: jnp.ndarray,
        ferr_j: jnp.ndarray,
        time_np: np.ndarray,
        verbose: bool = True,
        log: Optional[logging.Logger] = None,
        chunked: bool = False,
        chunk_size: int = 4096,
        chunk_threshold: int = 200_000,
        chunk_auto: bool = False,
    ) -> Tuple[List[SeasonSummary], np.ndarray, np.ndarray]:

        log = logger if log is None else log

        t_total0 = time.perf_counter()  # ★ total start

        seasons = self.splitter.split(time_np)
        n_seasons = len(seasons)

        teff_k = self.config.teff_init * np.power(
            self.config.common_ratio,
            np.arange(self.config.teff_grid_n),
        )

        if verbose:
            log.info("Season split: %d season(s) (gap=%.3f)", n_seasons, float(self.config.gap))
            log.info(
                "teff grid: n=%d, teff_init=%.6g, ratio=%.6g",
                int(self.config.teff_grid_n),
                float(self.config.teff_init),
                float(self.config.common_ratio),
            )

        season_summaries: List[SeasonSummary] = []
        all_clusters: List[np.ndarray] = []
        all_grid_metrics: List[np.ndarray] = []

        for season_idx, season in enumerate(seasons):
            t_season0 = time.perf_counter()  # ★ season start

            # JAX indexing: make indices a JAX array (safer across backends)
            idx = jnp.asarray(season.indices)
            t_season = time_j[idx]
            r_season = residual_j[idx]
            e_season = ferr_j[idx]
            w_season = 1.0 / (e_season ** 2)

            if verbose:
                log.info(
                    "[%d/%d] Season %d: start=%.6f end=%.6f n_pts=%d",
                    season_idx + 1, n_seasons, season_idx,
                    float(season.start), float(season.end),
                    int(season.indices.size),
                )

            # build (t0, teff) grids for this season
            t0_flat_list: List[np.ndarray] = []
            teff_flat_list: List[np.ndarray] = []

            for teff in teff_k:
                dt0 = self.config.dt0_coeff * float(teff)
                t0_vals = self.splitter.build_t0_grid(season, dt0)
                if t0_vals.size == 0:
                    continue

                t0_flat_list.append(t0_vals.astype(float, copy=False))
                teff_flat_list.append(np.full_like(t0_vals, float(teff), dtype=float))

            if len(t0_flat_list) == 0:
                if verbose:
                    dt_season = time.perf_counter() - t_season0
                    log.info("  -> grid: n_grid=0 (skipped), elapsed=%.3fs", dt_season)

                clusters = np.zeros((0, 3), dtype=float)
                season_summaries.append(
                    SeasonSummary(
                        season_idx=season_idx,
                        t_start=season.start,
                        t_end=season.end,
                        n_grid=0,
                        clusters=clusters,
                        grid_metrics=np.zeros((0, 9), dtype=float),
                    )
                )
                all_clusters.append(clusters)
                all_grid_metrics.append(np.zeros((0, 9), dtype=float))
                continue

            n_grid = int(sum(arr.size for arr in t0_flat_list))
            if verbose:
                log.info("  -> grid: n_grid=%d", n_grid)

            t0_flat_np = np.concatenate(t0_flat_list).astype(float, copy=False)
            teff_flat_np = np.concatenate(teff_flat_list).astype(float, copy=False)

            # JAX arrays for the grid
            t0_flat = jnp.asarray(t0_flat_np, dtype=jnp.float64)
            teff_flat = jnp.asarray(teff_flat_np, dtype=jnp.float64)

            # ---- timed block: grid run
            t_grid0 = time.perf_counter()

            used_cpp_backend = self.config.grid_backend == "cpp"
            if used_cpp_backend:
                if _cpp_grid is None:
                    raise RuntimeError(
                        "FinderConfig(grid_backend='cpp') requires the compiled jacscanomaly._cpp_grid extension."
                    )
                if verbose:
                    log.info("  -> grid exec: cpp")
                t_season_np = np.asarray(device_get(t_season), dtype=float)
                r_season_np = np.asarray(device_get(r_season), dtype=float)
                w_season_np = np.asarray(device_get(w_season), dtype=float)
                dchi2_np, nwin_np, ncontrib_np, neff_np, peak_np, rho1_np, longest_np = _cpp_grid.run_grid(
                    t_season_np,
                    r_season_np,
                    w_season_np,
                    t0_flat_np,
                    teff_flat_np,
                    sigma=float(self.config.sigma),
                    teff_coeff=float(self.config.teff_coeff),
                    min_pts=int(self.config.min_pts_in_window),
                )
                t0_np_out = t0_flat_np
                teff_np_out = teff_flat_np
            elif self.config.grid_chunk_auto:
                if verbose:
                    log.info("  -> grid exec: chunked(auto,size=%d)", self.config.grid_chunk_size)
                t0_out, teff_out, dchi2, n_window, n_contrib, n_eff, peak_frac, rho1, longest_run = GridRunner.run_auto(
                    t_season, r_season, w_season,
                    t0_flat, teff_flat,
                    sigma=self.config.sigma,
                    teff_coeff=self.config.teff_coeff,
                    min_pts=self.config.min_pts_in_window,
                    chunk_size=self.config.grid_chunk_size,
                    threshold=self.config.grid_chunk_threshold,
                )
            elif self.config.grid_chunked:
                if verbose:
                    log.info("  -> grid exec: chunked(forced,size=%d)", self.config.grid_chunk_size)
                t0_out, teff_out, dchi2, n_window, n_contrib, n_eff, peak_frac, rho1, longest_run = GridRunner.run_chunked(
                    t_season, r_season, w_season,
                    t0_flat, teff_flat,
                    sigma=self.config.sigma,
                    teff_coeff=self.config.teff_coeff,
                    min_pts=self.config.min_pts_in_window,
                    chunk_size=self.config.grid_chunk_size,
                )
            else:
                if verbose:
                    log.info("  -> grid exec: vmap")
                t0_out, teff_out, dchi2, n_window, n_contrib, n_eff, peak_frac, rho1, longest_run = GridRunner.run(
                    t_season, r_season, w_season,
                    t0_flat, teff_flat,
                    sigma=self.config.sigma,
                    teff_coeff=self.config.teff_coeff,
                    min_pts=self.config.min_pts_in_window,
                )

            if used_cpp_backend:
                dt_grid = time.perf_counter() - t_grid0
                dt_copy = 0.0
            else:
                block_until_ready(dchi2)
                dt_grid = time.perf_counter() - t_grid0

                # bring results back to CPU in one call
                t_copy0 = time.perf_counter()
                t0_np_out, teff_np_out, dchi2_np, nwin_np, ncontrib_np, neff_np, peak_np, rho1_np, longest_np = device_get(
                    (t0_out, teff_out, dchi2, n_window, n_contrib, n_eff, peak_frac, rho1, longest_run)
                )
                dt_copy = time.perf_counter() - t_copy0

            grid_metrics = np.column_stack(
                [
                    np.asarray(t0_np_out, dtype=float),
                    np.asarray(teff_np_out, dtype=float),
                    np.asarray(dchi2_np, dtype=float),
                    np.asarray(nwin_np, dtype=float),
                    np.asarray(ncontrib_np, dtype=float),
                    np.asarray(neff_np, dtype=float),
                    np.asarray(peak_np, dtype=float),
                    np.asarray(rho1_np, dtype=float),
                    np.asarray(longest_np, dtype=float),
                ]
            )
            dchi2_extract = np.asarray(dchi2_np, dtype=float)
            dchi2_extract = np.where(np.isfinite(dchi2_extract), dchi2_extract, -np.inf)
            criteria = self.config.candidate_criteria
            if criteria is not None:
                keep = np.ones_like(dchi2_extract, dtype=bool)
                if criteria.min_dchi2 is not None:
                    keep &= dchi2_extract >= float(criteria.min_dchi2)
                if criteria.min_n_eff is not None:
                    keep &= np.asarray(neff_np, dtype=float) >= float(criteria.min_n_eff)
                if criteria.min_n_contrib is not None:
                    keep &= np.asarray(ncontrib_np, dtype=float) >= int(criteria.min_n_contrib)
                if criteria.min_n_window is not None:
                    keep &= np.asarray(nwin_np, dtype=float) >= int(criteria.min_n_window)
                if criteria.min_longest_run is not None:
                    keep &= np.asarray(longest_np, dtype=float) >= int(criteria.min_longest_run)
                if criteria.max_peak_frac is not None:
                    keep &= np.asarray(peak_np, dtype=float) <= float(criteria.max_peak_frac)
                dchi2_extract = np.where(keep, dchi2_extract, -np.inf)

            # ---- timed block: extract clusters
            t_ext0 = time.perf_counter()
            clusters = self.extractor.iterative_anomaly_extraction(
                np.asarray(t0_np_out),
                np.asarray(teff_np_out),
                dchi2_extract,
            )
            dt_ext = time.perf_counter() - t_ext0

            dt_season = time.perf_counter() - t_season0  # ★ season end

            if verbose:
                log.info(
                    "  -> extracted clusters: %d | grid=%.3fs copy=%.3fs extract=%.3fs | season total=%.3fs",
                    int(clusters.shape[0]),
                    dt_grid, dt_copy, dt_ext,
                    dt_season,
                )

            all_clusters.append(clusters)
            all_grid_metrics.append(grid_metrics)
            season_summaries.append(
                SeasonSummary(
                    season_idx=season_idx,
                    t_start=season.start,
                    t_end=season.end,
                    n_grid=int(t0_flat.shape[0]),
                    clusters=clusters,
                    grid_metrics=grid_metrics,
                )
            )

        # flatten clusters across seasons
        if len(all_clusters) > 0:
            flat = np.array([c for cs in all_clusters for c in cs], dtype=float)
        else:
            flat = np.zeros((0, 3), dtype=float)

        if flat.size:
            flat = flat[np.isfinite(flat).all(axis=1)]
            flat = flat[flat[:, 2] != 0]

        non_empty_metrics = [g for g in all_grid_metrics if g.size > 0]
        if non_empty_metrics:
            grid_metrics_all = np.concatenate(non_empty_metrics, axis=0).astype(float, copy=False)
        else:
            grid_metrics_all = np.zeros((0, 9), dtype=float)

        dt_total = time.perf_counter() - t_total0  # ★ total end

        if verbose:
            log.info("Total clusters (flattened): %d", int(flat.shape[0]))
            log.info("Total elapsed: %.3fs", dt_total)

        return season_summaries, flat, grid_metrics_all
