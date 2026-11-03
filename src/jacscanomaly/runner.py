# jacscanomaly/runner.py
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

    def __post_init__(self) -> None:
        self._grids = {}  # time samples -> the (t0, teff) grid of every season
        self._reuse = {}  # (time, ferr, season) -> recent (z, cluster rows)

    def _season_grids(self, time_np: np.ndarray):
        """Seasons and their (t0, teff) grids; they depend on the times only."""
        key = _array_key(time_np)
        if key not in self._grids:
            teff_k = self.config.teff_init * np.power(
                self.config.common_ratio,
                np.arange(self.config.teff_grid_n),
            )
            out = []
            for season in self.splitter.split(time_np):
                t0_list, scales, steps = [], [], []
                for teff in teff_k:
                    dt0 = self.config.dt0_coeff * float(teff)
                    t0_vals = self.splitter.build_t0_grid(season, dt0)
                    if t0_vals.size == 0:
                        continue
                    t0_list.append(t0_vals.astype(float, copy=False))
                    scales.append(float(teff))
                    steps.append(float(dt0))
                if t0_list:
                    t0_flat = np.concatenate(t0_list)
                    teff_flat = np.concatenate(
                        [np.full(v.size, s) for v, s in zip(t0_list, scales)]
                    )
                else:
                    t0_flat = teff_flat = np.zeros(0)
                out.append((season, t0_list, scales, steps, t0_flat, teff_flat))
            if len(self._grids) >= 4:
                self._grids.clear()
            self._grids[key] = out
        return key, self._grids[key]

    def _season(self, grid, t, r, w, *, verbose=False, log=None):
        """Grid metrics (n_grid, 9) and extracted clusters (n, 3) of one season."""
        season, t0_list, scales, steps, t0_flat_np, teff_flat_np = grid
        backend = self.config.grid_backend
        t_grid0 = time.perf_counter()
        if backend == "cpp":
            if _cpp_grid is None:
                raise RuntimeError(
                    "FinderConfig(grid_backend='cpp') requires the compiled jacscanomaly._cpp_grid extension."
                )
            if verbose:
                log.info("  -> grid exec: cpp")
            dchi2_np, nwin_np, ncontrib_np, neff_np, peak_np, rho1_np, longest_np = _cpp_grid.run_grid(
                t, r, w, t0_flat_np, teff_flat_np,
                sigma=float(self.config.sigma),
                teff_coeff=float(self.config.teff_coeff),
                min_pts=int(self.config.min_pts_in_window),
            )
            t0_np_out, teff_np_out = t0_flat_np, teff_flat_np
        elif backend == "jax":
            args = (jnp.asarray(t), jnp.asarray(r), jnp.asarray(w),
                    jnp.asarray(t0_flat_np, dtype=jnp.float64),
                    jnp.asarray(teff_flat_np, dtype=jnp.float64))
            kw = dict(sigma=self.config.sigma, teff_coeff=self.config.teff_coeff,
                      min_pts=self.config.min_pts_in_window)
            if self.config.grid_chunk_auto:
                if verbose:
                    log.info("  -> grid exec: chunked(auto,size=%d)", self.config.grid_chunk_size)
                out = GridRunner.run_auto(*args, chunk_size=self.config.grid_chunk_size,
                                          threshold=self.config.grid_chunk_threshold, **kw)
            elif self.config.grid_chunked:
                if verbose:
                    log.info("  -> grid exec: chunked(forced,size=%d)", self.config.grid_chunk_size)
                out = GridRunner.run_chunked(*args, chunk_size=self.config.grid_chunk_size, **kw)
            else:
                if verbose:
                    log.info("  -> grid exec: vmap")
                out = GridRunner.run(*args, **kw)
            block_until_ready(out[2])
            (t0_np_out, teff_np_out, dchi2_np, nwin_np, ncontrib_np, neff_np, peak_np,
             rho1_np, longest_np) = device_get(out)
        else:
            raise ValueError(
                f"Unknown grid_backend '{backend}'. Valid options are 'cpp' and 'jax'."
            )
        dt_grid = time.perf_counter() - t_grid0

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
        # Extract the raw cluster population before applying scientific
        # candidate-quality criteria. The raw population is needed for an
        # unbiased estimate of the scan background used by the score.
        dchi2_extract = np.asarray(dchi2_np, dtype=float)
        dchi2_extract = np.where(np.isfinite(dchi2_extract), dchi2_extract, -np.inf)

        t_ext0 = time.perf_counter()
        if backend == "cpp" and hasattr(_cpp_grid, "extract_clusters"):
            clusters = _cpp_grid.extract_clusters(
                np.asarray(t0_np_out),
                np.asarray(teff_np_out),
                dchi2_extract,
                sigma_overlap=float(self.extractor.sigma_overlap),
                min_points=int(self.extractor.min_points),
            )
        else:
            clusters = self.extractor.iterative_anomaly_extraction(
                np.asarray(t0_np_out),
                np.asarray(teff_np_out),
                dchi2_extract,
            )
        dt_ext = time.perf_counter() - t_ext0

        if verbose:
            log.info("  -> extracted clusters: %d | grid=%.3fs extract=%.3fs",
                     int(clusters.shape[0]), dt_grid, dt_ext)
        return np.asarray(clusters, dtype=float).reshape(-1, 3), grid_metrics

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
        t_total0 = time.perf_counter()

        time_all = np.asarray(device_get(time_j), dtype=float)
        residual_all = np.asarray(device_get(residual_j), dtype=float)
        ferr_all = np.asarray(device_get(ferr_j), dtype=float)
        _, grids = self._season_grids(np.asarray(time_np, dtype=float))
        n_seasons = len(grids)

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

        for season_idx, grid in enumerate(grids):
            season, t0_flat = grid[0], grid[4]
            idx = season.indices
            if verbose:
                log.info(
                    "[%d/%d] Season %d: start=%.6f end=%.6f n_pts=%d",
                    season_idx + 1, n_seasons, season_idx,
                    float(season.start), float(season.end), int(idx.size),
                )
            if t0_flat.size == 0:
                if verbose:
                    log.info("  -> grid: n_grid=0 (skipped)")
                clusters = np.zeros((0, 3), dtype=float)
                grid_metrics = np.zeros((0, 9), dtype=float)
            else:
                if verbose:
                    log.info("  -> grid: n_grid=%d", int(t0_flat.size))
                e = ferr_all[idx]
                clusters, grid_metrics = self._season(
                    grid, time_all[idx], residual_all[idx], 1.0 / (e ** 2),
                    verbose=verbose, log=log,
                )
            all_clusters.append(clusters)
            all_grid_metrics.append(grid_metrics)
            season_summaries.append(
                SeasonSummary(
                    season_idx=season_idx,
                    t_start=season.start,
                    t_end=season.end,
                    n_grid=int(t0_flat.size),
                    clusters=clusters,
                    grid_metrics=grid_metrics,
                )
            )

        flat = _flat_clusters(all_clusters)
        non_empty_metrics = [g for g in all_grid_metrics if g.size > 0]
        if non_empty_metrics:
            grid_metrics_all = np.concatenate(non_empty_metrics, axis=0).astype(float, copy=False)
        else:
            grid_metrics_all = np.zeros((0, 9), dtype=float)

        if verbose:
            log.info("Total clusters (flattened): %d", int(flat.shape[0]))
            log.info("Total elapsed: %.3fs", time.perf_counter() - t_total0)

        return season_summaries, flat, grid_metrics_all

    def cluster_rows(self, time_np, residual_np, ferr_np, *, reuse_tol: float = 0.0,
                     keep: int = 16) -> np.ndarray:
        """The grid rows (t0, teff, dchi2, quality...) of every season's clusters.

        A season whose normalised residual ``z = r / ferr`` is within
        ``reuse_tol`` of a recent scan of the same samples, in
        ``min_c ||z - z_old - c||`` (a window fit has its own constant), takes
        that scan's clusters: in every window ``|d sqrt(dchi2)| <= reuse_tol``,
        so ``|d dchi2| <= 2 reuse_tol sqrt(dchi2) + reuse_tol**2``.
        """
        time_np, residual_np, ferr_np = (np.asarray(x, dtype=float)
                                         for x in (time_np, residual_np, ferr_np))
        key, grids = self._season_grids(time_np)
        key = (key, _array_key(ferr_np))
        if len(self._reuse) > 64 and not any(k[:2] == key for k in self._reuse):
            self._reuse.clear()
        out = []
        for season_idx, grid in enumerate(grids):
            season, t0_flat = grid[0], grid[4]
            if t0_flat.size == 0:
                continue
            idx = season.indices
            e = ferr_np[idx]
            z = residual_np[idx] / e
            recent = self._reuse.setdefault((*key, season_idx), [])
            hit = None
            for i, (z_old, rows) in enumerate(recent):
                dz = z - z_old
                if np.linalg.norm(dz - dz.mean()) <= reuse_tol:
                    hit = recent.pop(i)
                    break
            if hit is None:
                clusters, metrics = self._season(grid, time_np[idx], residual_np[idx], 1.0 / (e ** 2))
                hit = (z, _rows_of(clusters, metrics))
            recent.insert(0, hit)
            del recent[keep:]
            out.append(hit[1])
        rows = np.concatenate(out) if out else np.zeros((0, 9))
        return rows[np.isfinite(rows[:, :3]).all(axis=1) & (rows[:, 2] != 0)]


def _array_key(x: np.ndarray):
    x = np.ascontiguousarray(x, dtype=float)
    return (x.size, hash(x.tobytes()))


def _flat_clusters(all_clusters) -> np.ndarray:
    """All seasons' clusters, finite and with ``dchi2 != 0``."""
    parts = [np.asarray(c, dtype=float).reshape(-1, 3) for c in all_clusters]
    flat = np.concatenate(parts) if parts else np.zeros((0, 3), dtype=float)
    return flat[np.isfinite(flat).all(axis=1) & (flat[:, 2] != 0)]


def _rows_of(clusters: np.ndarray, metrics: np.ndarray) -> np.ndarray:
    """The grid row of each cluster (clusters sit on grid points), else the nearest."""
    rows = np.empty((clusters.shape[0], 9))
    for teff in np.unique(clusters[:, 1]):
        level = np.flatnonzero(metrics[:, 1] == teff)
        mine = np.flatnonzero(clusters[:, 1] == teff)
        exact = np.zeros(mine.size, bool)
        if level.size:
            t0 = metrics[level, 0]
            j = np.clip(np.searchsorted(t0, clusters[mine, 0]), 0, level.size - 1)
            exact = t0[j] == clusters[mine, 0]
            rows[mine[exact]] = metrics[level[j[exact]]]
        for i in mine[~exact]:
            d = np.abs(metrics[:, 0] - clusters[i, 0]) + np.abs(metrics[:, 1] - clusters[i, 1])
            rows[i] = metrics[int(np.argmin(d))]
    rows[:, :3] = clusters
    return rows
