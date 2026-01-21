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
from .utils import get_chi2_anom, get_chi2_flat

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
        get_chi2_anom(t0, teff, time, flux, ferr) -> (chi2_total, chi2s_per_point)
        get_chi2_flat(time, flux, ferr)           -> (chi2_total, chi2s_per_point)
      where chi2s_per_point is expected to be (residual / error)^2 per datum.

    - Outside the evaluation window, we "ignore" points by inflating uncertainties
      (ferr -> big), so their contribution to chi2 becomes negligible.
    """

    @staticmethod
    @partial(jit, static_argnames=("sigma", "teff_coeff", "min_pts"))
    def _grid_point(
        t0_ref: jnp.ndarray,
        teff_ref: jnp.ndarray,
        time: jnp.ndarray,
        flux: jnp.ndarray,
        ferr: jnp.ndarray,
        *,
        sigma: float = 3.0,
        teff_coeff: float = 3.0,
        min_pts: int = 4,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        mask = (time > (t0_ref - teff_coeff * teff_ref)) & (time < (t0_ref + teff_coeff * teff_ref))
        n_pts = jnp.sum(mask.astype(jnp.int32))

        def valid(_):
            big = jnp.asarray(1e12, dtype=ferr.dtype)
            ferr_eff = jnp.where(mask, ferr, big)

            chi2_anom, chi2s_anom = get_chi2_anom(t0_ref, teff_ref, time, flux, ferr_eff)
            chi2_flat, chi2s_flat = get_chi2_flat(time, flux, ferr_eff)

            dchi2 = chi2_flat - chi2_anom

            diff = chi2s_flat - chi2s_anom
            n_out = jnp.sum((diff > (sigma**2)) & mask).astype(jnp.int32)

            return dchi2.astype(jnp.float32), n_out

        def invalid(_):
            return jnp.asarray(0.0, jnp.float32), jnp.asarray(0, jnp.int32)

        return lax.cond(n_pts >= min_pts, valid, invalid, operand=None)

    @staticmethod
    @partial(jit, static_argnames=("sigma", "teff_coeff", "min_pts"))
    def run(
        time: jnp.ndarray,
        flux: jnp.ndarray,
        ferr: jnp.ndarray,
        t0_flat: jnp.ndarray,
        teff_flat: jnp.ndarray,
        *,
        sigma: float = 3.0,
        teff_coeff: float = 3.0,
        min_pts: int = 4,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        dchi2, n_out = vmap(
            lambda t0_ref, teff_ref: GridRunner._grid_point(
                t0_ref, teff_ref, time, flux, ferr,
                sigma=sigma, teff_coeff=teff_coeff, min_pts=min_pts
            )
        )(t0_flat, teff_flat)

        return t0_flat, teff_flat, dchi2, n_out


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
    ) -> Tuple[List[SeasonSummary], np.ndarray]:

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

        for season_idx, season in enumerate(seasons):
            t_season0 = time.perf_counter()  # ★ season start

            # JAX indexing: make indices a JAX array (safer across backends)
            idx = jnp.asarray(season.indices)
            t_season = time_j[idx]
            r_season = residual_j[idx]
            e_season = ferr_j[idx]

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
                    )
                )
                all_clusters.append(clusters)
                continue

            n_grid = int(sum(arr.size for arr in t0_flat_list))
            if verbose:
                log.info("  -> grid: n_grid=%d", n_grid)

            # JAX arrays for the grid
            t0_flat = jnp.asarray(np.concatenate(t0_flat_list), dtype=jnp.float64)
            teff_flat = jnp.asarray(np.concatenate(teff_flat_list), dtype=jnp.float64)

            # ---- timed block: grid run
            t_grid0 = time.perf_counter()
            t0_out, teff_out, dchi2, n_out = GridRunner.run(
                t_season, r_season, e_season,
                t0_flat, teff_flat,
                sigma=self.config.sigma,
                teff_coeff=self.config.teff_coeff,
                min_pts=self.config.min_pts_in_window,
            )
            block_until_ready(dchi2)
            dt_grid = time.perf_counter() - t_grid0

            # bring results back to CPU in one call
            t_copy0 = time.perf_counter()
            t0_np_out, teff_np_out, dchi2_np = device_get((t0_out, teff_out, dchi2))
            dt_copy = time.perf_counter() - t_copy0

            # ---- timed block: extract clusters
            t_ext0 = time.perf_counter()
            clusters = self.extractor.iterative_anomaly_extraction(
                np.asarray(t0_np_out),
                np.asarray(teff_np_out),
                np.asarray(dchi2_np),
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
            season_summaries.append(
                SeasonSummary(
                    season_idx=season_idx,
                    t_start=season.start,
                    t_end=season.end,
                    n_grid=int(t0_flat.shape[0]),
                    clusters=clusters,
                )
            )

        # flatten clusters across seasons
        if len(all_clusters) > 0:
            flat = np.array([c for cs in all_clusters for c in cs], dtype=float)
        else:
            flat = np.zeros((0, 3), dtype=float)

        if flat.size:
            flat = flat[flat[:, 2] != 0]

        dt_total = time.perf_counter() - t_total0  # ★ total end

        if verbose:
            log.info("Total clusters (flattened): %d", int(flat.shape[0]))
            log.info("Total elapsed: %.3fs", dt_total)

        return season_summaries, flat
