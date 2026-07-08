from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import numpy as np

from .seasons import SeasonSplitter


@dataclass(frozen=True)
class TemplateFreeSearchConfig:
    """
    Configuration for template-free residual anomaly searches.

    The current scanner optionally recalibrates residual z-scores within each
    season using iterative sigma clipping, then grows high-z seeds to local
    zero-crossing windows and reports candidates above a chi2 threshold.
    """

    gap: float = 100.0
    renormalize_z: bool = False
    sigma_clip_threshold: float = 3.0
    sigma_clip_max_iter: int = 5
    seed_z_threshold: float = 3.0
    zero_crossings_each_side: int = 2
    candidate_chi2_threshold: float = 150.0

    # Legacy fixed/hybrid scan options are retained for configuration
    # compatibility with existing notebooks, but are not used by the scanner.
    fixed_window_points: int = 6
    fixed_chi2_threshold: float = 100.0

    hybrid_seed_chi2_threshold: float = 30.0
    hybrid_reduced_chi2_threshold: float = 5.0
    hybrid_extension_points: int = 24

    reduced_min_points: int = 6
    reduced_max_points: Optional[int] = None
    run_blind_reduced_chi2: bool = False

    max_candidates_per_season: int = 20


@dataclass(frozen=True)
class TemplateFreeCandidate:
    """
    A template-free anomaly candidate measured directly from residual chi2.
    """

    kind: str
    season_idx: int
    start_index: int
    end_index: int
    t_start: float
    t_end: float
    t_center: float
    n_points: int
    chi2: float
    reduced_chi2: float
    max_abs_z: float
    seed_start_index: Optional[int] = None
    seed_end_index: Optional[int] = None
    seed_chi2: Optional[float] = None


@dataclass(frozen=True)
class TemplateFreeSearchResult:
    """
    Result of a template-free residual anomaly search.
    """

    time: np.ndarray
    residual: np.ndarray
    ferr: np.ndarray
    z: np.ndarray
    candidates: tuple[TemplateFreeCandidate, ...]
    fixed_window_candidates: tuple[TemplateFreeCandidate, ...]
    hybrid_candidates: tuple[TemplateFreeCandidate, ...]
    blind_reduced_candidates: tuple[TemplateFreeCandidate, ...]
    best: Optional[TemplateFreeCandidate]

    def plot(
        self,
        *,
        show: bool = True,
        ax=None,
        xlim: Optional[tuple[float, float]] = None,
        candidate: Optional[TemplateFreeCandidate] = None,
        top_n: int = 8,
        zoom: bool = False,
        zoom_pad: float = 5.0,
        use_normalized_residual: bool = True,
    ):
        """
        Plot residuals and highlight template-free candidates.

        Returns (fig, ax).
        """
        import matplotlib.pyplot as plt

        cand = self.best if candidate is None else candidate
        y = self.z if use_normalized_residual else self.residual
        yerr = None if use_normalized_residual else self.ferr
        ylabel = "residual / error" if use_normalized_residual else "residual"

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if yerr is None:
            ax.plot(self.time, y, ".", ms=4, zorder=1)
        else:
            ax.errorbar(self.time, y, yerr=yerr, fmt=".", ms=4, zorder=1)

        ax.axhline(0.0, lw=1, c="0.5", zorder=0)

        labeled_candidate = False
        for item in self.candidates[: max(0, int(top_n))]:
            color = "C3" if item == cand else "C1"
            alpha = 0.25 if item == cand else 0.12
            label = None
            if not labeled_candidate:
                label = "anomaly candidate"
                labeled_candidate = True
            ax.axvspan(item.t_start, item.t_end, color=color, alpha=alpha, zorder=2, label=label)
            ax.axvline(item.t_center, color=color, lw=1, alpha=0.8, zorder=3)

        if cand is not None:
            title = (
                f"anomaly candidate: chi2={cand.chi2:.1f}, "
                f"chi2/n={cand.reduced_chi2:.2f}, n={cand.n_points}"
            )
            ax.set_title(title)

        if xlim is not None:
            ax.set_xlim(xlim)
        elif zoom and cand is not None:
            pad = float(zoom_pad)
            ax.set_xlim(cand.t_start - pad, cand.t_end + pad)

        ax.set_xlabel("time")
        ax.set_ylabel(ylabel)
        ax.minorticks_on()

        if self.candidates:
            ax.legend()

        if show:
            plt.show()
        return fig, ax


class TemplateFreeScanner:
    """
    Template-free anomaly scanner operating on residual z-score excursions.
    """

    def __init__(self, config: Optional[TemplateFreeSearchConfig] = None):
        self.config = TemplateFreeSearchConfig() if config is None else config

    def run(self, time, residual, ferr) -> TemplateFreeSearchResult:
        time_np = np.asarray(time, dtype=float)
        residual_np = np.asarray(residual, dtype=float)
        ferr_np = np.asarray(ferr, dtype=float)

        if time_np.ndim != 1 or residual_np.ndim != 1 or ferr_np.ndim != 1:
            raise ValueError("time/residual/ferr must be 1D arrays.")
        if not (len(time_np) == len(residual_np) == len(ferr_np)):
            raise ValueError("time/residual/ferr must have the same length.")
        if np.any(~np.isfinite(time_np)) or np.any(~np.isfinite(residual_np)) or np.any(~np.isfinite(ferr_np)):
            raise ValueError("time/residual/ferr must be finite.")
        if np.any(ferr_np <= 0):
            raise ValueError("ferr must be positive.")

        z_raw = residual_np / ferr_np
        z = np.empty_like(z_raw)
        seasons = SeasonSplitter(gap=self.config.gap).split(time_np)

        candidates: List[TemplateFreeCandidate] = []

        for season_idx, season in enumerate(seasons):
            idx = np.asarray(season.indices, dtype=int)
            t_s = time_np[idx]
            z_s = (
                self._sigma_clipped_z(z_raw[idx])
                if self.config.renormalize_z
                else np.asarray(z_raw[idx], dtype=float)
            )
            z[idx] = z_s
            candidates.extend(self._scan_season(season_idx, idx, t_s, z_s))

        candidates = sorted(candidates, key=lambda c: c.chi2, reverse=True)
        best = max(candidates, key=lambda c: c.chi2, default=None)

        return TemplateFreeSearchResult(
            time=time_np,
            residual=residual_np,
            ferr=ferr_np,
            z=z,
            candidates=tuple(candidates),
            fixed_window_candidates=(),
            hybrid_candidates=(),
            blind_reduced_candidates=(),
            best=best,
        )

    def _sigma_clipped_z(self, z_raw):
        clip = float(self.config.sigma_clip_threshold)
        max_iter = int(self.config.sigma_clip_max_iter)
        if clip <= 0:
            raise ValueError("sigma_clip_threshold must be positive.")
        if max_iter < 0:
            raise ValueError("sigma_clip_max_iter must be non-negative.")
        if z_raw.size == 0:
            return np.asarray(z_raw, dtype=float)

        retained = np.ones(z_raw.size, dtype=bool)
        center = float(np.median(z_raw))
        scale = self._safe_std(z_raw)

        for _ in range(max_iter):
            sample = z_raw[retained]
            if sample.size == 0:
                break
            center = float(np.median(sample))
            scale = self._safe_std(sample)
            next_retained = np.abs((z_raw - center) / scale) <= clip
            if np.array_equal(next_retained, retained):
                break
            retained = next_retained

        sample = z_raw[retained]
        if sample.size > 0:
            center = float(np.median(sample))
            scale = self._safe_std(sample)
        return (z_raw - center) / scale

    @staticmethod
    def _safe_std(values):
        scale = float(np.std(values))
        if not np.isfinite(scale) or scale <= 0.0:
            return 1.0
        return scale

    def _scan_season(self, season_idx, global_idx, time, z):
        if z.size == 0:
            return []

        seed_threshold = float(self.config.seed_z_threshold)
        chi2_threshold = float(self.config.candidate_chi2_threshold)
        n_cross = int(self.config.zero_crossings_each_side)
        if seed_threshold <= 0:
            raise ValueError("seed_z_threshold must be positive.")
        if n_cross < 0:
            raise ValueError("zero_crossings_each_side must be non-negative.")

        chi2 = z * z
        csum = np.concatenate([[0.0], np.cumsum(chi2)])
        crossings = self._zero_crossings(z)
        seeds = np.flatnonzero(np.abs(z) > seed_threshold)

        windows_by_extent = {}
        for seed in seeds:
            start, end = self._zero_crossing_window(int(seed), z.size, crossings, n_cross)
            total = float(csum[end] - csum[start])
            if total > chi2_threshold:
                key = (start, end)
                previous = windows_by_extent.get(key)
                if previous is None or abs(z[seed]) > abs(z[previous[2]]):
                    windows_by_extent[key] = (start, end, int(seed), total)

        windows = list(windows_by_extent.values())
        candidates = self._merge_overlapping_windows(season_idx, global_idx, time, z, windows)
        max_candidates = max(0, int(self.config.max_candidates_per_season))
        if max_candidates:
            candidates = sorted(candidates, key=lambda c: c.chi2, reverse=True)[:max_candidates]
        return candidates

    @staticmethod
    def _zero_crossings(z):
        signs = np.sign(z)
        crossings = []
        last_nonzero = None
        last_index = None
        zero_start = None

        for idx, sign in enumerate(signs):
            if sign == 0:
                if zero_start is None:
                    zero_start = idx
                continue

            if last_nonzero is not None and sign != last_nonzero:
                boundary = zero_start if zero_start is not None else idx - 1
                crossings.append(int(boundary))
            last_nonzero = sign
            last_index = idx
            zero_start = None

        if last_index is None:
            return np.asarray([], dtype=int)
        return np.asarray(crossings, dtype=int)

    @staticmethod
    def _zero_crossing_window(seed, n_points, crossings, n_cross):
        if n_cross == 0:
            return seed, seed + 1

        insert = int(np.searchsorted(crossings, seed, side="left"))
        if insert >= n_cross:
            start = int(crossings[insert - n_cross]) + 1
        else:
            start = 0

        if crossings.size - insert >= n_cross:
            end = int(crossings[insert + n_cross - 1]) + 1
        else:
            end = n_points
        start = min(start, seed)
        end = max(end, seed + 1)
        return start, end

    def _merge_overlapping_windows(self, season_idx, global_idx, time, z, windows):
        if not windows:
            return []

        ordered = sorted(windows, key=lambda item: (item[0], item[1], item[2]))
        merged = []
        cur_end, cur_windows = ordered[0][1], [ordered[0]]

        for start, end, seed, total in ordered[1:]:
            if start < cur_end:
                cur_end = max(cur_end, end)
                cur_windows.append((start, end, seed, total))
            else:
                merged.append(cur_windows)
                cur_end, cur_windows = end, [(start, end, seed, total)]
        merged.append(cur_windows)

        candidates = []
        for group in merged:
            start, end, seed, total = max(group, key=lambda item: (item[3], abs(z[item[2]])))
            candidates.append(
                self._make_candidate(
                    kind="zero_crossing",
                    season_idx=season_idx,
                    global_idx=global_idx,
                    time=time,
                    z=z,
                    start=start,
                    end=end,
                    chi2=total,
                    seed_start=seed,
                    seed_end=seed + 1,
                    seed_chi2=float(z[seed] * z[seed]),
                )
            )
        return candidates

    def _make_candidate(
        self,
        *,
        kind,
        season_idx,
        global_idx,
        time,
        z,
        start,
        end,
        chi2,
        seed_start=None,
        seed_end=None,
        seed_chi2=None,
    ) -> TemplateFreeCandidate:
        n_points = int(end - start)
        start_index = int(global_idx[start])
        end_index = int(global_idx[end - 1])
        t_start = float(time[start])
        t_end = float(time[end - 1])
        return TemplateFreeCandidate(
            kind=str(kind),
            season_idx=int(season_idx),
            start_index=start_index,
            end_index=end_index,
            t_start=t_start,
            t_end=t_end,
            t_center=0.5 * (t_start + t_end),
            n_points=n_points,
            chi2=float(chi2),
            reduced_chi2=float(chi2) / n_points,
            max_abs_z=float(np.max(np.abs(z[start:end]))),
            seed_start_index=None if seed_start is None else int(global_idx[seed_start]),
            seed_end_index=None if seed_end is None else int(global_idx[seed_end - 1]),
            seed_chi2=None if seed_chi2 is None else float(seed_chi2),
        )
