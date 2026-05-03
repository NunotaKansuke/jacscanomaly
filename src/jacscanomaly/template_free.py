from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import numpy as np

from .seasons import SeasonSplitter


@dataclass(frozen=True)
class TemplateFreeSearchConfig:
    """
    Configuration for template-free residual anomaly searches.

    The default mode is a hybrid:
    1. Flag high-significance fixed-length windows.
    2. For lower-significance seed windows, search an extended local region
       for the subsequence with the largest reduced chi2.
    """

    gap: float = 100.0
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
    Template-free anomaly scanner operating on normalized residuals.
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

        z = residual_np / ferr_np
        seasons = SeasonSplitter(gap=self.config.gap).split(time_np)

        fixed: List[TemplateFreeCandidate] = []
        hybrid: List[TemplateFreeCandidate] = []
        blind: List[TemplateFreeCandidate] = []

        for season_idx, season in enumerate(seasons):
            idx = np.asarray(season.indices, dtype=int)
            t_s = time_np[idx]
            z_s = z[idx]
            chi2_s = z_s * z_s

            fixed_s, seeds_s = self._fixed_window_scan(season_idx, idx, t_s, z_s, chi2_s)
            fixed.extend(fixed_s)

            for seed in seeds_s:
                cand = self._hybrid_extended_scan(season_idx, idx, t_s, z_s, chi2_s, seed)
                if cand is not None:
                    hybrid.append(cand)

            if self.config.run_blind_reduced_chi2:
                cand = self._best_reduced_chi2_window(season_idx, idx, t_s, z_s, chi2_s)
                if cand is not None:
                    blind.append(cand)

        candidates = self._dedupe_candidates([*fixed, *hybrid, *blind])
        best = max(candidates, key=lambda c: c.chi2, default=None)

        return TemplateFreeSearchResult(
            time=time_np,
            residual=residual_np,
            ferr=ferr_np,
            z=z,
            candidates=tuple(candidates),
            fixed_window_candidates=tuple(fixed),
            hybrid_candidates=tuple(hybrid),
            blind_reduced_candidates=tuple(blind),
            best=best,
        )

    def _fixed_window_scan(self, season_idx, global_idx, time, z, chi2):
        k = int(self.config.fixed_window_points)
        if k <= 0:
            raise ValueError("fixed_window_points must be positive.")
        if chi2.size < k:
            return [], []

        csum = np.concatenate([[0.0], np.cumsum(chi2)])
        window_chi2 = csum[k:] - csum[:-k]

        high = np.flatnonzero(window_chi2 >= float(self.config.fixed_chi2_threshold))
        seed = np.flatnonzero(
            (window_chi2 >= float(self.config.hybrid_seed_chi2_threshold))
            & (window_chi2 < float(self.config.fixed_chi2_threshold))
        )

        fixed = [
            self._make_candidate(
                kind="fixed",
                season_idx=season_idx,
                global_idx=global_idx,
                time=time,
                z=z,
                start=int(i),
                end=int(i + k),
                chi2=float(window_chi2[i]),
            )
            for i in high
        ]
        fixed = self._select_non_overlapping(fixed)

        seed_windows = [(int(i), int(i + k), float(window_chi2[i])) for i in seed]
        seed_windows = self._select_seed_windows(seed_windows)
        return fixed, seed_windows

    def _hybrid_extended_scan(self, season_idx, global_idx, time, z, chi2, seed):
        seed_start, seed_end, seed_chi2 = seed
        radius = max(0, int(self.config.hybrid_extension_points))
        lo = max(0, seed_start - radius)
        hi = min(chi2.size, seed_end + radius)

        cand = self._best_reduced_chi2_window(
            season_idx,
            global_idx,
            time,
            z,
            chi2,
            lo=lo,
            hi=hi,
            require_overlap=(seed_start, seed_end),
            kind="hybrid",
            seed=(seed_start, seed_end, seed_chi2),
        )
        if cand is None:
            return None
        if cand.reduced_chi2 < float(self.config.hybrid_reduced_chi2_threshold):
            return None
        return cand

    def _best_reduced_chi2_window(
        self,
        season_idx,
        global_idx,
        time,
        z,
        chi2,
        *,
        lo: int = 0,
        hi: Optional[int] = None,
        require_overlap: Optional[tuple[int, int]] = None,
        kind: str = "blind_reduced",
        seed: Optional[tuple[int, int, float]] = None,
    ):
        hi = chi2.size if hi is None else int(hi)
        lo = int(lo)
        if hi <= lo:
            return None

        min_points = max(1, int(self.config.reduced_min_points))
        max_points = self.config.reduced_max_points
        if max_points is not None:
            max_points = max(min_points, int(max_points))

        csum = np.concatenate([[0.0], np.cumsum(chi2)])
        best = None

        for start in range(lo, hi):
            end_min = start + min_points
            if end_min > hi:
                break
            end_max = hi if max_points is None else min(hi, start + max_points)

            for end in range(end_min, end_max + 1):
                if require_overlap is not None:
                    seed_start, seed_end = require_overlap
                    if end <= seed_start or start >= seed_end:
                        continue
                total = float(csum[end] - csum[start])
                n_points = end - start
                reduced = total / n_points
                if best is None or reduced > best[0]:
                    best = (reduced, total, start, end)

        if best is None:
            return None

        reduced, total, start, end = best
        seed_start = seed_end = seed_chi2 = None
        if seed is not None:
            seed_start, seed_end, seed_chi2 = seed

        return self._make_candidate(
            kind=kind,
            season_idx=season_idx,
            global_idx=global_idx,
            time=time,
            z=z,
            start=start,
            end=end,
            chi2=total,
            seed_start=seed_start,
            seed_end=seed_end,
            seed_chi2=seed_chi2,
        )

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

    def _select_non_overlapping(self, candidates):
        ordered = sorted(candidates, key=lambda c: c.chi2, reverse=True)
        selected = []
        for cand in ordered:
            overlaps = any(
                cand.start_index <= kept.end_index and kept.start_index <= cand.end_index
                for kept in selected
            )
            if not overlaps:
                selected.append(cand)
            if len(selected) >= int(self.config.max_candidates_per_season):
                break
        return sorted(selected, key=lambda c: c.t_center)

    def _select_seed_windows(self, seed_windows):
        ordered = sorted(seed_windows, key=lambda item: item[2], reverse=True)
        selected = []
        for start, end, chi2 in ordered:
            overlaps = any(start < kept_end and kept_start < end for kept_start, kept_end, _ in selected)
            if not overlaps:
                selected.append((start, end, chi2))
            if len(selected) >= int(self.config.max_candidates_per_season):
                break
        return selected

    def _dedupe_candidates(self, candidates):
        ordered = sorted(candidates, key=lambda c: c.chi2, reverse=True)
        selected = []
        for cand in ordered:
            overlaps = any(
                cand.start_index <= kept.end_index and kept.start_index <= cand.end_index
                for kept in selected
            )
            if not overlaps:
                selected.append(cand)
        return sorted(selected, key=lambda c: c.chi2, reverse=True)
