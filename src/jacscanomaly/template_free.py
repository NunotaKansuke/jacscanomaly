from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import numpy as np

from .seasons import SeasonSplitter


@dataclass(frozen=True)
class TemplateFreeSearchConfig:
    """
    Configuration for template-free residual anomaly searches.

    The scanner splits each season into segments bounded by z=0 sign
    crossings. Segments whose peak |z| exceeds ``seed_z_threshold`` qualify
    and immediately gain their neighboring segment on each side, so every
    window reaches the second zero crossing away from its seed. Windows are
    then joined whenever the gap between their edges is small compared to the
    wider window (``max(bridge_floor_points, bridge_fraction * larger
    width)``), which bridges both single-point noise crossings and the slow
    zero transitions of smooth residual structures. A merged window becomes a
    candidate when its total chi2 exceeds ``candidate_chi2_threshold``.

    Optionally, z-scores are recalibrated within each season by iterative
    sigma clipping before scanning (``renormalize_z``).
    """

    gap: float = 100.0

    renormalize_z: bool = False
    sigma_clip_threshold: float = 3.0
    sigma_clip_max_iter: int = 5
    seed_z_threshold: float = 5.0
    bridge_floor_points: int = 2
    bridge_fraction: float = 0.25
    candidate_chi2_threshold: float = 150.0

    # Legacy options retained for configuration compatibility with existing
    # notebooks, but not used by the current scanner.
    zero_crossings_each_side: int = 2
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
            t_seed = (
                float(self.time[item.seed_start_index])
                if item.seed_start_index is not None
                else item.t_center
            )
            ax.axvline(t_seed, color=color, lw=1, alpha=0.8, zorder=3)

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
        floor_points = int(self.config.bridge_floor_points)
        fraction = float(self.config.bridge_fraction)
        if seed_threshold <= 0:
            raise ValueError("seed_z_threshold must be positive.")
        if floor_points < 0:
            raise ValueError("bridge_floor_points must be non-negative.")
        if fraction < 0:
            raise ValueError("bridge_fraction must be non-negative.")

        # Segment the season at z=0 sign crossings.
        crossings = np.flatnonzero(z[:-1] * z[1:] < 0.0)
        bounds = np.concatenate([[0], crossings + 1, [z.size]])
        segment_peak = np.maximum.reduceat(np.abs(z), bounds[:-1])
        qualifying = np.flatnonzero(segment_peak > seed_threshold)
        if qualifying.size == 0:
            return []

        # Each qualifying segment gains its neighboring segment on each side,
        # so the window reaches the second zero crossing away from its peak.
        n_segments = bounds.size - 1
        windows = [
            (int(bounds[max(0, i - 1)]), int(bounds[min(n_segments, i + 2)]))
            for i in qualifying
        ]
        windows = self._join_windows(windows, floor_points, fraction)

        csum = np.concatenate([[0.0], np.cumsum(z * z)])
        candidates = []
        for start, end in windows:
            total = float(csum[end] - csum[start])
            if total <= chi2_threshold:
                continue
            seed = start + int(np.argmax(np.abs(z[start:end])))
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

        max_candidates = max(0, int(self.config.max_candidates_per_season))
        if max_candidates:
            candidates = sorted(candidates, key=lambda c: c.chi2, reverse=True)[:max_candidates]
        return candidates

    @staticmethod
    def _join_windows(windows, floor_points, fraction):
        """
        Join windows whose edge gap is small compared to the wider window.

        The allowance ``max(floor_points, fraction * larger width)`` bridges
        single-point noise crossings and, for wide structures, the slow z=0
        transitions between their lobes. Joining is iterated to a fixed point
        because merged windows grow and may enable further bridges.
        """
        joined = sorted(windows)
        changed = True
        while changed:
            changed = False
            merged = [list(joined[0])]
            for start, end in joined[1:]:
                previous = merged[-1]
                gap = start - previous[1]
                allowance = max(
                    floor_points,
                    fraction * max(previous[1] - previous[0], end - start),
                )
                if gap <= allowance:
                    previous[1] = max(previous[1], end)
                    changed = True
                else:
                    merged.append([start, end])
            joined = merged
        return [(start, end) for start, end in joined]

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

