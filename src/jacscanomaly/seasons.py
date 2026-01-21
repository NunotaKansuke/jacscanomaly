from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class Season:
    """
    A contiguous observing season determined by time gaps.

    Attributes
    ----------
    start, end : float
        Season time range (min/max in the sorted time series for this season).
    indices : np.ndarray
        Indices into the *original* input array that belong to this season.

    Notes
    -----
    - `indices` correspond to the original input ordering, but the season boundaries
      (start/end) are computed after sorting by time.
    """
    start: float
    end: float
    indices: np.ndarray  # shape (N_season,), dtype int


@dataclass
class SeasonSplitter:
    """
    Split a time series into seasons separated by large time gaps.

    A new season starts when the time difference between consecutive *sorted* points
    exceeds `gap`.

    Parameters
    ----------
    gap : float
        Threshold for splitting seasons. If dt > gap, a season break occurs.
    """
    gap: float = 100.0

    def split(self, time) -> List[Season]:
        """
        Split input `time` into seasons.

        Parameters
        ----------
        time : array-like
            1D time array. NumPy is recommended, but any array-like is accepted.

        Returns
        -------
        seasons : list[Season]
            List of Season objects. If the input is empty, returns an empty list.

        Notes
        -----
        - Seasons are computed on `time` sorted in ascending order.
        - `Season.indices` always reference the original input positions.
        """
        t = np.asarray(time)

        if t.ndim != 1:
            raise ValueError(f"time must be 1D, got shape {t.shape}")
        if t.size == 0:
            return []
        if not np.all(np.isfinite(t)):
            raise ValueError("time must contain only finite values.")

        order = np.argsort(t)
        t_sorted = t[order]

        dt = np.diff(t_sorted)
        breaks = np.where(dt > self.gap)[0]

        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, t_sorted.size - 1]

        seasons: List[Season] = []
        for s, e in zip(starts, ends):
            # indices in sorted array for this season
            idx_sorted = np.arange(s, e + 1, dtype=int)
            # map back to original indices
            idx_orig = order[idx_sorted]
            seasons.append(
                Season(
                    start=float(t_sorted[s]),
                    end=float(t_sorted[e]),
                    indices=idx_orig.astype(int, copy=False),
                )
            )

        return seasons

    @staticmethod
    def build_t0_grid(season: Season, dt0: float) -> np.ndarray:
        """
        Build the t0 grid within a season.

        Parameters
        ----------
        season : Season
            Season object returned by :meth:`split`.
        dt0 : float
            Grid spacing. Must be positive.

        Returns
        -------
        t0_vals : np.ndarray
            Array of t0 grid points in [season.start, season.end) with step dt0.
            Returns an empty array if season.start >= season.end.

        Raises
        ------
        ValueError
            If dt0 <= 0.
        """
        if dt0 <= 0:
            raise ValueError(f"dt0 must be positive, got {dt0}.")
        if season.start >= season.end:
            return np.zeros((0,), dtype=float)
        return np.arange(season.start, season.end, dt0, dtype=float)
