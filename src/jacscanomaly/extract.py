from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class ResultExtractor:
    """
    Cluster extractor for grid-scan candidates.

    Given arrays of (t0, teff, delta_chi2) evaluated on a grid,
    this class groups overlapping candidates and returns one representative
    (the maximum delta_chi2 point) per cluster.

    Overlap definition
    ------------------
    Two candidates i and j are considered overlapping if:

        |t0_i - t0_j| < sigma_overlap * (teff_i + teff_j)

    Notes
    -----
    - This operates on CPU / NumPy arrays (no JAX).
    - Returned `clusters` has shape (K, 3) with rows [t0_best, teff_best, dchi2_best].
    """

    sigma_overlap: float = 3.0
    min_points: int = 3

    def _overlap_with_max(
        self,
        t0: np.ndarray,
        teff: np.ndarray,
        dchi2: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """
        Compute the overlap mask around the current maximum dchi2 point.

        Returns
        -------
        overlap_mask : np.ndarray of bool, shape (N,)
            Mask selecting points overlapping with the maximum point.
        i_max : int
            Index of the maximum point within the provided arrays.
        """
        i_max = int(np.nanargmax(dchi2))
        t0_max = t0[i_max]
        teff_max = teff[i_max]
        overlap_mask = np.abs(t0 - t0_max) < self.sigma_overlap * (teff + teff_max)
        return overlap_mask, i_max

    def iterative_anomaly_extraction(
        self,
        t0_list,
        teff_list,
        dchi2_list,
    ) -> np.ndarray:
        """
        Iteratively extract non-overlapping clusters from grid results.

        Parameters
        ----------
        t0_list, teff_list, dchi2_list
            1D arrays (or array-like) of equal length.

        Returns
        -------
        clusters : np.ndarray, shape (K, 3)
            Each row is [t0, teff, dchi2] for the best (max dchi2) point
            in each extracted cluster.
            Returns an empty array with shape (0, 3) if nothing is extractable.

        Stopping conditions
        -------------------
        - No remaining candidates.
        - The best remaining candidate is non-finite.
        - Remaining candidate count drops below `min_points`.
        """
        t0 = np.asarray(t0_list, dtype=float)
        teff = np.asarray(teff_list, dtype=float)
        dchi2 = np.asarray(dchi2_list, dtype=float)

        if t0.size == 0:
            return np.zeros((0, 3), dtype=float)

        if not (t0.shape == teff.shape == dchi2.shape):
            raise ValueError(
                f"Input arrays must have the same shape, got "
                f"t0={t0.shape}, teff={teff.shape}, dchi2={dchi2.shape}"
            )

        clusters: List[List[float]] = []
        remaining = np.ones_like(dchi2, dtype=bool)

        while True:
            if not np.any(remaining):
                break

            # pick the best remaining point
            dchi2_rem = np.where(remaining, dchi2, -np.inf)
            i_max_global = int(np.argmax(dchi2_rem))

            if not np.isfinite(dchi2[i_max_global]):
                break

            # overlap mask in the "compressed" remaining arrays
            overlap_mask, _ = self._overlap_with_max(
                t0[remaining], teff[remaining], dchi2[remaining]
            )

            # expand to full mask
            full_mask = np.zeros_like(remaining)
            full_mask[np.where(remaining)[0][overlap_mask]] = True

            # choose the best representative in this cluster
            cluster_dchi2 = dchi2[full_mask]
            cluster_t0 = t0[full_mask]
            cluster_teff = teff[full_mask]

            i_local_max = int(np.argmax(cluster_dchi2))
            clusters.append(
                [float(cluster_t0[i_local_max]), float(cluster_teff[i_local_max]), float(cluster_dchi2[i_local_max])]
            )

            # remove this cluster from remaining
            remaining &= ~full_mask

            if int(np.sum(remaining)) < self.min_points:
                break

        return np.asarray(clusters, dtype=float)
