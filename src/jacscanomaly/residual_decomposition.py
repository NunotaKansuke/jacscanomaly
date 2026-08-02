"""Cheap time-binned separation of smooth and localized residual structure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinnedResidualDecomposition:
    """A robust low-frequency residual estimate and its local remainder."""

    smooth_z: np.ndarray
    binned_z: np.ndarray
    local_z: np.ndarray
    bin_time: np.ndarray
    bin_z: np.ndarray
    smooth_bin_z: np.ndarray
    cadence: float
    bin_width: float
    trend_half_width: float


def decompose_binned_residual(
    time,
    standardized_residual,
    *,
    characteristic_scale: float,
    bin_scale_fraction: float = 0.5,
    trend_scale_factor: float = 3.0,
    min_cadences_per_bin: float = 3.0,
    min_trend_bins: int = 5,
) -> BinnedResidualDecomposition:
    """Separate a residual into a smooth trend and localized structure.

    The first reduction uses medians in fixed-duration time bins, so a few
    caustic samples cannot drag the representative value of a densely sampled
    bin. Interpolating those bin medians back to the original cadence defines
    the within-bin local remainder used by the mask. A second, time-domain
    running median supplies the slower component used by physical-effect
    diagnostics; it is not subtracted when deciding the point mask.

    This function only decomposes the residual.  It deliberately does not
    decide whether the smooth component is PSPL, FSPL, or parallax.
    """

    t = np.asarray(time, dtype=float).reshape(-1)
    z = np.asarray(standardized_residual, dtype=float).reshape(-1)
    if t.size != z.size:
        raise ValueError("time and standardized_residual must have the same length.")
    if t.size == 0:
        empty = np.asarray([], dtype=float)
        return BinnedResidualDecomposition(
            smooth_z=empty,
            binned_z=empty,
            local_z=empty,
            bin_time=empty,
            bin_z=empty,
            smooth_bin_z=empty,
            cadence=0.0,
            bin_width=0.0,
            trend_half_width=0.0,
        )

    finite = np.isfinite(t) & np.isfinite(z)
    if not np.any(finite):
        zeros = np.zeros(t.shape, dtype=float)
        return BinnedResidualDecomposition(
            smooth_z=zeros,
            binned_z=zeros,
            local_z=zeros,
            bin_time=np.asarray([], dtype=float),
            bin_z=np.asarray([], dtype=float),
            smooth_bin_z=np.asarray([], dtype=float),
            cadence=0.0,
            bin_width=0.0,
            trend_half_width=0.0,
        )

    finite_indices = np.flatnonzero(finite)
    order = np.argsort(t[finite], kind="stable")
    sorted_indices = finite_indices[order]
    ts = t[sorted_indices]
    zs = z[sorted_indices]

    positive_dt = np.diff(ts)
    positive_dt = positive_dt[np.isfinite(positive_dt) & (positive_dt > 0.0)]
    cadence = float(np.median(positive_dt)) if positive_dt.size else 0.0
    scale = abs(float(characteristic_scale))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = cadence if cadence > 0.0 else 1.0

    cadence_width = max(float(min_cadences_per_bin), 1.0) * cadence
    bin_width = max(
        max(float(bin_scale_fraction), 0.0) * scale,
        cadence_width,
        np.finfo(float).eps * max(1.0, float(np.max(np.abs(ts)))),
    )

    origin = float(ts[0])
    bin_id = np.floor((ts - origin) / bin_width).astype(np.int64)
    unique_bins, starts = np.unique(bin_id, return_index=True)
    ends = np.r_[starts[1:], ts.size]
    bin_time = np.empty(unique_bins.size, dtype=float)
    bin_z = np.empty(unique_bins.size, dtype=float)
    for i, (start, end) in enumerate(zip(starts, ends)):
        sl = slice(int(start), int(end))
        bin_time[i] = float(np.median(ts[sl]))
        bin_z[i] = float(np.median(zs[sl]))

    trend_half_width = max(float(trend_scale_factor), 1.0) * scale
    min_bins = max(int(min_trend_bins), 1)
    smooth_bin_z = np.empty_like(bin_z)
    for i, center in enumerate(bin_time):
        nearby = np.flatnonzero(np.abs(bin_time - center) <= trend_half_width)
        if nearby.size < min_bins:
            distance_order = np.argsort(np.abs(bin_time - center), kind="stable")
            nearby = distance_order[: min(min_bins, bin_time.size)]
        smooth_bin_z[i] = float(np.median(bin_z[nearby]))

    if bin_time.size == 1:
        binned_sorted = np.full(ts.shape, bin_z[0], dtype=float)
        smooth_sorted = np.full(ts.shape, smooth_bin_z[0], dtype=float)
    else:
        binned_sorted = np.interp(ts, bin_time, bin_z)
        smooth_sorted = np.interp(ts, bin_time, smooth_bin_z)

    smooth_z = np.zeros(t.shape, dtype=float)
    binned_z = np.zeros(t.shape, dtype=float)
    local_z = np.zeros(t.shape, dtype=float)
    smooth_z[sorted_indices] = smooth_sorted
    binned_z[sorted_indices] = binned_sorted
    local_z[sorted_indices] = zs - binned_sorted
    return BinnedResidualDecomposition(
        smooth_z=smooth_z,
        binned_z=binned_z,
        local_z=local_z,
        bin_time=bin_time,
        bin_z=bin_z,
        smooth_bin_z=smooth_bin_z,
        cadence=cadence,
        bin_width=bin_width,
        trend_half_width=trend_half_width,
    )


__all__ = ["BinnedResidualDecomposition", "decompose_binned_residual"]
