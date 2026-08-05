"""Shared event-width and observed-signal scale utilities.

The fitted ``|tE*u0|`` is the normal event-width coordinate.  Observed
profile scales remain available as a guarded fallback for poorly constrained
high-magnification fits and for anomaly morphology.  Both representations are
kept here so routing, masking, feature reporting, and plots can share the same
edge/censoring semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class ObservedSignalScale:
    """Robust width measured from a weighted residual/profile signal.

    ``t_left`` and ``t_right`` delimit the central weighted 80 percent of the
    signal.  ``half_width`` is half of that interval and is the common base
    scale for downstream windows.  ``censored`` means that the data do not
    close the interval on one or both sides; such a scale is a lower bound and
    should not be used for an automatic physical classification.
    """

    t_center: float
    t_left: float
    t_right: float
    half_width: float
    cadence: float
    n_points: int
    n_weighted_points: int
    left_coverage: float
    right_coverage: float
    asymmetry: float
    valid: bool = True
    censored: bool = False
    source: str = "observed_residual"

    @classmethod
    def invalid(cls, *, source: str = "observed_residual") -> "ObservedSignalScale":
        """Return a stable empty value for an unmeasurable signal."""
        nan = float("nan")
        return cls(
            t_center=nan,
            t_left=nan,
            t_right=nan,
            half_width=nan,
            cadence=nan,
            n_points=0,
            n_weighted_points=0,
            left_coverage=0.0,
            right_coverage=0.0,
            asymmetry=nan,
            valid=False,
            censored=True,
            source=str(source),
        )

    @property
    def width(self) -> float:
        """Full central signal width."""
        return float(2.0 * self.half_width)

    @property
    def points_per_width(self) -> float:
        """Number of cadence intervals across the measured width."""
        if not np.isfinite(self.cadence) or self.cadence <= 0.0:
            return float("nan")
        return float(self.width / self.cadence)

    def bounds(self, *, padding: float = 0.0) -> tuple[float, float]:
        """Return the measured interval with symmetric scale padding."""
        pad = max(float(padding), 0.0) * max(float(self.half_width), 0.0)
        return float(self.t_left - pad), float(self.t_right + pad)

    def summary_dict(self) -> dict[str, object]:
        """Return JSON-safe diagnostics."""
        return {
            "t_center": float(self.t_center),
            "t_left": float(self.t_left),
            "t_right": float(self.t_right),
            "width": float(self.width),
            "half_width": float(self.half_width),
            "cadence": float(self.cadence),
            "points_per_width": float(self.points_per_width),
            "n_points": int(self.n_points),
            "n_weighted_points": int(self.n_weighted_points),
            "left_coverage": float(self.left_coverage),
            "right_coverage": float(self.right_coverage),
            "asymmetry": float(self.asymmetry),
            "valid": bool(self.valid),
            "censored": bool(self.censored),
            "source": str(self.source),
        }


@dataclass(frozen=True)
class ResolvedSignalScale:
    """One event-width decision shared by plots and event-level windows.

    The primary scale is the model's ``|tE*u0|``.  It is deliberately kept
    separate from :class:`ObservedSignalScale`: the latter is evidence used
    only when the fitted scale is not trustworthy.  ``left_half_width`` and
    ``right_half_width`` preserve one-sided information for censored data
    instead of silently inventing a symmetric width.
    """

    center: float
    left_half_width: float
    right_half_width: float
    source: str
    valid: bool = True
    censored: bool = False
    tE: float = float("nan")
    u0: float = float("nan")
    t_eff: float = float("nan")
    fallback_reasons: tuple[str, ...] = ()

    @classmethod
    def invalid(cls, *, center: float = float("nan"), reason: str = "invalid"):
        return cls(
            center=float(center),
            left_half_width=float("nan"),
            right_half_width=float("nan"),
            source="invalid",
            valid=False,
            censored=True,
            fallback_reasons=(str(reason),),
        )

    @property
    def half_width(self) -> float:
        """Symmetric half-width needed to contain both observed sides."""
        return float(max(self.left_half_width, self.right_half_width))

    @property
    def width(self) -> float:
        """Full asymmetric width between the two resolved bounds."""
        return float(self.left_half_width + self.right_half_width)

    def bounds(
        self,
        *,
        padding: float = 0.0,
        symmetric: bool = False,
    ) -> tuple[float, float]:
        """Return bounds, optionally preserving a censored one-sided interval.

        ``padding`` is a fraction of the resolved symmetric half-width.  A
        symmetric interval is useful for the main plot; an asymmetric one is
        safer when a signal reaches the edge of the observed data.
        """
        if not self.valid or not np.all(
            np.isfinite(
                [self.center, self.left_half_width, self.right_half_width]
            )
        ):
            return float("nan"), float("nan")
        pad = max(float(padding), 0.0) * max(self.half_width, 0.0)
        if symmetric:
            half = self.half_width + pad
            return float(self.center - half), float(self.center + half)
        return (
            float(self.center - self.left_half_width - pad),
            float(self.center + self.right_half_width + pad),
        )

    def summary_dict(self) -> dict[str, object]:
        """Return JSON-safe diagnostics for the selected width source."""
        return {
            "center": float(self.center),
            "left_half_width": float(self.left_half_width),
            "right_half_width": float(self.right_half_width),
            "half_width": float(self.half_width),
            "width": float(self.width),
            "source": str(self.source),
            "valid": bool(self.valid),
            "censored": bool(self.censored),
            "tE": float(self.tE),
            "u0": float(self.u0),
            "t_eff": float(self.t_eff),
            "fallback_reasons": list(self.fallback_reasons),
        }


def _fit_scale_parameters(fit_or_params: Any) -> tuple[Any, np.ndarray]:
    fit = fit_or_params if hasattr(fit_or_params, "params") else None
    params = np.asarray(
        getattr(fit_or_params, "params", fit_or_params),
        dtype=float,
    ).reshape(-1)
    if params.size < 3:
        raise ValueError("A signal-scale fit must contain (t0, tE, u0).")
    return fit, params


def _observed_fallback_extents(
    center: float,
    observed_scale: Optional[ObservedSignalScale],
    time: Optional[np.ndarray],
    *,
    minimum: float,
) -> tuple[float, float, bool, str]:
    if observed_scale is not None and observed_scale.valid:
        left = float(center) - float(observed_scale.t_left)
        right = float(observed_scale.t_right) - float(center)
        if left >= 0.0 and right >= 0.0 and np.isfinite(left + right):
            left = max(left, minimum) if left > 0.0 else 0.0
            right = max(right, minimum) if right > 0.0 else 0.0
            return (
                left,
                right,
                bool(observed_scale.censored),
                "observed_censored"
                if observed_scale.censored
                else "observed_data",
            )
        # The residual interval does not contain the requested model centre.
        # Use its own symmetric width rather than creating a negative side.
        width = max(float(observed_scale.half_width), minimum)
        return width, width, bool(observed_scale.censored), "observed_data"

    if time is not None:
        tv = np.asarray(time, dtype=float).reshape(-1)
        tv = tv[np.isfinite(tv)]
        if tv.size:
            left = max(float(center) - float(np.min(tv)), 0.0)
            right = max(float(np.max(tv)) - float(center), 0.0)
            if left > 0.0 or right > 0.0:
                if left > 0.0:
                    left = max(left, minimum)
                if right > 0.0:
                    right = max(right, minimum)
                return left, right, True, "data_support_censored"

    return float("nan"), float("nan"), True, "unavailable"


def resolve_event_signal_scale(
    fit_or_params: Any,
    *,
    observed_scale: Optional[ObservedSignalScale] = None,
    time: Optional[np.ndarray] = None,
    center: Optional[float] = None,
    minimum_half_width: float = 0.0,
    max_tE_to_data_span: float = 8.0,
    condition_limit: float = 1.0e10,
) -> ResolvedSignalScale:
    """Resolve an event width with ``|tE*u0|`` as the normal path.

    A very large ``tE`` relative to the available time support is treated as
    unresolved.  This is the high-magnification failure mode where the fit
    can trade an enormous ``tE`` against a tiny ``u0``.  If the fit reports a
    bound, optimizer failure, or an ill-conditioned native Jacobian, the same
    fallback is used.  The fallback preserves one-sided observed support when
    the data are censored at an edge.
    """
    fit, params = _fit_scale_parameters(fit_or_params)
    t0 = float(params[0] if center is None else center)
    tE = abs(float(params[1]))
    u0 = abs(float(params[2]))
    t_eff = tE * u0
    reasons: list[str] = []

    if not np.isfinite(t0):
        reasons.append("nonfinite_t0")
    if not np.isfinite(tE) or tE <= 0.0:
        reasons.append("nonfinite_tE")
    if not np.isfinite(u0) or u0 <= 0.0:
        reasons.append("nonfinite_u0")
    if not np.isfinite(t_eff) or t_eff <= 0.0:
        reasons.append("nonfinite_tE_u0")

    if fit is not None:
        optimizer_success = getattr(fit, "optimizer_success", None)
        if optimizer_success is False:
            reasons.append("optimizer_failed")
        diagnostics = getattr(fit, "diagnostics", None)
        if bool(getattr(diagnostics, "parameter_at_bound", False)):
            reasons.append("parameter_at_bound")
        condition = float(getattr(diagnostics, "jacobian_condition", np.nan))
        if np.isfinite(condition) and condition > float(condition_limit):
            reasons.append("ill_conditioned_fit")

    if time is not None and np.isfinite(tE) and tE > 0.0:
        tv = np.asarray(time, dtype=float).reshape(-1)
        tv = tv[np.isfinite(tv)]
        if tv.size >= 2:
            span = float(np.ptp(tv))
            if (
                span > 0.0
                and np.isfinite(max_tE_to_data_span)
                and max_tE_to_data_span > 0.0
                and tE > float(max_tE_to_data_span) * span
            ):
                reasons.append("tE_unresolved_vs_data_span")

    minimum = max(float(minimum_half_width), 0.0)
    if not reasons:
        width = max(float(t_eff), minimum)
        return ResolvedSignalScale(
            center=t0,
            left_half_width=width,
            right_half_width=width,
            source="model_tE_u0",
            tE=tE,
            u0=u0,
            t_eff=t_eff,
        )

    left, right, censored, source = _observed_fallback_extents(
        t0,
        observed_scale,
        time,
        minimum=minimum,
    )
    if not np.isfinite(left + right):
        return ResolvedSignalScale.invalid(center=t0, reason=";".join(reasons))
    return ResolvedSignalScale(
        center=t0,
        left_half_width=float(left),
        right_half_width=float(right),
        source=source,
        censored=bool(censored),
        tE=tE,
        u0=u0,
        t_eff=t_eff,
        fallback_reasons=tuple(reasons),
    )


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    x = np.asarray(values, dtype=float)[order]
    w = np.asarray(weights, dtype=float)[order]
    cumulative = np.cumsum(w)
    total = float(cumulative[-1])
    if total <= 0.0 or not np.isfinite(total):
        return float("nan")
    return float(np.interp(float(q) * total, cumulative, x))


def measure_observed_signal_scale(
    time: np.ndarray,
    signal: np.ndarray,
    *,
    valid_mask: Optional[np.ndarray] = None,
    threshold: float = 3.0,
    low_quantile: float = 0.10,
    high_quantile: float = 0.90,
    min_points: int = 4,
    edge_cadences: float = 1.0,
    source: str = "observed_residual",
) -> ObservedSignalScale:
    """Measure a robust central interval from a residual/profile signal.

    The weight is based on the excess absolute standardized signal above a
    noise floor.  This makes the result insensitive to the fitted ``u0`` and
    avoids defining a width from a single threshold crossing.  If significant
    weight reaches a data edge, ``censored`` is set and the result is treated
    as a lower bound by routing code.
    """
    t = np.asarray(time, dtype=float).reshape(-1)
    y = np.asarray(signal, dtype=float).reshape(-1)
    if t.size != y.size:
        raise ValueError("time and signal must have the same length.")
    if valid_mask is None:
        valid = np.ones(t.size, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if valid.size != t.size:
            raise ValueError("valid_mask must have the same length as time.")
    valid &= np.isfinite(t) & np.isfinite(y)
    if np.count_nonzero(valid) < max(int(min_points), 1):
        return ObservedSignalScale.invalid(source=source)

    tv = t[valid]
    yv = np.abs(y[valid])
    order = np.argsort(tv)
    tv = tv[order]
    yv = yv[order]
    steps = np.diff(tv)
    positive_steps = steps[np.isfinite(steps) & (steps > 0.0)]
    cadence = float(np.median(positive_steps)) if positive_steps.size else float("nan")

    excess = np.maximum(yv - max(float(threshold), 0.0), 0.0)
    weights = excess * excess
    weighted = weights > 0.0
    if np.count_nonzero(weighted) < max(int(min_points), 1):
        return ObservedSignalScale.invalid(source=source)

    # Prevent one pathological point from becoming the entire time scale.
    positive_weights = weights[weighted]
    cap = float(np.nanpercentile(positive_weights, 95.0))
    if np.isfinite(cap) and cap > 0.0:
        weights = np.minimum(weights, cap)

    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        return ObservedSignalScale.invalid(source=source)
    t_center = _weighted_quantile(tv, weights, 0.50)
    t_left = _weighted_quantile(tv, weights, float(low_quantile))
    t_right = _weighted_quantile(tv, weights, float(high_quantile))
    if not np.all(np.isfinite([t_center, t_left, t_right])) or t_right <= t_left:
        return ObservedSignalScale.invalid(source=source)

    half_width = 0.5 * (t_right - t_left)
    left_span = max(t_center - t_left, 0.0)
    right_span = max(t_right - t_center, 0.0)
    asymmetry = (
        float(right_span / left_span)
        if left_span > 0.0
        else float("inf")
    )
    edge = max(float(edge_cadences), 0.0) * (
        cadence if np.isfinite(cadence) and cadence > 0.0 else 0.0
    )
    data_left, data_right = float(tv[0]), float(tv[-1])
    weighted_tv = tv[weighted]
    touches_left = bool(weighted_tv.size and weighted_tv[0] <= data_left + edge)
    touches_right = bool(weighted_tv.size and weighted_tv[-1] >= data_right - edge)
    censored = touches_left or touches_right

    return ObservedSignalScale(
        t_center=float(t_center),
        t_left=float(t_left),
        t_right=float(t_right),
        half_width=float(max(half_width, 0.0)),
        cadence=float(cadence),
        n_points=int(tv.size),
        n_weighted_points=int(np.count_nonzero(weighted)),
        left_coverage=float(0.0 if touches_left else 1.0),
        right_coverage=float(0.0 if touches_right else 1.0),
        asymmetry=float(asymmetry),
        valid=True,
        censored=bool(censored),
        source=str(source),
    )


def measure_observed_magnification_scale(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    center: float,
    valid_mask: Optional[np.ndarray] = None,
    search_half_width: float = 300.0,
    n_bins: int = 256,
    baseline_quantile: float = 0.20,
    threshold_fraction: float = 0.05,
    source: str = "observed_magnification",
) -> ObservedSignalScale:
    """Measure the observed duration of the broad magnification envelope.

    This is deliberately separate from :func:`measure_observed_signal_scale`,
    which measures the compact planet residual.  The light curve is binned by
    time and represented by per-bin medians, so a short planetary excursion
    cannot set the width of the broad event.  The returned interval is the
    contiguous above-baseline envelope around ``center`` and does not use
    ``tE`` or ``tE*u0``.
    """
    t = np.asarray(time, dtype=float).reshape(-1)
    y = np.asarray(flux, dtype=float).reshape(-1)
    if t.size != y.size or t.size == 0 or not np.isfinite(center):
        return ObservedSignalScale.invalid(source=source)
    if valid_mask is None:
        valid = np.ones(t.size, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if valid.size != t.size:
            raise ValueError("valid_mask must have the same length as time.")
    valid &= np.isfinite(t) & np.isfinite(y)
    half_search = max(float(search_half_width), 1.0)
    valid &= np.abs(t - float(center)) <= half_search
    if np.count_nonzero(valid) < 8:
        return ObservedSignalScale.invalid(source=source)

    tv = t[valid]
    yv = y[valid]
    order = np.argsort(tv)
    tv = tv[order]
    yv = yv[order]
    data_left, data_right = float(tv[0]), float(tv[-1])
    if data_right <= data_left:
        return ObservedSignalScale.invalid(source=source)

    bins = max(16, min(int(n_bins), int(tv.size)))
    edges = np.linspace(data_left, data_right, bins + 1)
    bin_index = np.searchsorted(edges, tv, side="right") - 1
    bin_index = np.clip(bin_index, 0, bins - 1)
    centres: list[float] = []
    profiles: list[float] = []
    for index in range(bins):
        selected = bin_index == index
        if np.any(selected):
            centres.append(float(np.median(tv[selected])))
            profiles.append(float(np.median(yv[selected])))
    if len(centres) < 8:
        return ObservedSignalScale.invalid(source=source)

    profile_t = np.asarray(centres, dtype=float)
    profile_y = np.asarray(profiles, dtype=float)
    baseline = float(np.nanpercentile(profile_y, float(baseline_quantile) * 100.0))
    excess = np.maximum(profile_y - baseline, 0.0)
    peak_excess = float(np.max(excess)) if excess.size else 0.0
    if not np.isfinite(peak_excess) or peak_excess <= 0.0:
        return ObservedSignalScale.invalid(source=source)

    robust_noise = 1.4826 * float(
        np.nanmedian(np.abs(profile_y - np.nanmedian(profile_y)))
    )
    threshold = max(float(threshold_fraction) * peak_excess, 3.0 * robust_noise)
    above = excess >= threshold
    if not np.any(above):
        return ObservedSignalScale.invalid(source=source)

    # Use the broad above-baseline envelope.  The plot is centred on the
    # requested physical t0 later; this interval only supplies the observed
    # duration and its asymmetry.
    above_indices = np.flatnonzero(above)
    left_index = int(above_indices[0])
    right_index = int(above_indices[-1])
    t_left = float(profile_t[left_index])
    t_right = float(profile_t[right_index])
    if t_right <= t_left:
        return ObservedSignalScale.invalid(source=source)

    weights = np.maximum(excess - threshold, 0.0) ** 2
    if np.sum(weights) > 0.0:
        t_center = _weighted_quantile(profile_t, weights, 0.50)
    else:
        t_center = 0.5 * (t_left + t_right)
    cadence_values = np.diff(profile_t)
    positive_cadence = cadence_values[cadence_values > 0.0]
    cadence = float(np.median(positive_cadence)) if positive_cadence.size else float("nan")
    edge = cadence if np.isfinite(cadence) else 0.0
    touches_left = t_left <= data_left + edge
    touches_right = t_right >= data_right - edge
    left_span = max(t_center - t_left, 0.0)
    right_span = max(t_right - t_center, 0.0)
    return ObservedSignalScale(
        t_center=float(t_center),
        t_left=t_left,
        t_right=t_right,
        half_width=0.5 * (t_right - t_left),
        cadence=cadence,
        n_points=int(tv.size),
        n_weighted_points=int(np.count_nonzero(weights > 0.0)),
        left_coverage=0.0 if touches_left else 1.0,
        right_coverage=0.0 if touches_right else 1.0,
        asymmetry=float(right_span / left_span) if left_span > 0.0 else float("inf"),
        valid=True,
        censored=bool(touches_left or touches_right),
        source=str(source),
    )


def interval_mask(
    time: np.ndarray,
    scale: ObservedSignalScale,
    *,
    padding: float = 0.0,
) -> np.ndarray:
    """Return points inside a measured scale interval."""
    t = np.asarray(time, dtype=float)
    if not scale.valid or not np.isfinite(scale.t_center) or not np.isfinite(scale.half_width):
        return np.zeros(t.shape, dtype=bool)
    left, right = scale.bounds(padding=padding)
    return (t >= left) & (t <= right)


__all__ = [
    "ObservedSignalScale",
    "ResolvedSignalScale",
    "interval_mask",
    "measure_observed_magnification_scale",
    "measure_observed_signal_scale",
    "resolve_event_signal_scale",
]
