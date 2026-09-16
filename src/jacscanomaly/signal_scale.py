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


def _contiguous_runs(
    mask: np.ndarray,
    time: np.ndarray,
    *,
    max_gap: float,
) -> tuple[np.ndarray, ...]:
    """Return above-threshold runs without joining separated observations."""
    active = np.flatnonzero(np.asarray(mask, dtype=bool))
    if active.size == 0:
        return ()
    gaps = np.diff(np.asarray(time, dtype=float)[active])
    breaks = np.flatnonzero(
        ~np.isfinite(gaps) | (gaps > max(float(max_gap), 0.0))
    ) + 1
    return tuple(np.split(active, breaks))


def _run_containing(
    runs: tuple[np.ndarray, ...],
    index: int,
) -> Optional[np.ndarray]:
    """Return the run containing one array index, if it exists."""
    for run in runs:
        if run.size and int(run[0]) <= int(index) <= int(run[-1]):
            return run
    return None


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
    allow_point_fallback: bool = False,
    source: str = "observed_magnification",
) -> ObservedSignalScale:
    """Measure the observed duration of the broad magnification envelope.

    This is deliberately separate from :func:`measure_observed_signal_scale`,
    which measures the compact planet residual.  The light curve is binned by
    time and represented by per-bin medians, so a short planetary excursion
    cannot set the width of the broad event.  The returned interval is the
    above-baseline component containing the requested event.  If
    ``allow_point_fallback`` is true, a component unresolved by the broad
    profile falls back to a point-level run around the nearby observed peak;
    isolated excursions elsewhere in the search window cannot define the
    width.  This does not use ``tE`` or ``tE*u0``.
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

    cadence_values = np.diff(profile_t)
    positive_cadence = cadence_values[cadence_values > 0.0]
    cadence = (
        float(np.median(positive_cadence))
        if positive_cadence.size
        else float("nan")
    )
    nominal_bin_width = max(
        (data_right - data_left) / max(float(bins), 1.0),
        np.finfo(float).eps,
    )
    runs = _contiguous_runs(
        above,
        profile_t,
        max_gap=3.0 * nominal_bin_width,
    )

    # Anchor the profile on the event centre.  If the profile has washed out
    # a sharp peak, try the highest observed point as the secondary anchor;
    # never use the first and last above-threshold bins from unrelated times.
    center_index = int(np.argmin(np.abs(profile_t - float(center))))
    selected_run = _run_containing(runs, center_index)
    if selected_run is None or selected_run.size < 2:
        peak_index = int(np.nanargmax(yv))
        peak_time = float(tv[peak_index])
        peak_anchor_tolerance = max(
            4.0 * nominal_bin_width,
            0.25 * half_search,
        )
        if abs(peak_time - float(center)) <= peak_anchor_tolerance:
            peak_profile_index = int(np.argmin(np.abs(profile_t - peak_time)))
            candidate_run = _run_containing(runs, peak_profile_index)
            if candidate_run is not None and candidate_run.size >= 2:
                selected_run = candidate_run

    if selected_run is not None and selected_run.size >= 2:
        selected = np.zeros(profile_t.size, dtype=bool)
        selected[selected_run] = True
        weights = np.maximum(excess - threshold, 0.0) ** 2
        weights[~selected] = 0.0
        t_left = float(profile_t[selected_run[0]])
        t_right = float(profile_t[selected_run[-1]])
        if t_right > t_left:
            t_center = (
                _weighted_quantile(profile_t, weights, 0.50)
                if np.sum(weights) > 0.0
                else 0.5 * (t_left + t_right)
            )
            edge = nominal_bin_width
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
                asymmetry=(
                    float(right_span / left_span)
                    if left_span > 0.0
                    else float("inf")
                ),
                valid=True,
                censored=bool(touches_left or touches_right),
                source=str(source),
            )

    if not allow_point_fallback:
        return ObservedSignalScale.invalid(source=source)

    # The broad median profile intentionally suppresses a very short event.
    # Recover such an event from the consecutive raw observations around its
    # peak, while using the local median/MAD as a noise floor.  The run
    # requirement prevents one noisy point from becoming an event width.
    raw_cadence_values = np.diff(tv)
    raw_positive_cadence = raw_cadence_values[raw_cadence_values > 0.0]
    raw_cadence = (
        float(np.median(raw_positive_cadence))
        if raw_positive_cadence.size
        else float("nan")
    )
    raw_baseline = float(np.nanmedian(yv))
    raw_noise = 1.4826 * float(np.nanmedian(np.abs(yv - raw_baseline)))
    raw_excess = np.maximum(yv - raw_baseline, 0.0)
    raw_peak_excess = float(np.max(raw_excess)) if raw_excess.size else 0.0
    raw_threshold = max(
        float(threshold_fraction) * raw_peak_excess,
        3.0 * raw_noise,
    )
    raw_above = yv - raw_baseline >= raw_threshold
    raw_runs = _contiguous_runs(
        raw_above,
        tv,
        max_gap=(3.0 * raw_cadence if np.isfinite(raw_cadence) else 0.0),
    )
    raw_peak_index = int(np.nanargmax(yv))
    if abs(float(tv[raw_peak_index]) - float(center)) > peak_anchor_tolerance:
        return ObservedSignalScale.invalid(source=source)
    raw_run = _run_containing(raw_runs, raw_peak_index)
    if raw_run is None or raw_run.size < 3:
        return ObservedSignalScale.invalid(source=source)

    raw_weights = np.maximum(yv - raw_baseline - raw_threshold, 0.0) ** 2
    raw_weights[~np.isin(np.arange(tv.size), raw_run)] = 0.0
    t_left = float(tv[raw_run[0]])
    t_right = float(tv[raw_run[-1]])
    if t_right <= t_left:
        return ObservedSignalScale.invalid(source=source)
    t_center = _weighted_quantile(tv, raw_weights, 0.50)
    edge = raw_cadence if np.isfinite(raw_cadence) else 0.0
    touches_left = t_left <= data_left + edge
    touches_right = t_right >= data_right - edge
    left_span = max(t_center - t_left, 0.0)
    right_span = max(t_right - t_center, 0.0)
    return ObservedSignalScale(
        t_center=float(t_center),
        t_left=t_left,
        t_right=t_right,
        half_width=0.5 * (t_right - t_left),
        cadence=raw_cadence,
        n_points=int(tv.size),
        n_weighted_points=int(np.count_nonzero(raw_weights > 0.0)),
        left_coverage=0.0 if touches_left else 1.0,
        right_coverage=0.0 if touches_right else 1.0,
        asymmetry=(
            float(right_span / left_span)
            if left_span > 0.0
            else float("inf")
        ),
        valid=True,
        censored=bool(touches_left or touches_right),
        source=str(source),
    )


def measure_observed_magnification_fwhm(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    center: float,
    valid_mask: Optional[np.ndarray] = None,
    search_half_width: float = 300.0,
    baseline_quantile: float = 0.20,
    peak_points: int = 5,
    min_points: int = 3,
    max_gap_cadences: float = 3.0,
    min_peak_snr: float = 3.0,
    source: str = "observed_magnification_fwhm",
) -> ObservedSignalScale:
    """Measure a direct, observed FWHM without time binning.

    The baseline is a lower flux quantile in the local search window.  The
    peak is the median of the brightest few points, which avoids letting one
    outlier set the half-maximum level.  The width is measured at half of the
    *excess* above baseline, not at half of the absolute flux.  Only a
    contiguous run around the requested centre is accepted; an isolated
    excursion elsewhere in the window cannot define the event width.

    A run must contain at least ``min_points`` observations.  If the direct
    FWHM cannot be resolved, an invalid scale is returned so the caller can
    make an explicit fallback decision rather than silently treating a model
    width as an observed measurement.
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
    if np.count_nonzero(valid) < max(int(min_points) + 2, 8):
        return ObservedSignalScale.invalid(source=source)

    tv = t[valid]
    yv = y[valid]
    order = np.argsort(tv, kind="stable")
    tv = tv[order]
    yv = yv[order]
    data_left, data_right = float(tv[0]), float(tv[-1])
    if data_right <= data_left:
        return ObservedSignalScale.invalid(source=source)

    cadence_values = np.diff(tv)
    positive_cadence = cadence_values[
        np.isfinite(cadence_values) & (cadence_values > 0.0)
    ]
    cadence = (
        float(np.median(positive_cadence))
        if positive_cadence.size
        else float("nan")
    )
    if not np.isfinite(cadence) or cadence <= 0.0:
        return ObservedSignalScale.invalid(source=source)

    baseline = float(
        np.nanpercentile(yv, float(baseline_quantile) * 100.0)
    )
    noise = 1.4826 * float(np.nanmedian(np.abs(yv - np.nanmedian(yv))))
    # Estimate the peak locally around the PSPL centre.  Using the brightest
    # points in the whole search window would let a stronger, unrelated
    # excursion determine the central event's half-maximum level.
    peak_anchor_half_width = min(
        half_search,
        max(4.0 * cadence, 0.10 * half_search),
    )
    peak_region = np.abs(tv - float(center)) <= peak_anchor_half_width
    peak_values = yv[peak_region]
    if peak_values.size < 1:
        return ObservedSignalScale.invalid(source=source)
    n_peak = max(1, min(int(peak_points), peak_values.size))
    brightest = np.partition(peak_values, peak_values.size - n_peak)[-n_peak:]
    peak = float(np.nanmedian(brightest))
    peak_excess = peak - baseline
    if not np.isfinite(peak_excess) or peak_excess <= 0.0:
        return ObservedSignalScale.invalid(source=source)
    if np.isfinite(noise) and noise > 0.0:
        if peak_excess < max(float(min_peak_snr), 0.0) * noise:
            return ObservedSignalScale.invalid(source=source)

    half_level = baseline + 0.5 * peak_excess
    above = yv >= half_level
    runs = _contiguous_runs(
        above,
        tv,
        max_gap=max(float(max_gap_cadences), 0.0) * cadence,
    )
    min_run_points = max(int(min_points), 2)
    runs = tuple(run for run in runs if run.size >= min_run_points)
    if not runs:
        return ObservedSignalScale.invalid(source=source)

    center_index = int(np.argmin(np.abs(tv - float(center))))
    selected_run = _run_containing(runs, center_index)
    if selected_run is None:
        anchor_tolerance = max(4.0 * cadence, 0.25 * half_search)
        eligible = [
            run
            for run in runs
            if abs(
                float(tv[int(run[np.argmax(yv[run])])]) - float(center)
            )
            <= anchor_tolerance
        ]
        if not eligible:
            return ObservedSignalScale.invalid(source=source)
        selected_run = max(
            eligible,
            key=lambda run: float(np.max(yv[run])),
        )

    def crossing(left_index: int, right_index: int) -> float:
        left_time = float(tv[left_index])
        right_time = float(tv[right_index])
        left_flux = float(yv[left_index])
        right_flux = float(yv[right_index])
        denominator = right_flux - left_flux
        if not np.isfinite(denominator) or denominator == 0.0:
            return 0.5 * (left_time + right_time)
        fraction = (half_level - left_flux) / denominator
        fraction = float(np.clip(fraction, 0.0, 1.0))
        return left_time + fraction * (right_time - left_time)

    first = int(selected_run[0])
    last = int(selected_run[-1])
    touches_left = first == 0
    touches_right = last == tv.size - 1
    t_left = float(tv[first]) if touches_left else crossing(first - 1, first)
    t_right = float(tv[last]) if touches_right else crossing(last, last + 1)
    if not np.isfinite(t_left + t_right) or t_right <= t_left:
        return ObservedSignalScale.invalid(source=source)

    t_center = 0.5 * (t_left + t_right)
    left_span = max(t_center - t_left, 0.0)
    right_span = max(t_right - t_center, 0.0)
    edge = cadence
    return ObservedSignalScale(
        t_center=float(t_center),
        t_left=float(t_left),
        t_right=float(t_right),
        half_width=0.5 * (t_right - t_left),
        cadence=float(cadence),
        n_points=int(tv.size),
        n_weighted_points=int(selected_run.size),
        left_coverage=0.0 if touches_left else 1.0,
        right_coverage=0.0 if touches_right else 1.0,
        asymmetry=(
            float(right_span / left_span) if left_span > 0.0 else float("inf")
        ),
        valid=True,
        censored=bool(
            touches_left or touches_right
            or t_left <= data_left + edge
            or t_right >= data_right - edge
        ),
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
    "measure_observed_magnification_fwhm",
    "measure_observed_magnification_scale",
    "measure_observed_signal_scale",
    "resolve_event_signal_scale",
]
