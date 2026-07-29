"""Contiguous anomaly segmentation and contamination-aware refitting."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ContaminationConfig:
    """Controls for the two-state dynamic-programming segmentation."""

    baseline_scale: float = 1.0
    anomaly_scale: float = 6.0
    student_dof: float = 4.0
    transition_penalty: float = 1.5
    occupancy_penalty: float = 0.75
    span_penalty: float = 0.10
    season_gap: float = 100.0
    max_anomaly_fraction: float = 0.50
    max_anomaly_span_fraction: float = 0.60
    protected_anomaly_penalty: float = 2.0
    max_protected_anomaly_fraction: float = 0.35
    anomaly_weight: float = 0.05
    min_weight: float = 0.02
    weight_damping: float = 0.5
    max_iter: int = 8
    min_weight_change: float = 1.0e-3
    min_parameter_change: float = 2.0e-3


@dataclass(frozen=True)
class ContaminationSegmentation:
    """MAP segmentation and diagnostics for one residual vector."""

    state: np.ndarray
    anomaly_probability: np.ndarray
    blocks: tuple[tuple[int, int], ...]
    objective: float
    anomaly_fraction: float
    anomaly_span_fraction: float
    protected_fraction: float
    protected_anomaly_fraction: float = 0.0
    protected_components: tuple[str, ...] = ()
    protected_component_anomaly_fractions: tuple[float, ...] = ()
    protected_component_retained_fractions: tuple[float, ...] = ()
    diagnostics: tuple[str, ...] = ()
    contamination_penalty: float = 0.0
    effective_occupancy_penalty: float = 0.0

    @property
    def anomaly_mask(self) -> np.ndarray:
        return self.state.astype(bool, copy=True)


@dataclass(frozen=True)
class RobustFitIteration:
    """One segmentation/refit iteration."""

    iteration: int
    fit: object
    segmentation: ContaminationSegmentation
    weights: np.ndarray
    parameter_change: float
    weight_change: float


@dataclass(frozen=True)
class RobustFitResult:
    """Result of :func:`robust_refine_with_fitter`."""

    fit: object
    initial_fit: object
    final_weights: np.ndarray
    segmentation: ContaminationSegmentation
    iterations: tuple[RobustFitIteration, ...]
    converged: bool
    segmentation_stable: bool = False


def _student_nll(z: np.ndarray, scale: float, dof: float) -> np.ndarray:
    scale = max(float(scale), 1.0e-8)
    nu = max(float(dof), 1.0e-3)
    x = np.asarray(z, dtype=float) / scale
    return 0.5 * (nu + 1.0) * np.log1p((x * x) / nu) + np.log(scale)


def _blocks_from_state(time: np.ndarray, state: np.ndarray, season_gap: float) -> tuple[tuple[int, int], ...]:
    t = np.asarray(time, dtype=float)
    active = np.asarray(state, dtype=bool)
    blocks: list[tuple[int, int]] = []
    if t.size != active.size:
        raise ValueError("time and state must have the same length.")
    season_start = np.r_[False, np.diff(t) > float(season_gap)]
    starts = np.flatnonzero(active & (~np.r_[False, active[:-1]] | season_start))
    for start in starts:
        end = int(start)
        while end + 1 < active.size and active[end + 1]:
            if float(t[end + 1] - t[end]) > float(season_gap):
                break
            end += 1
        blocks.append((int(start), end))
    return tuple(blocks)


def _span_fraction(time: np.ndarray, blocks: tuple[tuple[int, int], ...]) -> float:
    if time.size < 2 or not blocks:
        return 0.0
    total = max(float(np.nanmax(time) - np.nanmin(time)), 1.0e-12)
    span = sum(max(float(time[end] - time[start]), 0.0) for start, end in blocks)
    return float(span / total)


def _run_dp(
    time: np.ndarray,
    z: np.ndarray,
    protected: np.ndarray,
    cfg: ContaminationConfig,
    occupancy_penalty: float,
    protected_penalty: float,
    forced_anomaly: np.ndarray,
) -> tuple[np.ndarray, float]:
    n = z.size
    emission_b = _student_nll(z, cfg.baseline_scale, cfg.student_dof)
    emission_a = _student_nll(z, cfg.anomaly_scale, cfg.student_dof) + float(occupancy_penalty)
    emission_a = emission_a + np.asarray(protected, dtype=float) * float(protected_penalty)
    # Planet blocks detected before the physical fallback are excluded from
    # the single-lens fit by forcing the DP anomaly state. A large finite
    # emission keeps every objective numerical and comparable.
    emission_b = np.where(np.asarray(forced_anomaly, dtype=bool), 1.0e12, emission_b)
    dp = np.full((n, 2), np.inf, dtype=float)
    parent = np.zeros((n, 2), dtype=np.int8)
    dp[0] = [emission_b[0], emission_a[0]]
    for i in range(1, n):
        gap = float(time[i] - time[i - 1])
        transition = 0.0 if gap > float(cfg.season_gap) else float(cfg.transition_penalty)
        previous = dp[i - 1]
        for state in (0, 1):
            same = previous[state]
            switch = previous[1 - state] + transition
            if same <= switch:
                dp[i, state] = same
                parent[i, state] = state
            else:
                dp[i, state] = switch
                parent[i, state] = 1 - state
            span_cost = 0.0
            if state == 1 and i > 0 and gap <= float(cfg.season_gap):
                span_cost = float(cfg.span_penalty) * max(gap, 0.0)
            dp[i, state] += (emission_b[i] if state == 0 else emission_a[i]) + span_cost
    if not np.isfinite(np.min(dp[-1])):
        return np.zeros(n, dtype=np.int8), float("inf")
    state = np.zeros(n, dtype=np.int8)
    state[-1] = int(np.argmin(dp[-1]))
    for i in range(n - 1, 0, -1):
        state[i - 1] = parent[i, state[i]]
    return state, float(dp[-1, state[-1]])


def contamination_objective(
    time: np.ndarray,
    standardized_residual: np.ndarray,
    state: np.ndarray,
    *,
    config: ContaminationConfig = ContaminationConfig(),
    occupancy_penalty: Optional[float] = None,
    protected_mask: Optional[np.ndarray] = None,
    protected_masks: Optional[Sequence[np.ndarray]] = None,
    protected_penalty: Optional[float] = None,
) -> float:
    """Evaluate the common contamination likelihood on original errors."""
    t = np.asarray(time, dtype=float)
    z = np.asarray(standardized_residual, dtype=float)
    s = np.asarray(state, dtype=bool)
    occupancy = float(config.occupancy_penalty if occupancy_penalty is None else occupancy_penalty)
    support_strength = _protected_strength(
        t.size,
        protected_mask=protected_mask,
        protected_masks=protected_masks,
    )
    support_penalty = float(
        config.protected_anomaly_penalty if protected_penalty is None else protected_penalty
    )
    value = float(np.sum(np.where(s, _student_nll(z, config.anomaly_scale, config.student_dof) + occupancy, _student_nll(z, config.baseline_scale, config.student_dof))))
    value += float(np.sum(s.astype(float) * support_strength * support_penalty))
    if s.size > 1:
        gaps = np.diff(t)
        connected = gaps <= float(config.season_gap)
        transitions = connected & (s[1:] != s[:-1])
        anomaly_spans = connected & s[1:] & s[:-1]
        value += float(config.transition_penalty) * float(
            np.count_nonzero(transitions)
        )
        value += float(config.span_penalty) * float(
            np.sum(np.maximum(gaps[anomaly_spans], 0.0))
        )
    return float(value)


def segment_anomaly_dp(
    time: np.ndarray,
    standardized_residual: np.ndarray,
    *,
    config: ContaminationConfig = ContaminationConfig(),
    protected_mask: Optional[np.ndarray] = None,
    protected_masks: Optional[Sequence[np.ndarray]] = None,
    forced_anomaly_mask: Optional[np.ndarray] = None,
) -> ContaminationSegmentation:
    """Segment residuals into baseline (0) and anomaly (1) states.

    The transition cost makes anomalies contiguous, while the occupancy and
    span guards stop a broad physical residual from being discarded as one
    enormous planet block.  Season-sized gaps reset the transition cost.
    """
    t = np.asarray(time, dtype=float).reshape(-1)
    z = np.asarray(standardized_residual, dtype=float).reshape(-1)
    if t.size != z.size:
        raise ValueError("time and standardized_residual must have the same length.")
    if t.size == 0:
        return ContaminationSegmentation(
            state=np.zeros(0, dtype=np.int8),
            anomaly_probability=np.zeros(0, dtype=float),
            blocks=(),
            objective=0.0,
            anomaly_fraction=0.0,
            anomaly_span_fraction=0.0,
            protected_fraction=0.0,
            protected_anomaly_fraction=0.0,
            contamination_penalty=0.0,
            effective_occupancy_penalty=float(config.occupancy_penalty),
        )
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(z)):
        raise ValueError("time and standardized_residual must be finite.")
    forced_anomaly = (
        np.zeros(t.size, dtype=bool)
        if forced_anomaly_mask is None
        else np.asarray(forced_anomaly_mask, dtype=bool).reshape(-1)
    )
    if forced_anomaly.size != t.size:
        raise ValueError("forced_anomaly_mask must have the same length as time.")
    protected_components = _protected_components(
        t.size,
        protected_mask=protected_mask,
        protected_masks=protected_masks,
    )
    protected_strength = (
        np.sum(np.asarray(protected_components, dtype=float), axis=0)
        if protected_components
        else np.zeros(t.size, dtype=float)
    )
    protected = protected_strength > 0.0

    # Increase the occupancy cost when the first solution violates the broad
    # residual guard.  This keeps the DP itself simple and gives the result a
    # useful, explicit diagnostic instead of silently accepting a global mask.
    occupancy = float(config.occupancy_penalty)
    protected_penalty = float(config.protected_anomaly_penalty)
    chosen_state = np.zeros(t.size, dtype=np.int8)
    objective = float("inf")
    diagnostics: list[str] = []
    for _ in range(8):
        state, objective = _run_dp(
            t,
            z,
            protected_strength,
            config,
            occupancy,
            protected_penalty,
            forced_anomaly,
        )
        blocks = _blocks_from_state(t, state, config.season_gap)
        fraction = float(np.mean(state))
        span_fraction = _span_fraction(t, blocks)
        protected_anomaly_fraction = float(np.mean(state[protected])) if np.any(protected) else 0.0
        component_anomaly_fractions = tuple(
            float(np.mean(state[component > 0.0]))
            for component in protected_components
            if np.any(component > 0.0)
        )
        worst_component_fraction = max(component_anomaly_fractions, default=0.0)
        chosen_state = state
        if (
            fraction <= float(config.max_anomaly_fraction)
            and span_fraction <= float(config.max_anomaly_span_fraction)
            and protected_anomaly_fraction <= float(config.max_protected_anomaly_fraction)
            and worst_component_fraction <= float(config.max_protected_anomaly_fraction)
        ):
            break
        if (
            protected_anomaly_fraction > float(config.max_protected_anomaly_fraction)
            or worst_component_fraction > float(config.max_protected_anomaly_fraction)
        ):
            protected_penalty = max(protected_penalty * 1.8, protected_penalty + 0.5)
            diagnostics.append("protected_support_guard_increased_penalty")
        else:
            occupancy = max(occupancy * 1.8, occupancy + 0.5)
            diagnostics.append("broad_anomaly_guard_increased_penalty")
    blocks = _blocks_from_state(t, chosen_state, config.season_gap)
    fraction = float(np.mean(chosen_state))
    span_fraction = _span_fraction(t, blocks)
    protected_anomaly_fraction = float(np.mean(chosen_state[protected])) if np.any(protected) else 0.0
    component_anomaly_fractions = tuple(
        float(np.mean(chosen_state[component > 0.0]))
        for component in protected_components
        if np.any(component > 0.0)
    )
    component_retained_fractions = tuple(1.0 - value for value in component_anomaly_fractions)
    if fraction > float(config.max_anomaly_fraction) or span_fraction > float(config.max_anomaly_span_fraction):
        chosen_state = np.zeros(t.size, dtype=np.int8)
        blocks = ()
        fraction = 0.0
        span_fraction = 0.0
        protected_anomaly_fraction = 0.0
        diagnostics.append("broad_residual_protected")
    elif diagnostics and fraction == 0.0:
        diagnostics.append("broad_residual_protected")
    elif protected_anomaly_fraction > float(config.max_protected_anomaly_fraction):
        # Keep the DP solution visible for diagnostics. Protected support is
        # discouraged by a finite emission penalty; it is never made
        # impossible with an ``inf`` emission or post-hoc hard masking.
        diagnostics.append("protected_support_penalty_active")

    # MAP is the primary output.  The local emission contrast is a stable,
    # inexpensive soft diagnostic for callers that want posterior-like weights.
    eb = _student_nll(z, config.baseline_scale, config.student_dof)
    ea = _student_nll(z, config.anomaly_scale, config.student_dof) + occupancy
    margin = np.clip(eb - ea, -60.0, 60.0)
    probability = 1.0 / (1.0 + np.exp(-margin))
    probability = np.where(protected, 0.0, probability)
    probability = np.where(forced_anomaly, 1.0, probability)
    probability = np.where(chosen_state.astype(bool), np.maximum(probability, 0.5), np.minimum(probability, 0.5))
    penalty = contamination_objective(
        t,
        z,
        chosen_state,
        config=config,
        # Adaptive penalties are constraint-search mechanics. Attempt ranking
        # must use one canonical objective shared by every seed.
        occupancy_penalty=config.occupancy_penalty,
        protected_masks=protected_components,
        protected_penalty=config.protected_anomaly_penalty,
    )
    emissions = np.where(
        chosen_state.astype(bool),
        _student_nll(z, config.anomaly_scale, config.student_dof),
        _student_nll(z, config.baseline_scale, config.student_dof),
    )
    regularization = max(float(penalty - np.sum(emissions)), 0.0)
    return ContaminationSegmentation(
        state=chosen_state,
        anomaly_probability=probability,
        blocks=blocks,
        objective=float(penalty),
        anomaly_fraction=fraction,
        anomaly_span_fraction=span_fraction,
        protected_fraction=float(np.mean(protected)),
        protected_anomaly_fraction=protected_anomaly_fraction,
        protected_components=tuple(
            f"support_{index}" for index in range(len(protected_components))
        ),
        protected_component_anomaly_fractions=component_anomaly_fractions,
        protected_component_retained_fractions=component_retained_fractions,
        diagnostics=tuple(dict.fromkeys(diagnostics)),
        contamination_penalty=regularization,
        effective_occupancy_penalty=float(occupancy),
    )


def segmentation_weights(
    segmentation: ContaminationSegmentation,
    *,
    anomaly_weight: float = 0.05,
    min_weight: float = 0.02,
) -> np.ndarray:
    """Convert a segmentation into inverse-variance weights."""
    p = np.asarray(segmentation.anomaly_probability, dtype=float)
    weight = 1.0 - p * (1.0 - float(anomaly_weight))
    weight = np.where(segmentation.state.astype(bool), float(anomaly_weight), weight)
    return np.clip(weight, float(min_weight), 1.0)


def protected_support_mask(
    time: np.ndarray,
    effect: str,
    seed_parameters: Optional[np.ndarray],
    *,
    fspl_support_sigma: float = 3.0,
    parallax_peak_exclusion: float = 3.0,
) -> np.ndarray:
    """Return points that must remain available to the physical model.

    FSPL protects both sides of the finite-source peak.  Parallax protects the
    wings and only leaves a compact peak region eligible for contamination.
    Mixed effects are normally passed to the segmenter as separate support
    components.  This helper remains a single-effect primitive.
    """
    t = np.asarray(time, dtype=float).reshape(-1)
    p = None if seed_parameters is None else np.asarray(seed_parameters, dtype=float).reshape(-1)
    if p is None or p.size < 3:
        return np.ones(t.size, dtype=bool)
    t0, tE, u0 = float(p[0]), abs(float(p[1])), abs(float(p[2]))
    if tE <= 0.0:
        return np.ones(t.size, dtype=bool)
    effect_name = str(effect).lower()
    fspl_mask = np.zeros(t.size, dtype=bool)
    if "fspl" in effect_name and p.size >= 4:
        raw_rho = float(p[3])
        rho = np.exp(raw_rho) if raw_rho < 0.0 else raw_rho
        t_star = max(rho * tE, 1.0e-6)
        width = max(float(fspl_support_sigma) * t_star, 0.25 * u0 * tE, 1.0e-6)
        fspl_mask = np.abs(t - t0) <= width
    parallax_mask = np.zeros(t.size, dtype=bool)
    if "parallax" in effect_name:
        width = max(float(parallax_peak_exclusion) * max(u0 * tE, 1.0e-6), 1.0e-6)
        parallax_mask = np.abs(t - t0) > width
    if "mixed" in effect_name or (fspl_mask.any() and parallax_mask.any()):
        return fspl_mask | parallax_mask
    if fspl_mask.any():
        return fspl_mask
    if parallax_mask.any():
        return parallax_mask
    return np.ones(t.size, dtype=bool)


def _protected_components(
    size: int,
    *,
    protected_mask: Optional[np.ndarray] = None,
    protected_masks: Optional[Sequence[np.ndarray]] = None,
) -> tuple[np.ndarray, ...]:
    """Validate and preserve independent protected-support components."""
    masks: list[np.ndarray] = []
    if protected_mask is not None:
        masks.append(np.asarray(protected_mask, dtype=float).reshape(-1))
    if protected_masks is not None:
        masks.extend(np.asarray(mask, dtype=float).reshape(-1) for mask in protected_masks)
    for mask in masks:
        if mask.size != size:
            raise ValueError("Every protected mask must have the same length as time.")
        if not np.all(np.isfinite(mask)) or np.any(mask < 0.0):
            raise ValueError("Protected masks must be finite and non-negative.")
    if not masks:
        return ()
    return tuple(masks)


def _protected_strength(
    size: int,
    *,
    protected_mask: Optional[np.ndarray] = None,
    protected_masks: Optional[Sequence[np.ndarray]] = None,
) -> np.ndarray:
    """Convert independent protected supports into an additive penalty field."""
    components = _protected_components(
        size,
        protected_mask=protected_mask,
        protected_masks=protected_masks,
    )
    if not components:
        return np.zeros(size, dtype=float)
    return np.sum(np.asarray(components, dtype=float), axis=0)


def _parameter_vector(fit: object) -> np.ndarray:
    """Return the one canonical fallback seed representation.

    The contract is physical ``tE`` and physical parallax components, with
    only ``rho`` represented logarithmically.  Optimizer-private ``log_tE``
    values must never leak into the next robust-refit call.
    """
    value = np.asarray(getattr(fit, "params"), dtype=float).reshape(-1).copy()
    names = tuple(getattr(fit, "param_names", ()))
    if "rho" in names:
        index = names.index("rho")
        value[index] = np.log(max(abs(float(value[index])), 1.0e-12))
    return value


def scaled_parameter_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return a dimensionless distance for raw single-lens parameters.

    The raw contracts are ``(t0, tE, u0[, logrho][, piEN, piEE])``.  A plain
    Euclidean norm is dominated by the absolute time origin and is therefore
    not a useful basin-stability diagnostic.
    """
    a = np.asarray(first, dtype=float).reshape(-1)
    b = np.asarray(second, dtype=float).reshape(-1)
    if a.size != b.size or a.size < 3:
        raise ValueError("Scaled parameter distance requires equal vectors with at least three entries.")
    tE_a = max(abs(float(a[1])), 1.0e-8)
    tE_b = max(abs(float(b[1])), 1.0e-8)
    u0_scale = max(abs(float(a[2])), abs(float(b[2])), 1.0e-3)
    components = [
        (float(a[0]) - float(b[0])) / max(tE_a, tE_b),
        np.log(tE_a / tE_b),
        (float(a[2]) - float(b[2])) / u0_scale,
    ]
    if a.size == 4:
        components.append(float(a[3]) - float(b[3]))
    elif a.size == 5:
        components.extend([float(a[3]) - float(b[3]), float(a[4]) - float(b[4])])
    elif a.size >= 6:
        components.extend(
            [
                float(a[3]) - float(b[3]),
                float(a[4]) - float(b[4]),
                float(a[5]) - float(b[5]),
            ]
        )
        components.extend((a[6:] - b[6:]).tolist())
    return float(np.linalg.norm(np.asarray(components, dtype=float)) / np.sqrt(float(len(components))))


def _canonicalize_fit(fit: object, ferr: np.ndarray) -> object:
    """Keep optimizer internals weighted but expose original-error fit metrics."""
    if not hasattr(fit, "residual") or not hasattr(fit, "model_flux") or not hasattr(fit, "chi2"):
        return fit
    residual = np.asarray(fit.residual, dtype=float)
    original_ferr = np.maximum(np.asarray(ferr, dtype=float), 1.0e-12)
    chi2 = float(np.sum((residual / original_ferr) ** 2))
    if not hasattr(fit, "__dataclass_fields__"):
        return fit
    import jax.numpy as jnp

    return replace(
        fit,
        ferr=original_ferr,
        chi2=jnp.asarray(chi2),
        chi2_dof=jnp.asarray(chi2 / max(original_ferr.size - len(tuple(fit.param_names)), 1)),
    )


def robust_refine_with_fitter(
    fitter,
    time,
    flux,
    ferr,
    x0,
    *,
    config: ContaminationConfig = ContaminationConfig(),
    protected_mask: Optional[np.ndarray] = None,
    protected_masks: Optional[Sequence[np.ndarray]] = None,
    initial_standardized_residual: Optional[np.ndarray] = None,
    initial_anomaly_mask: Optional[np.ndarray] = None,
    forced_anomaly_mask: Optional[np.ndarray] = None,
) -> RobustFitResult:
    """Alternate DP segmentation and an existing single-lens fitter.

    The fitter is intentionally supplied by the caller, so the fallback works
    with the existing PSPL, FSPL, and parallax classes without replacing their
    low-level model implementations.  Anomaly points are represented by
    inflated errors during the next fit; no fixed PSPL planet mask is used.
    """
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    fe = np.maximum(np.asarray(ferr, dtype=float), 1.0e-12)
    current_x0 = np.asarray(x0, dtype=float)
    forced_mask = (
        np.zeros(t.size, dtype=bool)
        if forced_anomaly_mask is None
        else np.asarray(forced_anomaly_mask, dtype=bool).reshape(-1)
    )
    if forced_mask.size != t.size:
        raise ValueError("forced_anomaly_mask must match time.")
    if initial_anomaly_mask is not None:
        initial_mask = np.asarray(initial_anomaly_mask, dtype=bool).reshape(-1)
        if initial_mask.size != t.size:
            raise ValueError("initial_anomaly_mask must match time.")
        initial_mask |= forced_mask
        weights = np.where(initial_mask, float(config.anomaly_weight), 1.0)
        weights = np.clip(weights, float(config.min_weight), 1.0)
    elif initial_standardized_residual is None:
        weights = np.where(forced_mask, float(config.anomaly_weight), 1.0)
    else:
        initial_z = np.asarray(initial_standardized_residual, dtype=float).reshape(-1)
        if initial_z.size != t.size or not np.all(np.isfinite(initial_z)):
            raise ValueError(
                "initial_standardized_residual must be finite and match time."
            )
        initial_segmentation = segment_anomaly_dp(
            t,
            initial_z,
            config=config,
            protected_mask=protected_mask,
            protected_masks=protected_masks,
            forced_anomaly_mask=forced_anomaly_mask,
        )
        weights = segmentation_weights(
            initial_segmentation,
            anomaly_weight=config.anomaly_weight,
            min_weight=config.min_weight,
        )
    initial_fit = _canonicalize_fit(
        fitter.fit(t, f, fe / np.sqrt(weights), current_x0),
        fe,
    )
    current_fit = initial_fit
    iterations: list[RobustFitIteration] = []
    converged = False
    segmentation = segment_anomaly_dp(
        t,
        np.asarray(current_fit.residual, dtype=float) / fe,
        config=config,
        protected_mask=protected_mask,
        protected_masks=protected_masks,
        forced_anomaly_mask=forced_anomaly_mask,
    )
    for iteration in range(max(1, int(config.max_iter))):
        z = np.asarray(current_fit.residual, dtype=float) / fe
        segmentation = segment_anomaly_dp(
            t,
            z,
            config=config,
            protected_mask=protected_mask,
            protected_masks=protected_masks,
            forced_anomaly_mask=forced_anomaly_mask,
        )
        raw_next_weights = segmentation_weights(
            segmentation,
            anomaly_weight=config.anomaly_weight,
            min_weight=config.min_weight,
        )
        damping = float(np.clip(config.weight_damping, 0.0, 1.0))
        next_weights = weights + damping * (raw_next_weights - weights)
        next_x0 = _parameter_vector(current_fit)
        next_fit = _canonicalize_fit(
            fitter.fit(t, f, fe / np.sqrt(next_weights), next_x0),
            fe,
        )
        old_params = _parameter_vector(current_fit)
        new_params = _parameter_vector(next_fit)
        parameter_change = scaled_parameter_distance(old_params, new_params)
        # A contiguous block boundary commonly jitters by one or two cadence
        # points after an otherwise stable refit.  A max norm interprets that
        # harmless local change as a full ``1 -> anomaly_weight`` restart and
        # can therefore never converge on long, densely sampled light curves.
        # The mean absolute change measures the fraction of support that
        # actually moved while still detecting broad mask changes.
        weight_change = float(np.mean(np.abs(next_weights - weights)))
        iterations.append(
            RobustFitIteration(
                iteration=iteration,
                fit=next_fit,
                segmentation=segmentation,
                weights=next_weights.copy(),
                parameter_change=parameter_change,
                weight_change=weight_change,
            )
        )
        current_fit = next_fit
        weights = next_weights
        if (
            weight_change <= float(config.min_weight_change)
            and parameter_change <= float(config.min_parameter_change)
        ):
            converged = True
            break
    # The returned objective and state must be evaluated at the returned fit,
    # not at the pre-refit residual used to produce the last optimizer weights.
    final_segmentation = segment_anomaly_dp(
        t,
        np.asarray(current_fit.residual, dtype=float) / fe,
        config=config,
        protected_mask=protected_mask,
        protected_masks=protected_masks,
        forced_anomaly_mask=forced_anomaly_mask,
    )
    final_weights = segmentation_weights(
        final_segmentation,
        anomaly_weight=config.anomaly_weight,
        min_weight=config.min_weight,
    )
    final_state_change = float(
        np.mean(final_segmentation.state != segmentation.state)
    )
    final_weight_change = float(np.mean(np.abs(final_weights - weights)))
    segmentation_stable = bool(
        final_state_change <= 0.01
        and final_weight_change <= float(config.min_weight_change)
    )
    converged = bool(converged and segmentation_stable)
    return RobustFitResult(
        fit=current_fit,
        initial_fit=initial_fit,
        final_weights=final_weights,
        segmentation=final_segmentation,
        iterations=tuple(iterations),
        converged=converged,
        segmentation_stable=segmentation_stable,
    )


__all__ = [
    "ContaminationConfig",
    "ContaminationSegmentation",
    "RobustFitIteration",
    "RobustFitResult",
    "protected_support_mask",
    "scaled_parameter_distance",
    "contamination_objective",
    "robust_refine_with_fitter",
    "segment_anomaly_dp",
    "segmentation_weights",
]
