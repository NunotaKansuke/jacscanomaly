"""Routing policy for physical residual detector outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from .effect_detection import EffectCandidate


@dataclass(frozen=True)
class RoutingThresholds:
    """Conservative defaults for the three-stage detector router.

    Scores are approximate ``Delta chi2`` values.  They are deliberately
    configuration values rather than constants embedded in the detector, so a
    simulation-derived calibration can replace them without changing the
    physical score implementation.
    """

    exact_probe_score: float = 9.0
    fallback_score: float = 25.0
    min_coverage: float = 0.20
    max_condition_number: float = 25.0
    max_point_influence: float = 0.50
    max_block_influence: float = 0.75
    min_subset_stability: float = 0.25
    min_planet_mask_retention: float = 0.50
    max_planet_overlap: float = 0.35
    min_parallax_support_effective_blocks: float = 2.0
    """Minimum effective broad-residual blocks for parallax support."""
    min_parallax_support_evidence: float = 2.0
    """Minimum block-level null-excess significance for automatic fallback."""
    min_parallax_support_span_days: float = 7.0
    """Minimum non-local residual span for automatic parallax fallback."""
    min_parallax_fallback_tE: float = 20.0
    """Legacy fallback when a manually constructed candidate has no scale."""
    duration_probe_score: float = 0.999
    """Morphology score denoting an unconditional short/long exact probe."""
    residual_probe_score: float = 0.50
    """Residual-shape score needed to probe an intermediate-duration event."""
    coherent_parallax_rescue_score: float = 4.0
    allow_rank_deficient_exact_probe: bool = True
    exact_probe_available: bool = False


def route_candidate(
    candidate: EffectCandidate,
    thresholds: RoutingThresholds = RoutingThresholds(),
) -> EffectCandidate:
    """Attach ``skip``, ``exact_probe`` or ``fallback`` to one candidate.

    Routing is intentionally monotonic: a physical candidate with a high score
    can still be demoted to ``exact_probe`` when its geometry is poorly
    conditioned or its score is concentrated in one compact block.
    """
    reasons = list(candidate.reason_codes)
    score = float(candidate.score)
    quality_ok = (
        candidate.coverage >= thresholds.min_coverage
        and np.isfinite(candidate.condition_number)
        and candidate.condition_number <= thresholds.max_condition_number
        and candidate.max_point_influence <= thresholds.max_point_influence
        and candidate.max_block_influence <= thresholds.max_block_influence
        and candidate.subset_stability >= thresholds.min_subset_stability
    )
    if candidate.effect == "fspl" and candidate.morphology in {
        "fspl_even_peak",
        "fspl_flattened_peak",
    }:
        # FSPL is intrinsically compact and its PSPL nuisance basis is often
        # ill-conditioned near high magnification. Its discriminating quality
        # test is the even central morphology, not parallax-style wing
        # conditioning.
        quality_ok = (
            candidate.coverage >= 0.02
            and candidate.max_point_influence <= thresholds.max_point_influence
            and candidate.subset_stability >= thresholds.min_subset_stability
        )
    hard_geometry_failure = (
        candidate.coverage < thresholds.min_coverage
        or not np.isfinite(candidate.condition_number)
        or candidate.condition_number > thresholds.max_condition_number
    )

    compact_dominated = "compact_block_dominated" in candidate.reason_codes
    score_without_planet = (
        float(candidate.score_without_planet)
        if np.isfinite(candidate.score_without_planet)
        else float(candidate.score_without_compact_blocks)
    )
    robust_score = min(
        score,
        float(candidate.score_without_compact_blocks),
        score_without_planet,
    )
    clear_physical_morphology = candidate.morphology in {
        "fspl_even_peak",
        "fspl_flattened_peak",
        "parallax_coherent_wings",
    }
    ambiguous_mixed = candidate.morphology in {
        "mixed_or_planet",
        "fspl_partial_peak",
    }
    planet_mask_conflict = bool(
        "planet_morphology_dominated" in candidate.reason_codes
        or candidate.planet_overlap > thresholds.max_planet_overlap
        or (
            score > 0.0
            and score_without_planet / score
            < thresholds.min_planet_mask_retention
        )
    )
    parallax_planet_conflict = bool(
        "parallax" in candidate.effect and planet_mask_conflict
    )
    planet_dominated = (
        candidate.morphology == "planet_like"
        or (
            not clear_physical_morphology
            and not ambiguous_mixed
            and planet_mask_conflict
        )
    )
    parallax_support = candidate.parallax_support
    parallax_support_available = bool(
        "parallax" in candidate.effect
        and parallax_support is not None
        and parallax_support.valid
    )
    parallax_support_ok = bool(
        parallax_support_available
        and np.isfinite(parallax_support.effective_blocks)
        and parallax_support.effective_blocks
        >= thresholds.min_parallax_support_effective_blocks
        and np.isfinite(parallax_support.evidence)
        and parallax_support.evidence >= thresholds.min_parallax_support_evidence
        and np.isfinite(parallax_support.span)
        and parallax_support.span >= thresholds.min_parallax_support_span_days
    )
    morphology_uncertain = (
        "non_fspl_peak_shape" in candidate.reason_codes
        or "parallax_wings_incoherent" in candidate.reason_codes
        or "parallax_too_local" in candidate.reason_codes
        or "parallax_support_unresolved" in candidate.reason_codes
        or (
            "parallax" not in candidate.effect
            and "signal_scale_censored" in candidate.reason_codes
        )
        or (
            "parallax" not in candidate.effect
            and "signal_scale_unmeasurable" in candidate.reason_codes
        )
        or candidate.morphology == "mixed_or_planet"
        or candidate.morphology == "fspl_partial_peak"
        or parallax_planet_conflict
    )
    parallax_tE = (
        abs(float(np.asarray(candidate.seed_parameters).reshape(-1)[1]))
        if (
            "parallax" in candidate.effect
            and candidate.seed_parameters is not None
            and np.asarray(candidate.seed_parameters).size >= 2
        )
        else float("nan")
    )
    short_parallax_event = bool(
        (
            parallax_support is not None
            and not parallax_support_ok
        )
        or (
            parallax_support is None
            and np.isfinite(parallax_tE)
            and parallax_tE < thresholds.min_parallax_fallback_tE
        )
    )
    coherent_parallax_rescue = bool(
        "parallax" in candidate.effect
        and candidate.morphology == "parallax_coherent_wings"
        and (
            parallax_support_ok
            or (
                parallax_support is None
                and np.isfinite(parallax_tE)
                and parallax_tE >= thresholds.min_parallax_fallback_tE
            )
        )
        and score >= thresholds.coherent_parallax_rescue_score
    )
    morphology_diagnostics = candidate.morphology_diagnostics or {}
    if candidate.effect == "fspl":
        duration_probe_score = float(
            morphology_diagnostics.get("fspl_short_score", 0.0)
        )
        residual_probe_score = float(
            morphology_diagnostics.get("fspl_residual_score", 0.0)
        )
        duration_probe_reason = "short_duration_fspl_exact_probe"
        residual_probe_reason = "fspl_residual_shape_exact_probe"
    elif "parallax" in candidate.effect:
        duration_probe_score = float(
            morphology_diagnostics.get("parallax_long_score", 0.0)
        )
        residual_probe_score = float(
            morphology_diagnostics.get("parallax_residual_score", 0.0)
        )
        duration_probe_reason = "long_duration_parallax_exact_probe"
        residual_probe_reason = "parallax_residual_shape_exact_probe"
    else:
        duration_probe_score = 0.0
        residual_probe_score = 0.0
        duration_probe_reason = "duration_exact_probe"
        residual_probe_reason = "residual_shape_exact_probe"
    duration_probe = bool(
        np.isfinite(duration_probe_score)
        and duration_probe_score >= thresholds.duration_probe_score
    )
    residual_probe = bool(
        np.isfinite(residual_probe_score)
        and residual_probe_score >= thresholds.residual_probe_score
    )
    if planet_dominated:
        decision = "planet"
        reasons.append("routed_as_planet_anomaly")
    elif duration_probe or residual_probe:
        # Duration is a deliberately broad recall gate: run a bounded exact
        # comparison, but do not promote the candidate to the expensive
        # robust fallback until the physical score and diagnostics agree.
        decision = "exact_probe"
        reasons.append(
            duration_probe_reason if duration_probe else residual_probe_reason
        )
        if not thresholds.exact_probe_available:
            reasons.append("exact_probe_unavailable")
    elif (
        score < thresholds.exact_probe_score
        and robust_score < thresholds.exact_probe_score
        and not coherent_parallax_rescue
    ):
        decision = "skip"
        reasons.append("score_below_exact_probe")
    elif coherent_parallax_rescue and score < thresholds.exact_probe_score:
        decision = "exact_probe"
        reasons.append("coherent_long_parallax_low_score_rescue")
        if not thresholds.exact_probe_available:
            reasons.append("exact_probe_unavailable")
    elif (
        compact_dominated
        and not clear_physical_morphology
        and candidate.score_without_compact_blocks < thresholds.fallback_score
    ):
        decision = "exact_probe"
        reasons.append("compact_block_only")
        if not thresholds.exact_probe_available:
            reasons.append("exact_probe_unavailable")
    elif (
        morphology_uncertain
        or short_parallax_event
        or score < thresholds.fallback_score
        or not quality_ok
    ):
        decision = "exact_probe"
        reasons.append(
            "parallax_planet_conflict_requires_model_comparison"
            if parallax_planet_conflict
            else "morphology_requires_model_comparison"
            if morphology_uncertain
            else "short_event_parallax_requires_model_comparison"
            if short_parallax_event
            else "boundary_score"
            if score < thresholds.fallback_score
            else "diagnostic_uncertainty"
        )
        if hard_geometry_failure:
            reasons.append("geometry_requires_exact_probe")
        if not thresholds.exact_probe_available:
            reasons.append("exact_probe_unavailable")
    else:
        decision = "fallback"
        reasons.append("strong_physical_score")

    if candidate.max_point_influence > thresholds.max_point_influence:
        reasons.append("point_influence_high")
    if candidate.max_block_influence > thresholds.max_block_influence:
        reasons.append("block_influence_high")
    if candidate.subset_stability < thresholds.min_subset_stability:
        reasons.append("subset_stability_low")
    if candidate.effective_rank <= 0:
        reasons.append("no_effective_rank")
        if decision == "fallback":
            decision = "exact_probe" if thresholds.allow_rank_deficient_exact_probe else "skip"

    return candidate.with_decision(decision, reasons)


def route_candidates(
    candidates: Iterable[EffectCandidate],
    thresholds: RoutingThresholds = RoutingThresholds(),
) -> tuple[EffectCandidate, ...]:
    """Route all candidates, preserving detector order."""
    return tuple(route_candidate(candidate, thresholds) for candidate in candidates)


def select_fallback_candidates(
    candidates: Sequence[EffectCandidate],
    *,
    thresholds: RoutingThresholds = RoutingThresholds(),
    max_candidates: Optional[int] = None,
) -> tuple[EffectCandidate, ...]:
    """Route and rank candidates that should enter the expensive fallback."""
    routed = [route_candidate(candidate, thresholds) for candidate in candidates]
    selected = [candidate for candidate in routed if candidate.decision == "fallback"]
    selected.sort(key=lambda candidate: candidate.score, reverse=True)
    if max_candidates is not None:
        selected = selected[: max(0, int(max_candidates))]
    return tuple(selected)


def routing_pareto_curve(
    candidates: Sequence[EffectCandidate],
    labels: Sequence[bool],
    thresholds: Sequence[float],
) -> tuple[dict[str, float], ...]:
    """Evaluate score-only fallback recall versus fallback rate.

    This small calibration helper keeps the planned threshold selection
    reproducible.  ``labels`` should represent a counterfactual detectable
    physical effect, not merely whether the initial fit happened to converge.
    """
    scores = np.asarray([float(candidate.score) for candidate in candidates])
    truth = np.asarray(labels, dtype=bool)
    if scores.size != truth.size:
        raise ValueError("candidates and labels must have the same length.")
    n_truth = max(int(np.count_nonzero(truth)), 1)
    n_null = max(int(np.count_nonzero(~truth)), 1)
    rows = []
    for threshold in thresholds:
        routed = scores >= float(threshold)
        rows.append(
            {
                "threshold": float(threshold),
                "recall": float(np.count_nonzero(routed & truth) / n_truth),
                "fallback_rate": float(np.count_nonzero(routed & ~truth) / n_null),
                "wasted_fit_rate": float(
                    np.count_nonzero(routed & ~truth) / max(int(np.count_nonzero(routed)), 1)
                ),
            }
        )
    return tuple(rows)


__all__ = [
    "RoutingThresholds",
    "route_candidate",
    "route_candidates",
    "routing_pareto_curve",
    "select_fallback_candidates",
]
