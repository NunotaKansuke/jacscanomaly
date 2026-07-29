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
    hard_geometry_failure = (
        candidate.coverage < thresholds.min_coverage
        or not np.isfinite(candidate.condition_number)
        or candidate.condition_number > thresholds.max_condition_number
    )

    compact_dominated = "compact_block_dominated" in candidate.reason_codes
    robust_score = min(score, float(candidate.score_without_compact_blocks))
    if score < thresholds.exact_probe_score and robust_score < thresholds.exact_probe_score:
        decision = "skip"
        reasons.append("score_below_exact_probe")
    elif compact_dominated and candidate.score_without_compact_blocks < thresholds.fallback_score:
        decision = "exact_probe"
        reasons.append("compact_block_only")
        if not thresholds.exact_probe_available:
            reasons.append("exact_probe_unavailable")
    elif robust_score >= thresholds.fallback_score and not quality_ok and not thresholds.exact_probe_available:
        # Fail open for a strong physical signal while the exact-probe executor
        # is unavailable.  Raw subset score differences alone must not create
        # a dead end; compact-only candidates were handled above.
        decision = "fallback"
        reasons.append("fallback_after_probe_unavailable")
    elif score < thresholds.fallback_score or not quality_ok:
        decision = "exact_probe"
        reasons.append("boundary_score" if score < thresholds.fallback_score else "diagnostic_uncertainty")
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
