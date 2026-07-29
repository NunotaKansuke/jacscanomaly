"""Explicit before/after planet pipeline around physical-effect fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PlanetCandidateMatch:
    """Provenance link between a before and after extracted candidate."""

    before: object | None
    after: object | None
    category: str
    interval_iou: float
    peak_time_difference: float
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class EffectAwareFinderResult:
    """Result of :meth:`Finder.run_effect_aware`."""

    initial_fit: object
    selected_fit: object
    effect_candidates: tuple[object, ...]
    routing_decision: object
    fallback_result: object | None
    planet_before: object | None
    planet_after: object | None
    candidate_matches: tuple[PlanetCandidateMatch, ...]
    final_candidates: tuple[object, ...]
    reason_codes: tuple[str, ...]
    diagnostics: dict[str, object]


def _candidate_interval(candidate) -> tuple[float, float]:
    return float(candidate.t_start), float(candidate.t_end)


def _iou(first, second) -> float:
    a0, a1 = _candidate_interval(first)
    b0, b1 = _candidate_interval(second)
    intersection = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return float(intersection / union) if union > 0.0 else 0.0


def _same_season(t0: float, t1: float, season_gap: float) -> bool:
    # A full season id is not stored by PlanetSignalCandidate.  This local
    # guard captures the intended matching rule without copying light curves.
    return abs(float(t1) - float(t0)) <= float(season_gap)


def match_planet_candidates(before, after, *, season_gap: float, iou_threshold: float = 0.25, cadence_factor: float = 3.0) -> tuple[PlanetCandidateMatch, ...]:
    before_rows = tuple(() if before is None else before.candidates)
    after_rows = tuple(() if after is None else after.candidates)
    used_after: set[int] = set()
    matches: list[PlanetCandidateMatch] = []
    for old in before_rows:
        options = []
        for index, new in enumerate(after_rows):
            if index in used_after or not _same_season(old.peak_time, new.peak_time, season_gap):
                continue
            overlap = _iou(old, new)
            peak_delta = abs(float(old.peak_time) - float(new.peak_time))
            cadence = max(
                (float(old.t_end) - float(old.t_start)) / max(int(old.n_points), 1),
                (float(new.t_end) - float(new.t_start)) / max(int(new.n_points), 1),
                np.finfo(float).eps,
            )
            if overlap >= iou_threshold or peak_delta <= cadence_factor * cadence:
                options.append((overlap, -peak_delta, index, new, peak_delta))
        if options:
            overlap, _, index, new, peak_delta = max(options)
            used_after.add(index)
            changed = abs(float(old.chi2) - float(new.chi2)) > max(0.5 * abs(float(old.chi2)), 25.0)
            matches.append(PlanetCandidateMatch(old, new, "changed" if changed else "survived", overlap, peak_delta, ("matched_by_interval_or_peak",)))
        else:
            matches.append(PlanetCandidateMatch(old, None, "explained_by_single_lens_effect", 0.0, float("inf"), ("missing_after_fallback",)))
    for index, new in enumerate(after_rows):
        if index not in used_after:
            matches.append(PlanetCandidateMatch(None, new, "revealed_after_fallback", 0.0, float("inf"), ("new_after_fallback",)))
    return tuple(matches)


__all__ = ["EffectAwareFinderResult", "PlanetCandidateMatch", "match_planet_candidates"]
