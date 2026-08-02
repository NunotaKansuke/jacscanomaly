"""Public configuration and result types for the complete anomaly pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .effect_routing import RoutingThresholds
from .planet_signal import (
    PlanetFeatureConfig,
    PlanetFeatureResult,
    PlanetSignalConfig,
    PlanetSignalResult,
)
from .singlelens_fallback import FallbackConfig
from .template_free import TemplateFreeSearchConfig, TemplateFreeSearchResult


@dataclass(frozen=True)
class AnomalyPipelineConfig:
    """Configuration for :meth:`Finder.run_anomaly_pipeline`.

    The final residual pass is always frozen by the orchestrator.  Its mask is
    a measurement window and is deliberately kept separate from points that
    were actually excluded by an accepted continuation fit.
    """

    planet: PlanetSignalConfig = field(default_factory=PlanetSignalConfig)
    final_measurement: PlanetSignalConfig = field(
        default_factory=PlanetSignalConfig.residual_measurement
    )
    features: PlanetFeatureConfig = field(default_factory=PlanetFeatureConfig)
    template_free: TemplateFreeSearchConfig | None = None
    routing_thresholds: RoutingThresholds | None = None
    fallback: FallbackConfig = field(default_factory=FallbackConfig)
    planet_fast_mode: bool = True
    post_physical_max_refits: int = 3
    post_refinement_max_chi2_ratio: float = 1.25
    post_refinement_max_chi2_delta: float = 0.5
    candidate_merge_cadence_factor: float = 3.0


@dataclass(frozen=True)
class AnomalyCandidate:
    """One unified adopted-residual anomaly candidate.

    Peak/dip measurements provide the preferred timing and sign.  A matching
    template-free window supplements those measurements with chi-square and
    point-count statistics.  ``sources`` preserves that provenance.
    """

    rank: int
    kind: str
    t_center: float
    t_start: float
    t_end: float
    timescale: float
    max_abs_z: float
    signed_z: float | None
    fractional_deviation: float | None
    chi2: float | None
    reduced_chi2: float | None
    n_points: int | None
    sources: tuple[str, ...]
    fit_excluded: bool
    adopted_model: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready dictionary for notebooks and reports."""

        return {
            "rank": int(self.rank),
            "kind": self.kind,
            "t_center": float(self.t_center),
            "t_start": float(self.t_start),
            "t_end": float(self.t_end),
            "timescale": float(self.timescale),
            "half_width": float(max(self.t_end - self.t_start, 0.0) / 2.0),
            "max_abs_z": float(self.max_abs_z),
            "signed_z": None if self.signed_z is None else float(self.signed_z),
            "fractional_deviation": (
                None
                if self.fractional_deviation is None
                else float(self.fractional_deviation)
            ),
            "chi2": None if self.chi2 is None else float(self.chi2),
            "reduced_chi2": (
                None if self.reduced_chi2 is None else float(self.reduced_chi2)
            ),
            "n_points": None if self.n_points is None else int(self.n_points),
            "sources": list(self.sources),
            "fit_excluded": bool(self.fit_excluded),
            "adopted_model": self.adopted_model,
        }


@dataclass(frozen=True)
class AnomalyPipelineResult:
    """Complete adopted-model anomaly result returned by the one-line API."""

    effect_aware: object
    adopted_fit: object
    post_physical_result: object | None
    fit_exclusion_mask: np.ndarray
    final_measurement: PlanetSignalResult
    features: PlanetFeatureResult
    template_free: TemplateFreeSearchResult
    candidates: tuple[AnomalyCandidate, ...]
    reason_codes: tuple[str, ...]
    diagnostics: dict[str, object]

    @property
    def has_anomaly_candidate(self) -> bool:
        """Whether the final adopted residual produced at least one candidate."""

        return bool(self.candidates)

    @property
    def anomaly_candidates(self) -> list[dict[str, object]]:
        """Ranked, JSON-ready candidate dictionaries."""

        return [candidate.to_dict() for candidate in self.candidates]

    @property
    def best_anomaly_candidate(self) -> dict[str, object] | None:
        """Highest-ranked candidate dictionary, or ``None``."""

        return self.candidates[0].to_dict() if self.candidates else None

    @property
    def display_signal_mask(self) -> np.ndarray:
        """Points actually excluded by the adopted continuation fit."""

        return self.fit_exclusion_mask

    @property
    def measurement_mask(self) -> np.ndarray:
        """Final frozen-residual window used only for feature measurement."""

        return np.asarray(self.final_measurement.signal_mask, dtype=bool)


def build_anomaly_candidates(
    features: PlanetFeatureResult,
    template_free: TemplateFreeSearchResult,
    *,
    time: np.ndarray,
    fit_exclusion_mask: np.ndarray,
    adopted_model: str,
    cadence_factor: float = 3.0,
) -> tuple[AnomalyCandidate, ...]:
    """Merge feature and template-free detections into ranked candidates."""

    time_np = np.asarray(time, dtype=float).reshape(-1)
    excluded = np.asarray(fit_exclusion_mask, dtype=bool).reshape(-1)
    if time_np.size != excluded.size:
        raise ValueError("fit_exclusion_mask must match time.")
    positive_steps = np.diff(np.sort(time_np))
    positive_steps = positive_steps[
        np.isfinite(positive_steps) & (positive_steps > 0.0)
    ]
    cadence = float(np.median(positive_steps)) if positive_steps.size else 0.0

    rows: list[dict[str, Any]] = []
    for feature in tuple(getattr(features, "features", ())):
        rows.append(
            {
                "source": "final_residual_feature",
                "object": feature,
                "start": float(feature.t_start),
                "end": float(feature.t_end),
                "center": float(feature.time),
            }
        )
    for candidate in tuple(getattr(template_free, "candidates", ())):
        rows.append(
            {
                "source": "template_free",
                "object": candidate,
                "start": float(candidate.t_start),
                "end": float(candidate.t_end),
                "center": float(candidate.t_center),
            }
        )
    if not rows:
        return ()

    parents = list(range(len(rows)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = root(first), root(second)
        if first_root != second_root:
            parents[second_root] = first_root

    cadence_tolerance = max(0.0, float(cadence_factor)) * cadence
    for first in range(len(rows)):
        for second in range(first + 1, len(rows)):
            if rows[first]["source"] == rows[second]["source"]:
                continue
            overlap = max(rows[first]["start"], rows[second]["start"]) <= min(
                rows[first]["end"], rows[second]["end"]
            )
            first_width = max(rows[first]["end"] - rows[first]["start"], 0.0)
            second_width = max(rows[second]["end"] - rows[second]["start"], 0.0)
            center_tolerance = max(
                cadence_tolerance,
                0.5 * max(first_width, second_width),
            )
            centers_close = (
                abs(rows[first]["center"] - rows[second]["center"])
                <= center_tolerance
            )
            if overlap or centers_close:
                union(first, second)

    groups: dict[int, list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(root(index), []).append(row)

    unranked: list[dict[str, Any]] = []
    for group in groups.values():
        feature_rows = [
            row for row in group if row["source"] == "final_residual_feature"
        ]
        free_rows = [row for row in group if row["source"] == "template_free"]
        primary_feature = max(
            feature_rows,
            key=lambda row: float(row["object"].strength),
            default=None,
        )
        primary_free = max(
            free_rows,
            key=lambda row: float(row["object"].chi2),
            default=None,
        )
        primary = primary_feature or primary_free
        obj = primary["object"]
        if primary_feature is not None:
            t_center = float(obj.time)
            signed_z = float(obj.signed_z)
            fractional_deviation = float(obj.fractional_deviation)
        else:
            t_center = float(obj.t_center)
            start_index = max(0, int(obj.start_index))
            end_index = min(len(template_free.z) - 1, int(obj.end_index))
            local_z = np.asarray(template_free.z, dtype=float)[
                start_index : end_index + 1
            ]
            signed_z = (
                float(local_z[np.argmax(np.abs(local_z))]) if local_z.size else None
            )
            fractional_deviation = None
        strengths = [
            float(row["object"].strength) for row in feature_rows
        ] + [float(row["object"].max_abs_z) for row in free_rows]
        max_abs_z = max(strengths)
        t_start = float(primary["start"])
        t_end = float(primary["end"])
        free_obj = None if primary_free is None else primary_free["object"]
        component_start = min(float(row["start"]) for row in group)
        component_end = max(float(row["end"]) for row in group)
        in_interval = (time_np >= component_start) & (time_np <= component_end)
        unranked.append(
            {
                "kind": str(obj.kind),
                "t_center": t_center,
                "t_start": t_start,
                "t_end": t_end,
                "timescale": max(t_end - t_start, 0.0),
                "max_abs_z": max_abs_z,
                "signed_z": signed_z,
                "fractional_deviation": fractional_deviation,
                "chi2": None if free_obj is None else float(free_obj.chi2),
                "reduced_chi2": (
                    None if free_obj is None else float(free_obj.reduced_chi2)
                ),
                "n_points": None if free_obj is None else int(free_obj.n_points),
                "sources": tuple(
                    source
                    for source in ("final_residual_feature", "template_free")
                    if any(row["source"] == source for row in group)
                ),
                "fit_excluded": bool(np.any(excluded & in_interval)),
                "adopted_model": str(adopted_model),
            }
        )

    unranked.sort(
        key=lambda row: (
            -float(row["max_abs_z"]),
            -float(row["chi2"] or 0.0),
            float(row["t_center"]),
        )
    )
    return tuple(
        AnomalyCandidate(rank=rank, **row)
        for rank, row in enumerate(unranked, start=1)
    )


__all__ = [
    "AnomalyCandidate",
    "AnomalyPipelineConfig",
    "AnomalyPipelineResult",
    "build_anomaly_candidates",
]
