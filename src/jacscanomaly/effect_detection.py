"""Physical residual detectors for finite-source and parallax effects.

The detectors in this module are deliberately cheaper than a nonlinear
single-lens fit.  They operate on a PSPL fit, remove the local PSPL nuisance
subspace, and then measure how much of the remaining standardized residual can
be explained by a physical tangent or template.

The public functions accept NumPy-like arrays wherever possible.  The default
survey detector uses analytic NumPy tangents and VBMicrolensing's native FSPL
backend, avoiding event-shape JAX compilation.  The microjax FSPL backend is
available explicitly for cross-validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from .signal_scale import ObservedSignalScale, measure_observed_signal_scale


def _json_safe(value):
    """Recursively convert detector diagnostics to JSON-native values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class ProjectionDiagnostics:
    """Numerical diagnostics for a nuisance-subspace projection."""

    n_points: int
    n_parameters: int
    effective_rank: int
    condition_number: float
    singular_values: tuple[float, ...]


@dataclass(frozen=True)
class EffectCandidate:
    """Physical detector output used by the routing policy.

    ``decision`` is intentionally left as ``"unrouted"`` by detector
    functions.  :func:`jacscanomaly.effect_routing.route_candidate` applies a
    configurable policy without changing the measured diagnostics.
    """

    effect: str
    score: float
    score_without_compact_blocks: float
    effective_rank: int
    condition_number: float
    coverage: float
    max_point_influence: float
    max_block_influence: float
    subset_stability: float
    best_template_or_direction: Optional[np.ndarray] = None
    seed_parameters: Optional[np.ndarray] = None
    decision: str = "unrouted"
    reason_codes: tuple[str, ...] = ()
    subset_scores: tuple[float, ...] = ()
    subset_diagnostics: tuple[dict[str, object], ...] = ()
    compact_block_mask: Optional[np.ndarray] = None
    score_without_planet: float = float("nan")
    planet_overlap: float = 0.0
    morphology: str = "unclassified"
    observed_signal_scale: Optional[ObservedSignalScale] = None

    def with_decision(self, decision: str, reason_codes: Iterable[str]) -> "EffectCandidate":
        """Return a copy with the routing decision attached."""
        return EffectCandidate(
            effect=self.effect,
            score=float(self.score),
            score_without_compact_blocks=float(self.score_without_compact_blocks),
            effective_rank=int(self.effective_rank),
            condition_number=float(self.condition_number),
            coverage=float(self.coverage),
            max_point_influence=float(self.max_point_influence),
            max_block_influence=float(self.max_block_influence),
            subset_stability=float(self.subset_stability),
            best_template_or_direction=None
            if self.best_template_or_direction is None
            else np.asarray(self.best_template_or_direction, dtype=float).copy(),
            seed_parameters=None
            if self.seed_parameters is None
            else np.asarray(self.seed_parameters, dtype=float).copy(),
            decision=str(decision),
            reason_codes=tuple(dict.fromkeys(str(code) for code in reason_codes)),
            subset_scores=tuple(float(x) for x in self.subset_scores),
            subset_diagnostics=tuple(dict(row) for row in self.subset_diagnostics),
            compact_block_mask=None
            if self.compact_block_mask is None
            else np.asarray(self.compact_block_mask, dtype=bool).copy(),
            score_without_planet=float(self.score_without_planet),
            planet_overlap=float(self.planet_overlap),
            morphology=str(self.morphology),
            observed_signal_scale=self.observed_signal_scale,
        )

    def with_probe(
        self,
        *,
        score: float,
        seed_parameters: Optional[np.ndarray] = None,
        decision: str = "unrouted",
        reason_codes: Iterable[str] = (),
    ) -> "EffectCandidate":
        """Return a candidate updated with an exact-probe improvement."""
        return EffectCandidate(
            effect=self.effect,
            score=float(score),
            score_without_compact_blocks=float(self.score_without_compact_blocks),
            effective_rank=self.effective_rank,
            condition_number=self.condition_number,
            coverage=self.coverage,
            max_point_influence=self.max_point_influence,
            max_block_influence=self.max_block_influence,
            subset_stability=self.subset_stability,
            best_template_or_direction=self.best_template_or_direction,
            seed_parameters=self.seed_parameters if seed_parameters is None else seed_parameters,
            decision=decision,
            reason_codes=tuple(dict.fromkeys((*self.reason_codes, *tuple(reason_codes)))),
            subset_scores=self.subset_scores,
            subset_diagnostics=self.subset_diagnostics,
            compact_block_mask=self.compact_block_mask,
            score_without_planet=self.score_without_planet,
            planet_overlap=self.planet_overlap,
            morphology=self.morphology,
            observed_signal_scale=self.observed_signal_scale,
        )

    def summary_dict(self) -> dict[str, object]:
        """Return JSON-friendly scalar diagnostics."""
        row: dict[str, object] = {
            "effect": self.effect,
            "score": float(self.score),
            "score_without_compact_blocks": float(self.score_without_compact_blocks),
            "effective_rank": int(self.effective_rank),
            "condition_number": float(self.condition_number),
            "coverage": float(self.coverage),
            "max_point_influence": float(self.max_point_influence),
            "max_block_influence": float(self.max_block_influence),
            "subset_stability": float(self.subset_stability),
            "score_without_planet": float(self.score_without_planet),
            "planet_overlap": float(self.planet_overlap),
            "morphology": self.morphology,
            "observed_signal_scale": (
                None
                if self.observed_signal_scale is None
                else self.observed_signal_scale.summary_dict()
            ),
            "decision": self.decision,
            "reason_codes": list(self.reason_codes),
            "subset_diagnostics": _json_safe(self.subset_diagnostics),
        }
        if self.best_template_or_direction is not None:
            row["best_template_or_direction"] = np.asarray(
                self.best_template_or_direction, dtype=float
            ).tolist()
        if self.seed_parameters is not None:
            row["seed_parameters"] = np.asarray(self.seed_parameters, dtype=float).tolist()
        return row


@dataclass(frozen=True)
class _ProjectedScore:
    score: float
    rank: int
    condition_number: float
    fitted: np.ndarray
    support: np.ndarray
    point_influence: float
    coefficient: float = 0.0
    information: float = 0.0
    unmatched_energy: float = 0.0


def _as_2d(matrix: Optional[np.ndarray], n: int) -> np.ndarray:
    if matrix is None:
        return np.empty((n, 0), dtype=float)
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] != n:
        raise ValueError("Jacobian/template arrays must have shape (n_points, n_columns).")
    return arr


def _orthonormal_basis(matrix: np.ndarray, rtol: float) -> tuple[np.ndarray, ProjectionDiagnostics]:
    """Build a rank-aware orthonormal basis without materializing ``P_perp``."""
    n, p = matrix.shape
    if p == 0:
        return np.empty((n, 0), dtype=float), ProjectionDiagnostics(
            n_points=n,
            n_parameters=0,
            effective_rank=0,
            condition_number=1.0,
            singular_values=(),
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Nuisance Jacobian contains non-finite values.")

    # SVD gives stable rank and condition diagnostics even when the PSPL
    # tangent directions are nearly collinear with flux parameters.
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    if singular.size == 0:
        rank = 0
    else:
        cutoff = float(rtol) * max(float(singular[0]), 1.0)
        rank = int(np.count_nonzero(singular > cutoff))
    q = u[:, :rank]
    nonzero = singular[singular > (float(rtol) * max(float(singular[0]), 1.0))] if singular.size else singular
    condition = float(nonzero[0] / nonzero[-1]) if nonzero.size else float("inf")
    return q, ProjectionDiagnostics(
        n_points=n,
        n_parameters=p,
        effective_rank=rank,
        condition_number=condition,
        singular_values=tuple(float(x) for x in singular),
    )


def project_out_nuisance(
    vector: np.ndarray,
    nuisance_jacobian: Optional[np.ndarray],
    *,
    rtol: float = 1.0e-10,
) -> tuple[np.ndarray, ProjectionDiagnostics]:
    """Project a vector orthogonally away from a nuisance Jacobian.

    The calculation is ``v - Q(Q.T @ v)``.  No dense ``n x n`` projection
    matrix is constructed.
    """
    v = np.asarray(vector, dtype=float).reshape(-1)
    J = _as_2d(nuisance_jacobian, v.size)
    if not np.all(np.isfinite(v)):
        raise ValueError("vector contains non-finite values.")
    q, diagnostics = _orthonormal_basis(J, rtol)
    return v - q @ (q.T @ v), diagnostics


def _projected_score(
    z: np.ndarray,
    template: np.ndarray,
    nuisance_jacobian: Optional[np.ndarray],
    mask: np.ndarray,
    *,
    rtol: float,
) -> _ProjectedScore:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return _ProjectedScore(
            0.0,
            0,
            float("inf"),
            np.zeros_like(z),
            np.zeros_like(z, dtype=bool),
            0.0,
        )

    zs = np.asarray(z[indices], dtype=float)
    hs = np.asarray(template[indices], dtype=float)
    Js = _as_2d(nuisance_jacobian, z.size)[indices]
    q, diagnostics = _orthonormal_basis(Js, rtol)
    zp = zs - q @ (q.T @ zs)
    hp = hs - q @ (q.T @ hs)
    gram = float(hp @ hp)
    if not np.isfinite(gram) or gram <= 0.0:
        return _ProjectedScore(
            0.0,
            diagnostics.effective_rank,
            diagnostics.condition_number,
            np.zeros_like(z),
            np.zeros_like(z, dtype=bool),
            0.0,
            0.0,
            gram,
            float(np.sum(zp * zp)),
        )

    b = float(hp @ zp)
    coefficient = b / gram
    fitted = np.zeros_like(z)
    fitted[indices] = coefficient * hp
    point_energy = fitted * fitted
    total_energy = float(np.sum(point_energy))
    max_point = float(np.max(point_energy) / total_energy) if total_energy > 0.0 else 0.0
    support = np.zeros_like(z, dtype=bool)
    threshold = max(float(np.max(np.abs(hp))) * 1.0e-3, 1.0e-12)
    support[indices] = np.abs(hp) >= threshold
    return _ProjectedScore(
        score=float(b * coefficient),
        rank=diagnostics.effective_rank,
        condition_number=diagnostics.condition_number,
        fitted=fitted,
        support=support,
        point_influence=max_point,
        coefficient=float(coefficient),
        information=gram,
        unmatched_energy=float(max(np.sum(zp * zp) - b * coefficient, 0.0)),
    )


def _projected_template_scores(
    z: np.ndarray,
    templates: np.ndarray,
    nuisance_jacobian: Optional[np.ndarray],
    mask: np.ndarray,
    *,
    rtol: float,
) -> np.ndarray:
    """Score a template bank with one nuisance decomposition.

    The nuisance Jacobian and the valid-data mask are shared by all FSPL
    templates.  Projecting the whole bank at once avoids one SVD per template
    while remaining algebraically equivalent to :func:`_projected_score`.
    """
    residual = np.asarray(z, dtype=float).reshape(-1)
    bank = np.asarray(templates, dtype=float)
    if bank.ndim == 1:
        bank = bank[None, :]
    if bank.ndim != 2 or bank.shape[1] != residual.size:
        raise ValueError("templates must have shape (n_templates, n_points).")
    use = np.asarray(mask, dtype=bool).reshape(-1)
    if use.size != residual.size:
        raise ValueError("mask must have the same length as z.")
    indices = np.flatnonzero(use)
    if indices.size == 0:
        return np.zeros(bank.shape[0], dtype=float)

    zs = residual[indices]
    hs = bank[:, indices]
    Js = _as_2d(nuisance_jacobian, residual.size)[indices]
    q, _ = _orthonormal_basis(Js, rtol)
    zp = zs - q @ (q.T @ zs)
    # Templates are rows, so their nuisance coefficients are ``hs @ q``.
    hp = hs - (hs @ q) @ q.T
    gram = np.einsum("ij,ij->i", hp, hp)
    matched = hp @ zp
    scores = np.zeros(bank.shape[0], dtype=float)
    valid = np.isfinite(gram) & np.isfinite(matched) & (gram > 0.0)
    scores[valid] = matched[valid] * matched[valid] / gram[valid]
    return scores


def _subset_stability(scores: Sequence[float]) -> float:
    finite = np.asarray([x for x in scores if np.isfinite(x)], dtype=float)
    if finite.size < 2:
        return 1.0 if finite.size else 0.0
    mean = float(np.mean(finite))
    if mean <= 0.0:
        return 0.0
    return float(np.clip(1.0 - np.std(finite) / (mean + 1.0e-12), 0.0, 1.0))


def _direction_stability(rows: Sequence[dict[str, object]], full_direction: np.ndarray) -> float:
    """Compare physical directions and normalized amplitudes, not raw scores."""
    valid = [row for row in rows if bool(row.get("valid", False))]
    if not valid:
        return 0.0
    direction = np.asarray(full_direction, dtype=float).reshape(-1)
    norm = max(float(np.linalg.norm(direction)), 1.0e-30)
    direction = direction / norm
    cosines = []
    amplitudes = []
    for row in valid:
        row_direction = np.asarray(row.get("direction", ()), dtype=float).reshape(-1)
        if row_direction.size != direction.size:
            continue
        row_norm = float(np.linalg.norm(row_direction))
        if row_norm <= 0.0:
            continue
        cosines.append(float(np.clip(np.dot(direction, row_direction / row_norm), -1.0, 1.0)))
        amplitudes.append(float(row.get("normalized_amplitude", 0.0)))
    if not cosines:
        return 0.0
    # A single subset with a reversed physical direction is not a stable
    # detection even when the mean cosine looks acceptable.  Use the worst
    # valid subset so sign-flip injections fail open to fallback/probe.
    direction_score = float(np.min([(cosine + 1.0) * 0.5 for cosine in cosines]))
    amplitude_score = 1.0
    finite_amp = np.asarray([x for x in amplitudes if np.isfinite(x) and x > 0.0], dtype=float)
    if finite_amp.size >= 2:
        amplitude_score = float(np.clip(np.min(finite_amp) / np.max(finite_amp), 0.0, 1.0))
    return float(np.clip(direction_score * amplitude_score, 0.0, 1.0))


def find_compact_blocks(
    time: np.ndarray,
    standardized_residual: np.ndarray,
    *,
    sigma: float = 5.0,
    max_blocks: int = 1,
    max_span: float = 2.0,
    max_gap: Optional[float] = None,
) -> np.ndarray:
    """Find at most a few compact high-residual blocks.

    This helper is intentionally conservative: a block wider than
    ``max_span`` or separated by a season-sized gap is never removed.  That
    prevents a broad parallax wing from being silently treated as a planet
    mask.
    """
    t = np.asarray(time, dtype=float).reshape(-1)
    z = np.asarray(standardized_residual, dtype=float).reshape(-1)
    if t.size != z.size:
        raise ValueError("time and standardized_residual must have the same length.")
    if t.size == 0:
        return np.zeros(0, dtype=bool)
    active = np.isfinite(t) & np.isfinite(z) & (np.abs(z) >= float(sigma))
    blocks: list[tuple[float, int, int]] = []
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    for start in starts:
        end = int(start)
        while end + 1 < t.size and active[end + 1]:
            if max_gap is not None and (t[end + 1] - t[end]) > float(max_gap):
                break
            end += 1
        span = float(t[end] - t[start])
        if span <= float(max_span):
            strength = float(np.sum(np.abs(z[start : end + 1])))
            blocks.append((strength, int(start), end))
    blocks.sort(reverse=True)
    selected = blocks[: max(0, int(max_blocks))]
    mask = np.zeros(t.size, dtype=bool)
    for _, start, end in selected:
        mask[start : end + 1] = True
    return mask


def _block_influence(time: np.ndarray, z: np.ndarray, fitted: np.ndarray, *, max_span: float) -> float:
    mask = find_compact_blocks(time, z, sigma=5.0, max_blocks=1, max_span=max_span)
    energy = fitted * fitted
    total = float(np.sum(energy))
    return float(np.sum(energy[mask]) / total) if total > 0.0 else 0.0


def _validated_mask(mask: Optional[np.ndarray], size: int, name: str) -> np.ndarray:
    if mask is None:
        return np.zeros(size, dtype=bool)
    value = np.asarray(mask, dtype=bool).reshape(-1)
    if value.size != size:
        raise ValueError(f"{name} must have the same length as time.")
    return value


def _energy_overlap(values: np.ndarray, mask: np.ndarray) -> float:
    energy = np.square(np.asarray(values, dtype=float))
    total = float(np.sum(energy))
    return float(np.sum(energy[mask]) / total) if total > 0.0 else 0.0


def _weighted_span(time: np.ndarray, weights: np.ndarray) -> float:
    t = np.asarray(time, dtype=float)
    w = np.maximum(np.asarray(weights, dtype=float), 0.0)
    valid = np.isfinite(t) & np.isfinite(w) & (w > 0.0)
    if np.count_nonzero(valid) < 2:
        return 0.0
    order = np.argsort(t[valid])
    ts = t[valid][order]
    ws = w[valid][order]
    cumulative = np.cumsum(ws)
    cumulative /= cumulative[-1]
    lo = float(np.interp(0.05, cumulative, ts))
    hi = float(np.interp(0.95, cumulative, ts))
    return max(hi - lo, 0.0)


def _central_symmetry(
    time: np.ndarray,
    residual: np.ndarray,
    *,
    t0: float,
    half_width: float,
    valid_mask: np.ndarray,
) -> float:
    """Measure whether a central residual has the even morphology of FSPL.

    The comparison is made at matched distances on the two sides of ``t0``.
    A smooth same-sign swelling scores near one; a one-sided peak, dip/peak
    pair, or caustic structure scores near zero.
    """
    t = np.asarray(time, dtype=float)
    z = np.asarray(residual, dtype=float)
    use = (
        np.asarray(valid_mask, dtype=bool)
        & np.isfinite(t)
        & np.isfinite(z)
        & (np.abs(t - float(t0)) <= max(float(half_width), 1.0e-12))
    )
    left = np.flatnonzero(use & (t < t0))
    right = np.flatnonzero(use & (t > t0))
    if left.size < 4 or right.size < 4:
        return 0.0
    left_x = (t0 - t[left])[::-1]
    left_z = z[left][::-1]
    right_x = t[right] - t0
    right_z = z[right]
    radius_max = min(float(left_x[-1]), float(right_x[-1]))
    if radius_max <= 0.0:
        return 0.0
    radius = np.linspace(0.0, radius_max, min(128, left.size, right.size))
    lhs = np.interp(radius, left_x, left_z)
    rhs = np.interp(radius, right_x, right_z)
    denominator = float(np.sum(lhs * lhs + rhs * rhs))
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(1.0 - np.sum(np.square(lhs - rhs)) / denominator, 0.0, 1.0))


def _fspl_signed_topology(
    time: np.ndarray,
    residual: np.ndarray,
    *,
    t0: float,
    t_star: float,
    valid_mask: np.ndarray,
) -> dict[str, float | int | bool]:
    """Measure the PSPL-residual topology of a finite source crossing."""
    t = np.asarray(time, dtype=float)
    z = np.asarray(residual, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(t) & np.isfinite(z)
    radius = np.abs(t - float(t0))
    scale = max(float(t_star), 1.0e-12)
    core = valid & (radius <= 0.5 * scale)
    left = valid & (t < t0) & (radius > 0.5 * scale) & (radius <= 1.5 * scale)
    right = valid & (t > t0) & (radius > 0.5 * scale) & (radius <= 1.5 * scale)

    def mean(mask: np.ndarray) -> float:
        return float(np.mean(z[mask])) if np.any(mask) else float("nan")

    core_mean = mean(core)
    left_mean = mean(left)
    right_mean = mean(right)
    valid_shape = bool(
        np.count_nonzero(core) >= 4
        and np.count_nonzero(left) >= 4
        and np.count_nonzero(right) >= 4
        and core_mean <= -3.0
        and left_mean > 0.0
        and right_mean > 0.0
    )
    partial_shape = bool(
        np.count_nonzero(core) >= 4
        and core_mean <= -3.0
        and (
            (
                np.count_nonzero(left) >= 4
                and left_mean > 0.0
                and np.count_nonzero(right) == 0
            )
            or (
                np.count_nonzero(right) >= 4
                and right_mean > 0.0
                and np.count_nonzero(left) == 0
            )
        )
    )
    return {
        "core_mean_z": core_mean,
        "left_shoulder_mean_z": left_mean,
        "right_shoulder_mean_z": right_mean,
        "core_points": int(np.count_nonzero(core)),
        "left_shoulder_points": int(np.count_nonzero(left)),
        "right_shoulder_points": int(np.count_nonzero(right)),
        "valid": valid_shape,
        "partial": partial_shape,
    }


def _fspl_sparse_high_snr_topology(
    topology: dict[str, float | int | bool],
    *,
    symmetry: float,
    template_explained_fraction: float,
) -> bool:
    """Accept a sparsely sampled but unambiguous signed FSPL peak."""
    shoulder_values = np.asarray(
        [
            topology.get("left_shoulder_mean_z", np.nan),
            topology.get("right_shoulder_mean_z", np.nan),
        ],
        dtype=float,
    )
    return bool(
        symmetry >= 0.95
        and template_explained_fraction >= 0.20
        and int(topology.get("core_points", 0)) >= 2
        and int(topology.get("left_shoulder_points", 0)) >= 2
        and int(topology.get("right_shoulder_points", 0)) >= 2
        and float(topology.get("core_mean_z", 0.0)) <= -5.0
        and np.all(np.isfinite(shoulder_values))
        and np.all(shoulder_values >= 3.0)
    )


def _candidate_from_template(
    *,
    effect: str,
    time: np.ndarray,
    z: np.ndarray,
    nuisance_jacobian: np.ndarray,
    template: np.ndarray,
    seed_parameters: Optional[np.ndarray],
    best_template_or_direction: Optional[np.ndarray],
    compact_mask: np.ndarray,
    subset_masks: Sequence[np.ndarray],
    rtol: float,
    max_compact_span: float,
    min_coverage: float,
    planet_mask: Optional[np.ndarray] = None,
) -> EffectCandidate:
    all_mask = np.isfinite(z) & np.isfinite(template) & np.isfinite(time)
    full = _projected_score(z, template, nuisance_jacobian, all_mask, rtol=rtol)
    without = _projected_score(
        z,
        template,
        nuisance_jacobian,
        all_mask & ~compact_mask,
        rtol=rtol,
    )
    planet = _validated_mask(planet_mask, time.size, "planet_mask")
    without_planet = _projected_score(
        z,
        template,
        nuisance_jacobian,
        all_mask & ~planet,
        rtol=rtol,
    )
    profile_peak = (
        float(np.nanmax(np.abs(full.fitted[all_mask])))
        if np.any(all_mask)
        else 0.0
    )
    observed_scale = measure_observed_signal_scale(
        time,
        full.fitted,
        valid_mask=all_mask,
        threshold=max(1.0e-3, 0.05 * profile_peak),
        source=f"{effect}_profile",
    )
    subset_rows: list[dict[str, object]] = []
    for index, mask in enumerate(subset_masks):
        use = all_mask & mask
        if np.count_nonzero(use) < max(3, nuisance_jacobian.shape[1] + 1):
            subset_rows.append({"name": f"subset_{index}", "valid": False, "reason": "too_few_points"})
            continue
        subset = _projected_score(z, template, nuisance_jacobian, use, rtol=rtol)
        valid_subset = bool(
            subset.rank > 0
            and subset.information >= max(full.information * 1.0e-3, 1.0e-30)
            and np.isfinite(subset.score)
        )
        subset_rows.append(
            {
                "name": f"subset_{index}",
                "n_points": int(np.count_nonzero(use)),
                "score": float(subset.score),
                "information": float(subset.information),
                "normalized_score": float(subset.score / max(subset.information, 1.0e-30)),
                "normalized_amplitude": float(abs(subset.coefficient) * np.sqrt(max(subset.information, 0.0))),
                "coefficient": float(subset.coefficient),
                "direction": np.asarray([subset.coefficient], dtype=float),
                "coverage": float(np.count_nonzero(subset.support & use) / max(np.count_nonzero(use), 1)),
                "unmatched_energy": float(subset.unmatched_energy),
                "valid": valid_subset,
                "reason": "ok" if valid_subset else "insufficient_subset_information",
            }
        )
    subset_scores = tuple(float(row.get("score", 0.0)) for row in subset_rows)
    support_count = int(np.count_nonzero(full.support))
    coverage = float(support_count / max(np.count_nonzero(all_mask), 1))
    if seed_parameters is not None and len(seed_parameters) >= 4:
        seed = np.asarray(seed_parameters, dtype=float)
        t0, tE, u0 = float(seed[0]), abs(float(seed[1])), abs(float(seed[2]))
        rho = float(np.exp(seed[3])) if seed[3] < 0.0 else float(seed[3])
        if observed_scale.valid:
            # Morphology is measured in observed time, not from the
            # ``tE*u0`` or ``rho*tE`` parameter products.  Keep a cadence-sized
            # floor so sparse events do not collapse to one sample.
            cadence = (
                observed_scale.cadence
                if np.isfinite(observed_scale.cadence)
                else 0.0
            )
            width = max(observed_scale.half_width, cadence, 1.0e-12)
            topology_half_width = max(0.35 * width, cadence, 1.0e-12)
            t0 = float(observed_scale.t_center)
        else:
            # Compatibility fallback for a detector with too little usable
            # residual support.  A valid observed scale is required for an
            # automatic physical classification below.
            width = max(3.0 * rho * tE, 0.25 * u0 * tE, 1.0e-12)
            topology_half_width = max(3.0 * rho * tE, 1.0e-12)
        central = np.abs(time - t0) <= width
        left = central & (time <= t0)
        right = central & (time >= t0)
        left_coverage = float(np.count_nonzero(full.support & left) / max(np.count_nonzero(left), 1))
        right_coverage = float(np.count_nonzero(full.support & right) / max(np.count_nonzero(right), 1))
        coverage = min(coverage, left_coverage, right_coverage)
        subset_rows.extend(
            [
                {"name": "peak_left", "coverage": left_coverage, "n_points": int(np.count_nonzero(left)), "valid": bool(np.count_nonzero(left) > 0)},
                {"name": "peak_right", "coverage": right_coverage, "n_points": int(np.count_nonzero(right)), "valid": bool(np.count_nonzero(right) > 0)},
            ]
        )
        symmetry = _central_symmetry(
            time,
            z,
            t0=t0,
            half_width=width,
            # FSPL is itself a central anomaly. The known anomaly mask is a
            # competing-planet label, not an exclusion mask for this
            # morphology measurement.
            valid_mask=all_mask,
        )
        topology = _fspl_signed_topology(
            time,
            z,
            t0=t0,
            t_star=topology_half_width,
            valid_mask=all_mask,
        )
        template_explained_fraction = float(
            full.score
            / max(full.score + full.unmatched_energy, 1.0e-30)
        )
        shoulder_values = np.asarray(
            [
                topology.get("left_shoulder_mean_z", np.nan),
                topology.get("right_shoulder_mean_z", np.nan),
            ],
            dtype=float,
        )
        # Roman can resolve an extremely narrow finite-source crossing with
        # only two samples in each signed region. Requiring four samples then
        # turns the canonical central-dip/two-positive-shoulder topology into
        # a compact "planet" block. Keep this exception deliberately strict:
        # both shoulders must be independently significant, the central dip
        # strong, and the full peak highly symmetric.
        sparse_high_snr_topology = _fspl_sparse_high_snr_topology(
            topology,
            symmetry=symmetry,
            template_explained_fraction=template_explained_fraction,
        )
        fspl_shape_ok = bool(
            observed_scale.valid
            and not observed_scale.censored
            and symmetry >= 0.95
            and template_explained_fraction >= 0.20
            and (topology["valid"] or sparse_high_snr_topology)
        )
        fspl_flattened_peak = bool(
            observed_scale.valid
            and not observed_scale.censored
            and symmetry >= 0.95
            and template_explained_fraction >= 0.40
            and np.all(np.isfinite(shoulder_values))
            and float(topology.get("core_mean_z", 0.0)) <= -5.0
            and np.all(shoulder_values < 0.0)
            and abs(float(topology.get("core_mean_z", 0.0)))
            >= 1.5 * float(np.max(np.abs(shoulder_values)))
        )
        fspl_partial = bool(
            symmetry >= 0.95
            and template_explained_fraction >= 0.20
            and topology["partial"]
        )
        subset_rows.append(
            {
                "name": "fspl_morphology",
                "central_symmetry": symmetry,
                "template_explained_fraction": template_explained_fraction,
                "signed_template_coefficient": float(full.coefficient),
                **topology,
                "valid": fspl_shape_ok or fspl_flattened_peak,
                "sparse_high_snr": sparse_high_snr_topology,
                "reason": (
                    "sparse_high_snr_central_dip_with_two_shoulders"
                    if sparse_high_snr_topology
                    else "central_dip_with_two_shoulders"
                    if fspl_shape_ok
                    else "symmetric_flattened_peak"
                    if fspl_flattened_peak
                    else "non_fspl_peak_shape"
                ),
            }
        )
    else:
        symmetry = float("nan")
        fspl_shape_ok = False
        fspl_partial = False
        fspl_flattened_peak = False
    reasons: list[str] = []
    if full.rank <= 0:
        reasons.append("rank_deficient")
    if not np.isfinite(full.condition_number) or full.condition_number > 1.0e10:
        reasons.append("ill_conditioned")
    if coverage < float(min_coverage):
        reasons.append("insufficient_coverage")
    if full.score <= 0.0:
        reasons.append("non_positive_score")
    if not observed_scale.valid:
        reasons.append("signal_scale_unmeasurable")
    elif observed_scale.censored:
        reasons.append("signal_scale_censored")
    if compact_mask.any() and full.score > 0.0 and without.score < 0.5 * full.score:
        reasons.append("compact_block_dominated")
    planet_overlap = _energy_overlap(full.fitted, planet)
    planet_retention = float(
        without_planet.score / max(full.score, 1.0e-30)
    ) if full.score > 0.0 else 0.0
    if planet.any() and (planet_retention < 0.5 or planet_overlap > 0.35):
        reasons.append("planet_morphology_dominated")
    if np.isfinite(symmetry) and not (fspl_shape_ok or fspl_flattened_peak):
        reasons.append("non_fspl_peak_shape")
    if fspl_partial:
        reasons.append("fspl_one_sided_coverage")
    stability = _direction_stability(subset_rows, np.asarray([full.coefficient]))
    if not any(bool(row.get("valid", False)) for row in subset_rows):
        stability = 0.0
    if stability < 0.25:
        reasons.append("subset_unstable")
    if not reasons:
        reasons.append("physical_support_available")
    return EffectCandidate(
        effect=effect,
        score=float(full.score),
        score_without_compact_blocks=float(without.score),
        effective_rank=int(full.rank),
        condition_number=float(full.condition_number),
        coverage=coverage,
        max_point_influence=float(full.point_influence),
        max_block_influence=_block_influence(time, z, full.fitted, max_span=max_compact_span),
        subset_stability=stability,
        best_template_or_direction=best_template_or_direction,
        seed_parameters=seed_parameters,
        reason_codes=tuple(reasons),
        subset_scores=subset_scores,
        subset_diagnostics=tuple(subset_rows),
        compact_block_mask=np.asarray(compact_mask, dtype=bool) | planet,
        score_without_planet=float(without_planet.score),
        planet_overlap=planet_overlap,
        morphology=(
            "fspl_even_peak"
            if fspl_shape_ok
            else "fspl_flattened_peak"
            if fspl_flattened_peak
            else "fspl_partial_peak"
            if fspl_partial
            else "planet_like"
            if (
                "planet_morphology_dominated" in reasons
                and without_planet.score < 25.0
            )
            else "mixed_or_planet"
            if "planet_morphology_dominated" in reasons
            else "ambiguous"
        ),
        observed_signal_scale=observed_scale,
    )


def _time_subsets(time: np.ndarray) -> tuple[np.ndarray, ...]:
    n = time.size
    if n < 4:
        return (np.ones(n, dtype=bool),)
    order = np.argsort(time)
    masks = []
    for lo, hi in ((0, n // 2), (n // 2, n), (0, max(1, n // 3)), (2 * n // 3, n)):
        mask = np.zeros(n, dtype=bool)
        mask[order[lo:hi]] = True
        masks.append(mask)
    return tuple(masks)


def _named_time_subsets(
    time: np.ndarray,
    seed_parameters: Optional[np.ndarray] = None,
    observed_scale: Optional[ObservedSignalScale] = None,
) -> tuple[tuple[str, np.ndarray], ...]:
    """Return physically interpretable, ordered subsets."""
    n = time.size
    order = np.argsort(time)
    if n < 4:
        return (("full", np.ones(n, dtype=bool)),)
    cut = max(1, n // 2)
    masks = []
    for name, lo, hi in (
        ("pre", 0, cut),
        ("post", cut, n),
        ("early_wing", 0, max(1, n // 3)),
        ("late_wing", min(n - 1, 2 * n // 3), n),
    ):
        mask = np.zeros(n, dtype=bool)
        mask[order[lo:hi]] = True
        masks.append((name, mask))
    if observed_scale is not None and observed_scale.valid:
        masks.extend(
            [
                ("pre_event_wing", time < float(observed_scale.t_left)),
                ("post_event_wing", time > float(observed_scale.t_right)),
            ]
        )
    elif seed_parameters is not None and np.asarray(seed_parameters).size >= 3:
        t0, tE, u0 = (
            float(value)
            for value in np.asarray(seed_parameters, dtype=float).reshape(-1)[:3]
        )
        peak_half_width = max(abs(tE) * max(abs(u0), 0.10), 0.5)
        masks.extend(
            [
                ("pre_event_wing", time < t0 - peak_half_width),
                ("post_event_wing", time > t0 + peak_half_width),
            ]
        )
    return tuple(masks)


def parallax_score_test(
    time: np.ndarray,
    standardized_residual: np.ndarray,
    nuisance_jacobian: np.ndarray,
    parallax_jacobian: np.ndarray,
    *,
    effect: str = "annual_parallax",
    seed_parameters: Optional[np.ndarray] = None,
    compact_mask: Optional[np.ndarray] = None,
    compact_sigma: float = 5.0,
    compact_max_blocks: int = 1,
    compact_max_span: float = 2.0,
    projection_rtol: float = 1.0e-10,
    min_coverage: float = 0.20,
    planet_mask: Optional[np.ndarray] = None,
) -> EffectCandidate:
    """Run the projected linear parallax score test.

    ``parallax_jacobian`` must already be in standardized-residual units, and
    should contain the derivatives with respect to the two parallax
    components at the null model.  This low-level form is useful for unit
    tests and for annual and spacecraft geometry alike.
    """
    t = np.asarray(time, dtype=float).reshape(-1)
    z = np.asarray(standardized_residual, dtype=float).reshape(-1)
    H = _as_2d(parallax_jacobian, t.size)
    J = _as_2d(nuisance_jacobian, t.size)
    if z.size != t.size:
        raise ValueError("time and standardized_residual must have the same length.")
    if H.shape[1] == 0:
        raise ValueError("parallax_jacobian must have at least one column.")
    if compact_mask is None:
        gap = float(np.nanpercentile(np.diff(np.sort(t)), 95)) * 5.0 if t.size > 2 else None
        compact = find_compact_blocks(
            t,
            z,
            sigma=compact_sigma,
            max_blocks=compact_max_blocks,
            max_span=compact_max_span,
            max_gap=gap,
        )
    else:
        compact = np.asarray(compact_mask, dtype=bool).reshape(-1)
        if compact.size != t.size:
            raise ValueError("compact_mask must have the same length as time.")
    planet = _validated_mask(planet_mask, t.size, "planet_mask")

    # The score test is the sum over the two projected tangent directions.
    all_mask = np.isfinite(t) & np.isfinite(z) & np.all(np.isfinite(H), axis=1)
    indices = np.flatnonzero(all_mask)
    q, nuisance_diag = _orthonormal_basis(J[indices], projection_rtol)
    zp = z[indices] - q @ (q.T @ z[indices])
    Hp = H[indices] - q @ (q.T @ H[indices])
    gram = Hp.T @ Hp
    b = Hp.T @ zp
    covariance = np.linalg.pinv(gram, rcond=projection_rtol)
    beta = covariance @ b
    fitted = np.zeros_like(z)
    fitted[indices] = Hp @ beta
    score = float(b @ beta)
    energy = fitted * fitted
    observed_scale = measure_observed_signal_scale(
        t,
        fitted,
        valid_mask=all_mask,
        threshold=max(
            1.0e-3,
            0.05 * float(np.nanmax(np.abs(fitted[all_mask])))
            if np.any(all_mask)
            else 1.0e-3,
        ),
        source=f"{effect}_profile",
    )
    max_point = float(np.max(energy) / np.sum(energy)) if np.sum(energy) > 0 else 0.0
    support = np.abs(fitted) >= max(float(np.max(np.abs(fitted))) * 1.0e-3, 1.0e-12)
    coverage = float(np.count_nonzero(support) / max(indices.size, 1))
    def matrix_score(mask: np.ndarray) -> float:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return 0.0
        qs, _ = _orthonormal_basis(J[idx], projection_rtol)
        zps = z[idx] - qs @ (qs.T @ z[idx])
        hps = H[idx] - qs @ (qs.T @ H[idx])
        gs = hps.T @ hps
        bs = hps.T @ zps
        return float(bs @ (np.linalg.pinv(gs, rcond=projection_rtol) @ bs))

    without_score = matrix_score(all_mask & ~compact)
    without_planet_score = matrix_score(all_mask & ~planet)
    h_singular = np.linalg.svd(Hp, compute_uv=False) if Hp.size else np.empty(0)
    h_cutoff = projection_rtol * max(float(h_singular[0]), 1.0) if h_singular.size else np.inf
    h_nonzero = h_singular[h_singular > h_cutoff]
    effective_rank = int(h_nonzero.size)
    condition_number = float(h_nonzero[0] / h_nonzero[-1]) if h_nonzero.size else float("inf")
    full_information = float(np.trace(gram))
    subset_rows: list[dict[str, object]] = []
    for name, mask in _named_time_subsets(
        t,
        seed_parameters,
        observed_scale=observed_scale,
    ):
        use = all_mask & mask & ~planet
        if np.count_nonzero(use) < max(3, J.shape[1] + 1):
            subset_rows.append({"name": name, "valid": False, "reason": "too_few_points"})
            continue
        idx = np.flatnonzero(use)
        qs, _ = _orthonormal_basis(J[idx], projection_rtol)
        zps = z[idx] - qs @ (qs.T @ z[idx])
        hps = H[idx] - qs @ (qs.T @ H[idx])
        gs = hps.T @ hps
        bs = hps.T @ zps
        subset_covariance = np.linalg.pinv(gs, rcond=projection_rtol)
        subset_beta = subset_covariance @ bs
        subset_score = float(bs @ subset_beta)
        singular = np.linalg.svd(hps, compute_uv=False) if hps.size else np.empty(0)
        cutoff = projection_rtol * max(float(singular[0]), 1.0) if singular.size else np.inf
        nonzero = singular[singular > cutoff]
        subset_rank = int(nonzero.size)
        subset_condition = float(nonzero[0] / nonzero[-1]) if nonzero.size else float("inf")
        subset_information = float(np.trace(gs))
        subset_direction = subset_beta / max(float(np.linalg.norm(subset_beta)), 1.0e-30)
        valid = bool(
            subset_rank == effective_rank
            and subset_information >= max(full_information * 1.0e-3, 1.0e-30)
            and np.isfinite(subset_score)
        )
        subset_rows.append(
            {
                "name": name,
                "n_points": int(idx.size),
                "score": subset_score,
                "information": subset_information,
                "normalized_score": float(subset_score / max(subset_information, 1.0e-30)),
                "normalized_amplitude": float(np.sqrt(max(subset_score, 0.0) / max(subset_information, 1.0e-30))),
                "direction": np.asarray(subset_direction, dtype=float),
                "effective_rank": subset_rank,
                "condition_number": subset_condition,
                "valid": valid,
                "reason": "ok" if valid else "insufficient_subset_information",
            }
        )
    subset_rows.append(
        {
            "name": "without_compact_blocks",
            "score": float(without_score),
            "information": full_information,
            "normalized_score": float(without_score / max(full_information, 1.0e-30)),
            "retained_fraction": float(without_score / max(score, 1.0e-30)) if score > 0 else 0.0,
            "valid": bool(without_score >= 0.5 * score if score > 0 else False),
            "reason": "ok" if without_score >= 0.5 * score else "compact_block_dominated",
        }
    )
    direction = beta / max(float(np.linalg.norm(beta)), 1.0e-30)
    for row in subset_rows:
        if "direction" not in row and row.get("name") != "without_compact_blocks":
            row["direction"] = np.asarray(direction, dtype=float)
    subset_scores = tuple(float(row.get("score", 0.0)) for row in subset_rows if "score" in row)
    stability = _direction_stability(subset_rows, direction)
    wing_rows = {
        str(row.get("name")): row
        for row in subset_rows
        if row.get("name") in {"pre_event_wing", "post_event_wing"}
    }
    wing_coherent = False
    if len(wing_rows) == 2 and all(
        bool(row.get("valid", False)) for row in wing_rows.values()
    ):
        wing_directions = [
            np.asarray(row["direction"], dtype=float)
            for row in wing_rows.values()
        ]
        wing_coherent = bool(
            np.dot(wing_directions[0], wing_directions[1]) >= 0.5
        )
    if seed_parameters is not None and np.asarray(seed_parameters).size >= 2:
        event_tE = max(
            abs(float(np.asarray(seed_parameters).reshape(-1)[1])),
            1.0e-12,
        )
        duration_te = _weighted_span(t, energy) / event_tE
    else:
        duration_te = float("nan")
    duration = _weighted_span(t, energy)
    duration_over_signal_scale = (
        float(duration / max(observed_scale.width, 1.0e-12))
        if observed_scale.valid and observed_scale.width > 0.0
        else float("nan")
    )
    signal_scale_ok = bool(
        observed_scale.valid
        and not observed_scale.censored
        and np.isfinite(observed_scale.points_per_width)
        and observed_scale.points_per_width >= 6.0
    )
    subset_rows.append(
        {
            "name": "parallax_morphology",
            "duration_over_tE": duration_te,
            "duration_over_signal_scale": duration_over_signal_scale,
            "signal_scale_points_per_width": observed_scale.points_per_width,
            "signal_scale_censored": observed_scale.censored,
            "wing_coherent": wing_coherent,
            "valid": bool(wing_coherent and signal_scale_ok),
            "reason": (
                "broad_coherent_distortion"
                if wing_coherent and signal_scale_ok
                else "signal_scale_censored"
                if observed_scale.censored
                else "local_or_incoherent_distortion"
            ),
        }
    )
    reasons: list[str] = []
    if effective_rank < H.shape[1]:
        reasons.append("rank_deficient")
    if not np.isfinite(condition_number) or condition_number > 1.0e10:
        reasons.append("ill_conditioned")
    if coverage < float(min_coverage):
        reasons.append("insufficient_coverage")
    if score <= 0.0:
        reasons.append("non_positive_score")
    if compact.any() and score > 0.0 and without_score < 0.5 * score:
        reasons.append("compact_block_dominated")
    planet_overlap = _energy_overlap(fitted, planet)
    planet_retention = float(
        without_planet_score / max(score, 1.0e-30)
    ) if score > 0.0 else 0.0
    if planet.any() and (planet_retention < 0.5 or planet_overlap > 0.35):
        reasons.append("planet_morphology_dominated")
    if seed_parameters is not None and not wing_coherent:
        reasons.append("parallax_wings_incoherent")
    if observed_scale.censored:
        reasons.append("signal_scale_censored")
    elif not observed_scale.valid:
        reasons.append("signal_scale_unmeasurable")
    elif not signal_scale_ok:
        reasons.append("parallax_too_local")
    if stability < 0.25:
        reasons.append("subset_unstable")
    if not reasons:
        reasons.append("physical_support_available")
    return EffectCandidate(
        effect=effect,
        score=score,
        score_without_compact_blocks=without_score,
        effective_rank=effective_rank,
        condition_number=condition_number,
        coverage=coverage,
        max_point_influence=max_point,
        max_block_influence=_block_influence(t, z, fitted, max_span=compact_max_span),
        subset_stability=stability,
        best_template_or_direction=np.asarray(direction, dtype=float),
        seed_parameters=None if seed_parameters is None else np.asarray(seed_parameters, dtype=float),
        reason_codes=tuple(reasons),
        subset_scores=tuple(float(x) for x in subset_scores),
        subset_diagnostics=tuple(
            {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in row.items()
            }
            for row in subset_rows
        ),
        compact_block_mask=compact | planet,
        score_without_planet=float(without_planet_score),
        planet_overlap=planet_overlap,
        morphology=(
            "parallax_coherent_wings"
            if wing_coherent and signal_scale_ok
            else "planet_like"
            if (
                "planet_morphology_dominated" in reasons
                and without_planet_score < 25.0
            )
            else "mixed_or_planet"
            if "planet_morphology_dominated" in reasons
            else "ambiguous"
        ),
        observed_signal_scale=observed_scale,
    )


def _pspl_nuisance_and_parallax_jacobians(
    fit,
    time: np.ndarray,
    projector,
    *,
    space: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(fit.params, dtype=float)[:3]
    t = np.asarray(time, dtype=float)
    A0, dA_pspl = _pspl_magnification_and_jacobian(t, p)

    # Parallax offsets are exactly linear in (piEN, piEE), so evaluating the
    # two unit directions gives the tangent without autodiff or XLA.  This is
    # both cheaper and more transparent than recompiling a shape-specific JAX
    # Jacobian for every survey event.
    if space:
        d_tau, d_beta = _space_parallax_unit_offsets_numpy(t, projector)
        is_gulls = (
            type(projector).__name__ == "GullsSpaceParallaxProjector"
        )
        earth_projector = None if is_gulls else projector.earth
    else:
        d_tau, d_beta = _earth_parallax_unit_offsets_numpy(t, projector)
        is_gulls = False
        earth_projector = projector

    t0, tE, u0 = (float(value) for value in p)
    if is_gulls:
        tau = (t - t0) / tE
    else:
        light_time = _earth_light_time_numpy(t, earth_projector)
        light_time_t0 = float(
            _earth_light_time_numpy(np.asarray([t0]), earth_projector)[0]
        )
        tau = (t + light_time - t0 - light_time_t0) / tE
    beta = np.full_like(tau, u0)
    separation = np.sqrt(tau * tau + beta * beta)
    safe_u = np.maximum(separation, 1.0e-12)
    dA_du = -8.0 / (
        safe_u * safe_u * np.power(safe_u * safe_u + 4.0, 1.5)
    )
    du_dpi = (
        tau[:, None] * d_tau + beta[:, None] * d_beta
    ) / safe_u[:, None]
    dA_pi = dA_du[:, None] * du_dpi
    fs = float(np.asarray(fit.fs))
    ferr = np.maximum(np.asarray(fit.ferr, dtype=float), 1.0e-12)
    nuisance = np.column_stack(
        [
            fs * dA_pspl / ferr[:, None],
            A0 / ferr,
            np.ones_like(ferr) / ferr,
        ]
    )
    parallax = fs * dA_pi / ferr[:, None]
    z = np.asarray(fit.residual, dtype=float) / ferr
    return z, nuisance, parallax


def _interp_uniform_numpy(
    query: np.ndarray,
    x0: float,
    step: float,
    values: np.ndarray,
) -> np.ndarray:
    q = np.atleast_1d(np.asarray(query, dtype=float))
    table = np.asarray(values, dtype=float)
    coordinate = (q - float(x0)) / float(step)
    lower = np.floor(coordinate).astype(np.int64)
    lower = np.clip(lower, 0, table.shape[0] - 2)
    weight = coordinate - lower
    shape = (weight.size,) + (1,) * (table.ndim - 1)
    return table[lower] + (table[lower + 1] - table[lower]) * weight.reshape(shape)


def _interp_linear_numpy(
    query: np.ndarray,
    grid: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    q = np.atleast_1d(np.asarray(query, dtype=float))
    x = np.asarray(grid, dtype=float)
    table = np.asarray(values, dtype=float)
    upper = np.searchsorted(x, q, side="right")
    upper = np.clip(upper, 1, x.size - 1)
    lower = upper - 1
    denominator = np.where(x[upper] != x[lower], x[upper] - x[lower], 1.0)
    weight = (q - x[lower]) / denominator
    shape = (weight.size,) + (1,) * (table.ndim - 1)
    return table[lower] + (table[upper] - table[lower]) * weight.reshape(shape)


def _light_time_corrected_numpy(
    time: np.ndarray,
    projector,
) -> np.ndarray:
    observed = np.asarray(time, dtype=float)
    emitted = observed.copy()
    rv = np.asarray(projector.rv, dtype=float)
    line_of_sight = np.asarray(projector.n_hat, dtype=float)
    for _ in range(int(projector.light_time_iters)):
        current = _interp_uniform_numpy(
            emitted, float(projector.t0), float(projector.dt), rv
        )
        delay = (
            np.sum(current[:, :3] * line_of_sight, axis=-1)
            * float(projector.au_c_day)
        )
        emitted = observed - delay
    return emitted


def _earth_light_time_numpy(time: np.ndarray, projector) -> np.ndarray:
    absolute_time = np.asarray(time, dtype=float) + float(projector.time_add)
    if projector.use_HJD:
        return np.zeros_like(absolute_time)
    rv = _interp_uniform_numpy(
        absolute_time,
        float(projector.t0),
        float(projector.dt),
        np.asarray(projector.rv, dtype=float),
    )
    return (
        np.sum(rv[:, :3] * np.asarray(projector.n_hat, dtype=float), axis=-1)
        * float(projector.au_c_day)
    )


def _earth_parallax_unit_offsets_numpy(
    time: np.ndarray,
    projector,
) -> tuple[np.ndarray, np.ndarray]:
    absolute_time = np.asarray(time, dtype=float) + float(projector.time_add)
    evaluation_time = (
        _light_time_corrected_numpy(absolute_time, projector)
        if projector.use_HJD
        else absolute_time
    )
    position = _interp_uniform_numpy(
        evaluation_time,
        float(projector.t0),
        float(projector.dt),
        np.asarray(projector.rv, dtype=float),
    )[:, :3]
    east = np.asarray(projector.sky_east, dtype=float)
    north = np.asarray(projector.sky_north, dtype=float)
    projected = -np.column_stack([position @ east, position @ north])
    displacement = -(
        np.asarray(projector.E_ref, dtype=float)[None, :]
        - projected
        + np.asarray(projector.V_ref, dtype=float)[None, :]
        * (absolute_time - float(projector.tref))[:, None]
    )
    # Columns correspond to unit piEN and unit piEE.
    d_tau = np.column_stack([displacement[:, 1], displacement[:, 0]])
    d_beta = np.column_stack([displacement[:, 0], -displacement[:, 1]])
    return d_tau, d_beta


def _space_parallax_unit_offsets_numpy(
    time: np.ndarray,
    projector,
) -> tuple[np.ndarray, np.ndarray]:
    if type(projector).__name__ == "GullsSpaceParallaxProjector":
        absolute_time = np.asarray(time, dtype=float) + float(projector.time_add)
        position = _interp_linear_numpy(
            absolute_time,
            np.asarray(projector.t, dtype=float),
            np.asarray(projector.r, dtype=float),
        )
        north = np.asarray(projector.sky_north, dtype=float)
        east = np.asarray(projector.sky_east, dtype=float)
        projected = np.column_stack([position @ north, position @ east])
        displacement = (
            projected
            - np.asarray(projector.NE_ref, dtype=float)[None, :]
            - np.asarray(projector.NE_vref, dtype=float)[None, :]
            * (absolute_time - float(projector.tref))[:, None]
        )
        d_n, d_e = displacement[:, 0], displacement[:, 1]
        return (
            np.column_stack([-d_n, -d_e]),
            np.column_stack([-d_e, d_n]),
        )

    earth_tau, earth_beta = _earth_parallax_unit_offsets_numpy(
        time, projector.earth
    )
    absolute_time = (
        np.asarray(time, dtype=float) + float(projector.earth.time_add)
    )
    satellite_position = _interp_linear_numpy(
        absolute_time,
        np.asarray(projector.sat_t, dtype=float),
        np.asarray(projector.sat_r, dtype=float),
    )
    displacement = -np.column_stack(
        [
            satellite_position @ np.asarray(projector.earth.sky_east, dtype=float),
            satellite_position @ np.asarray(projector.earth.sky_north, dtype=float),
        ]
    )
    satellite_tau = np.column_stack(
        [displacement[:, 1], displacement[:, 0]]
    )
    satellite_beta = np.column_stack(
        [displacement[:, 0], -displacement[:, 1]]
    )
    return earth_tau + satellite_tau, earth_beta + satellite_beta


def detect_parallax_from_pspl_fit(
    fit,
    projector,
    *,
    space: bool = False,
    planet_mask: Optional[np.ndarray] = None,
    **kwargs,
) -> EffectCandidate:
    """Build the exact observer-geometry tangent and run a parallax score test."""
    params = np.asarray(fit.params, dtype=float)
    if params.size < 3:
        raise ValueError("A PSPL fit must contain at least (t0, tE, u0).")
    z, nuisance, parallax = _pspl_nuisance_and_parallax_jacobians(
        fit,
        np.asarray(fit.time, dtype=float),
        projector,
        space=space,
    )
    return parallax_score_test(
        np.asarray(fit.time, dtype=float),
        z,
        nuisance,
        parallax,
        effect="space_parallax" if space else "annual_parallax",
        seed_parameters=params[:3],
        planet_mask=planet_mask,
        **kwargs,
    )


@lru_cache(maxsize=1)
def _native_fspl_magnifier():
    """Return a process-local VBMicrolensing ESPL evaluator."""
    try:
        import VBMicrolensing
    except ImportError as exc:  # pragma: no cover - required package dependency
        raise ImportError(
            "VBMicrolensing>=5.5 is required for the native FSPL detector."
        ) from exc
    magnifier = VBMicrolensing.VBMicrolensing()
    magnifier.Tol = 1.0e-4
    magnifier.RelTol = 1.0e-4
    table = Path(VBMicrolensing.__file__).resolve().parent / "data" / "ESPL.tbl"
    if table.is_file():
        magnifier.LoadESPLTable(str(table))
    return magnifier


def _native_fspl_magnification(u: np.ndarray, rho: float) -> np.ndarray:
    """Evaluate an exact uniform-source curve through the native C++ backend."""
    magnifier = _native_fspl_magnifier()
    rho_safe = max(float(rho), 1.0e-12)
    values = np.asarray(
        [magnifier.ESPLMag(float(value), rho_safe) for value in np.abs(u)],
        dtype=float,
    )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise RuntimeError("VBMicrolensing returned an invalid FSPL magnification.")
    return values


def _pspl_magnification_and_jacobian(
    time: np.ndarray,
    params: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return PSPL magnification and its analytic ``(t0, tE, u0)`` Jacobian."""
    t = np.asarray(time, dtype=float).reshape(-1)
    p = np.asarray(params, dtype=float).reshape(-1)
    if p.size < 3:
        raise ValueError("params must contain (t0, tE, u0).")
    t0, tE, u0 = (float(value) for value in p[:3])
    if not np.isfinite(tE) or tE <= 0.0:
        raise ValueError("PSPL tE must be positive and finite.")
    tau = (t - t0) / tE
    separation = np.sqrt(tau * tau + u0 * u0)
    safe_u = np.maximum(separation, 1.0e-12)
    magnification = (safe_u * safe_u + 2.0) / (
        safe_u * np.sqrt(safe_u * safe_u + 4.0)
    )
    dA_du = -8.0 / (
        safe_u * safe_u * np.power(safe_u * safe_u + 4.0, 1.5)
    )
    du = np.column_stack(
        [
            -tau / (safe_u * tE),
            -(tau * tau) / (safe_u * tE),
            np.full_like(safe_u, u0) / safe_u,
        ]
    )
    return magnification, dA_du[:, None] * du


def build_fspl_template_bank(
    time: np.ndarray,
    pspl_params: Sequence[float],
    *,
    rho_over_u0: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
    tE_factors: Sequence[float] = (0.5, 1.0, 2.0),
    u0_signs: Sequence[float] = (-1.0, 1.0),
    N_fft: int = 1024,
    backend: str = "native",
    native_support_rho: Optional[float] = 10.0,
    native_support_floor: float = 3.0,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Create a joint ``tE`` / ``rho / |u0|`` FSPL template bank.

    The returned templates are magnification differences.  Each metadata row
    is ``(tE_factor, rho_over_u0, u0_sign, rho, tE, u0)`` and can be converted
    directly to a FSPL ``(t0, tE, u0, logrho)`` seed.

    ``backend="native"`` (the default) evaluates the physical model through
    VBMicrolensing's C++ implementation and does not invoke JAX.  The
    ``"microjax"`` backend is retained for cross-validation.  Opposite
    ``u0`` signs have identical rectilinear FSPL magnification, so each
    ``(tE, rho/|u0|)`` curve is evaluated and profiled once and then reused
    for the requested seed signs.  Native evaluation is restricted by
    default to ``u <= max(native_support_floor, native_support_rho * rho)``;
    outside that region the finite-source correction is asymptotically
    negligible and the exact PSPL value is used.  Pass
    ``native_support_rho=None`` to cross-check against full-curve evaluation.
    """
    t = np.asarray(time, dtype=float)
    p = np.asarray(pspl_params, dtype=float).reshape(-1)
    if p.size < 3:
        raise ValueError("pspl_params must contain (t0, tE, u0).")
    t0, tE0, u00 = p[:3]
    if tE0 <= 0.0:
        raise ValueError("PSPL tE must be positive.")
    backend_name = str(backend).lower()
    if backend_name not in {"native", "microjax"}:
        raise ValueError("backend must be 'native' or 'microjax'.")
    u0_abs = max(abs(float(u00)), 1.0e-3)
    templates: list[np.ndarray] = []
    metadata: list[np.ndarray] = []
    for tE_factor in tE_factors:
        tE = float(tE0) * float(tE_factor)
        u = np.sqrt(((t - t0) / tE) ** 2 + u0_abs * u0_abs)
        A_pspl = (u * u + 2.0) / (u * np.sqrt(u * u + 4.0))
        for ratio in rho_over_u0:
            # Match the native fitter's validated ESPL domain.  Flat or
            # weakly constrained PSPL baselines can have very large |u0|;
            # allowing rho=ratio*|u0| beyond this bound makes VBM emit one
            # diagnostic per evaluated point and carries no useful routing
            # information.
            rho = min(max(float(ratio) * u0_abs, 1.0e-6), 10.0)
            if backend_name == "native":
                if native_support_rho is None:
                    support = np.ones_like(u, dtype=bool)
                else:
                    support_limit = max(
                        float(native_support_floor),
                        float(native_support_rho) * rho,
                    )
                    support = u <= support_limit
                A_fspl = A_pspl.copy()
                if np.any(support):
                    A_fspl[support] = _native_fspl_magnification(
                        u[support], rho
                    )
            else:
                import jax.numpy as jnp
                from .magnification import A_fspl_from_u

                A_fspl = np.asarray(
                    A_fspl_from_u(jnp.asarray(u), rho, N_fft=N_fft),
                    dtype=float,
                )
            # Profile the local PSPL source and blend once per physical curve.
            # This is equivalent to the previous least-squares solve but avoids
            # repeating it for the sign-degenerate seed.
            centered = A_pspl - np.mean(A_pspl)
            denominator = float(centered @ centered)
            if denominator <= 0.0 or not np.isfinite(denominator):
                raise RuntimeError("PSPL template profile is singular.")
            slope = float(centered @ (A_fspl - np.mean(A_fspl)) / denominator)
            profiled = A_fspl - (slope * A_pspl + np.mean(A_fspl) - slope * np.mean(A_pspl))
            for sign in u0_signs:
                u0 = float(sign) * u0_abs
                templates.append(profiled)
                metadata.append(np.asarray([tE_factor, ratio, sign, rho, tE, u0], dtype=float))
    if not templates:
        raise ValueError("FSPL template bank is empty.")
    return np.asarray(templates, dtype=float), tuple(metadata)


def detect_fspl_from_pspl_fit(
    fit,
    *,
    rho_over_u0: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
    tE_factors: Sequence[float] = (0.5, 1.0, 2.0),
    u0_signs: Sequence[float] = (-1.0, 1.0),
    N_fft: int = 1024,
    backend: str = "native",
    native_support_rho: Optional[float] = 10.0,
    native_support_floor: float = 3.0,
    compact_sigma: float = 5.0,
    compact_max_blocks: int = 1,
    compact_max_span: float = 2.0,
    projection_rtol: float = 1.0e-10,
    min_coverage: float = 0.05,
    planet_mask: Optional[np.ndarray] = None,
) -> EffectCandidate:
    """Score a PSPL residual against a projected joint FSPL template bank."""
    t = np.asarray(fit.time, dtype=float)
    ferr = np.maximum(np.asarray(fit.ferr, dtype=float), 1.0e-12)
    p = np.asarray(fit.params, dtype=float)[:3]
    fs = float(np.asarray(fit.fs))
    z = np.asarray(fit.residual, dtype=float) / ferr
    A0, dA = _pspl_magnification_and_jacobian(t, p)
    nuisance = np.column_stack(
        [
            fs * dA / ferr[:, None],
            A0 / ferr,
            np.ones_like(ferr) / ferr,
        ]
    )
    bank, metadata = build_fspl_template_bank(
        t,
        p,
        rho_over_u0=rho_over_u0,
        tE_factors=tE_factors,
        u0_signs=u0_signs,
        N_fft=N_fft,
        backend=backend,
        native_support_rho=native_support_rho,
        native_support_floor=native_support_floor,
    )
    compact = find_compact_blocks(
        t,
        z,
        sigma=compact_sigma,
        max_blocks=compact_max_blocks,
        max_span=compact_max_span,
    )
    standardized_bank = fs * bank / ferr[None, :]
    valid = (
        np.isfinite(t)
        & np.isfinite(z)
        & np.all(np.isfinite(nuisance), axis=1)
        & np.all(np.isfinite(standardized_bank), axis=0)
    )
    scores = _projected_template_scores(
        z,
        standardized_bank,
        nuisance,
        valid,
        rtol=projection_rtol,
    )
    best_index = int(np.argmax(scores)) if scores.size else -1
    if best_index < 0 or not np.isfinite(scores[best_index]):
        raise RuntimeError("FSPL template bank produced no finite score.")
    h_best = standardized_bank[best_index]
    seed_meta = metadata[best_index]
    seed = np.asarray([p[0], seed_meta[4], seed_meta[5], np.log(seed_meta[3])], dtype=float)
    return _candidate_from_template(
        effect="fspl",
        time=t,
        z=z,
        nuisance_jacobian=nuisance,
        template=h_best,
        seed_parameters=seed,
        best_template_or_direction=seed_meta,
        compact_mask=compact,
        subset_masks=_time_subsets(t),
        rtol=projection_rtol,
        max_compact_span=compact_max_span,
        min_coverage=min_coverage,
        planet_mask=planet_mask,
    )


def detect_physical_effects(
    fit,
    *,
    parallax_projector=None,
    space_parallax_projector=None,
    include_fspl: bool = True,
    skip_unavailable: bool = True,
    planet_mask: Optional[np.ndarray] = None,
    **fspl_kwargs,
) -> tuple[EffectCandidate, ...]:
    """Run available detector-only probes against a PSPL fit.

    FSPL magnification is an optional runtime dependency in this repository.
    When ``skip_unavailable`` is true, a missing FSPL backend leaves the
    parallax detector usable and the direct FSPL function still reports the
    dependency error to callers that explicitly request it.
    """
    candidates: list[EffectCandidate] = []
    if parallax_projector is not None:
        candidates.append(
            detect_parallax_from_pspl_fit(
                fit, parallax_projector, planet_mask=planet_mask
            )
        )
    if space_parallax_projector is not None:
        candidates.append(
            detect_parallax_from_pspl_fit(
                fit,
                space_parallax_projector,
                space=True,
                planet_mask=planet_mask,
            )
        )
    if include_fspl:
        try:
            candidates.append(
                detect_fspl_from_pspl_fit(
                    fit, planet_mask=planet_mask, **fspl_kwargs
                )
            )
        except ImportError:
            if not skip_unavailable:
                raise
    return tuple(candidates)


__all__ = [
    "EffectCandidate",
    "ProjectionDiagnostics",
    "build_fspl_template_bank",
    "detect_fspl_from_pspl_fit",
    "detect_parallax_from_pspl_fit",
    "detect_physical_effects",
    "find_compact_blocks",
    "parallax_score_test",
    "project_out_nuisance",
]
