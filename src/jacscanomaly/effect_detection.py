"""Physical residual detectors for finite-source and parallax effects.

The detectors in this module are deliberately cheaper than a nonlinear
single-lens fit.  They operate on a PSPL fit, remove the local PSPL nuisance
subspace, and then measure how much of the remaining standardized residual can
be explained by a physical tangent or template.

The public functions accept NumPy-like arrays wherever possible.  JAX is used
only to obtain the model Jacobians and the existing magnification functions;
the projection and score calculations are small CPU linear-algebra problems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .singlelens_model import (
    A_pspl_func,
    A_pspl_parallax_func,
    A_pspl_space_parallax_func,
)
from .magnification import A_fspl_from_u
from .trajectory import u_rectilinear


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
            "decision": self.decision,
            "reason_codes": list(self.reason_codes),
            "subset_diagnostics": [dict(row) for row in self.subset_diagnostics],
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
        return _ProjectedScore(0.0, 0, float("inf"), np.zeros_like(z), np.zeros_like(z), 0.0)

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
            np.zeros_like(z),
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
        width = max(3.0 * rho * tE, 0.25 * u0 * tE, 1.0e-12)
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
    reasons: list[str] = []
    if full.rank <= 0:
        reasons.append("rank_deficient")
    if not np.isfinite(full.condition_number) or full.condition_number > 1.0e10:
        reasons.append("ill_conditioned")
    if coverage < float(min_coverage):
        reasons.append("insufficient_coverage")
    if full.score <= 0.0:
        reasons.append("non_positive_score")
    if compact_mask.any() and full.score > 0.0 and without.score < 0.5 * full.score:
        reasons.append("compact_block_dominated")
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
        compact_block_mask=compact_mask,
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


def _named_time_subsets(time: np.ndarray) -> tuple[tuple[str, np.ndarray], ...]:
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
    h_singular = np.linalg.svd(Hp, compute_uv=False) if Hp.size else np.empty(0)
    h_cutoff = projection_rtol * max(float(h_singular[0]), 1.0) if h_singular.size else np.inf
    h_nonzero = h_singular[h_singular > h_cutoff]
    effective_rank = int(h_nonzero.size)
    condition_number = float(h_nonzero[0] / h_nonzero[-1]) if h_nonzero.size else float("inf")
    full_information = float(np.trace(gram))
    subset_rows: list[dict[str, object]] = []
    for name, mask in _named_time_subsets(t):
        use = all_mask & mask
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
        compact_block_mask=compact,
    )


def _pspl_nuisance_and_parallax_jacobians(
    fit,
    time: np.ndarray,
    projector,
    *,
    space: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = jnp.asarray(np.asarray(fit.params, dtype=float)[:3])
    t = jnp.asarray(time)
    if space:
        def model_fn(q):
            return A_pspl_space_parallax_func(q, t, projector)
    else:
        def model_fn(q):
            return A_pspl_parallax_func(q, t, projector)
    A0 = A_pspl_func(p, t)
    dA_pspl = jax.jacfwd(lambda q: A_pspl_func(q, t))(p)
    p0 = jnp.concatenate([p, jnp.zeros(2, dtype=p.dtype)])
    dA_pi = jax.jacfwd(model_fn)(p0)[:, 3:]
    fs = float(np.asarray(fit.fs))
    ferr = np.maximum(np.asarray(fit.ferr, dtype=float), 1.0e-12)
    nuisance = np.column_stack(
        [
            fs * np.asarray(dA_pspl) / ferr[:, None],
            np.asarray(A0) / ferr,
            np.ones_like(ferr) / ferr,
        ]
    )
    parallax = fs * np.asarray(dA_pi) / ferr[:, None]
    z = np.asarray(fit.residual, dtype=float) / ferr
    return z, nuisance, parallax


def detect_parallax_from_pspl_fit(
    fit,
    projector,
    *,
    space: bool = False,
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
        **kwargs,
    )


def build_fspl_template_bank(
    time: np.ndarray,
    pspl_params: Sequence[float],
    *,
    rho_over_u0: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
    tE_factors: Sequence[float] = (0.5, 1.0, 2.0),
    u0_signs: Sequence[float] = (-1.0, 1.0),
    N_fft: int = 1024,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Create a joint ``tE`` / ``rho / |u0|`` FSPL template bank.

    The returned templates are magnification differences.  Each metadata row
    is ``(tE_factor, rho_over_u0, u0_sign, rho, tE, u0)`` and can be converted
    directly to a FSPL ``(t0, tE, u0, logrho)`` seed.
    """
    t = np.asarray(time, dtype=float)
    p = np.asarray(pspl_params, dtype=float).reshape(-1)
    if p.size < 3:
        raise ValueError("pspl_params must contain (t0, tE, u0).")
    t0, tE0, u00 = p[:3]
    if tE0 <= 0.0:
        raise ValueError("PSPL tE must be positive.")
    u0_abs = max(abs(float(u00)), 1.0e-3)
    templates: list[np.ndarray] = []
    metadata: list[np.ndarray] = []
    for tE_factor in tE_factors:
        tE = float(tE0) * float(tE_factor)
        for ratio in rho_over_u0:
            rho = max(float(ratio) * u0_abs, 1.0e-6)
            for sign in u0_signs:
                u0 = float(sign) * u0_abs
                u = u_rectilinear(t0, tE, u0, jnp.asarray(t))
                A_fspl = np.asarray(A_fspl_from_u(u, rho, N_fft=N_fft), dtype=float)
                A_pspl = np.asarray(A_pspl_func(jnp.asarray([t0, tE, u0]), jnp.asarray(t)), dtype=float)
                # Profile the local PSPL nuisance for every FSPL curve.  A
                # raw ``A_fspl - A_pspl`` difference is biased when the best
                # local source/blend and PSPL amplitude differ, especially
                # for rho/|u0| families away from the baseline seed.
                design = np.column_stack([A_pspl, np.ones_like(A_pspl)])
                coefficients, *_ = np.linalg.lstsq(design, A_fspl, rcond=None)
                templates.append(A_fspl - design @ coefficients)
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
    compact_sigma: float = 5.0,
    compact_max_blocks: int = 1,
    compact_max_span: float = 2.0,
    projection_rtol: float = 1.0e-10,
    min_coverage: float = 0.05,
) -> EffectCandidate:
    """Score a PSPL residual against a projected joint FSPL template bank."""
    t = np.asarray(fit.time, dtype=float)
    ferr = np.maximum(np.asarray(fit.ferr, dtype=float), 1.0e-12)
    p = np.asarray(fit.params, dtype=float)[:3]
    fs = float(np.asarray(fit.fs))
    z = np.asarray(fit.residual, dtype=float) / ferr
    A0 = np.asarray(A_pspl_func(jnp.asarray(p), jnp.asarray(t)), dtype=float)
    dA = np.asarray(jax.jacfwd(lambda q: A_pspl_func(q, jnp.asarray(t)))(jnp.asarray(p)))
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
    )
    compact = find_compact_blocks(
        t,
        z,
        sigma=compact_sigma,
        max_blocks=compact_max_blocks,
        max_span=compact_max_span,
    )
    best_score = -np.inf
    best_index = -1
    best_projected: Optional[_ProjectedScore] = None
    for i, template in enumerate(bank):
        h = fs * template / ferr
        score = _projected_score(
            z,
            h,
            nuisance,
            np.isfinite(t) & np.isfinite(z) & np.isfinite(h),
            rtol=projection_rtol,
        )
        if score.score > best_score:
            best_score = score.score
            best_index = i
            best_projected = score
    if best_index < 0 or best_projected is None:
        raise RuntimeError("FSPL template bank produced no finite score.")
    h_best = fs * bank[best_index] / ferr
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
    )


def detect_physical_effects(
    fit,
    *,
    parallax_projector=None,
    space_parallax_projector=None,
    include_fspl: bool = True,
    skip_unavailable: bool = True,
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
        candidates.append(detect_parallax_from_pspl_fit(fit, parallax_projector))
    if space_parallax_projector is not None:
        candidates.append(
            detect_parallax_from_pspl_fit(fit, space_parallax_projector, space=True)
        )
    if include_fspl:
        try:
            candidates.append(detect_fspl_from_pspl_fit(fit, **fspl_kwargs))
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
