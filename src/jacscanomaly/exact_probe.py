"""Cheap exact forward-model probes between detection and nonlinear fallback."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional, Sequence

import jax.numpy as jnp
import numpy as np

from .effect_detection import EffectCandidate, build_fspl_template_bank
from .magnification import A_fspl_from_u
from .singlelens_model import (
    A_pspl_func,
    A_pspl_parallax_func,
    A_pspl_space_parallax_func,
)
from .trajectory import u_rectilinear


@dataclass(frozen=True)
class ExactProbeEvaluation:
    """One exact template evaluation with profiled flux parameters."""

    effect: str
    parameters: np.ndarray
    chi2: float
    delta_chi2: float
    fs: float
    fb: float
    n_points: int


@dataclass(frozen=True)
class ExactProbeResult:
    """Summary of an exact probe run."""

    pre_candidate: EffectCandidate
    best: Optional[ExactProbeEvaluation]
    evaluations: tuple[ExactProbeEvaluation, ...]
    promoted_candidate: EffectCandidate
    decision: str
    reason_codes: tuple[str, ...]
    runtime_seconds: float


def _solve_fs_fb(A: np.ndarray, flux: np.ndarray, ferr: np.ndarray) -> tuple[float, float]:
    w = 1.0 / np.maximum(ferr, 1.0e-12) ** 2
    x_mean = float(np.sum(w * A) / np.sum(w))
    y_mean = float(np.sum(w * flux) / np.sum(w))
    xc = A - x_mean
    yc = flux - y_mean
    denom = float(np.sum(w * xc * xc))
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan"), float("nan")
    fs = float(np.sum(w * xc * yc) / denom)
    return fs, float(y_mean - fs * x_mean)


def _evaluate(
    A: np.ndarray,
    fit,
    effect: str,
    parameters: Sequence[float],
    base_chi2: float,
    indices: Optional[np.ndarray] = None,
) -> ExactProbeEvaluation:
    A = np.asarray(A, dtype=float)
    if indices is None:
        flux = np.asarray(fit.flux, dtype=float)
        ferr = np.maximum(np.asarray(fit.ferr, dtype=float), 1.0e-12)
    else:
        indices = np.asarray(indices, dtype=int)
        flux = np.asarray(fit.flux, dtype=float)[indices]
        ferr = np.maximum(np.asarray(fit.ferr, dtype=float), 1.0e-12)[indices]
        if A.size != indices.size:
            A = A[indices]
    fs, fb = _solve_fs_fb(A, flux, ferr)
    if not np.isfinite(fs) or not np.isfinite(fb):
        chi2 = float("inf")
    else:
        chi2 = float(np.sum(((flux - (fs * A + fb)) / ferr) ** 2))
    return ExactProbeEvaluation(
        effect=effect,
        parameters=np.asarray(parameters, dtype=float),
        chi2=chi2,
        delta_chi2=float(base_chi2 - chi2),
        fs=fs,
        fb=fb,
        n_points=int(flux.size),
    )


def _candidate_result(
    pre: EffectCandidate,
    best: Optional[ExactProbeEvaluation],
    evaluations: list[ExactProbeEvaluation],
    started: float,
    threshold: float,
) -> ExactProbeResult:
    if best is None or not np.isfinite(best.delta_chi2) or best.delta_chi2 < float(threshold):
        promoted = pre.with_decision("skip", ("exact_probe_completed", "exact_probe_not_promoted"))
        decision = "skip"
        reasons = ("exact_probe_completed", "exact_probe_not_promoted")
    else:
        promoted = pre.with_probe(
            score=best.delta_chi2,
            seed_parameters=best.parameters,
            decision="fallback",
            reason_codes=("exact_probe_completed", "exact_probe_promoted"),
        )
        decision = "fallback"
        reasons = ("exact_probe_completed", "exact_probe_promoted")
    return ExactProbeResult(
        pre_candidate=pre,
        best=best,
        evaluations=tuple(evaluations),
        promoted_candidate=promoted,
        decision=decision,
        reason_codes=reasons,
        runtime_seconds=float(perf_counter() - started),
    )


def run_parallax_exact_probe(
    fit,
    projector,
    candidate: EffectCandidate,
    *,
    space: bool = False,
    radii: Sequence[float] = (0.0, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0),
    improvement_threshold: float = 9.0,
    **_,
) -> ExactProbeResult:
    """Evaluate a small signed direction/radius atlas with linear flux solves."""
    if projector is None:
        raise ValueError("A parallax exact probe requires an observer projector.")
    started = perf_counter()
    t = np.asarray(fit.time, dtype=float)
    base = np.asarray(fit.params, dtype=float)[:3]
    direction = np.asarray(candidate.best_template_or_direction, dtype=float).reshape(-1)
    if direction.size < 2 or not np.all(np.isfinite(direction[:2])):
        direction = np.asarray([1.0, 0.0])
    direction = direction[:2] / max(float(np.linalg.norm(direction[:2])), 1.0e-30)
    dirs = [direction, -direction, np.asarray([-direction[1], direction[0]]), np.asarray([direction[1], -direction[0]])]
    model = A_pspl_space_parallax_func if space else A_pspl_parallax_func
    ferr = np.maximum(np.asarray(fit.ferr, dtype=float), 1.0e-12)
    base_A = np.asarray(A_pspl_func(jnp.asarray(base), jnp.asarray(t)), dtype=float)
    base_model = float(np.asarray(fit.fs)) * base_A + float(np.asarray(fit.fb))
    base_chi2 = float(np.sum(((np.asarray(fit.flux) - base_model) / ferr) ** 2))
    evaluations: list[ExactProbeEvaluation] = []
    seen: set[tuple[float, ...]] = set()
    for radius in radii:
        for trial_direction in dirs:
            parameters = np.r_[base, float(radius) * trial_direction]
            key = tuple(np.round(parameters, 12))
            if key in seen:
                continue
            seen.add(key)
            A = np.asarray(model(jnp.asarray(parameters), jnp.asarray(t), projector), dtype=float)
            evaluations.append(_evaluate(A, fit, candidate.effect, parameters, base_chi2))
    best = max(evaluations, key=lambda row: row.delta_chi2, default=None)
    return _candidate_result(candidate, best, evaluations, started, improvement_threshold)


def run_fspl_exact_probe(
    fit,
    candidate: EffectCandidate,
    *,
    rho_over_u0: Sequence[float] = (0.2, 0.25, 0.35, 0.5, 0.75, 1.0, 1.25, 1.4, 1.5, 2.0, 3.0, 4.0),
    tE_factors: Sequence[float] = (0.4, 0.6, 0.75, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0),
    t0_offsets: Sequence[float] = (-0.5, -0.25, 0.0, 0.25, 0.5),
    N_fft: int = 1024,
    improvement_threshold: float = 9.0,
    **_,
) -> ExactProbeResult:
    """Evaluate an exact FSPL joint grid and profile only ``fs`` / ``fb``."""
    started = perf_counter()
    t = np.asarray(fit.time, dtype=float)
    base = np.asarray(fit.params, dtype=float)[:3]
    candidate_seed = base if candidate.seed_parameters is None else np.asarray(candidate.seed_parameters, dtype=float)
    rho_seed = (
        np.exp(candidate_seed[3]) if candidate_seed.size >= 4 and candidate_seed[3] < 0.0
        else (candidate_seed[3] if candidate_seed.size >= 4 else abs(candidate_seed[2]))
    )
    width = max(4.0 * abs(float(candidate_seed[1])) * max(abs(float(candidate_seed[2])), float(rho_seed)), 0.5)
    probe_indices = np.flatnonzero(np.abs(t - float(candidate_seed[0])) <= width)
    if probe_indices.size < 64:
        nearest = np.argsort(np.abs(t - float(candidate_seed[0])))[: min(t.size, 256)]
        probe_indices = np.sort(nearest)
    probe_time = t[probe_indices]
    bank, metadata = build_fspl_template_bank(
        probe_time, base, rho_over_u0=rho_over_u0, tE_factors=tE_factors,
        u0_signs=(-1.0, 1.0), N_fft=N_fft,
    )
    base_A = np.asarray(A_pspl_func(jnp.asarray(base), jnp.asarray(probe_time)), dtype=float)
    ferr = np.maximum(np.asarray(fit.ferr, dtype=float), 1.0e-12)[probe_indices]
    base_model = float(np.asarray(fit.fs)) * base_A + float(np.asarray(fit.fb))
    base_chi2 = float(np.sum(((np.asarray(fit.flux)[probe_indices] - base_model) / ferr) ** 2))
    evaluations: list[ExactProbeEvaluation] = []
    for _, meta in zip(bank, metadata):
        _, _, _, rho, tE, u0 = meta
        for offset in t0_offsets:
            t0 = float(base[0] + offset * max(abs(float(base[1])) * abs(float(base[2])), 1.0e-6))
            q = np.asarray([t0, tE, u0, np.log(rho)], dtype=float)
            u = np.asarray(u_rectilinear(t0, tE, u0, jnp.asarray(probe_time)), dtype=float)
            A = np.asarray(A_fspl_from_u(jnp.asarray(u), rho, N_fft=N_fft), dtype=float)
            evaluations.append(_evaluate(A, fit, candidate.effect, q, base_chi2, probe_indices))
    best = max(evaluations, key=lambda row: row.delta_chi2, default=None)
    return _candidate_result(candidate, best, evaluations, started, improvement_threshold)


def run_exact_probe(fit, projector, candidate: EffectCandidate, **kwargs) -> ExactProbeResult:
    """Dispatch an exact probe by effect class."""
    if candidate.effect == "fspl":
        return run_fspl_exact_probe(fit, candidate, **kwargs)
    return run_parallax_exact_probe(
        fit, projector, candidate, space=candidate.effect == "space_parallax", **kwargs
    )


__all__ = [
    "ExactProbeEvaluation",
    "ExactProbeResult",
    "run_exact_probe",
    "run_fspl_exact_probe",
    "run_parallax_exact_probe",
]
