"""Detector-seeded, contamination-aware single-lens fallback fitting."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Optional, Sequence

import numpy as np

from .contamination import (
    ContaminationConfig,
    RobustFitResult,
    protected_support_mask,
    robust_refine_with_fitter,
    scaled_parameter_distance,
)
from .effect_detection import EffectCandidate


@dataclass(frozen=True)
class FallbackConfig:
    """Controls structured multistart and robust fallback effort."""

    tE_factors: tuple[float, ...] = (0.75, 1.0, 1.5)
    u0_sign_flip: bool = True
    parallax_radii: tuple[float, ...] = (0.0, 0.05, 0.2, 0.5)
    parallax_angle_steps: int = 8
    max_seeds: int = 24
    contamination: ContaminationConfig = ContaminationConfig()
    max_point_parameter_change: float = 10.0
    max_basin_distance: float = 0.5
    parameter_dimension: Optional[int] = None
    default_logrho: float = -3.0
    u0_factors: tuple[float, ...] = (0.7, 1.0, 1.4)
    rho_over_u0: tuple[float, ...] = (0.35, 0.75, 1.5, 3.0)
    t_star_factors: tuple[float, ...] = (0.6, 1.0, 1.6)
    t0_offsets: tuple[float, ...] = (-0.25, 0.0, 0.25)
    max_piE: Optional[float] = None
    min_bic_improvement: float = 0.0
    # A compact FSPL crossing can be excluded from the complementary
    # (non-planet) selection set.  In that case the clean-region BIC is
    # intentionally blind to the very samples that distinguish FSPL from
    # PSPL.  Permit a rescue only when a numerically valid FSPL fit also wins
    # decisively on the complete light curve; the relatively large default
    # keeps this from turning a small local PSPL adjustment into an FSPL.
    min_fspl_full_bic_improvement: float = 1.0e3
    min_coherent_parallax_tE: float = 20.0


@dataclass(frozen=True)
class FallbackAttempt:
    """One detector seed and its robust fit result."""

    seed: np.ndarray
    result: RobustFitResult
    objective: float
    stable: bool
    original_chi2: float = float("inf")
    robust_objective: float = float("inf")
    contamination_penalty: float = float("inf")
    parameter_distance: float = float("inf")
    optimizer_success: bool = False
    parameter_at_bound: bool = False
    segmentation_stable: bool = False


@dataclass(frozen=True)
class FallbackResult:
    """Best robust fallback and all attempted seeds."""

    fit: object
    initial_fit: object
    effect: str
    attempts: tuple[FallbackAttempt, ...]
    selected_seed: np.ndarray
    success: bool
    reason_codes: tuple[str, ...]
    baseline_original_chi2: float = float("inf")
    selected_original_chi2: float = float("inf")
    selected_robust_objective: float = float("inf")
    baseline_effect_score: Optional[float] = None
    selected_effect_score: Optional[float] = None
    baseline_bic: float = float("inf")
    selected_bic: float = float("inf")
    bic_improvement: float = float("-inf")
    numerically_valid: bool = False
    model_spec: Optional[dict[str, object]] = None
    stage_results: tuple["FallbackResult", ...] = ()


@dataclass(frozen=True)
class EffectFitterSpec:
    """Effect-specific fitter and its raw parameter contract."""

    effect: str
    fitter: object
    parameter_dimension: int
    parameter_names: tuple[str, ...]
    raw_parameter_names: tuple[str, ...]
    backend: str
    convention: str


def _require_parallax_config(config, effect: str, tref: float) -> tuple[float, float, float]:
    if config.ra_deg is None or config.dec_deg is None:
        raise ValueError(f"{effect} fallback requires ra_deg and dec_deg.")
    resolved_tref = float(config.tref if config.tref is not None else tref)
    return float(config.ra_deg), float(config.dec_deg), resolved_tref


def make_effect_fitter(config, effect: str, tref: float) -> EffectFitterSpec:
    """Construct the fallback model specified by an effect candidate.

    This is intentionally separate from the baseline fitter stored on
    ``Finder``.  It validates sky/ephemeris metadata and exposes the parameter
    dimension before any fit call is attempted.
    """
    from .singlelens_fit import CPPVBMFSPLFitter, FSPLFitter
    from .parallax_backend import (
        Ephemeris,
        NativeFSPLAnnualParallaxFitter,
        NativeFSPLSpaceParallaxFitter,
        NativePSPLAnnualParallaxFitter,
        NativePSPLSpaceParallaxFitter,
        TimeSpec,
        default_earth_ephemeris,
        load_vbm_satellite_ephemeris,
    )

    effect = str(effect)
    if effect == "mixed":
        raise ValueError("mixed requires an explicit resolved effect before fitter construction.")
    backend = str(getattr(config, "single_fit_backend", "jax"))
    convention = str(getattr(config, "parallax_observer_convention", "earth_geocentric_offset"))
    if convention not in {
        "earth_geocentric_offset",
        "heliocentric_observer",
        "gulls",
    }:
        raise ValueError(f"Unsupported parallax_observer_convention={convention!r}.")
    time_spec = TimeSpec(
        scale=str(getattr(config, "parallax_time_scale", "jd")),
        offset=float(getattr(config, "parallax_time_offset", 0.0)),
    )
    extrapolation = str(getattr(config, "parallax_extrapolation", "reject"))
    earth = getattr(config, "parallax_earth_ephemeris", None)
    if earth is None:
        earth = default_earth_ephemeris(time_spec=time_spec)
    if getattr(earth, "extrapolation", extrapolation) != extrapolation:
        earth = replace(earth, extrapolation=extrapolation)

    def native_space_inputs():
        path = getattr(config, "satellite_ephemeris_path", None)
        if path is None:
            raise ValueError("space parallax fallback requires satellite_ephemeris_path.")
        satellite = load_vbm_satellite_ephemeris(
            path,
            time_spec=time_spec,
            extrapolation=extrapolation,
        )
        if convention == "earth_geocentric_offset":
            return earth, satellite, None
        observer = getattr(config, "parallax_observer_ephemeris", None)
        reference = getattr(config, "parallax_reference_ephemeris", None)
        if observer is None:
            # RTModel/GULLS satellite tables are Earth-relative perturbations.
            # Build a complete observer orbit explicitly at the table times.
            earth_r = np.column_stack([np.interp(satellite.time, earth.time, earth.position_au[:, j]) for j in range(3)])
            observer = Ephemeris(satellite.time, earth_r + satellite.position_au, origin="explicit_reference", time_spec=time_spec)
        if reference is None:
            reference = earth
        return earth, observer, reference

    if effect == "fspl":
        # The detector/exact-probe may profile only the peak, but the fallback
        # must refit the full light curve so the local profile window cannot
        # freeze the optimizer into a false convergence.
        if backend in {"cpp", "vbm_cpp"}:
            # Production runs are C++-only. Missing native support must fail
            # loudly instead of silently switching numerical backends.
            fitter = CPPVBMFSPLFitter()
            resolved_backend = "native_vbm_cpp_lm"
        else:
            fitter = FSPLFitter(profile_peak_only=False)
            resolved_backend = "jax"
        return EffectFitterSpec(effect, fitter, 4, ("t0", "tE", "u0", "rho"), ("t0", "tE", "u0", "logrho"), resolved_backend, convention)

    if effect == "annual_parallax":
        ra, dec, resolved_tref = _require_parallax_config(config, effect, tref)
        fitter = NativePSPLAnnualParallaxFitter(
            ra, dec, resolved_tref, time_spec=time_spec, earth_ephemeris=earth,
            maxiter=int(getattr(config, "vbm_cpp_maxiter", 300)), max_piE=float(config.max_piE),
        )
        return EffectFitterSpec(effect, fitter, 5, ("t0", "tE", "u0", "piEN", "piEE"), ("t0", "log_tE", "u0", "piEN", "piEE"), "native_cpp_scipy_trf", convention)

    if effect == "space_parallax":
        ra, dec, resolved_tref = _require_parallax_config(config, effect, tref)
        earth_input, observer_input, reference_input = native_space_inputs()
        fitter = NativePSPLSpaceParallaxFitter(
            ra, dec, resolved_tref, observer_input, convention=convention,
            time_spec=time_spec, earth_ephemeris=earth_input, reference_ephemeris=reference_input,
            maxiter=int(getattr(config, "vbm_cpp_maxiter", 300)), max_piE=float(config.max_piE),
        )
        return EffectFitterSpec(effect, fitter, 5, ("t0", "tE", "u0", "piEN", "piEE"), ("t0", "log_tE", "u0", "piEN", "piEE"), "native_cpp_scipy_trf", convention)

    if effect == "fspl_parallax":
        ra, dec, resolved_tref = _require_parallax_config(config, effect, tref)
        fitter = NativeFSPLAnnualParallaxFitter(
            ra, dec, resolved_tref, time_spec=time_spec, earth_ephemeris=earth,
            maxiter=int(getattr(config, "vbm_cpp_maxiter", 300)), max_piE=float(config.max_piE),
        )
        return EffectFitterSpec(effect, fitter, 6, ("t0", "tE", "u0", "rho", "piEN", "piEE"), ("t0", "log_tE", "u0", "log_rho", "piEN", "piEE"), "native_cpp_scipy_trf", convention)

    if effect == "fspl_space_parallax":
        ra, dec, resolved_tref = _require_parallax_config(config, effect, tref)
        earth_input, observer_input, reference_input = native_space_inputs()
        fitter = NativeFSPLSpaceParallaxFitter(
            ra, dec, resolved_tref, observer_input, convention=convention,
            time_spec=time_spec, earth_ephemeris=earth_input, reference_ephemeris=reference_input,
            maxiter=int(getattr(config, "vbm_cpp_maxiter", 300)), max_piE=float(config.max_piE),
        )
        return EffectFitterSpec(effect, fitter, 6, ("t0", "tE", "u0", "rho", "piEN", "piEE"), ("t0", "log_tE", "u0", "log_rho", "piEN", "piEE"), "native_cpp_scipy_trf", convention)
    raise ValueError(f"Unknown fallback effect '{effect}'.")


def _normalise_seed(seed: Sequence[float]) -> np.ndarray:
    value = np.asarray(seed, dtype=float).reshape(-1).copy()
    if value.size < 3:
        raise ValueError("A fallback seed must contain at least (t0, tE, u0).")
    if value[1] <= 0.0 or not np.all(np.isfinite(value)):
        raise ValueError("Fallback seeds must be finite and have positive tE.")
    return value


def _coerce_seed_dimension(seed: Sequence[float], dimension: Optional[int], default_logrho: float) -> np.ndarray:
    """Convert a physical/raw seed to exactly one fitter parameter dimension."""
    value = _normalise_seed(seed)
    if dimension is None or value.size == int(dimension):
        return value
    dimension = int(dimension)
    if value.size == 3 and dimension == 4:
        return np.r_[value, float(default_logrho)]
    if value.size == 3 and dimension == 5:
        return np.r_[value, 0.0, 0.0]
    if value.size == 3 and dimension == 6:
        return np.r_[value, float(default_logrho), 0.0, 0.0]
    if value.size == 4 and dimension == 5:
        return np.r_[value[:3], 0.0, 0.0]
    if value.size == 4 and dimension == 6:
        return np.r_[value, 0.0, 0.0]
    if value.size == 5 and dimension == 6:
        return np.r_[value[:3], float(default_logrho), value[3:]]
    raise ValueError(
        f"Cannot coerce seed with dimension {value.size} to required fitter dimension {dimension}."
    )


def detector_seed_parameters(
    base_seed: Sequence[float],
    candidates: Iterable[EffectCandidate] = (),
    *,
    config: FallbackConfig = FallbackConfig(),
) -> tuple[np.ndarray, ...]:
    """Expand detector outputs into a bounded structured multistart set."""
    base = _coerce_seed_dimension(base_seed, config.parameter_dimension, config.default_logrho)
    dimension = config.parameter_dimension or base.size
    seeds: list[np.ndarray] = [base]
    fspl_anchors: list[np.ndarray] = []
    parallax_anchors: list[np.ndarray] = []
    detector_anchors: list[np.ndarray] = []
    for effect in candidates:
        if effect.seed_parameters is None:
            continue
        detector_seed = np.asarray(effect.seed_parameters, dtype=float).reshape(-1)
        if detector_seed.size < 3 or not np.all(np.isfinite(detector_seed[:3])):
            raise ValueError("Detector seed must contain finite (t0, tE, u0).")
        if "parallax" in effect.effect and detector_seed.size == 3:
            direction = np.asarray(effect.best_template_or_direction, dtype=float).reshape(-1)
            if direction.size < 2 or not np.all(np.isfinite(direction[:2])):
                direction = np.zeros(2, dtype=float)
            norm = max(float(np.linalg.norm(direction[:2])), 1.0e-30)
            direction = direction[:2] / norm
            angles = np.linspace(0.0, 2.0 * np.pi, max(1, int(config.parallax_angle_steps)), endpoint=False)
            for radius in config.parallax_radii:
                for angle in angles:
                    rotated = np.asarray(
                        [
                            direction[0] * np.cos(angle) - direction[1] * np.sin(angle),
                            direction[0] * np.sin(angle) + direction[1] * np.cos(angle),
                        ],
                        dtype=float,
                    )
                    parallax_seed = _coerce_seed_dimension(
                        np.r_[detector_seed[:3], float(radius) * rotated],
                        config.parameter_dimension,
                        config.default_logrho,
                    )
                    seeds.append(parallax_seed)
                    detector_anchors.append(parallax_seed)
                    if dimension == 6:
                        parallax_anchors.append(parallax_seed)
        else:
            physical_seed = _coerce_seed_dimension(
                detector_seed,
                config.parameter_dimension,
                config.default_logrho,
            )
            seeds.append(physical_seed)
            detector_anchors.append(physical_seed)
            if dimension == 6 and effect.effect == "fspl":
                fspl_anchors.append(physical_seed)
            if effect.effect == "fspl" and dimension in (4, 6):
                # Independent starts near the exact-probe basin must appear
                # before the broad PSPL atlas; otherwise a small seed budget
                # contains one good FSPL seed and cannot reproduce its basin.
                local_variants: list[np.ndarray] = []
                for factor in sorted(
                    config.tE_factors,
                    key=lambda value: abs(np.log(max(float(value), 1.0e-12))),
                ):
                    if np.isclose(float(factor), 1.0):
                        continue
                    variant = physical_seed.copy()
                    variant[1] *= float(factor)
                    local_variants.append(variant)
                    break
                for factor in sorted(
                    config.u0_factors,
                    key=lambda value: abs(np.log(max(float(value), 1.0e-12))),
                ):
                    if np.isclose(float(factor), 1.0):
                        continue
                    variant = physical_seed.copy()
                    variant[2] *= float(factor)
                    local_variants.append(variant)
                    break
                # PSPL refits can partially absorb a finite-source peak and
                # bias the template-bank rho by an order-unity factor. Cover
                # both nearby and factor-e basins before spending the bounded
                # seed budget on the broad PSPL atlas.
                for delta_logrho in (-0.5, 0.5, 1.0):
                    variant = physical_seed.copy()
                    variant[3] += delta_logrho
                    local_variants.append(variant)
                if config.u0_sign_flip:
                    # Duplicate the large-rho basin across the exact FSPL
                    # u0-sign degeneracy so a bounded eight-seed run can
                    # independently reproduce the same physical solution.
                    variant = physical_seed.copy()
                    variant[2] *= -1.0
                    variant[3] += 1.0
                    local_variants.append(variant)
                seeds.extend(local_variants)

    if dimension == 6 and fspl_anchors and parallax_anchors:
        combined = [
            np.r_[fspl[:3], fspl[3], parallax[4:6]]
            for fspl in fspl_anchors
            for parallax in parallax_anchors
            if np.linalg.norm(parallax[4:6]) > 0.0
        ]
        # Joint seeds must survive a small seed budget; place them immediately
        # after the null baseline and before the single-effect atlas.
        seeds[1:1] = combined

    # Baseline perturbations are appended after detector-derived seeds so the
    # seed budget cannot discard the physical detector's best basin.
    for factor in config.tE_factors:
        for u0_factor in config.u0_factors:
            candidate = base.copy()
            candidate[1] *= float(factor)
            candidate[2] *= float(u0_factor)
            seeds.append(candidate)
            if config.u0_sign_flip:
                flipped = candidate.copy()
                flipped[2] *= -1.0
                seeds.append(flipped)
            if dimension in (4, 6):
                tstar0 = max(abs(base[1]) * np.exp(base[3]), 1.0e-8)
                for tstar_factor in config.t_star_factors:
                    for ratio in config.rho_over_u0:
                        fspl_seed = candidate.copy()
                        rho_ratio = max(float(ratio) * abs(fspl_seed[2]), 1.0e-6)
                        rho_tstar = max(
                            float(tstar_factor) * tstar0 / max(abs(fspl_seed[1]), 1.0e-8),
                            1.0e-6,
                        )
                        fspl_seed[3] = np.log(np.sqrt(rho_ratio * rho_tstar))
                        for offset in config.t0_offsets:
                            shifted = fspl_seed.copy()
                            shifted[0] += float(offset) * max(abs(float(candidate[1] * candidate[2])), 1.0e-6)
                            seeds.append(shifted)

    # Re-centre the same degeneracy families on each physical detector seed.
    # This is important for FSPL: perturbing only the PSPL baseline can spend
    # the entire seed budget in the wrong rho basin, leaving the detector's
    # good 4-D seed without an independent basin check.
    for anchor in tuple(detector_anchors):
        for factor in config.tE_factors:
            for u0_factor in config.u0_factors:
                candidate = anchor.copy()
                candidate[1] *= float(factor)
                candidate[2] *= float(u0_factor)
                seeds.append(candidate)
                if config.u0_sign_flip:
                    flipped = candidate.copy()
                    flipped[2] *= -1.0
                    seeds.append(flipped)
                if dimension in (4, 6):
                    tstar0 = max(abs(anchor[1]) * np.exp(anchor[3]), 1.0e-8)
                    for tstar_factor in config.t_star_factors:
                        for ratio in config.rho_over_u0:
                            fspl_seed = candidate.copy()
                            rho_ratio = max(
                                float(ratio) * abs(fspl_seed[2]),
                                1.0e-6,
                            )
                            rho_tstar = max(
                                float(tstar_factor) * tstar0
                                / max(abs(fspl_seed[1]), 1.0e-8),
                                1.0e-6,
                            )
                            fspl_seed[3] = np.log(
                                np.sqrt(rho_ratio * rho_tstar)
                            )
                            for offset in config.t0_offsets:
                                shifted = fspl_seed.copy()
                                shifted[0] += float(offset) * max(
                                    abs(float(candidate[1] * candidate[2])), 1.0e-6
                                )
                                seeds.append(shifted)

    unique: list[np.ndarray] = []
    for seed in seeds:
        value = _coerce_seed_dimension(seed, config.parameter_dimension, config.default_logrho)
        if not any(value.shape == other.shape and np.allclose(value, other, rtol=0.0, atol=1.0e-12) for other in unique):
            unique.append(value)
        if len(unique) >= max(1, int(config.max_seeds)):
            break
    return tuple(unique)


def _parameter_at_bound(parameters: np.ndarray, config: FallbackConfig) -> bool:
    """Detect a solution that is only held in place by a configured bound."""
    value = np.asarray(parameters, dtype=float).reshape(-1)
    if not np.all(np.isfinite(value)):
        return True
    if config.max_piE is not None and value.size >= 5:
        max_pi_e = abs(float(config.max_piE))
        if max_pi_e > 0.0 and np.any(np.abs(value[-2:]) >= max_pi_e * (1.0 - 1.0e-3)):
            return True
    if value.size in (4, 6) and (value[3] <= -40.0 or value[3] >= 10.0):
        return True
    return False


def _fit_raw_parameters(fit: object) -> np.ndarray:
    # Keep the fallback seed contract physical (tE and rho), even when the
    # optimizer stores log_tE/log_rho internally.  The native evaluator makes
    # that conversion exactly once at its optimizer boundary.
    value = np.asarray(getattr(fit, "params"), dtype=float).reshape(-1).copy()
    names = tuple(getattr(fit, "param_names", ()))
    if value.size in (4, 6) and "rho" in names:
        value[3] = np.log(max(abs(float(value[3])), 1e-12))
    return value


def _stage_basin_parameters(
    result: FallbackResult,
    *,
    dimension: int,
    limit: int = 3,
) -> tuple[np.ndarray, ...]:
    """Collect distinct fitted basins from one single-effect stage."""
    values = [_fit_raw_parameters(result.fit)]
    values.extend(_fit_raw_parameters(attempt.result.fit) for attempt in result.attempts)
    unique: list[np.ndarray] = []
    for value in values:
        value = np.asarray(value, dtype=float).reshape(-1)
        if value.size != dimension or not np.all(np.isfinite(value)):
            continue
        if not any(np.allclose(value, other, rtol=0.0, atol=1.0e-10) for other in unique):
            unique.append(value)
        if len(unique) >= max(1, int(limit)):
            break
    return tuple(unique)


def _compose_joint_stage_seeds(
    fspl_seeds: Sequence[np.ndarray],
    parallax_seeds: Sequence[np.ndarray],
) -> tuple[np.ndarray, ...]:
    """Compose FSPL rho and parallax vectors into genuine 6-D joint seeds."""
    composed: list[np.ndarray] = []
    for fspl_seed in fspl_seeds:
        fspl = np.asarray(fspl_seed, dtype=float).reshape(-1)
        if fspl.size != 4:
            continue
        for parallax_seed in parallax_seeds:
            parallax = np.asarray(parallax_seed, dtype=float).reshape(-1)
            if parallax.size != 5:
                continue
            shared_candidates = (
                fspl[:3],
                parallax[:3],
                np.asarray(
                    [
                        0.5 * (fspl[0] + parallax[0]),
                        np.sqrt(max(fspl[1], 1.0e-12) * max(parallax[1], 1.0e-12)),
                        0.5 * (fspl[2] + parallax[2]),
                    ],
                    dtype=float,
                ),
            )
            for shared in shared_candidates:
                value = np.r_[shared, fspl[3], parallax[3:5]]
                if not any(
                    np.allclose(value, other, rtol=0.0, atol=1.0e-10)
                    for other in composed
                ):
                    composed.append(value)
    return tuple(composed)


def run_robust_fallback(
    fitter,
    time,
    flux,
    ferr,
    base_seed: Sequence[float],
    *,
    candidates: Iterable[EffectCandidate] = (),
    extra_seeds: Iterable[Sequence[float]] = (),
    effect: str = "mixed",
    config: FallbackConfig = FallbackConfig(),
    protected_mask: Optional[np.ndarray] = None,
    protected_masks: Optional[Sequence[np.ndarray]] = None,
    known_anomaly_mask: Optional[np.ndarray] = None,
    soft_anomaly_mask: Optional[np.ndarray] = None,
    selection_exclusion_mask: Optional[np.ndarray] = None,
    baseline_fit: Optional[object] = None,
    effect_score_fn=None,
    model_spec: Optional[EffectFitterSpec] = None,
) -> FallbackResult:
    """Run robust alternating fits from detector and degeneracy seeds.

    ``soft_anomaly_mask`` protects ambiguous planet wings through capped fit
    weights.  ``selection_exclusion_mask`` additionally keeps those points
    out of the physical-model ranking, so a model cannot win by explaining the
    very contamination it is meant to be protected from.
    """
    candidate_tuple = tuple(candidates)
    generated_seeds = detector_seed_parameters(base_seed, candidate_tuple, config=config)
    seeds: list[np.ndarray] = []
    for seed in tuple(extra_seeds) + generated_seeds:
        value = _coerce_seed_dimension(seed, config.parameter_dimension, config.default_logrho)
        if not any(np.allclose(value, other, rtol=0.0, atol=1.0e-12) for other in seeds):
            seeds.append(value)
        if len(seeds) >= max(1, int(config.max_seeds)):
            break
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    fe = np.maximum(np.asarray(ferr, dtype=float), 1.0e-12)
    attempts: list[FallbackAttempt] = []
    errors: list[str] = []
    known = None
    initial_standardized_residual = None
    if baseline_fit is not None and hasattr(baseline_fit, "residual"):
        baseline_residual = np.asarray(baseline_fit.residual, dtype=float).reshape(-1)
        if baseline_residual.size == t.size and np.all(np.isfinite(baseline_residual)):
            initial_standardized_residual = baseline_residual / fe
    compact_masks = [
        np.asarray(candidate.compact_block_mask, dtype=bool)
        for candidate in candidate_tuple
        if "parallax" in candidate.effect
        and candidate.compact_block_mask is not None
        and np.asarray(candidate.compact_block_mask).size == t.size
    ]
    initial_anomaly_mask = (
        np.logical_or.reduce(compact_masks) if compact_masks else None
    )
    if known_anomaly_mask is not None:
        known = np.asarray(known_anomaly_mask, dtype=bool).reshape(-1)
        if known.size != t.size:
            raise ValueError("known_anomaly_mask must match the light curve.")
        initial_anomaly_mask = (
            known.copy()
            if initial_anomaly_mask is None
            else np.asarray(initial_anomaly_mask, dtype=bool) | known
        )
    soft = None
    if soft_anomaly_mask is not None:
        soft = np.asarray(soft_anomaly_mask, dtype=bool).reshape(-1)
        if soft.size != t.size:
            raise ValueError("soft_anomaly_mask must match the light curve.")
        if initial_anomaly_mask is None:
            initial_anomaly_mask = soft.copy()
        else:
            initial_anomaly_mask = (
                np.asarray(initial_anomaly_mask, dtype=bool) | soft
            )
    selection_exclusion = (
        np.zeros(t.size, dtype=bool)
        if selection_exclusion_mask is None
        else np.asarray(selection_exclusion_mask, dtype=bool).reshape(-1)
    )
    if selection_exclusion.size != t.size:
        raise ValueError("selection_exclusion_mask must match the light curve.")
    if known is not None:
        selection_exclusion |= known
    if soft is not None:
        selection_exclusion |= soft
    if "parallax" not in effect:
        # A compact FSPL peak is itself the physical signal; do not pre-mask it.
        initial_standardized_residual = None
    for seed in seeds:
        if protected_mask is None and protected_masks is None:
            masks = tuple(
                protected_support_mask(
                    t,
                    candidate.effect,
                    candidate.seed_parameters,
                    observed_scale=getattr(candidate, "observed_signal_scale", None),
                )
                for candidate in candidate_tuple
                if candidate.seed_parameters is not None
                and (
                    candidate.effect == effect
                    or candidate.effect in effect
                    or effect == "mixed"
                )
            )
        else:
            masks = protected_masks
            if protected_mask is not None:
                masks = (np.asarray(protected_mask, dtype=bool),)
        try:
            result = robust_refine_with_fitter(
                fitter,
                t,
                f,
                fe,
                seed,
                config=config.contamination,
                protected_masks=masks,
                soft_anomaly_mask=soft,
                initial_standardized_residual=initial_standardized_residual,
                initial_anomaly_mask=initial_anomaly_mask,
                forced_anomaly_mask=known_anomaly_mask,
            )
            original_chi2 = float(np.asarray(result.fit.chi2))
            robust_objective = float(result.segmentation.objective)
            contamination_penalty = float(result.segmentation.contamination_penalty)
            initial = np.asarray(seed, dtype=float)
            final = _fit_raw_parameters(result.fit)
            parameter_distance = scaled_parameter_distance(initial, final)
            parameter_at_bound = _parameter_at_bound(final, config)
            optimizer_success_value = getattr(result.fit, "optimizer_success", None)
            optimizer_success = bool(
                np.all(np.isfinite(final))
                and optimizer_success_value is not None
                and optimizer_success_value
            )
            stable = bool(
                optimizer_success
                and parameter_distance <= config.max_point_parameter_change
            )
            attempts.append(
                FallbackAttempt(
                    seed=seed.copy(),
                    result=result,
                    objective=robust_objective,
                    stable=stable,
                    original_chi2=original_chi2,
                    robust_objective=robust_objective,
                    contamination_penalty=contamination_penalty,
                    parameter_distance=parameter_distance,
                    optimizer_success=optimizer_success,
                    parameter_at_bound=parameter_at_bound,
                    segmentation_stable=bool(result.segmentation_stable),
                )
            )
        except Exception as exc:  # keep another basin alive
            errors.append(f"{type(exc).__name__}: {exc}")
    if not attempts:
        detail = errors[0] if errors else "no valid seeds"
        raise RuntimeError(f"All robust fallback seeds failed: {detail}")

    selection_keep = ~selection_exclusion

    def selection_chi2(attempt: FallbackAttempt) -> float:
        if not np.any(selection_exclusion):
            return float(attempt.original_chi2)
        residual = np.asarray(
            getattr(attempt.result.fit, "residual", ()),
            dtype=float,
        ).reshape(-1)
        if (
            residual.size == t.size
            and np.count_nonzero(selection_keep) > 0
            and np.all(np.isfinite(residual[selection_keep]))
        ):
            return float(
                np.sum(
                    np.square(
                        residual[selection_keep] / fe[selection_keep]
                    )
                )
            )
        return float(attempt.original_chi2)

    attempts.sort(
        key=lambda attempt: (
            not attempt.optimizer_success,
            attempt.parameter_at_bound,
            selection_chi2(attempt),
            attempt.original_chi2,
        )
    )
    converged = [
        attempt for attempt in attempts
        if (
            attempt.stable
            and attempt.result.converged
            and attempt.result.segmentation_stable
            and not attempt.parameter_at_bound
            and np.isfinite(attempt.robust_objective)
        )
    ]
    valid_attempts = [
        attempt
        for attempt in attempts
        if (
            attempt.optimizer_success
            and not attempt.parameter_at_bound
            and np.isfinite(attempt.original_chi2)
            and np.all(np.isfinite(_fit_raw_parameters(attempt.result.fit)))
        )
    ]
    ranked = valid_attempts if valid_attempts else attempts
    ranked.sort(
        key=lambda attempt: (
            selection_chi2(attempt),
            attempt.original_chi2,
            attempt.robust_objective,
        )
    )
    best = ranked[0]
    reasons = ["robust_fallback_completed"]
    if not best.stable:
        reasons.append("parameter_change_large")
    if not best.result.converged:
        reasons.append("alternating_not_converged")
    if not best.optimizer_success:
        reasons.append("optimizer_failed")
    if len(converged) < 2:
        reasons.append("single_seed_only")
    if best.parameter_at_bound:
        reasons.append("parameter_at_bound")
    if not best.result.segmentation_stable:
        reasons.append("contamination_sensitive")
    minimum_retained = 1.0 - float(
        config.contamination.max_protected_anomaly_fraction
    )
    clear_fspl_morphology = any(
        candidate.effect == "fspl"
        and candidate.morphology in {
            "fspl_even_peak",
            "fspl_flattened_peak",
        }
        for candidate in candidate_tuple
    )
    # Near a season/data boundary only one FSPL shoulder may be observed.
    # Accept that partial geometry only when the measured signed topology is
    # still exceptionally coherent: a symmetric central dip, a majority of
    # the residual energy explained by the FSPL tangent, and at least one
    # energy explained by the FSPL tangent, and at least one positive
    # shoulder. This keeps generic compact planetary peaks out of the
    # exception while allowing a censored source crossing to remain
    # identifiable.
    supported_partial_fspl_morphology = False
    for candidate in candidate_tuple:
        if candidate.effect != "fspl" or candidate.morphology != "fspl_partial_peak":
            continue
        morphology_rows = [
            row for row in candidate.subset_diagnostics
            if row.get("name") == "fspl_morphology"
        ]
        if not morphology_rows:
            continue
        row = morphology_rows[-1]
        shoulder_values = []
        for key in ("left_shoulder_mean_z", "right_shoulder_mean_z"):
            try:
                value = float(row.get(key, float("nan")))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                shoulder_values.append(value)
        shoulder_strength = max(shoulder_values, default=float("-inf"))
        supported_partial_fspl_morphology = bool(
            row.get("partial")
            and float(row.get("central_symmetry", 0.0)) >= 0.9
            and float(row.get("template_explained_fraction", 0.0)) >= 0.6
            and float(row.get("core_mean_z", 0.0)) <= -5.0
            and shoulder_strength >= 3.0
        )
        if supported_partial_fspl_morphology:
            break
    component_identifiable = all(
        retained >= minimum_retained
        for retained in best.result.segmentation.protected_component_retained_fractions
    )
    # A finite-source crossing occupies the same central support that the
    # contamination segmenter is designed to protect from being fitted away.
    # Once the independent signed-topology gate has established a symmetric
    # central dip with two positive shoulders, do not veto an otherwise
    # reproducible, BIC-improving FSPL fit merely because that support remains
    # prominent in an intermediate robust-fit residual.
    fspl_morphology_identifiable = bool(
        clear_fspl_morphology or supported_partial_fspl_morphology
    )
    support_identifiable = bool(
        component_identifiable or fspl_morphology_identifiable
    )
    if not support_identifiable:
        reasons.append("insufficient_identifiability")
    elif fspl_morphology_identifiable and not component_identifiable:
        reasons.append("identifiability_supported_by_fspl_morphology")
    if best.result.segmentation.diagnostics:
        reasons.extend(best.result.segmentation.diagnostics)
    if baseline_fit is not None:
        baseline_original_chi2 = float(np.asarray(getattr(baseline_fit, "chi2", np.inf)))
    else:
        baseline_original_chi2 = min(
            float(attempt.result.initial_fit.chi2) for attempt in attempts
            if np.isfinite(float(np.asarray(attempt.result.initial_fit.chi2)))
        ) if any(np.isfinite(float(np.asarray(attempt.result.initial_fit.chi2))) for attempt in attempts) else float("inf")
    baseline_dimension = int(
        np.asarray(getattr(baseline_fit, "params", ())).size
        if baseline_fit is not None
        else 3
    )
    selected_dimension = int(
        np.asarray(getattr(best.result.fit, "params", ())).size
    )
    n_data = max(int(t.size), 1)
    n_selection = max(int(np.count_nonzero(selection_keep)), 1)
    delta_chi2 = baseline_original_chi2 - best.original_chi2
    baseline_selection_chi2 = baseline_original_chi2
    if baseline_fit is not None and np.any(selection_exclusion):
        baseline_residual_for_selection = np.asarray(
            getattr(baseline_fit, "residual", ()),
            dtype=float,
        ).reshape(-1)
        if (
            baseline_residual_for_selection.size == t.size
            and np.all(
                np.isfinite(
                    baseline_residual_for_selection[selection_keep]
                )
            )
        ):
            baseline_selection_chi2 = float(
                np.sum(
                    np.square(
                        baseline_residual_for_selection[selection_keep]
                        / fe[selection_keep]
                    )
                )
            )
    selected_selection_chi2 = selection_chi2(best)
    baseline_bic = float(
        baseline_selection_chi2
        + baseline_dimension * np.log(n_selection)
    )
    selected_bic = float(
        selected_selection_chi2
        + selected_dimension * np.log(n_selection)
    )
    bic_improvement = float(baseline_bic - selected_bic)
    model_selection_improved = bool(
        np.isfinite(bic_improvement)
        and bic_improvement > float(config.min_bic_improvement)
    )
    clean_region_improved = True
    if np.any(selection_exclusion):
        baseline_residual = np.asarray(
            getattr(baseline_fit, "residual", np.full(t.size, np.nan)),
            dtype=float,
        ).reshape(-1)
        selected_residual = np.asarray(
            getattr(best.result.fit, "residual", np.full(t.size, np.nan)),
            dtype=float,
        ).reshape(-1)
        clean = ~selection_exclusion
        if (
            baseline_residual.size != t.size
            or selected_residual.size != t.size
            or np.count_nonzero(clean) <= selected_dimension
        ):
            clean_region_improved = False
        else:
            baseline_clean_chi2 = float(
                np.sum(np.square(baseline_residual[clean] / fe[clean]))
            )
            selected_clean_chi2 = float(
                np.sum(np.square(selected_residual[clean] / fe[clean]))
            )
            clean_bic_improvement = float(
                baseline_clean_chi2
                - selected_clean_chi2
                - max(selected_dimension - baseline_dimension, 0)
                * np.log(max(int(np.count_nonzero(clean)), 1))
            )
            clean_region_improved = bool(
                np.isfinite(clean_bic_improvement)
                and clean_bic_improvement
                > float(config.min_bic_improvement)
            )
        # Once a localized planet interval has been established, the adopted
        # single-lens family must be selected on the complementary data. A
        # correct physical model intentionally leaves the planet in its
        # full-cadence residual and can therefore have a worse all-point chi2
        # than a biased PSPL that partially absorbs that planet.
        model_selection_improved = clean_region_improved
    if not model_selection_improved:
        reasons.append("original_bic_not_improved")
    if not clean_region_improved:
        reasons.append("non_planet_region_bic_not_improved")
    clean_region_acceptable = bool(clean_region_improved)

    baseline_score = None
    selected_score = None
    if effect_score_fn is not None:
        try:
            if baseline_fit is not None:
                try:
                    baseline_score = float(effect_score_fn(baseline_fit, effect))
                except TypeError:
                    baseline_score = float(effect_score_fn(baseline_fit))
            else:
                baseline_score = baseline_original_chi2
            try:
                selected_score = float(effect_score_fn(best.result.fit, effect))
            except TypeError:
                selected_score = float(effect_score_fn(best.result.fit))
        except Exception as exc:
            reasons.append(f"effect_score_evaluation_failed:{type(exc).__name__}")
    else:
        reasons.append("effect_score_not_checked")
    score_reduced = bool(
        baseline_score is not None and selected_score is not None
        and np.isfinite(baseline_score) and np.isfinite(selected_score)
        and selected_score < baseline_score
    )

    if baseline_score is not None and not score_reduced:
        reasons.append("effect_score_not_reduced")
        reasons.append("objective_not_improved")

    independent = [attempt for attempt in converged if attempt is not best]
    best_parameters = _fit_raw_parameters(best.result.fit)

    def basin_distance(attempt: FallbackAttempt) -> float:
        other = _fit_raw_parameters(attempt.result.fit)
        distance = scaled_parameter_distance(best_parameters, other)
        if "fspl" in str(effect) and other.size >= 3:
            # Rectilinear FSPL magnification is exactly invariant under the
            # u0 sign. Count the mirrored optimizer basin as an independent
            # reproduction of the same physical solution.
            mirrored = other.copy()
            mirrored[2] *= -1.0
            distance = min(
                distance,
                scaled_parameter_distance(best_parameters, mirrored),
            )
        return distance

    basin_reproduced = any(
        basin_distance(attempt) <= config.max_basin_distance
        for attempt in independent
    )
    if not basin_reproduced:
        reasons.append("basin_not_reproduced")

    score_ratio = (
        float(selected_score / baseline_score)
        if (
            score_reduced
            and baseline_score is not None
            and selected_score is not None
            and baseline_score > 0.0
        )
        else float("inf")
    )
    overwhelming_improvement = bool(
        score_ratio <= 1.0e-2
        and np.isfinite(delta_chi2)
        and delta_chi2 >= max(1_000.0, 0.5 * baseline_original_chi2)
    )
    # A textbook finite-source topology is independent evidence that the
    # central support is physical, rather than contamination.  Native VBM
    # fits can land on the same solution from every seed while the alternating
    # segmenter continues to toggle points inside that support.  Permit that
    # narrow case only after an overwhelming full-data and effect-score gain.
    clear_fspl_topology_acceptance = bool(
        effect == "fspl"
        and clear_fspl_morphology
        and overwhelming_improvement
        and best.stable
        and best.optimizer_success
        and not best.parameter_at_bound
        and support_identifiable
        and model_selection_improved
        and clean_region_acceptable
    )
    if clear_fspl_topology_acceptance and not (
        best.result.converged
        and len(converged) >= 2
        and best.result.segmentation_stable
        and basin_reproduced
    ):
        reasons.append("clear_fspl_topology_overrides_contamination_stability")

    selected_dof = max(n_data - selected_dimension, 1)
    selected_reduced_chi2 = float(best.original_chi2 / selected_dof)
    fractional_chi2_improvement = float(
        delta_chi2 / baseline_original_chi2
        if baseline_original_chi2 > 0.0
        else float("-inf")
    )
    # When a real planet remains in the residual, the selected single-lens
    # model need not reach chi2/dof ~= 1.  It must, however, either provide an
    # acceptable global fit or remove a dominant fraction of the baseline
    # mismatch.  This prevents a flexible physical model from shaving a small
    # part off an overwhelmingly planetary residual and being adopted merely
    # because the absolute delta-chi2 is large.
    independently_supported_long_parallax = bool(
        "parallax" in str(effect)
        and abs(float(np.asarray(best.result.fit.params, dtype=float)[1]))
        >= 0.95 * float(config.min_coherent_parallax_tE)
        and any(
            candidate.effect in {
                "annual_parallax",
                "space_parallax",
                "fspl_parallax",
                "fspl_space_parallax",
            }
            and (
                candidate.morphology == "parallax_coherent_wings"
                or "exact_probe_promoted" in candidate.reason_codes
            )
            for candidate in candidate_tuple
        )
    )
    known_planet_parallax_acceptance = bool(
        "parallax" in str(effect)
        and known is not None
        and np.any(known)
        and independently_supported_long_parallax
        and score_ratio <= 1.0e-2
        and best.stable
        and best.optimizer_success
        and not best.parameter_at_bound
        and support_identifiable
        and model_selection_improved
        and clean_region_acceptable
    )
    if known_planet_parallax_acceptance and not (
        best.result.converged
        and len(converged) >= 2
        and best.result.segmentation_stable
        and basin_reproduced
    ):
        reasons.append(
            "known_planet_parallax_evidence_overrides_contamination_stability"
        )
    global_fit_acceptable = bool(
        selected_reduced_chi2 <= 2.0
        or fractional_chi2_improvement >= 0.5
        or independently_supported_long_parallax
    )
    if not global_fit_acceptable:
        reasons.append("global_fit_improvement_insufficient")
    model_topology_acceptable = bool(
        "fspl" not in str(effect) or fspl_morphology_identifiable
    )
    if not model_topology_acceptable:
        reasons.append("fspl_topology_not_supported")
    fitted_tE = abs(float(np.asarray(best.result.fit.params, dtype=float)[1]))
    parallax_duration_acceptable = bool(
        "parallax" not in str(effect)
        or fitted_tE >= 0.95 * float(config.min_coherent_parallax_tE)
        or overwhelming_improvement
    )
    if not parallax_duration_acceptable:
        reasons.append("short_event_parallax_not_independently_supported")
    # Long-duration parallax can overlap the protected anomaly support over
    # much of the event.  Override only that support-identifiability veto when
    # independent optimizer basins reproduce an excellent global solution and
    # the physical-effect residual is essentially eliminated.
    overwhelming_parallax_acceptance = bool(
        "parallax" in str(effect)
        and score_ratio <= 1.0e-3
        and overwhelming_improvement
        and selected_reduced_chi2 <= 2.0
        and best.stable
        and best.result.converged
        and best.optimizer_success
        and len(converged) >= 2
        and best.result.segmentation_stable
        and not best.parameter_at_bound
        and model_selection_improved
        and clean_region_acceptable
        and basin_reproduced
        and global_fit_acceptable
        and model_topology_acceptable
        and parallax_duration_acceptable
    )
    support_acceptable = bool(
        support_identifiable or overwhelming_parallax_acceptance
    )
    if overwhelming_parallax_acceptance and not support_identifiable:
        reasons.append("overwhelming_parallax_evidence_overrides_support_guard")

    strict_acceptance = bool(
        best.stable
        and best.result.converged
        and best.optimizer_success
        and len(converged) >= 2
        and best.result.segmentation_stable
        and not best.parameter_at_bound
        and support_acceptable
        and score_reduced
        and model_selection_improved
        and clean_region_acceptable
        and basin_reproduced
        and global_fit_acceptable
        and model_topology_acceptable
        and parallax_duration_acceptable
    )
    diagnostic_acceptance = bool(
        strict_acceptance
        or clear_fspl_topology_acceptance
        or overwhelming_parallax_acceptance
        or known_planet_parallax_acceptance
    )
    numerically_valid = bool(
        best.optimizer_success
        and not best.parameter_at_bound
        and np.isfinite(best.original_chi2)
        and np.all(np.isfinite(_fit_raw_parameters(best.result.fit)))
    )
    # The ordinary BIC above is deliberately evaluated on the complementary
    # clean region whenever a planet interval is known.  That is the right
    # guard for parallax (a parallax model must not win by fitting a planet),
    # but it is incomplete for FSPL: the finite-source crossing is itself a
    # compact signal and is therefore part of the excluded interval.  A
    # valid FSPL fit can consequently improve the full light curve by orders
    # of magnitude while having no clean-region gain.  Re-open that narrow
    # case using a *full-data* BIC and an independently strong FSPL detector
    # score.  The score, stable detector support, and chi2/dof guard are all
    # required so a toy/local planet-only fit cannot pass this rescue.
    full_bic_improvement = float(
        baseline_original_chi2
        + baseline_dimension * np.log(n_data)
        - best.original_chi2
        - selected_dimension * np.log(n_data)
    )
    fspl_detector_support = any(
        candidate.effect == "fspl"
        and float(candidate.score) >= float(
            max(1.0e3, config.min_fspl_full_bic_improvement)
        )
        and float(candidate.coverage) >= 0.02
        and float(candidate.subset_stability) >= 0.75
        and candidate.morphology not in {"planet_like", "mixed_or_planet"}
        for candidate in candidate_tuple
    )
    fspl_full_fit_acceptance = bool(
        "fspl" in str(effect)
        and numerically_valid
        and fspl_detector_support
        and selected_reduced_chi2 <= 2.0
        and np.isfinite(full_bic_improvement)
        and full_bic_improvement
        >= float(config.min_fspl_full_bic_improvement)
    )
    if fspl_full_fit_acceptance:
        reasons.append("fspl_full_data_bic_rescue")
        reasons.append("clean_region_bic_not_applicable_to_fspl_support")
    # Physical diagnostics decide whether this expensive fit should run.
    # Once it has run, selection is deliberately model-based: retain every
    # numerically valid, non-boundary solution whose BIC improves on its
    # parent baseline. Morphology, segmentation convergence, and independent
    # basin reproduction remain recorded diagnostics, but no longer veto a
    # better single-lens model.
    success = bool(
        numerically_valid
        and (
            (model_selection_improved and clean_region_acceptable)
            or fspl_full_fit_acceptance
        )
    )
    if success:
        reasons.append("accepted_by_postfit_validity_and_bic")
        if not diagnostic_acceptance:
            reasons.append("diagnostic_warnings_do_not_veto_postfit_model")
    if not success:
        reasons.append("fallback_acceptance_failed")
    return FallbackResult(
        fit=best.result.fit,
        initial_fit=best.result.initial_fit,
        effect=str(effect),
        attempts=tuple(attempts),
        selected_seed=best.seed.copy(),
        success=success,
        reason_codes=tuple(dict.fromkeys(reasons)),
        baseline_original_chi2=baseline_original_chi2,
        selected_original_chi2=best.original_chi2,
        selected_robust_objective=best.robust_objective,
        baseline_effect_score=baseline_score,
        selected_effect_score=selected_score,
        baseline_bic=baseline_bic,
        selected_bic=selected_bic,
        bic_improvement=bic_improvement,
        numerically_valid=numerically_valid,
        model_spec=None if model_spec is None else {
            "effect": model_spec.effect,
            "parameter_dimension": model_spec.parameter_dimension,
            "parameter_names": list(model_spec.parameter_names),
            "raw_parameter_names": list(model_spec.raw_parameter_names),
            "backend": model_spec.backend,
            "convention": model_spec.convention,
        },
    )


def run_staged_joint_fallback(
    config,
    time,
    flux,
    ferr,
    base_seed: Sequence[float],
    *,
    candidates: Iterable[EffectCandidate] = (),
    effect: str = "fspl_space_parallax",
    fallback_config: FallbackConfig = FallbackConfig(),
    protected_mask: Optional[np.ndarray] = None,
    protected_masks: Optional[Sequence[np.ndarray]] = None,
    known_anomaly_mask: Optional[np.ndarray] = None,
    soft_anomaly_mask: Optional[np.ndarray] = None,
    selection_exclusion_mask: Optional[np.ndarray] = None,
    baseline_fit: Optional[object] = None,
    effect_score_fn=None,
) -> FallbackResult:
    """Fit single effects first and use their basins in the joint fit."""
    candidate_tuple = tuple(candidates)
    stage_results: list[FallbackResult] = []
    stage_seeds: dict[str, tuple[np.ndarray, ...]] = {}
    stage_errors: list[str] = []
    score_fn = effect_score_fn
    for stage_effect in ("fspl", "annual_parallax", "space_parallax"):
        if not any(candidate.effect == stage_effect for candidate in candidate_tuple):
            continue
        try:
            stage_spec = make_effect_fitter(config, stage_effect, float(np.median(time)))
            stage_cfg = replace(fallback_config, parameter_dimension=stage_spec.parameter_dimension)
            stage_result = run_robust_fallback(
                stage_spec.fitter, time, flux, ferr, base_seed,
                candidates=tuple(candidate for candidate in candidate_tuple if candidate.effect == stage_effect),
                effect=stage_effect, config=stage_cfg, protected_mask=protected_mask,
                protected_masks=protected_masks,
                known_anomaly_mask=known_anomaly_mask,
                soft_anomaly_mask=soft_anomaly_mask,
                selection_exclusion_mask=selection_exclusion_mask,
                baseline_fit=baseline_fit, effect_score_fn=score_fn, model_spec=stage_spec,
            )
            stage_results.append(stage_result)
            stage_seeds[stage_effect] = _stage_basin_parameters(
                stage_result,
                dimension=stage_spec.parameter_dimension,
            )
        except Exception as exc:
            stage_errors.append(
                f"{stage_effect}:{type(exc).__name__}:{exc}"
            )
            continue

    try:
        joint_spec = make_effect_fitter(config, effect, float(np.median(time)))
        joint_cfg = replace(fallback_config, parameter_dimension=joint_spec.parameter_dimension)
        parallax_effect = (
            "space_parallax" if "space_parallax" in stage_seeds else "annual_parallax"
        )
        composed_seeds = _compose_joint_stage_seeds(
            stage_seeds.get("fspl", ()),
            stage_seeds.get(parallax_effect, ()),
        )
        if composed_seeds:
            seed = composed_seeds[0]
        else:
            seed = _coerce_seed_dimension(
                base_seed,
                joint_spec.parameter_dimension,
                joint_cfg.default_logrho,
            )
        bridged_stage_seeds = tuple(
            seed_value
            for values in stage_seeds.values()
            for seed_value in values
        )
        joint_result = run_robust_fallback(
            joint_spec.fitter, time, flux, ferr, seed,
            candidates=candidate_tuple,
            extra_seeds=composed_seeds + bridged_stage_seeds,
            effect=effect,
            config=joint_cfg,
            protected_mask=protected_mask,
            protected_masks=protected_masks,
            baseline_fit=baseline_fit,
            known_anomaly_mask=known_anomaly_mask,
            soft_anomaly_mask=soft_anomaly_mask,
            selection_exclusion_mask=selection_exclusion_mask,
            effect_score_fn=score_fn,
            model_spec=joint_spec,
        )
    except Exception as exc:
        stage_errors.append(
            f"{effect}:{type(exc).__name__}:{exc}"
        )
        joint_result = None
    # A mixed detector decision does not prove that both effects are present.
    # Compare every numerically valid, PSPL-improving model on the same
    # selection data and retain the lowest BIC. This naturally falls back to
    # FSPL/parallax when the joint component does not earn its extra
    # parameter.
    successful_models = [
        result
        for result in (
            *stage_results,
            *((joint_result,) if joint_result is not None else ()),
        )
        if result.success
    ]
    if successful_models:
        selected_model = min(
            successful_models,
            key=lambda result: (
                result.selected_bic
                if np.isfinite(result.selected_bic)
                else float("inf"),
                result.selected_original_chi2,
                result.selected_robust_objective,
            ),
        )
        selected_joint = bool(selected_model is joint_result)
        reasons = tuple(
            dict.fromkeys(
                (
                    *selected_model.reason_codes,
                    "selected_by_hierarchical_bic",
                    *(
                        ()
                        if selected_joint
                        else (
                            "joint_model_not_preferred_by_bic",
                            "accepted_lower_order_model",
                            # Backward-compatible provenance label.
                            "accepted_single_effect_stage",
                        )
                    ),
                )
            )
        )
        return replace(
            selected_model,
            reason_codes=reasons,
            stage_results=tuple(stage_results),
        )
    if joint_result is None:
        detail = stage_errors[0] if stage_errors else "no accepted stage"
        raise RuntimeError(
            f"Joint fallback failed and no single-effect stage was accepted: {detail}"
        )
    if stage_errors:
        return replace(
            joint_result,
            stage_results=tuple(stage_results),
            reason_codes=tuple(
                dict.fromkeys(
                    (
                        *joint_result.reason_codes,
                        *(f"stage_failed:{error}" for error in stage_errors),
                    )
                )
            ),
        )
    return replace(joint_result, stage_results=tuple(stage_results))


__all__ = [
    "FallbackAttempt",
    "FallbackConfig",
    "FallbackResult",
    "EffectFitterSpec",
    "detector_seed_parameters",
    "make_effect_fitter",
    "run_robust_fallback",
    "run_staged_joint_fallback",
]
