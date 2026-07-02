from __future__ import annotations

import numpy as np

from .pspl import angle_of, q_grid_from_width, r_major, r_minor, u_abs, u_vec
from .types import AtomFitResult, PlanetClassConfig, PSPLParams, SeedCandidate


def seeds_from_atom(
    fit: AtomFitResult,
    pspl: PSPLParams,
    config: PlanetClassConfig,
) -> tuple[SeedCandidate, ...]:
    if fit.delta_chi2 < float(config.min_delta_chi2_for_seed):
        return ()
    if fit.class_label == "major_image_bump":
        return positive_bump_seeds(fit, pspl, config)
    if fit.class_label == "minor_image_dip":
        return negative_dip_seeds(fit, pspl, config)
    if fit.class_label == "central_caustic":
        return central_caustic_seeds(fit, pspl, config)
    if fit.class_label == "second_source":
        return second_pspl_seeds(fit, pspl)
    return ()


def positive_bump_seeds(
    fit: AtomFitResult,
    pspl: PSPLParams,
    config: PlanetClassConfig,
) -> tuple[SeedCandidate, ...]:
    t_peak = float(fit.params.get("t_peak", fit.params.get("t0_2", pspl.t0)))
    width = float(fit.params.get("width", fit.params.get("tE_2", pspl.tE)))
    u = float(u_abs(t_peak, pspl))
    s_wide = float(r_major(u))
    s_close = 1.0 / max(s_wide, 1e-12)
    alpha = angle_of(u_vec(t_peak, pspl))
    q_grid = q_grid_from_width(
        width,
        pspl.tE,
        factors=config.q_width_factors,
        q_floor=config.q_floor,
        q_ceil=config.q_ceil,
    )
    seeds: list[SeedCandidate] = []
    for q in q_grid:
        warnings = tuple(fit.warnings + (("q clipped" ,) if q in (config.q_floor, config.q_ceil) else ()))
        seeds.append(
            SeedCandidate(
                model_type="2L1S",
                class_label="major_image_bump",
                params={"s": s_wide, "q": float(q), "alpha": alpha},
                score=float(fit.score),
                source_atom=fit.atom_name,
                degeneracy_tag="wide_major",
                warnings=warnings,
            )
        )
        seeds.append(
            SeedCandidate(
                model_type="2L1S",
                class_label="major_image_bump_close_counterpart",
                params={"s": s_close, "q": float(q), "alpha": alpha + np.pi},
                score=float(fit.score) - 1.0,
                source_atom=fit.atom_name,
                degeneracy_tag="close_counterpart",
                warnings=warnings,
            )
        )
    return tuple(seeds)


def negative_dip_seeds(
    fit: AtomFitResult,
    pspl: PSPLParams,
    config: PlanetClassConfig,
) -> tuple[SeedCandidate, ...]:
    t_peak = float(fit.params.get("t_peak", pspl.t0))
    width = float(fit.params.get("width", pspl.tE))
    u = float(u_abs(t_peak, pspl))
    s_close = float(r_minor(u))
    s_wide = 1.0 / max(s_close, 1e-12)
    alpha = angle_of(-u_vec(t_peak, pspl))
    q_grid = q_grid_from_width(
        width,
        pspl.tE,
        factors=config.q_width_factors,
        q_floor=config.q_floor,
        q_ceil=config.q_ceil,
    )
    seeds: list[SeedCandidate] = []
    for q in q_grid:
        warnings = tuple(fit.warnings + (("q clipped" ,) if q in (config.q_floor, config.q_ceil) else ()))
        seeds.append(
            SeedCandidate(
                model_type="2L1S",
                class_label="minor_image_dip",
                params={"s": s_close, "q": float(q), "alpha": alpha},
                score=float(fit.score),
                source_atom=fit.atom_name,
                degeneracy_tag="close_minor",
                warnings=warnings,
            )
        )
        seeds.append(
            SeedCandidate(
                model_type="2L1S",
                class_label="minor_image_dip_wide_counterpart",
                params={"s": s_wide, "q": float(q), "alpha": alpha + np.pi},
                score=float(fit.score) - 1.0,
                source_atom=fit.atom_name,
                degeneracy_tag="wide_counterpart",
                warnings=warnings,
            )
        )
    return tuple(seeds)


def central_caustic_seeds(
    fit: AtomFitResult,
    pspl: PSPLParams,
    config: PlanetClassConfig,
) -> tuple[SeedCandidate, ...]:
    duration = float(fit.params.get("duration", 2.0 * fit.params.get("width", pspl.tE)))
    duration = max(duration, 0.0)
    s_grid = np.asarray(config.s_central_grid, dtype=float)
    q_grid = (duration / max(4.0 * pspl.tE, 1e-12)) * (s_grid - 1.0 / s_grid) ** 2
    q_grid = np.clip(q_grid, float(config.q_floor), float(config.q_ceil))

    n_alpha = max(1, int(config.alpha_grid_size_central))
    alpha_center = float(np.arctan2(pspl.u0, 0.0))
    alpha_grid = alpha_center + np.linspace(-np.pi, np.pi, n_alpha, endpoint=False)
    seeds: list[SeedCandidate] = []
    for s, q in zip(s_grid, q_grid):
        warnings = list(fit.warnings)
        if np.isclose(float(q), float(config.q_floor)) or np.isclose(float(q), float(config.q_ceil)):
            warnings.append("q clipped")
        if abs(float(s) - 1.0) < 0.1:
            warnings.append("central seed is near resonant separation")
        for alpha in alpha_grid:
            seeds.append(
                SeedCandidate(
                    model_type="2L1S",
                    class_label="central_caustic",
                    params={"s": float(s), "q": float(q), "alpha": float(alpha)},
                    score=float(fit.score),
                    source_atom=fit.atom_name,
                    degeneracy_tag="central_close" if float(s) < 1.0 else "central_wide",
                    warnings=tuple(warnings),
                )
            )
    return tuple(seeds)


def second_pspl_seeds(fit: AtomFitResult, pspl: PSPLParams) -> tuple[SeedCandidate, ...]:
    t0_2 = float(fit.params["t0_2"])
    tE_2 = float(fit.params["tE_2"])
    u0_2 = float(fit.params["u0_2"])
    fs_ratio = float(fit.params.get("Fs_2_over_Fs_1", np.nan))
    if not np.isfinite(fs_ratio):
        fs_ratio = float(fit.params.get("amplitude", np.nan)) / max(abs(pspl.Fs), 1e-12)
    q = (tE_2 / max(pspl.tE, 1e-12)) ** 2
    dx = (t0_2 - pspl.t0) / max(pspl.tE, 1e-12)
    dy_abs = u0_2 * np.sqrt(max(q, 0.0))
    seeds = [
        SeedCandidate(
            model_type="1L2S",
            class_label="second_source",
            params={
                "t0_1": pspl.t0,
                "u0_1": pspl.u0,
                "tE": pspl.tE,
                "t0_2": t0_2,
                "u0_2": u0_2,
                "q_flux": fs_ratio,
            },
            score=float(fit.score),
            source_atom=fit.atom_name,
            degeneracy_tag="binary_source",
            warnings=fit.warnings,
        )
    ]
    for sign in (1.0, -1.0):
        dy = pspl.u0 + sign * dy_abs
        seeds.append(
            SeedCandidate(
                model_type="2L1S",
                class_label="wide_repeating",
                params={"s": float(np.hypot(dx, dy)), "q": float(q), "alpha": float(np.arctan2(dy, dx))},
                score=float(fit.score) - 2.0,
                source_atom=fit.atom_name,
                degeneracy_tag="wide_repeating_plus" if sign > 0 else "wide_repeating_minus",
                warnings=fit.warnings,
            )
        )
    return tuple(seeds)
