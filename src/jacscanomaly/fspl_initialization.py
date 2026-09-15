"""Observed-duration finite-source grid initialization.

The default initializer is the data-centered route used by the 3008
benchmark: measure the observed brightening width, convert it to a set of
effective source-crossing times, build transit-shape seeds, and rank every
seed by the original-flux profiled chi-square.  A PSPL seed may be supplied
to locate the observed event, but no PSPL residual is used as the fitting
template.

The older PSPL-centered keyword grid remains as a compatibility path when a
caller explicitly supplies its legacy grid arguments.  It is not used by the
canonical FSPL fitters or Finder initialization.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional, Sequence

import numpy as np

from .fitters import FSPLFitter, _solve_fs_fb_numpy, _vbm_cpp
from .signal_scale import ObservedSignalScale, measure_observed_magnification_scale


_DEFAULT_CROSSING_TIME_FACTORS = (0.0625, 0.125, 0.25, 0.5, 1.0)
_DEFAULT_RHO_VALUES = (0.01, 0.3, 1.0, 3.0)
_DEFAULT_U0_OVER_RHO_VALUES = (0.05, 0.4, 0.8)
_DEFAULT_T0_OFFSET_WIDTHS = (-0.25, 0.0, 0.25)


def _pspl_magnification(u: np.ndarray) -> np.ndarray:
    """Return point-source magnification for a separation array."""
    safe_u = np.maximum(np.asarray(u, dtype=float), 1.0e-12)
    return (safe_u * safe_u + 2.0) / (
        safe_u * np.sqrt(safe_u * safe_u + 4.0)
    )


def _rectilinear_separation(
    time: np.ndarray,
    t0: float,
    tE: float,
    u0: float,
) -> np.ndarray:
    """Return the rectilinear lens-source separation for one grid point."""
    return np.sqrt(
        ((time - float(t0)) / max(abs(float(tE)), 1.0e-12)) ** 2
        + float(u0) ** 2
    )


def _profiled_chi2(
    magnification: np.ndarray,
    flux: np.ndarray,
    ferr: np.ndarray,
) -> float:
    """Profile ``Fs`` and ``Fb`` and return the original-data chi-square."""
    values = np.asarray(magnification, dtype=float).reshape(-1)
    if not np.all(np.isfinite(values)):
        return float("inf")
    fs, fb = _solve_fs_fb_numpy(values, flux, ferr)
    if not np.isfinite(fs) or not np.isfinite(fb):
        return float("inf")
    normalized = (np.asarray(flux, dtype=float) - (fs * values + fb)) / ferr
    if not np.all(np.isfinite(normalized)):
        return float("inf")
    return float(np.dot(normalized, normalized))


def _native_magnification(
    u: np.ndarray,
    rho: float,
    *,
    native_support_rho: Optional[float],
    native_support_floor: float,
) -> np.ndarray:
    """Evaluate the optional Python-bound native VBM backend."""
    from .effect_detection import _native_fspl_magnification

    separation = np.asarray(u, dtype=float)
    if native_support_rho is None:
        support = np.ones_like(separation, dtype=bool)
    else:
        support_limit = max(
            float(native_support_floor),
            float(native_support_rho) * float(rho),
        )
        support = separation <= support_limit

    values = _pspl_magnification(separation)
    if np.any(support):
        values[support] = _native_fspl_magnification(separation[support], rho)
    return values


def _microjax_magnification(
    u: np.ndarray,
    rho: float,
    *,
    N_fft: int,
) -> np.ndarray:
    """Evaluate the optional microjax cross-validation backend."""
    import jax.numpy as jnp

    from .magnification import A_fspl_from_u

    return np.asarray(
        A_fspl_from_u(jnp.asarray(u), float(rho), N_fft=int(N_fft)),
        dtype=float,
    )


def _legacy_fspl_template_initial_guesses(
    fit,
    *,
    top_k: int = 4,
    rho_over_u0: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
    tE_factors: Sequence[float] = (0.5, 1.0, 2.0),
    u0_factors: Sequence[float] = (0.5, 1.0, 2.0),
    t0_offsets: Sequence[float] = (0.0,),
    u0_signs: Sequence[float] = (-1.0, 1.0),
    N_fft: int = 1024,
    backend: str = "compiled",
    native_support_rho: Optional[float] = 10.0,
    native_support_floor: float = 3.0,
    magnification_tol: float = 1.0e-4,
    magnification_reltol: float = 1.0e-4,
) -> tuple[np.ndarray, ...]:
    """Return seeds from the pre-duration PSPL-centered compatibility grid.

    ``fit`` must expose ``time``, ``flux``, ``ferr``, and ``params``.  The
    first three entries of ``params`` are used as the center of the
    ``(t0, tE, u0, rho)`` grid.  ``t0_offsets`` are absolute offsets in the
    input time units, and ``u0_factors`` multiply the PSPL ``|u0|``.  For
    every grid point this function
    evaluates the FSPL magnification and solves the linear source/blend fluxes
    against the original observed flux.  It does not use PSPL residuals,
    PSPL fluxes, or a nuisance-subspace projection.

    The default ``compiled`` backend calls the same canonical
    :class:`~jacscanomaly.fitters.FSPLFitter` ``ESPLMag2`` path used by the
    subsequent nonlinear fit.  ``native`` and ``microjax`` are retained for
    explicit cross-checks.
    """
    limit = int(top_k)
    if limit < 1:
        raise ValueError("top_k must be at least one.")

    time = np.asarray(fit.time, dtype=float).reshape(-1)
    flux = np.asarray(fit.flux, dtype=float).reshape(-1)
    ferr = np.asarray(fit.ferr, dtype=float).reshape(-1)
    params = np.asarray(fit.params, dtype=float).reshape(-1)
    if params.size < 3:
        raise ValueError("fit.params must contain (t0, tE, u0).")
    if not (time.shape == flux.shape == ferr.shape):
        raise ValueError("fit time, flux, and ferr must have equal lengths.")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(flux)):
        raise ValueError("fit time and flux must be finite.")
    if not np.all(np.isfinite(ferr)) or np.any(ferr <= 0.0):
        raise ValueError("fit ferr must be finite and positive.")
    if not np.all(np.isfinite(params[:3])) or params[1] <= 0.0:
        raise ValueError("fit.params must contain a finite positive tE.")

    ratios = np.asarray(tuple(rho_over_u0), dtype=float).reshape(-1)
    timescale_factors = np.asarray(tuple(tE_factors), dtype=float).reshape(-1)
    impact_factors = np.asarray(tuple(u0_factors), dtype=float).reshape(-1)
    peak_offsets = np.asarray(tuple(t0_offsets), dtype=float).reshape(-1)
    signs = np.asarray(tuple(u0_signs), dtype=float).reshape(-1)
    if (
        ratios.size == 0
        or not np.all(np.isfinite(ratios))
        or np.any(ratios <= 0.0)
    ):
        raise ValueError("rho_over_u0 must contain positive finite values.")
    if (
        timescale_factors.size == 0
        or not np.all(np.isfinite(timescale_factors))
        or np.any(timescale_factors <= 0.0)
    ):
        raise ValueError("tE_factors must contain positive finite values.")
    if (
        impact_factors.size == 0
        or not np.all(np.isfinite(impact_factors))
        or np.any(impact_factors <= 0.0)
    ):
        raise ValueError("u0_factors must contain positive finite values.")
    if peak_offsets.size == 0 or not np.all(np.isfinite(peak_offsets)):
        raise ValueError("t0_offsets must contain finite values.")
    if signs.size == 0 or not np.all(np.isfinite(signs)) or np.any(signs == 0.0):
        raise ValueError("u0_signs must contain finite nonzero values.")
    if int(N_fft) < 1:
        raise ValueError("N_fft must be at least one.")
    if not np.isfinite(native_support_floor) or native_support_floor <= 0.0:
        raise ValueError("native_support_floor must be positive and finite.")
    if native_support_rho is not None and (
        not np.isfinite(native_support_rho) or native_support_rho <= 0.0
    ):
        raise ValueError("native_support_rho must be positive and finite or None.")

    backend_name = str(backend).lower()
    if backend_name not in {"compiled", "native", "microjax"}:
        raise ValueError("backend must be 'compiled', 'native', or 'microjax'.")

    t0 = float(params[0])
    tE0 = float(params[1])
    u0_abs = max(abs(float(params[2])), 1.0e-3)
    compiled_fitter = None
    if backend_name == "compiled":
        compiled_fitter = FSPLFitter(
            magnification_tol=float(magnification_tol),
            magnification_reltol=float(magnification_reltol),
        )

    ranked: list[tuple[float, np.ndarray]] = []
    for t0_offset in peak_offsets:
        candidate_t0 = t0 + float(t0_offset)
        for tE_factor in timescale_factors:
            tE = tE0 * float(tE_factor)
            for u0_factor in impact_factors:
                candidate_u0_abs = max(u0_abs * float(u0_factor), 1.0e-6)
                u_abs = _rectilinear_separation(
                    time,
                    candidate_t0,
                    tE,
                    candidate_u0_abs,
                )
                for ratio in ratios:
                    rho = min(
                        max(float(ratio) * candidate_u0_abs, 1.0e-6),
                        10.0,
                    )
                    if backend_name == "compiled":
                        assert compiled_fitter is not None
                        magnification = compiled_fitter._magnification(
                            time,
                            np.asarray(
                                [candidate_t0, tE, candidate_u0_abs, np.log(rho)],
                                dtype=float,
                            ),
                        )
                    elif backend_name == "native":
                        magnification = _native_magnification(
                            u_abs,
                            rho,
                            native_support_rho=native_support_rho,
                            native_support_floor=float(native_support_floor),
                        )
                    else:
                        magnification = _microjax_magnification(
                            u_abs,
                            rho,
                            N_fft=int(N_fft),
                        )

                    chi2 = _profiled_chi2(magnification, flux, ferr)
                    for sign in signs:
                        seed = np.asarray(
                            [
                                candidate_t0,
                                tE,
                                float(sign) * candidate_u0_abs,
                                np.log(rho),
                            ],
                            dtype=float,
                        )
                        ranked.append((chi2, seed))

    ranked.sort(key=lambda item: item[0])
    seeds: list[np.ndarray] = []
    for chi2, seed in ranked:
        if not np.isfinite(chi2):
            continue
        if not any(np.allclose(seed, other, rtol=0.0, atol=1.0e-12) for other in seeds):
            seeds.append(seed)
        if len(seeds) >= limit:
            break
    if not seeds:
        raise RuntimeError("FSPL grid initialization produced no finite chi2.")
    return tuple(seeds)


def _sequence_matches(values: Sequence[float], expected: Sequence[float]) -> bool:
    """Return whether two public keyword grids contain the same values."""
    try:
        actual = np.asarray(tuple(values), dtype=float).reshape(-1)
        reference = np.asarray(tuple(expected), dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return False
    return actual.shape == reference.shape and np.array_equal(actual, reference)


def _coerce_observed_scale(value: Any) -> Optional[ObservedSignalScale]:
    """Accept the native scale object and its JSON summary representation."""
    if value is None:
        return None
    if isinstance(value, ObservedSignalScale):
        return value
    if isinstance(value, Mapping):
        center = float(value.get("t_center", np.nan))
        width = float(value.get("width", np.nan))
        half_width = float(value.get("half_width", 0.5 * width))
        if not np.isfinite(width) and np.isfinite(half_width):
            width = 2.0 * half_width
        if not np.isfinite(half_width) and np.isfinite(width):
            half_width = 0.5 * width
        t_left = float(value.get("t_left", center - half_width))
        t_right = float(value.get("t_right", center + half_width))
        return ObservedSignalScale(
            t_center=center,
            t_left=t_left,
            t_right=t_right,
            half_width=half_width,
            cadence=float(value.get("cadence", np.nan)),
            n_points=int(value.get("n_points", 0)),
            n_weighted_points=int(value.get("n_weighted_points", 0)),
            left_coverage=float(value.get("left_coverage", 1.0)),
            right_coverage=float(value.get("right_coverage", 1.0)),
            asymmetry=float(value.get("asymmetry", np.nan)),
            valid=bool(value.get("valid", True)),
            censored=bool(value.get("censored", False)),
            source=str(value.get("source", "observed_scale")),
        )
    if all(hasattr(value, name) for name in ("t_center", "width", "valid")):
        return value
    raise TypeError("observed_scale must be an ObservedSignalScale or mapping.")


def _extract_pspl_parameters(
    fit: Any,
    pspl_params: Optional[Sequence[float]],
) -> Optional[np.ndarray]:
    """Return an optional ``(t0, tE, u0)`` locator without fitting PSPL."""
    value = pspl_params
    if value is None and fit is not None:
        value = getattr(fit, "params", None)
    if value is None:
        return None
    if hasattr(value, "params"):
        value = getattr(value, "params")
    params = np.asarray(value, dtype=float).reshape(-1)
    if params.size < 3 or not np.all(np.isfinite(params[:3])):
        return None
    if float(params[1]) == 0.0:
        return None
    return np.asarray(params[:3], dtype=float)


def _fallback_duration_scale(
    time: np.ndarray,
    flux: np.ndarray,
    pspl_params: Optional[np.ndarray],
) -> tuple[float, float]:
    """Return a conservative scale if the observed-width measurement fails."""
    finite = np.isfinite(time) & np.isfinite(flux)
    if not np.any(finite):
        raise RuntimeError("FSPL initialization has no finite observations.")
    tv = np.asarray(time[finite], dtype=float)
    yv = np.asarray(flux[finite], dtype=float)
    if pspl_params is None:
        center = float(tv[int(np.nanargmax(yv))])
    else:
        center = float(pspl_params[0])

    steps = np.diff(np.sort(tv))
    steps = steps[np.isfinite(steps) & (steps > 0.0)]
    cadence = float(np.median(steps)) if steps.size else 1.0
    if pspl_params is not None and np.isfinite(pspl_params[1]):
        width = 2.0 * abs(float(pspl_params[1])) * max(abs(float(pspl_params[2])), 1.0e-3)
    else:
        width = 4.0 * cadence
    width = max(width, 2.0 * cadence, 1.0e-6)
    return center, width


def _observed_duration_scale(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    pspl_params: Optional[np.ndarray],
    observed_scale: Optional[ObservedSignalScale],
) -> tuple[float, float]:
    """Resolve the observed center/full width used by the duration grid."""
    if observed_scale is not None and bool(getattr(observed_scale, "valid", False)):
        center = float(observed_scale.t_center)
        width = float(observed_scale.width)
        if np.isfinite(center) and np.isfinite(width) and width > 0.0:
            return center, width

    finite = np.isfinite(time) & np.isfinite(flux)
    if not np.any(finite):
        return _fallback_duration_scale(time, flux, pspl_params)
    tv = np.asarray(time[finite], dtype=float)
    yv = np.asarray(flux[finite], dtype=float)
    if pspl_params is not None:
        center = float(pspl_params[0])
        search_half_width = max(
            1.0,
            min(300.0, 20.0 * abs(float(pspl_params[1]))),
        )
    else:
        center = float(tv[int(np.nanargmax(yv))])
        steps = np.diff(np.sort(tv))
        steps = steps[np.isfinite(steps) & (steps > 0.0)]
        cadence = float(np.median(steps)) if steps.size else 1.0
        # The initializer needs the central brightening envelope, not the
        # multi-tE PSPL wing.  A short cadence-scaled window keeps ordinary
        # long-tE events on the same source-crossing scale as the benchmark.
        search_half_width = max(1.0, min(5.0, 200.0 * cadence))

    measured = measure_observed_magnification_scale(
        time,
        flux,
        center=center,
        search_half_width=search_half_width,
        n_bins=256,
        source="observed_magnification",
    )
    if measured.valid and np.isfinite(measured.width) and measured.width > 0.0:
        return float(measured.t_center), float(measured.width)
    return _fallback_duration_scale(time, flux, pspl_params)


def _build_fspl_duration_seed_grid(
    t_center: float,
    effective_crossing_time: float,
    *,
    rho_values: Sequence[float],
    u0_over_rho_values: Sequence[float],
    t0_offsets: Sequence[float],
) -> np.ndarray:
    """Build ``(t0, tE, u0, logrho)`` seeds from transit coordinates."""
    center = float(t_center)
    crossing_time = float(effective_crossing_time)
    rho_grid = np.asarray(tuple(rho_values), dtype=float).reshape(-1)
    impact_grid = np.asarray(tuple(u0_over_rho_values), dtype=float).reshape(-1)
    offset_grid = np.asarray(tuple(t0_offsets), dtype=float).reshape(-1)
    if not np.isfinite(center):
        raise ValueError("t_center must be finite.")
    if not np.isfinite(crossing_time) or crossing_time <= 0.0:
        raise ValueError("effective_crossing_time must be positive and finite.")
    if rho_grid.size == 0 or not np.all(np.isfinite(rho_grid)) or np.any(rho_grid <= 0.0):
        raise ValueError("rho_values must contain positive finite values.")
    if (
        impact_grid.size == 0
        or not np.all(np.isfinite(impact_grid))
        or np.any(impact_grid < 0.0)
        or np.any(impact_grid >= 1.0)
    ):
        raise ValueError("u0_over_rho_values must lie in [0, 1).")
    if offset_grid.size == 0 or not np.all(np.isfinite(offset_grid)):
        raise ValueError("t0_offsets must contain finite values.")

    seeds = []
    for t0_offset in offset_grid:
        for rho in rho_grid:
            for u0_over_rho in impact_grid:
                u0 = float(u0_over_rho * rho)
                crossing_scale = float(np.sqrt(rho * rho - u0 * u0))
                tE = crossing_time / crossing_scale
                seeds.append(
                    [center + float(t0_offset), tE, u0, float(np.log(rho))]
                )
    return np.asarray(seeds, dtype=float)


def fspl_template_initial_guesses(
    fit=None,
    *,
    top_k: int = 4,
    rho_over_u0: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
    tE_factors: Sequence[float] = (0.5, 1.0, 2.0),
    u0_factors: Sequence[float] = (0.5, 1.0, 2.0),
    t0_offsets: Sequence[float] = (0.0,),
    u0_signs: Sequence[float] = (-1.0, 1.0),
    N_fft: int = 1024,
    backend: str = "compiled",
    native_support_rho: Optional[float] = 10.0,
    native_support_floor: float = 3.0,
    magnification_tol: float = 1.0e-4,
    magnification_reltol: float = 1.0e-4,
    pspl_params: Optional[Sequence[float]] = None,
    observed_scale: Optional[ObservedSignalScale | Mapping[str, object]] = None,
    width_to_crossing_time: float = 1.0,
    crossing_time_factors: Sequence[float] = _DEFAULT_CROSSING_TIME_FACTORS,
    rho_values: Sequence[float] = _DEFAULT_RHO_VALUES,
    u0_over_rho_values: Sequence[float] = _DEFAULT_U0_OVER_RHO_VALUES,
    t0_offset_widths: Sequence[float] = _DEFAULT_T0_OFFSET_WIDTHS,
    espl_table_path: Optional[str] = None,
) -> tuple[np.ndarray, ...]:
    """Return FSPL seeds ranked by direct chi-square on the original flux.

    The default route uses the observed brightening duration.  For each
    crossing-time factor it builds the transit grid
    ``rho x (u0/rho) x t0-width-offset`` and evaluates every candidate with
    profiled ``Fs`` and ``Fb``.  With the benchmark defaults this is
    ``5 * 4 * 3 * 3 = 180`` seeds.  A supplied ``pspl_params`` is used only
    to locate the observed interval; the PSPL residual is never used as a
    template.  ``fit`` may expose ``observed_signal_scale`` to reuse a scale
    already measured by the detector.

    The legacy PSPL-centered keyword grid remains available when one of its
    legacy grid arguments is explicitly changed.  This preserves callers of
    the previous public API while keeping the canonical default route fixed
    to the duration grid.
    """
    limit = int(top_k)
    if limit < 1:
        raise ValueError("top_k must be at least one.")
    if fit is None:
        raise ValueError("fit must expose time, flux, and ferr.")

    backend_name = str(backend).lower()
    legacy_requested = (
        backend_name != "compiled"
        or int(N_fft) != 1024
        or not _sequence_matches(rho_over_u0, (0.25, 0.5, 1.0, 2.0, 4.0))
        or not _sequence_matches(tE_factors, (0.5, 1.0, 2.0))
        or not _sequence_matches(u0_factors, (0.5, 1.0, 2.0))
        or not _sequence_matches(t0_offsets, (0.0,))
        or not _sequence_matches(u0_signs, (-1.0, 1.0))
        or native_support_rho != 10.0
        or float(native_support_floor) != 3.0
    )
    if legacy_requested:
        return _legacy_fspl_template_initial_guesses(
            fit,
            top_k=limit,
            rho_over_u0=rho_over_u0,
            tE_factors=tE_factors,
            u0_factors=u0_factors,
            t0_offsets=t0_offsets,
            u0_signs=u0_signs,
            N_fft=N_fft,
            backend=backend,
            native_support_rho=native_support_rho,
            native_support_floor=native_support_floor,
            magnification_tol=magnification_tol,
            magnification_reltol=magnification_reltol,
        )

    time = np.asarray(getattr(fit, "time"), dtype=float).reshape(-1)
    flux = np.asarray(getattr(fit, "flux"), dtype=float).reshape(-1)
    ferr = np.asarray(getattr(fit, "ferr"), dtype=float).reshape(-1)
    if not (time.shape == flux.shape == ferr.shape) or time.size == 0:
        raise ValueError("fit time, flux, and ferr must have equal nonzero lengths.")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(flux)):
        raise ValueError("fit time and flux must be finite.")
    if not np.all(np.isfinite(ferr)) or np.any(ferr <= 0.0):
        raise ValueError("fit ferr must be finite and positive.")

    pspl = _extract_pspl_parameters(fit, pspl_params)
    supplied_scale = _coerce_observed_scale(
        observed_scale
        if observed_scale is not None
        else getattr(fit, "observed_signal_scale", None)
    )
    center, observed_width = _observed_duration_scale(
        time,
        flux,
        pspl_params=pspl,
        observed_scale=supplied_scale,
    )
    width_factor = float(width_to_crossing_time)
    if not np.isfinite(width_factor) or width_factor <= 0.0:
        raise ValueError("width_to_crossing_time must be positive and finite.")
    crossing_factors = np.asarray(tuple(crossing_time_factors), dtype=float).reshape(-1)
    offset_widths = np.asarray(tuple(t0_offset_widths), dtype=float).reshape(-1)
    if (
        crossing_factors.size == 0
        or not np.all(np.isfinite(crossing_factors))
        or np.any(crossing_factors <= 0.0)
    ):
        raise ValueError("crossing_time_factors must contain positive finite values.")
    if offset_widths.size == 0 or not np.all(np.isfinite(offset_widths)):
        raise ValueError("t0_offset_widths must contain finite values.")

    t0_offsets_grid = observed_width * offset_widths
    seeds = np.vstack(
        [
            _build_fspl_duration_seed_grid(
                center,
                observed_width * width_factor * float(crossing_factor),
                rho_values=rho_values,
                u0_over_rho_values=u0_over_rho_values,
                t0_offsets=t0_offsets_grid,
            )
            for crossing_factor in crossing_factors
        ]
    )

    compiled_fitter = FSPLFitter(
        magnification_tol=float(magnification_tol),
        magnification_reltol=float(magnification_reltol),
        espl_table_path=espl_table_path,
    )
    scores = None
    if _vbm_cpp is not None and hasattr(_vbm_cpp, "score_fspl_seeds"):
        scores = np.asarray(
            _vbm_cpp.score_fspl_seeds(
                time,
                flux,
                ferr,
                seeds,
                espl_table=compiled_fitter.espl_table_path,
                tol=float(magnification_tol),
                reltol=float(magnification_reltol),
            ),
            dtype=float,
        ).reshape(-1)
    if scores is None:
        scores = np.asarray(
            [
                _profiled_chi2(compiled_fitter._magnification(time, seed), flux, ferr)
                for seed in seeds
            ],
            dtype=float,
        )
    if scores.shape != (seeds.shape[0],):
        raise RuntimeError("FSPL duration-grid scorer returned an invalid score array.")

    ranked = [
        (float(chi2), np.asarray(seed, dtype=float))
        for chi2, seed in zip(scores, seeds)
        if np.isfinite(chi2)
    ]

    ranked.sort(key=lambda item: item[0])
    if not ranked:
        raise RuntimeError("FSPL duration-grid initialization produced no finite chi2.")
    return tuple(seed for _, seed in ranked[:limit])


__all__ = ["fspl_template_initial_guesses"]
