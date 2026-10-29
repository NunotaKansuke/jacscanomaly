"""Automatic initial guesses shared by every fitting entry point.

The finite-source seeds come from the observed-duration grid in
:mod:`jacscanomaly.fspl_initialization`: the observed brightening width fixes
the crossing time, and ``rho x (u0/rho) x t0-offset`` is scanned with profiled
source/blend fluxes.  A PSPL fit contributes only its ``t0`` as an event-centre
hint.  The standalone FSPL fitter, the parallax fitters and ``Finder`` all call
:func:`fspl_initial_guesses` instead of building their own copy.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Optional, Sequence

import numpy as np

from .config import FinderConfig
from .pspl_fft import PSPLFFTScanner

# Degeneracy-family grid shared by every automatic multistart (and by the
# detector fallback's FallbackConfig defaults).
SEED_TE_FACTORS = (0.75, 1.0, 1.5)
SEED_U0_FACTORS = (0.7, 1.0, 1.4)
SEED_MAX = 24
# A t0 hint is a rough "this brightening" locator: an FFT candidate matches it
# when its t0 lies within this many of its own Einstein times of the hint,
# but never farther than this fraction of the data span (so that a very wide
# template centred on another brightening cannot match).
T0_HINT_TE_COEFF = 1.0
T0_HINT_SPAN_FRACTION = 0.05
T0_HINT_PEAKS_PER_TEMPLATE = 8
# last PSPL FFT bank search, keyed by its data and settings
_SEARCH_CACHE: dict = {}



def fspl_initial_guesses(
    time,
    flux,
    ferr,
    *,
    top_k: int = 4,
    pspl_params=None,
    observed_scale=None,
    magnification_tol: float = 1.0e-4,
    magnification_reltol: float = 1.0e-4,
    espl_table_path: Optional[str] = None,
    parallax: bool = False,
) -> tuple[np.ndarray, ...]:
    """Return the ``top_k`` lowest-chi-square FSPL seeds.

    Seeds are ``(t0, tE, u0, ln rho)``; with ``parallax=True`` they are
    extended with ``(piEN, piEE) = (0, 0)``, the parallax-free point that the
    data constrain before any parallax is fitted.
    """
    # fspl_initialization imports the fitters, so import it lazily.
    from .fspl_initialization import fspl_template_initial_guesses

    proxy = SimpleNamespace(
        time=np.asarray(time, dtype=float),
        flux=np.asarray(flux, dtype=float),
        ferr=np.asarray(ferr, dtype=float),
        params=pspl_params,
    )
    seeds = fspl_template_initial_guesses(
        proxy,
        top_k=max(1, int(top_k)),
        pspl_params=pspl_params,
        observed_scale=observed_scale,
        magnification_tol=float(magnification_tol),
        magnification_reltol=float(magnification_reltol),
        espl_table_path=espl_table_path,
    )
    result = tuple(np.asarray(seed, dtype=float) for seed in seeds)
    if not result:
        raise RuntimeError("FSPL duration initializer returned no seeds.")
    if parallax:
        result = tuple(np.concatenate([seed, [0.0, 0.0]]) for seed in result)
    return result


def _pspl_seed(t0: float, tE: float, u0: float, parallax: bool) -> np.ndarray:
    values = [t0, tE, u0] + ([0.0, 0.0] if parallax else [])
    return np.asarray(values, dtype=float)


def pspl_initial_guesses(
    time,
    flux,
    ferr,
    *,
    config: Optional[FinderConfig] = None,
    parallax: bool = False,
    t0_hint: Optional[float] = None,
) -> tuple[np.ndarray, ...]:
    """Return PSPL seeds ``(t0, tE, u0)`` from a profiled ``tE -> (u0, t0)`` FFT bank.

    The scan ranks templates by profiled chi-square with free source and
    blend fluxes.  With ``parallax=True`` each seed is extended with
    ``(piEN, piEE) = (0, 0)``.  A rough ``t0_hint`` keeps only candidates
    within ``min(T0_HINT_TE_COEFF * tE, T0_HINT_SPAN_FRACTION * span)`` of it (all candidates if none match), so a
    different brightening cannot win the ranking.  ``config`` supplies the ``auto_init_*`` scan
    settings and defaults to :class:`FinderConfig`.
    """
    time = np.asarray(time, dtype=float)
    flux_np = np.asarray(flux, dtype=float)
    ferr_np = np.asarray(ferr, dtype=float)
    cfg = config if config is not None else FinderConfig()
    if cfg.auto_init_u0_min <= 0 or cfg.auto_init_u0_max <= 0:
        raise ValueError("auto_init_u0_min and auto_init_u0_max must be positive.")
    if cfg.auto_init_u0_max < cfg.auto_init_u0_min:
        raise ValueError("auto_init_u0_max must be >= auto_init_u0_min.")
    if cfg.auto_init_tE_min <= 0 or cfg.auto_init_tE_max <= 0:
        raise ValueError("auto_init_tE_min and auto_init_tE_max must be positive.")
    if cfg.auto_init_tE_max < cfg.auto_init_tE_min:
        raise ValueError("auto_init_tE_max must be >= auto_init_tE_min.")

    n_u0 = int(cfg.auto_init_u0_grid_n)
    n_tE = int(cfg.auto_init_fft_tE_grid_n)
    top_k = int(cfg.auto_init_fft_top_k)
    if n_tE < 1 or n_u0 < 1:
        raise ValueError("PSPL FFT template-grid sizes must be at least one.")
    if top_k < 1:
        raise ValueError("auto_init_fft_top_k must be at least one.")

    u0_grid = np.geomspace(
        float(cfg.auto_init_u0_min),
        float(cfg.auto_init_u0_max),
        n_u0,
    )
    tE_grid = np.geomspace(
        float(cfg.auto_init_tE_min),
        float(cfg.auto_init_tE_max),
        n_tE,
    )
    scanner = PSPLFFTScanner(
        grid_dt=cfg.auto_init_fft_grid_dt,
        max_grid_points=int(cfg.auto_init_fft_max_grid_points),
        fft_workers=int(cfg.auto_init_fft_workers),
    )
    # ``top_k`` is only a post-ranking truncation inside the FFT scanner.
    # Ask for the complete one-peak-per-template pool here so a handful of
    # gap/cancellation artefacts cannot hide the first usable seed.
    # With a t0 hint the global best peak of a template may be another
    # brightening, so keep several local peaks per template to find the hinted one.
    peaks = T0_HINT_PEAKS_PER_TEMPLATE if t0_hint is not None else 1
    candidate_pool_size = max(1, n_u0 * n_tE * peaks, int(cfg.auto_init_fft_top_k))
    # The PSPL and parallax fitters of one event scan the same data with the
    # same bank; the search is a pure function of these inputs, so reuse it.
    key = (
        hashlib.blake2b(b"".join(np.ascontiguousarray(x).tobytes() for x in (time, flux_np, ferr_np)),
                        digest_size=16).digest(),
        u0_grid.tobytes(), tE_grid.tobytes(), candidate_pool_size, peaks,
        float(cfg.auto_init_fft_grid_dt), int(cfg.auto_init_fft_max_grid_points),
    )
    search = _SEARCH_CACHE.get(key)
    if search is None:
        search = scanner.search_tE(
            time,
            flux_np,
            ferr_np,
            u0_grid=u0_grid,
            tE_grid=tE_grid,
            top_k=candidate_pool_size,
            peaks_per_template=peaks,
        )
        _SEARCH_CACHE.clear()
        _SEARCH_CACHE[key] = search

    candidates = filter_pspl_fft_candidates(
        search.candidates,
        config=cfg,
        time_np=time,
        flux_np=flux_np,
        limit=top_k,
        t0_hint=t0_hint,
    )
    if candidates:
        return tuple(
            _pspl_seed(candidate.t0, candidate.tE, candidate.u0, parallax)
            for candidate in candidates
        )

    # Keep the automatic workflow recoverable for flat or numerically
    # singular light curves. The subsequent fitter will decide whether
    # this conservative seed is usable.
    i_peak = int(np.nanargmax(flux_np))
    t0 = float(time[i_peak])
    teff = float(np.sqrt(float(cfg.auto_init_teff_min) * float(cfg.auto_init_teff_max)))
    u0 = float(np.sqrt(float(cfg.auto_init_u0_min) * float(cfg.auto_init_u0_max)))
    return (_pspl_seed(t0, teff / u0, u0, parallax),)



def filter_pspl_fft_candidates(
    candidates: Sequence,
    *,
    time_np: np.ndarray,
    flux_np: np.ndarray,
    limit: int,
    config: FinderConfig,
    t0_hint: Optional[float] = None,
) -> list:
    """Reject FFT seeds that are not supported by actual observations.

    The profiled FFT score can be dominated by a tiny PSPL wing in a long
    seasonal gap when the source and blend fluxes nearly cancel.  Apply the
    fitter's support condition before ranking, and reject that cancellation
    pathology at the initializer boundary.
    """
    cfg = config
    minimum_support = int(cfg.auto_init_min_t0_support_points)
    support_coeff = float(cfg.auto_init_t0_support_tE_coeff)
    nearest_coeff = float(cfg.auto_init_nearest_support_tE_coeff)
    cancellation_limit = float(cfg.auto_init_max_flux_cancellation_ratio)
    if minimum_support < 1 or support_coeff <= 0.0:
        raise ValueError(
            "PSPL automatic-initialization support settings must be positive."
        )
    if nearest_coeff <= 0.0:
        raise ValueError(
            "auto_init_nearest_support_tE_coeff must be positive."
        )

    time_values = np.asarray(time_np, dtype=float)
    flux_values = np.asarray(flux_np, dtype=float)
    median_flux = float(np.median(np.abs(flux_values)))
    selected = []
    max_selected = max(1, int(limit))
    span = float(np.ptp(time_values))

    for candidate in candidates:
        t0 = float(candidate.t0)
        t_e = abs(float(candidate.tE))
        u0 = abs(float(candidate.u0))
        fs = float(candidate.fs)
        fb = float(candidate.fb)
        if not all(np.isfinite(value) for value in (t0, t_e, u0, fs, fb)):
            continue
        if t_e <= 0.0 or u0 < float(cfg.auto_init_u0_min):
            continue

        distances = np.abs(time_values - t0)
        support = int(np.count_nonzero(distances <= support_coeff * t_e))
        if support < minimum_support:
            continue
        if float(np.min(distances)) > nearest_coeff * t_e:
            continue

        baseline = max(abs(fs + fb), median_flux, 1.0e-30)
        cancellation = (abs(fs) + abs(fb)) / baseline
        if np.isfinite(cancellation_limit) and cancellation > cancellation_limit:
            continue

        if t0_hint is not None and abs(t0 - float(t0_hint)) > min(
            T0_HINT_TE_COEFF * t_e, T0_HINT_SPAN_FRACTION * span
        ):
            continue

        selected.append(candidate)
        if len(selected) >= max_selected:
            break

    if not selected and t0_hint is not None:
        return filter_pspl_fft_candidates(
            candidates, time_np=time_np, flux_np=flux_np, limit=limit, config=config
        )

    return selected



def seed_family(
    anchor,
    *,
    tE_factors: Sequence[float],
    u0_factors: Sequence[float],
    u0_sign_flip: bool,
    t_star_factors: Sequence[float] = (),
    rho_over_u0: Sequence[float] = (),
    t0_offsets: Sequence[float] = (0.0,),
) -> list[np.ndarray]:
    """Degeneracy family around one anchor seed, in a fixed priority order.

    * ``tE x u0`` rescalings, each also with the opposite ``u0`` sign.  The sign
      is a genuine degeneracy for point lenses and, with parallax, selects
      physically different solutions, so both signs are always tried.
    * For finite-source anchors (pass ``t_star_factors``), a ``rho`` grid built
      from ``rho = sqrt((rho/u0) * (t_star / tE))`` with ``t0`` offsets in units
      of ``tE * u0`` (the 4th entry of ``anchor`` is ``ln rho``).
    """
    anchor = np.asarray(anchor, dtype=float).reshape(-1)
    seeds: list[np.ndarray] = []
    finite_source = len(tuple(t_star_factors)) > 0
    tstar0 = max(abs(anchor[1]) * np.exp(anchor[3]), 1.0e-8) if finite_source else 0.0
    for factor in tE_factors:
        for u0_factor in u0_factors:
            candidate = anchor.copy()
            candidate[1] *= float(factor)
            candidate[2] *= float(u0_factor)
            seeds.append(candidate)
            if u0_sign_flip:
                flipped = candidate.copy()
                flipped[2] *= -1.0
                seeds.append(flipped)
            for tstar_factor in t_star_factors:
                for ratio in rho_over_u0:
                    fspl_seed = candidate.copy()
                    rho_ratio = max(float(ratio) * abs(fspl_seed[2]), 1.0e-6)
                    rho_tstar = max(
                        float(tstar_factor) * tstar0 / max(abs(fspl_seed[1]), 1.0e-8),
                        1.0e-6,
                    )
                    fspl_seed[3] = np.log(np.sqrt(rho_ratio * rho_tstar))
                    for offset in t0_offsets:
                        shifted = fspl_seed.copy()
                        shifted[0] += float(offset) * max(abs(float(candidate[1] * candidate[2])), 1.0e-6)
                        seeds.append(shifted)
    return seeds


def automatic_seeds(
    time,
    flux,
    ferr,
    *,
    finite_source: bool,
    parallax: bool = False,
    max_seeds: int = SEED_MAX,
    pspl_params=None,
    observed_scale=None,
    magnification_tol: float = 1.0e-4,
    magnification_reltol: float = 1.0e-4,
    espl_table_path: Optional[str] = None,
    config: Optional[FinderConfig] = None,
) -> tuple[np.ndarray, ...]:
    """The one automatic multistart set used by every fitter when no seed is given.

    Anchors are the data-driven initial guesses (PSPL FFT bank, or the FSPL
    duration grid, whose ``rho`` comes from the observed duration), followed by
    (with ``parallax=True`` only) the ``tE x u0`` degeneracy family -- each with
    both ``u0`` signs -- around each anchor in priority order, capped at
    ``max_seeds``.  Assumptions:
    ``rho`` is not re-gridded around finite-source anchors, and parallax
    starts at ``piE = 0`` (no direction hint exists without a detector).
    ``pspl_params`` contributes only its ``t0``, as a rough locator of the
    brightening to fit (its ``tE`` and ``u0`` are never used).
    """
    if finite_source:
        anchors = fspl_initial_guesses(
            time, flux, ferr, top_k=4, pspl_params=pspl_params,
            observed_scale=observed_scale,
            magnification_tol=magnification_tol,
            magnification_reltol=magnification_reltol,
            espl_table_path=espl_table_path, parallax=parallax,
        )
    else:
        t0_hint = None
        if pspl_params is not None and np.size(pspl_params) >= 1 and np.isfinite(pspl_params[0]):
            t0_hint = float(pspl_params[0])
        anchors = pspl_initial_guesses(
            time, flux, ferr, config=config, parallax=parallax, t0_hint=t0_hint
        )
    seeds = list(anchors)
    # Without parallax the model depends on u0 only through |u|, so the sign
    # flip is an exact duplicate and the tE x u0 family was never part of the
    # point-lens fits: keep just the anchors.  With parallax the sign selects
    # physically different solutions, so the family is added.
    for anchor in anchors if parallax else ():
        seeds.extend(
            seed_family(
                anchor,
                tE_factors=SEED_TE_FACTORS,
                u0_factors=SEED_U0_FACTORS,
                u0_sign_flip=True,
            )
        )
    unique: list[np.ndarray] = []
    for seed in seeds:
        if not any(np.allclose(seed, other, rtol=0.0, atol=1.0e-12) for other in unique):
            unique.append(np.asarray(seed, dtype=float))
        if len(unique) >= max(1, int(max_seeds)):
            break
    return tuple(unique)


def best_fit_over_seeds(fit_one, seeds, label: str):
    """Fit every seed with ``fit_one(seed)`` and return the lowest-chi-square fit."""
    best, best_chi2, errors = None, np.inf, []
    for seed in seeds:
        try:
            candidate = fit_one(seed)
            chi2 = float(np.asarray(candidate.chi2))
        except Exception as exc:
            errors.append(exc)
            continue
        if np.isfinite(chi2) and chi2 < best_chi2:
            best, best_chi2 = candidate, chi2
    if best is None:
        message = f"All automatic {label} initial guesses failed."
        if errors:
            message += f" First error: {errors[0]}"
        raise RuntimeError(message)
    return best
