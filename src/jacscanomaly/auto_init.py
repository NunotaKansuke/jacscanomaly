"""Automatic initial guesses shared by every fitting entry point.

The finite-source seeds come from the observed-duration grid in
:mod:`jacscanomaly.fspl_initialization`: the observed brightening width fixes
the crossing time, and ``rho x (u0/rho) x t0-offset`` is scanned with profiled
source/blend fluxes.  A PSPL fit contributes only its ``t0`` as an event-centre
hint.  The standalone FSPL fitter, the parallax fitters and ``Finder`` all call
:func:`fspl_initial_guesses` instead of building their own copy.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional, Sequence

import numpy as np

from .config import FinderConfig
from .pspl_fft import PSPLFFTScanner



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
) -> tuple[np.ndarray, ...]:
    """Return PSPL seeds ``(t0, tE, u0)`` from a profiled ``tE -> (u0, t0)`` FFT bank.

    The scan ranks templates by profiled chi-square with free source and
    blend fluxes.  With ``parallax=True`` each seed is extended with
    ``(piEN, piEE) = (0, 0)``.  ``config`` supplies the ``auto_init_*`` scan
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
    candidate_pool_size = max(1, n_u0 * n_tE, int(cfg.auto_init_fft_top_k))
    search = scanner.search_tE(
        time,
        flux_np,
        ferr_np,
        u0_grid=u0_grid,
        tE_grid=tE_grid,
        top_k=candidate_pool_size,
    )

    candidates = filter_pspl_fft_candidates(
        search.candidates,
        config=cfg,
        time_np=time,
        flux_np=flux_np,
        limit=top_k,
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

        selected.append(candidate)
        if len(selected) >= max_selected:
            break

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
