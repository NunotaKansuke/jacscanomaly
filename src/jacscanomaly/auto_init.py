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
from typing import Optional

import numpy as np



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
