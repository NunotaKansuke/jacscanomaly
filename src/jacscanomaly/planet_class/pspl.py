from __future__ import annotations

import numpy as np

from ..planet_signal import PlanetSignalResult
from .types import PSPLParams


def pspl_params_from_result(result: PlanetSignalResult) -> PSPLParams:
    params = np.asarray(result.refined_fit.params, dtype=float)
    if params.size < 3:
        raise ValueError("Planet anomaly classification requires at least t0, tE, u0 baseline parameters.")
    return PSPLParams(
        t0=float(params[0]),
        tE=float(abs(params[1])),
        u0=float(params[2]),
        Fs=float(np.asarray(result.refined_fit.fs, dtype=float)),
        Fb=float(np.asarray(result.refined_fit.fb, dtype=float)),
    )


def u_vec(t: float | np.ndarray, pspl: PSPLParams) -> np.ndarray:
    t_arr = np.asarray(t, dtype=float)
    return np.stack(((t_arr - pspl.t0) / pspl.tE, np.full_like(t_arr, pspl.u0)), axis=0)


def u_abs(t: float | np.ndarray, pspl: PSPLParams) -> np.ndarray:
    v = u_vec(t, pspl)
    return np.sqrt(np.sum(v * v, axis=0))


def pspl_magnification_from_u(u: np.ndarray) -> np.ndarray:
    u_safe = np.maximum(np.asarray(u, dtype=float), 1e-12)
    return (u_safe * u_safe + 2.0) / (u_safe * np.sqrt(u_safe * u_safe + 4.0))


def pspl_flux(time: np.ndarray, pspl: PSPLParams) -> np.ndarray:
    return pspl.Fs * pspl_magnification_from_u(u_abs(time, pspl)) + pspl.Fb


def r_major(u: float | np.ndarray) -> np.ndarray:
    u_arr = np.asarray(u, dtype=float)
    return 0.5 * (np.sqrt(u_arr * u_arr + 4.0) + u_arr)


def r_minor(u: float | np.ndarray) -> np.ndarray:
    u_arr = np.asarray(u, dtype=float)
    return 0.5 * (np.sqrt(u_arr * u_arr + 4.0) - u_arr)


def angle_of(v: np.ndarray) -> float:
    arr = np.asarray(v, dtype=float)
    return float(np.arctan2(arr[1], arr[0]))
