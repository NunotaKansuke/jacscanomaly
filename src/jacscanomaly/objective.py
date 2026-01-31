from __future__ import annotations

import jax.numpy as jnp
from jax import jit
from typing import Callable

from .photometry import solve_fs_fb

@jit
def residual_norm(
    params: jnp.ndarray,
    data: tuple,
    A_func: Callable,
) -> jnp.ndarray:
    """
    Normalized residual for LM: (flux - model) / ferr
    """
    time, flux, ferr = data
    A = A_func(params, time)
    fs, fb = solve_fs_fb(A, flux, ferr)
    model_flux = fs * A + fb
    return (flux - model_flux) / ferr


@jit
def chi2(
    params: jnp.ndarray,
    data: tuple,
    A_func: Callable,
) -> jnp.ndarray:
    res = residual_norm(params, data, A_func)
    return jnp.sum(res**2)

