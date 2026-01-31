from __future__ import annotations

import jax.numpy as jnp
from jax import jit
from .photometry import solve_fs_fb

@jit
def residual_norm_from_A(A: jnp.ndarray, flux: jnp.ndarray, ferr: jnp.ndarray) -> jnp.ndarray:
    fs, fb = solve_fs_fb(A, flux, ferr)
    model_flux = fs * A + fb
    return (flux - model_flux) / ferr

@jit
def chi2_from_res(res: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(res**2)

