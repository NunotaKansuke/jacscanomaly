from __future__ import annotations

import jax.numpy as jnp

from .trajectory import u_rectilinear, u_parallax
from .magnification import A_pspl_from_u, A_fspl_from_u

def A_pspl_func(params: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
    t0, tE, u0 = params
    u = u_rectilinear(t0, tE, u0, time)
    return A_pspl_from_u(u)

def A_fspl_func(params: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
    t0, tE, u0, rho = params
    u = u_rectilinear(t0, tE, u0, time)
    return A_fspl_from_u(u, rho)

def A_fspl_logrho_func(q: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
    t0, tE, u0, logrho = q
    rho = jnp.exp(logrho)
    u = u_rectilinear(t0, tE, u0, time)
    return A_fspl_from_u(u, rho)

def A_pspl_parallax_func(params: jnp.ndarray, time: jnp.ndarray, P) -> jnp.ndarray:
    # params = (t0, tE, u0, piEN, piEE)
    t0, tE, u0, piEN, piEE = params
    u = u_parallax(time, t0, tE, u0, piEN, piEE, P)
    return A_pspl_from_u(u)

def A_fspl_parallax_func(params: jnp.ndarray, time: jnp.ndarray, P) -> jnp.ndarray:
    # params = (t0, tE, u0, rho, piEN, piEE)
    t0, tE, u0, rho, piEN, piEE = params
    u = u_parallax(time, t0, tE, u0, piEN, piEE, P)
    return A_fspl_from_u(u, rho)

def A_fspl_parallax_logrho_func(q: jnp.ndarray, time: jnp.ndarray, P) -> jnp.ndarray:
    # q = (t0, tE, u0, logrho, piEN, piEE)
    t0, tE, u0, logrho, piEN, piEE = q
    rho = jnp.exp(logrho)
    u = u_parallax(time, t0, tE, u0, piEN, piEE, P)
    return A_fspl_from_u(u, rho)
