from __future__ import annotations

import jax.numpy as jnp

from .trajectory import u_rectilinear, u_parallax, u_space_parallax
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

def A_fspl_parallax_logrho_peak_func(
    q: jnp.ndarray,
    time: jnp.ndarray,
    P,
    peak_indices: jnp.ndarray,
    *,
    N_fft: int = 1024,
) -> jnp.ndarray:
    """FSPL+parallax magnification with microjax evaluated only near peak.

    The indices are fixed before fitting, so JAX can compile a static slice.
    Outside that window the finite-source correction is negligible by design
    and we use the point-source magnification, while retaining every data
    point in the likelihood and in the fitted flux solution.
    """
    t0, tE, u0, logrho, piEN, piEE = q
    rho = jnp.exp(logrho)
    u = u_parallax(time, t0, tE, u0, piEN, piEE, P)
    A = A_pspl_from_u(u)
    A_peak = A_fspl_from_u(u[peak_indices], rho, N_fft=N_fft)
    return A.at[peak_indices].set(A_peak)

def A_pspl_space_parallax_func(params: jnp.ndarray, time: jnp.ndarray, P) -> jnp.ndarray:
    t0, tE, u0, piEN, piEE = params
    u = u_space_parallax(time, t0, tE, u0, piEN, piEE, P)
    return A_pspl_from_u(u)

def A_fspl_space_parallax_func(params: jnp.ndarray, time: jnp.ndarray, P) -> jnp.ndarray:
    t0, tE, u0, rho, piEN, piEE = params
    u = u_space_parallax(time, t0, tE, u0, piEN, piEE, P)
    return A_fspl_from_u(u, rho)

def A_fspl_space_parallax_logrho_func(q: jnp.ndarray, time: jnp.ndarray, P) -> jnp.ndarray:
    t0, tE, u0, logrho, piEN, piEE = q
    rho = jnp.exp(logrho)
    u = u_space_parallax(time, t0, tE, u0, piEN, piEE, P)
    return A_fspl_from_u(u, rho)


def A_cv_asymexp_logtau_func(q: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
    """
    CV template with linear rise and exponential decay.

    Optimizer parameters
    --------------------
    q = (t0, log_tau_rise, log_tau_decay)

    Returns
    -------
    A(t) with peak normalized to 1 at t=t0.
    """
    t0, log_tau_rise, log_tau_decay = q
    tau_rise = jnp.exp(log_tau_rise)
    tau_decay = jnp.exp(log_tau_decay)
    dt = time - t0
    rise = jnp.clip((dt + tau_rise) / tau_rise, 0.0, 1.0)
    decay = jnp.exp(-dt / tau_decay)
    return jnp.where(dt < 0.0, rise, decay)
