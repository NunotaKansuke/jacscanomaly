from __future__ import annotations

import jax.numpy as jnp

from .trajectory import u_rectilinear
from .magnification import A_pspl_from_u

def A_pspl_func(params: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
    t0, tE, u0 = params
    u = u_rectilinear(t0, tE, u0, time)
    return A_pspl_from_u(u)

