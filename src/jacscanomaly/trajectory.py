from __future__ import annotations

import jax.numpy as jnp
from jax import jit

@jit
def u_rectilinear(t0: jnp.ndarray, tE: jnp.ndarray, u0: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(u0**2 + ((t - t0) / tE) ** 2)

