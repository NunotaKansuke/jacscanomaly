from __future__ import annotations

import jax.numpy as jnp
from jax import jit

@jit
def A_pspl_from_u(u: jnp.ndarray) -> jnp.ndarray:
    # u can be scalar or array
    return (u**2 + 2) / (u * jnp.sqrt(u**2 + 4))

