from __future__ import annotations

import jax.numpy as jnp
from jax import jit

from microjax.fastlens import fspl_disk
_mag_fspl = fspl_disk()

@jit
def A_pspl_from_u(u: jnp.ndarray) -> jnp.ndarray:
    # u can be scalar or array
    return (u**2 + 2) / (u * jnp.sqrt(u**2 + 4))

@jit
def A_fspl_from_u(u, rho):
    return _mag_fspl.A(u, rho)
