from __future__ import annotations

import jax.numpy as jnp
from jax import jit

_mag_fspl = None


def _get_fspl_disk():
    global _mag_fspl
    if _mag_fspl is None:
        try:
            from microjax.fastlens import fspl_disk
        except ImportError as exc:
            raise ImportError(
                "microjax is required for FSPL magnification. Install it from "
                "https://github.com/ShotaMiyazaki94/microjax before using FSPL fitters."
            ) from exc
        _mag_fspl = fspl_disk()
    return _mag_fspl

@jit
def A_pspl_from_u(u: jnp.ndarray) -> jnp.ndarray:
    # u can be scalar or array
    return (u**2 + 2) / (u * jnp.sqrt(u**2 + 4))

def A_fspl_from_u(u, rho):
    return _get_fspl_disk().A(u, rho)
