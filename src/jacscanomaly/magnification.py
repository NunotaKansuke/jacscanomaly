from __future__ import annotations

import jax.numpy as jnp
from jax import jit

def _make_fspl_disk():
    try:
        from microjax.fastlens import fspl_disk
    except ImportError:
        return None
    return fspl_disk()


_mag_fspl = _make_fspl_disk()


def _get_fspl_disk():
    global _mag_fspl
    if _mag_fspl is None:
        _mag_fspl = _make_fspl_disk()
        if _mag_fspl is None:
            raise ImportError(
                "microjax is required for FSPL magnification. Install it from "
                "https://github.com/ShotaMiyazaki94/microjax before using FSPL fitters."
            )
    return _mag_fspl

@jit
def A_pspl_from_u(u: jnp.ndarray) -> jnp.ndarray:
    # u can be scalar or array
    return (u**2 + 2) / (u * jnp.sqrt(u**2 + 4))

@jit
def A_fspl_from_u(u, rho):
    return _get_fspl_disk().A(u, rho)
