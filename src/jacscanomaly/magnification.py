from __future__ import annotations

import jax.numpy as jnp
from jax import jit

_mag_fspl = {}


def _get_fspl_disk(N_fft: int = 1024):
    """Return a cached microjax uniform-disk magnifier.

    ``N_fft`` is exposed for fast, peak-only exploratory fits.  The default
    keeps the established 1024-point accuracy setting.
    """
    global _mag_fspl
    if N_fft not in _mag_fspl:
        try:
            from microjax.fastlens import fspl_disk
        except ImportError as exc:
            raise ImportError(
                "microjax is required for FSPL magnification. Install it from "
                "https://github.com/ShotaMiyazaki94/microjax before using FSPL fitters."
            ) from exc
        _mag_fspl[N_fft] = fspl_disk(N_fft=N_fft)
    return _mag_fspl[N_fft]

@jit
def A_pspl_from_u(u: jnp.ndarray) -> jnp.ndarray:
    # u can be scalar or array
    return (u**2 + 2) / (u * jnp.sqrt(u**2 + 4))

def A_fspl_from_u(u, rho, *, N_fft: int = 1024):
    return _get_fspl_disk(N_fft=N_fft).A(u, rho)
