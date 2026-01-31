from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax import jit
from importlib import resources

import jacscanomaly.parallax as parallax


@jit
def uvec_rectilinear(t0, tE, u0, t):
    tau = (t - t0) / tE
    beta = jnp.full_like(tau, u0)
    return tau, beta


@jit
def u_rectilinear(t0, tE, u0, t):
    tau, beta = uvec_rectilinear(t0, tE, u0, t)
    return jnp.sqrt(tau**2 + beta**2)


# -------- parallax: cached ephemeris + projector --------

_EPH = None

def _load_earth_orbital_parallax_array():
    p = resources.files("jacscanomaly.data").joinpath("earth_orbital_parallax_table.txt")
    with p.open("r") as f:
        return np.genfromtxt(f, skip_header=59, skip_footer=60)

def get_heliocentric_ephemeris():
    global _EPH
    if _EPH is None:
        arr = _load_earth_orbital_parallax_array()
        _EPH = parallax.HeliocentricEphemeris.from_horizons_table(arr)
    return _EPH

def make_parallax_projector(RA: float, Dec: float, tref: float):
    eph = get_heliocentric_ephemeris()
    return parallax.EarthOrbitalParallaxProjector(eph, RA, Dec, tref)


def u_parallax_tau_beta(t, t0, tE, u0, piEN, piEE, P):
    tau0 = (t - t0) / tE
    beta0 = jnp.full_like(tau0, u0)
    d_tau, d_beta = parallax.earth_orbital_parallax_offsets_jit(t, piEN, piEE, P)
    return tau0 + d_tau, beta0 + d_beta


def u_parallax(t, t0, tE, u0, piEN, piEE, P):
    tau, beta = u_parallax_tau_beta(t, t0, tE, u0, piEN, piEE, P)
    return jnp.sqrt(tau**2 + beta**2)

