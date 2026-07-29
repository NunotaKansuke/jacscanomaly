from __future__ import annotations

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
    return parallax.load_horizons_vectors_file(str(p))

def get_heliocentric_ephemeris():
    global _EPH
    if _EPH is None:
        arr = _load_earth_orbital_parallax_array()
        _EPH = parallax.HeliocentricEphemeris.from_horizons_table(arr)
    return _EPH

def make_parallax_projector(RA: float, Dec: float, tref: float, *, use_HJD: bool = True):
    eph = get_heliocentric_ephemeris()
    return parallax.EarthOrbitalParallaxProjector(eph, RA, Dec, tref, use_HJD=use_HJD)


def make_space_parallax_projector(
    RA: float,
    Dec: float,
    tref: float,
    satellite_ephemeris_path: str,
    *,
    use_HJD: bool = True,
    convention: str = "vbm",
):
    sat_table = parallax.load_vbm_satellite_file(satellite_ephemeris_path)
    sat = parallax.SatelliteEphemeris.from_radec_distance_table(sat_table)
    if convention == "gulls":
        # GULLS builds each observer orbit by adding its satellite
        # perturbation to the reference (Earth) orbit.  The RTModel-style
        # satellite tables contain that geocentric perturbation (Roman is
        # about 0.01 AU from Earth), not a complete heliocentric position.
        # Treating the table as the complete orbit suppresses the annual
        # displacement by roughly two orders of magnitude and drives piE to
        # its configured bound.
        earth = get_heliocentric_ephemeris()
        earth_r = parallax.interp_uniform_linear(
            sat.t,
            earth.t[0],
            earth.t[1] - earth.t[0],
            earth.r,
        )
        observer_r = earth_r + sat.r
        observer_v = jnp.gradient(observer_r, sat.t, axis=0)
        observer = parallax.SatelliteEphemeris(sat.t, observer_r, observer_v)
        return parallax.GullsSpaceParallaxProjector(observer, RA, Dec, tref)
    if convention != "vbm":
        raise ValueError("space parallax convention must be 'vbm' or 'gulls'.")
    eph = get_heliocentric_ephemeris()
    earth = parallax.EarthOrbitalParallaxProjector(eph, RA, Dec, tref, use_HJD=use_HJD)
    return parallax.SpaceOrbitalParallaxProjector(earth, sat)


def u_parallax_tau_beta(t, t0, tE, u0, piEN, piEE, P):
    lt = parallax.earth_observer_lighttravel_delay_jit(t, P)
    lt0 = parallax.earth_observer_lighttravel_delay_jit(jnp.asarray([t0], dtype=jnp.asarray(t).dtype), P)[0]
    tau0 = (t + lt - t0 - lt0) / tE
    beta0 = jnp.full_like(tau0, u0)
    d_tau, d_beta = parallax.earth_orbital_parallax_offsets_jit(t, piEN, piEE, P)
    return tau0 + d_tau, beta0 + d_beta


def u_parallax(t, t0, tE, u0, piEN, piEE, P):
    tau, beta = u_parallax_tau_beta(t, t0, tE, u0, piEN, piEE, P)
    return jnp.sqrt(tau**2 + beta**2)


def u_space_parallax_tau_beta(t, t0, tE, u0, piEN, piEE, P):
    if isinstance(P, parallax.GullsSpaceParallaxProjector):
        tau0 = (t - t0) / tE
    else:
        lt = parallax.earth_observer_lighttravel_delay_jit(t, P.earth)
        lt0 = parallax.earth_observer_lighttravel_delay_jit(jnp.asarray([t0], dtype=jnp.asarray(t).dtype), P.earth)[0]
        tau0 = (t + lt - t0 - lt0) / tE
    beta0 = jnp.full_like(tau0, u0)
    d_tau, d_beta = parallax.any_space_parallax_offsets_jit(t, piEN, piEE, P)
    return tau0 + d_tau, beta0 + d_beta


def u_space_parallax(t, t0, tE, u0, piEN, piEE, P):
    tau, beta = u_space_parallax_tau_beta(t, t0, tE, u0, piEN, piEE, P)
    return jnp.sqrt(tau**2 + beta**2)
