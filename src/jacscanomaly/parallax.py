from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

"""
Parallax utilities (JAX).

This module provides shared routines to project an observer ephemeris onto a
local (east, north) tangent plane at a fixed event sky position, and to
compute microlensing parallax offsets in (tau, beta).

Implemented:
- Earth-orbital ("annual") parallax.

Planned:
- Space-orbital parallax (same formalism with a spacecraft ephemeris).

Not included:
- Multi-observer baseline parallax (ground–space / space–space).
"""

# ============================================================
# Constants / units
# ============================================================
AU_KM = 149_597_870.700
DAY_S = 86400.0
ARCSEC_TO_RAD = jnp.deg2rad(1.0 / 3600.0)


def ang_arcsec_per_hour_to_rad_per_day(x):
    """Convert angular rate from arcsec/hour to rad/day."""
    return x * ARCSEC_TO_RAD * 24.0


# ============================================================
# Sky basis utilities (Earth/space common)
# ============================================================
def get_north_east(RA_deg, Dec_deg):
    """
    Construct local sky basis vectors (north, east) at a given sky position.

    Parameters
    ----------
    RA_deg, Dec_deg : float
        Event right ascension / declination in degrees.

    Returns
    -------
    sky_north, sky_east : ndarray, shape (3,), (3,)
        Orthonormal tangent-plane basis at the event direction.

    Notes
    -----
    Undefined at the celestial poles (|Dec| = 90 deg).
    """
    lam = jnp.deg2rad(RA_deg)
    bet = jnp.deg2rad(Dec_deg)

    earth_north = jnp.array([0.0, 0.0, 1.0], dtype=lam.dtype)
    event = jnp.array(
        [
            jnp.cos(lam) * jnp.cos(bet),
            jnp.sin(lam) * jnp.cos(bet),
            jnp.sin(bet),
        ],
        dtype=lam.dtype,
    )

    sky_east = jnp.cross(earth_north, event)
    sky_east = sky_east / jnp.linalg.norm(sky_east)
    sky_north = jnp.cross(event, sky_east)
    return sky_north, sky_east


# ============================================================
# Horizons -> Cartesian utilities (Earth/space common)
# ============================================================
def r_from_radec_delta(ra_deg, dec_deg, delta_au):
    """Convert (RA, Dec, Delta) to Cartesian position vector(s) in AU."""
    ra = jnp.deg2rad(ra_deg)
    dec = jnp.deg2rad(dec_deg)
    cd, sd = jnp.cos(dec), jnp.sin(dec)
    ca, sa = jnp.cos(ra), jnp.sin(ra)
    u = jnp.stack([cd * ca, cd * sa, sd], axis=-1)
    return delta_au[..., None] * u


def rdot_from_horizons_cols(
    ra_deg,
    dec_deg,
    delta_au,
    deldot_km_s,
    dRAcosD_arcsec_per_hour,
    dDecdt_arcsec_per_hour,
):
    """
    Build Cartesian velocity from Horizons-style columns.

    Notes
    -----
    Horizons reports d(RA*cosDec)/dt. We convert to dRA/dt by dividing by cosDec.
    Near |Dec| ~ 90 deg this becomes ill-defined; this implementation inserts NaNs.
    """
    ra = jnp.deg2rad(ra_deg)
    dec = jnp.deg2rad(dec_deg)
    cd, sd = jnp.cos(dec), jnp.sin(dec)
    ca, sa = jnp.cos(ra), jnp.sin(ra)

    u = jnp.stack([cd * ca, cd * sa, sd], axis=-1)

    dDec = ang_arcsec_per_hour_to_rad_per_day(dDecdt_arcsec_per_hour)
    dRAcosD = ang_arcsec_per_hour_to_rad_per_day(dRAcosD_arcsec_per_hour)

    cd_safe = jnp.where(jnp.abs(cd) < 1e-15, jnp.nan, cd)
    dRA = dRAcosD / cd_safe

    du = jnp.stack(
        [
            -cd * sa * dRA - sd * ca * dDec,
            cd * ca * dRA - sd * sa * dDec,
            cd * dDec,
        ],
        axis=-1,
    )

    deldot_au_per_day = deldot_km_s * (DAY_S / AU_KM)
    return deldot_au_per_day[..., None] * u + delta_au[..., None] * du


# ============================================================
# Interpolation utilities (Earth/space common)
# ============================================================
def interp_uniform_linear(xq, x0, dt, y):
    """
    Linear interpolation on a uniform grid.

    Notes
    -----
    Out-of-range queries are clamped to edge segments.
    """
    xq = jnp.atleast_1d(xq)
    u = (xq - x0) / dt
    i0 = jnp.floor(u).astype(jnp.int32)
    i0 = jnp.clip(i0, 0, y.shape[0] - 2)
    w = u - i0.astype(u.dtype)

    y0 = y[i0]
    y1 = y[i0 + 1]
    return y0 + (y1 - y0) * (w[:, None] if y.ndim == 2 else w)


# ============================================================
# Ephemeris container (Earth/space common)
# ============================================================
@jax.tree_util.register_pytree_node_class
@dataclass
class HeliocentricEphemeris:
    """
    Uniform ephemeris container.

    Attributes
    ----------
    t : (N,) time array (uniform spacing assumed)
    r : (N,3) position vectors [AU]
    v : (N,3) velocity vectors [AU/day]

    Notes
    -----
    The interpretation of (r, v) depends on how you queried Horizons (e.g.,
    Sun->Earth, Sun->spacecraft, etc.). Use a consistent inertial frame/units.
    """

    t: jnp.ndarray
    r: jnp.ndarray
    v: jnp.ndarray

    def tree_flatten(self):
        return (self.t, self.r, self.v), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

    @staticmethod
    def from_horizons_table(table_np):
        """
        Build ephemeris from a numeric table with columns:
        [t, RA_deg, Dec_deg, dRA*cosDec (arcsec/hr), dDec (arcsec/hr), Delta (AU), dDelta (km/s)].
        """
        tab = jnp.asarray(table_np)
        t, ra, dec, dRAcosD, dDecdt, delta, deldot = [tab[:, i] for i in range(7)]
        order = jnp.argsort(t)

        t = t[order]
        ra = ra[order]
        dec = dec[order]
        dRAcosD = dRAcosD[order]
        dDecdt = dDecdt[order]
        delta = delta[order]
        deldot = deldot[order]

        r = r_from_radec_delta(ra, dec, delta)
        v = rdot_from_horizons_cols(ra, dec, delta, deldot, dRAcosD, dDecdt)
        return HeliocentricEphemeris(t, r, v)


# ============================================================
# Earth-orbital parallax projector
# ============================================================
@jax.tree_util.register_pytree_node_class
class EarthOrbitalParallaxProjector:
    """
    Precompute projections for Earth-orbital ("annual") parallax offsets.

    Supply an Earth ephemeris (typically Sun->Earth in AU and AU/day) and an
    event sky position (RA, Dec). The same math can be reused for spacecraft
    by defining a separate wrapper (see bottom of file).
    """

    def __init__(self, eph: HeliocentricEphemeris, RA_deg, Dec_deg, tref):
        dtype = eph.t.dtype
        self.t0 = eph.t[0]
        self.dt = eph.t[1] - eph.t[0]
        self.tref = jnp.asarray(tref, dtype=dtype)

        self.sky_north, self.sky_east = get_north_east(RA_deg, Dec_deg)
        self.rv = jnp.concatenate([eph.r, eph.v], axis=-1)

        rv_ref = interp_uniform_linear(self.tref[None], self.t0, self.dt, self.rv)[0]
        r_ref, v_ref = rv_ref[:3], rv_ref[3:]

        self.E_ref = -jnp.stack([r_ref @ self.sky_east, r_ref @ self.sky_north])
        self.V_ref = -jnp.stack([v_ref @ self.sky_east, v_ref @ self.sky_north])

    def tree_flatten(self):
        return (
            self.t0,
            self.dt,
            self.tref,
            self.sky_north,
            self.sky_east,
            self.rv,
            self.E_ref,
            self.V_ref,
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        (
            obj.t0,
            obj.dt,
            obj.tref,
            obj.sky_north,
            obj.sky_east,
            obj.rv,
            obj.E_ref,
            obj.V_ref,
        ) = children
        return obj


# ============================================================
# Earth-orbital parallax API (main)
# ============================================================
def earth_orbital_parallax_offsets(t, piEN, piEE, P: EarthOrbitalParallaxProjector):
    """
    Compute Earth-orbital parallax offsets (d_tau, d_beta) at times `t`.

    Uses the standard microlensing transform:
      d_tau  =  pi_E,N * ds_N + pi_E,E * ds_E
      d_beta = -pi_E,E * ds_N + pi_E,N * ds_E
    """
    t = jnp.asarray(t, dtype=P.tref.dtype)

    rv_t = interp_uniform_linear(t, P.t0, P.dt, P.rv)
    r_t = rv_t[:, :3]

    E_t = -jnp.stack([r_t @ P.sky_east, r_t @ P.sky_north], axis=-1)
    ds = (P.E_ref[None] - E_t) + P.V_ref[None] * (t - P.tref)[:, None]

    d_tau = piEN * ds[:, 1] + piEE * ds[:, 0]
    d_beta = -piEE * ds[:, 1] + piEN * ds[:, 0]
    return d_tau, d_beta


earth_orbital_parallax_offsets_jit = jax.jit(earth_orbital_parallax_offsets)


# ============================================================
# Space-orbital parallax (placeholder / thin wrapper)
# ============================================================
@jax.tree_util.register_pytree_node_class
class SpaceOrbitalParallaxProjector(EarthOrbitalParallaxProjector):
    """
    Placeholder for space-orbital parallax.

    Currently identical to EarthOrbitalParallaxProjector; this distinct name
    keeps APIs explicit when spacecraft ephemerides are introduced.
    """
    pass


def space_orbital_parallax_offsets(t, piEN, piEE, P: SpaceOrbitalParallaxProjector):
    """Thin wrapper for spacecraft ephemerides (same math as Earth version)."""
    return earth_orbital_parallax_offsets(t, piEN, piEE, P)


space_orbital_parallax_offsets_jit = jax.jit(space_orbital_parallax_offsets)
