from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

# ----------------------------
# 1) Horizons vectors loader
# ----------------------------
def load_horizons_vectors_file(path: str) -> np.ndarray:
    """
    Read a Horizons 'GEOMETRIC cartesian states' table (CSV-like) and return
    a numeric array with columns:
        [t_jdtdb, x, y, z, vx, vy, vz]
    Units: t in days (JD TDB), position in AU, velocity in AU/day.

    Expected Horizons line format (example):
    2451544.500000000, A.D. 2000-Jan-01 00:00:00.0000, -1.7E-01, 8.8E-01, ... , RR,

    Notes:
    - Skips everything outside $$SOE ... $$EOE.
    - Ignores the Calendar Date column.
    - Ignores LT/RG/RR.
    """
    rows = []
    in_block = False
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not in_block:
                if s.startswith("$$SOE"):
                    in_block = True
                continue
            else:
                if s.startswith("$$EOE"):
                    break
                if not s or s.startswith("*"):
                    continue

                # Split CSV-ish line. Expect at least 11 columns; last may be empty due to trailing comma.
                parts = [p.strip() for p in s.split(",") if p.strip() != ""]
                # After removing empty, typical length is 11:
                # [JDTDB, 'A.D....', X, Y, Z, VX, VY, VZ, LT, RG, RR]
                if len(parts) < 8:
                    continue  # skip malformed lines safely

                try:
                    t = float(parts[0])
                    # parts[1] is calendar date string; ignore it
                    x = float(parts[2]); y = float(parts[3]); z = float(parts[4])
                    vx = float(parts[5]); vy = float(parts[6]); vz = float(parts[7])
                except ValueError:
                    continue

                rows.append((t, x, y, z, vx, vy, vz))

    if not rows:
        raise ValueError("No ephemeris rows parsed. Check file format and $$SOE/$$EOE markers.")
    return np.asarray(rows, dtype=np.float64)


# ----------------------------
# 2) Ephemeris constructor
# ----------------------------
@jax.tree_util.register_pytree_node_class
class HeliocentricEphemeris:
    """
    Uniform ephemeris container (t must be uniform grid for interp_uniform_linear).
    Here it can be barycentric too; the name is legacy.
    """
    def __init__(self, t: jnp.ndarray, r: jnp.ndarray, v: jnp.ndarray):
        self.t = t
        self.r = r
        self.v = v

    def tree_flatten(self):
        return (self.t, self.r, self.v), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

    @staticmethod
    def from_horizons_vectors_table(table_np: np.ndarray) -> "HeliocentricEphemeris":
        """
        table_np columns: [t, x, y, z, vx, vy, vz]
        """
        tab = jnp.asarray(table_np)
        t = tab[:, 0]
        r = tab[:, 1:4]
        v = tab[:, 4:7]
        order = jnp.argsort(t)
        return HeliocentricEphemeris(t[order], r[order], v[order])


# ----------------------------
# 3) Interp (your existing one)
# ----------------------------
def interp_uniform_linear(xq, x0, dt, y):
    xq = jnp.atleast_1d(xq)
    u = (xq - x0) / dt
    i0 = jnp.floor(u).astype(jnp.int32)
    i0 = jnp.clip(i0, 0, y.shape[0] - 2)
    w = u - i0.astype(u.dtype)
    y0 = y[i0]
    y1 = y[i0 + 1]
    return y0 + (y1 - y0) * (w[:, None] if y.ndim == 2 else w)


# ----------------------------
# 4) Sky basis + LOS unit vector
# ----------------------------
ARCSEC_TO_RAD = jnp.deg2rad(1.0 / 3600.0)
AU_C_DAY = 0.005775518331436995  # AU/c in days

def get_north_east(RA_deg, Dec_deg):
    lam = jnp.deg2rad(RA_deg)
    bet = jnp.deg2rad(Dec_deg)

    earth_north = jnp.array([0.0, 0.0, 1.0], dtype=lam.dtype)
    event = jnp.array(
        [jnp.cos(lam) * jnp.cos(bet), jnp.sin(lam) * jnp.cos(bet), jnp.sin(bet)],
        dtype=lam.dtype,
    )
    sky_east = jnp.cross(earth_north, event)
    sky_east = sky_east / jnp.linalg.norm(sky_east)
    sky_north = jnp.cross(event, sky_east)
    return sky_north, sky_east

def event_unit_vector(RA_deg, Dec_deg, dtype=jnp.float64):
    ra = jnp.deg2rad(jnp.asarray(RA_deg, dtype=dtype))
    dec = jnp.deg2rad(jnp.asarray(Dec_deg, dtype=dtype))
    cd, sd = jnp.cos(dec), jnp.sin(dec)
    ca, sa = jnp.cos(ra), jnp.sin(ra)
    return jnp.array([cd * ca, cd * sa, sd], dtype=dtype)


# ----------------------------
# 5) Light-time (optional, if you want HJD-style)
# ----------------------------
def light_time_corrected_time(t, t0, dt, rv, n_hat, au_c_day=AU_C_DAY, n_iter=5):
    t = jnp.asarray(t)
    t_emit = t

    def body(_, t_emit_curr):
        rv_curr = interp_uniform_linear(t_emit_curr, t0, dt, rv)
        r_curr = rv_curr[..., :3]
        lt = jnp.sum(r_curr * n_hat, axis=-1) * au_c_day
        return t - lt

    return jax.lax.fori_loop(0, n_iter, body, t_emit)


# ----------------------------
# 6) Projector: static use_HJD (jit/grad friendly)
# ----------------------------
@jax.tree_util.register_pytree_node_class
class EarthOrbitalParallaxProjector:
    def __init__(self, eph: HeliocentricEphemeris, RA_deg, Dec_deg, tref, *,
                 use_HJD: bool = True, light_time_iters: int = 5, au_c_day: float = AU_C_DAY):
        dtype = eph.t.dtype
        self.t0 = eph.t[0]
        self.dt = eph.t[1] - eph.t[0]
        self.tref = jnp.asarray(tref, dtype=dtype)

        self.use_HJD = bool(use_HJD)
        self.light_time_iters = int(light_time_iters)
        self.au_c_day = jnp.asarray(au_c_day, dtype=dtype)

        self.sky_north, self.sky_east = get_north_east(RA_deg, Dec_deg)
        self.n_hat = event_unit_vector(RA_deg, Dec_deg, dtype=dtype)

        self.rv = jnp.concatenate([eph.r, eph.v], axis=-1)

        # reference evaluation
        if self.use_HJD:
            tref_eval = light_time_corrected_time(
                self.tref[None], self.t0, self.dt, self.rv, self.n_hat,
                au_c_day=self.au_c_day, n_iter=self.light_time_iters
            )[0]
        else:
            tref_eval = self.tref

        rv_ref = interp_uniform_linear(tref_eval[None], self.t0, self.dt, self.rv)[0]
        r_ref, v_ref = rv_ref[:3], rv_ref[3:]

        self.E_ref = -jnp.stack([r_ref @ self.sky_east, r_ref @ self.sky_north])
        self.V_ref = -jnp.stack([v_ref @ self.sky_east, v_ref @ self.sky_north])

    def tree_flatten(self):
        children = (
            self.t0, self.dt, self.tref, self.au_c_day,
            self.sky_north, self.sky_east, self.n_hat,
            self.rv, self.E_ref, self.V_ref
        )
        aux = (self.use_HJD, self.light_time_iters)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        (
            obj.t0, obj.dt, obj.tref, obj.au_c_day,
            obj.sky_north, obj.sky_east, obj.n_hat,
            obj.rv, obj.E_ref, obj.V_ref
        ) = children
        (obj.use_HJD, obj.light_time_iters) = aux
        return obj


# ----------------------------
# 7) dtau, dbeta computation
# ----------------------------
def earth_orbital_parallax_offsets(t, piEN, piEE, P: EarthOrbitalParallaxProjector):
    t = jnp.asarray(t, dtype=P.tref.dtype)

    if P.use_HJD:
        t_eval = light_time_corrected_time(
            t, P.t0, P.dt, P.rv, P.n_hat,
            au_c_day=P.au_c_day, n_iter=P.light_time_iters
        )
    else:
        t_eval = t

    rv_t = interp_uniform_linear(t_eval, P.t0, P.dt, P.rv)
    r_t = rv_t[:, :3]

    E_t = -jnp.stack([r_t @ P.sky_east, r_t @ P.sky_north], axis=-1)
    ds = -((P.E_ref[None] - E_t) + P.V_ref[None] * (t - P.tref)[:, None])

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
