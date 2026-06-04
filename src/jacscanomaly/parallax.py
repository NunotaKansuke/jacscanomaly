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

    @staticmethod
    def from_horizons_table(table_np: np.ndarray) -> "HeliocentricEphemeris":
        """Backward-compatible alias for Horizons vector tables."""
        return HeliocentricEphemeris.from_horizons_vectors_table(table_np)


@jax.tree_util.register_pytree_node_class
class SatelliteEphemeris:
    """
    Satellite ephemeris in geocentric equatorial coordinates.

    The supported text format follows VBMicrolensing satellite tables used by
    RTModel/Roman simulations:

        JD  RA_deg  Dec_deg  distance_AU  optional_columns...

    Position vectors are stored in AU.
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
    def from_radec_distance_table(table_np: np.ndarray) -> "SatelliteEphemeris":
        tab = np.asarray(table_np, dtype=np.float64)
        if tab.ndim != 2 or tab.shape[1] < 4:
            raise ValueError("Satellite table must have columns [JD, RA_deg, Dec_deg, distance_AU].")

        order = np.argsort(tab[:, 0])
        tab = tab[order]
        t_np = tab[:, 0]
        ra = np.deg2rad(tab[:, 1])
        dec = np.deg2rad(tab[:, 2])
        dist = tab[:, 3]
        r_np = np.column_stack(
            [
                dist * np.cos(dec) * np.cos(ra),
                dist * np.cos(dec) * np.sin(ra),
                dist * np.sin(dec),
            ]
        )

        if t_np.size < 2:
            raise ValueError("Satellite ephemeris requires at least two rows.")
        v_np = np.gradient(r_np, t_np, axis=0, edge_order=1)
        return SatelliteEphemeris(jnp.asarray(t_np), jnp.asarray(r_np), jnp.asarray(v_np))

    @staticmethod
    def from_vbm_radec_distance_table(table_np: np.ndarray) -> "SatelliteEphemeris":
        tab = np.asarray(table_np, dtype=np.float64)
        if tab.ndim != 2 or tab.shape[1] < 4:
            raise ValueError("Satellite table must have columns [JD, RA_deg, Dec_deg, distance_AU].")

        order = np.argsort(tab[:, 0])
        tab = tab[order]
        t_np = tab[:, 0]
        r_np = []
        for _, ra, dec, dist in tab[:, :4]:
            unit = np.asarray(_vbm_radec_unit_vector(float(ra), float(dec)), dtype=float)
            r_np.append(float(dist) * unit)
        r_np = np.asarray(r_np, dtype=np.float64)

        if t_np.size < 2:
            raise ValueError("Satellite ephemeris requires at least two rows.")
        v_np = np.gradient(r_np, t_np, axis=0, edge_order=1)
        return SatelliteEphemeris(jnp.asarray(t_np), jnp.asarray(r_np), jnp.asarray(v_np))


def load_vbm_satellite_file(path: str) -> np.ndarray:
    """
    Load a VBMicrolensing-style satellite table.

    Returns a NumPy array with columns [JD, RA_deg, Dec_deg, distance_AU].
    Lines outside an optional ``$$SOE`` / ``$$EOE`` block are ignored when the
    markers are present.
    """
    rows = []
    saw_soe = False
    in_block = False
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("$$SOE"):
                saw_soe = True
                in_block = True
                continue
            if s.startswith("$$EOE"):
                break
            if saw_soe and not in_block:
                continue
            if s.startswith("*") or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 4:
                continue
            try:
                rows.append(tuple(float(x) for x in parts[:4]))
            except ValueError:
                continue

    if not rows:
        raise ValueError("No satellite ephemeris rows parsed.")
    return np.asarray(rows, dtype=np.float64)


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


def interp_linear(xq, x, y):
    xq = jnp.atleast_1d(xq)
    i1 = jnp.searchsorted(x, xq, side="right")
    i1 = jnp.clip(i1, 1, x.shape[0] - 1)
    i0 = i1 - 1
    x0 = x[i0]
    x1 = x[i1]
    den = jnp.where(x1 != x0, x1 - x0, jnp.asarray(1.0, x.dtype))
    w = (xq - x0) / den
    y0 = y[i0]
    y1 = y[i1]
    return y0 + (y1 - y0) * (w[:, None] if y.ndim == 2 else w)


# ----------------------------
# 4) Sky basis + LOS unit vector
# ----------------------------
ARCSEC_TO_RAD = jnp.deg2rad(1.0 / 3600.0)
AU_C_DAY = 0.005775518331436995  # AU/c in days
VB_EQ2000 = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64)
VB_QUAD2000 = jnp.array([0.0, 0.9174820003578725, -0.3977772982704228], dtype=jnp.float64)
VB_NORTH2000 = jnp.array([0.0, 0.3977772982704228, 0.9174820003578725], dtype=jnp.float64)


def _vbm_radec_unit_vector(ra_deg, dec_deg):
    ra = jnp.deg2rad(jnp.asarray(ra_deg, dtype=jnp.float64))
    dec = jnp.deg2rad(jnp.asarray(dec_deg, dtype=jnp.float64))
    return (
        jnp.cos(ra) * jnp.cos(dec) * VB_EQ2000
        + jnp.sin(ra) * jnp.cos(dec) * VB_QUAD2000
        + jnp.sin(dec) * VB_NORTH2000
    )


def load_vbm_sun_file(path: str) -> np.ndarray:
    """
    Load VBMicrolensing's SunEphemeris table as Earth heliocentric vectors.

    The table rows are ``JD RA_deg Dec_deg distance_AU phiprec`` for the Sun as
    seen from Earth. VB stores Earth heliocentric position as ``-distance *
    unit(RA, Dec)`` in its ecliptic J2000 basis.
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
            if s.startswith("$$EOE"):
                break
            parts = s.split()
            if len(parts) < 4:
                continue
            try:
                t, ra, dec, dist = (float(parts[i]) for i in range(4))
            except ValueError:
                continue
            unit = np.asarray(_vbm_radec_unit_vector(ra, dec), dtype=float)
            r = -dist * unit
            rows.append((t, r[0], r[1], r[2]))
    if not rows:
        raise ValueError("No VBMicrolensing Sun ephemeris rows parsed.")

    arr = np.asarray(rows, dtype=np.float64)
    t = arr[:, 0]
    r = arr[:, 1:4]
    v = np.gradient(r, t, axis=0, edge_order=1)
    return np.column_stack([t, r, v])

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


def get_vbm_south_west(RA_deg, Dec_deg):
    obj = _vbm_radec_unit_vector(RA_deg, Dec_deg)
    sp = jnp.dot(VB_NORTH2000, obj)
    south = -VB_NORTH2000 + sp * obj
    south = south / jnp.linalg.norm(south)
    west = jnp.cross(south, obj)
    return south, west, obj


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

        tref_user = jnp.asarray(tref, dtype=dtype)
        origin = jnp.asarray(2450000.0, dtype=dtype)
        self.time_add = jnp.where(tref_user < origin, origin, jnp.asarray(0.0, dtype=dtype))
        self.tref = tref_user + self.time_add

        self.use_HJD = bool(use_HJD)
        self.light_time_iters = int(light_time_iters)
        self.au_c_day = jnp.asarray(au_c_day, dtype=dtype)

        self.sky_north, self.sky_east = get_north_east(RA_deg, Dec_deg)
        self.n_hat = event_unit_vector(RA_deg, Dec_deg, dtype=dtype)

        self.rv = jnp.concatenate([eph.r, eph.v], axis=-1)

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
            self.t0, self.dt, self.tref, self.au_c_day, self.time_add,
            self.sky_north, self.sky_east, self.n_hat,
            self.rv, self.E_ref, self.V_ref
        )
        aux = (self.use_HJD, self.light_time_iters)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        (
            obj.t0, obj.dt, obj.tref, obj.au_c_day, obj.time_add,
            obj.sky_north, obj.sky_east, obj.n_hat,
            obj.rv, obj.E_ref, obj.V_ref
        ) = children
        (obj.use_HJD, obj.light_time_iters) = aux
        return obj


@jax.tree_util.register_pytree_node_class
class VBMEarthOrbitalParallaxProjector(EarthOrbitalParallaxProjector):
    """
    Earth parallax projector matching VBMicrolensing's SunEphemeris basis.
    """

    def __init__(self, eph: HeliocentricEphemeris, RA_deg, Dec_deg, tref, *,
                 use_HJD: bool = False, light_time_iters: int = 5, au_c_day: float = AU_C_DAY):
        super().__init__(
            eph,
            RA_deg,
            Dec_deg,
            tref,
            use_HJD=use_HJD,
            light_time_iters=light_time_iters,
            au_c_day=au_c_day,
        )
        south, west, obj = get_vbm_south_west(RA_deg, Dec_deg)
        self.sky_east = -west
        self.sky_north = -south
        self.n_hat = obj

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

# ----------------------------
# 7) dtau, dbeta computation
# ----------------------------
def earth_orbital_parallax_offsets(t, piEN, piEE, P: EarthOrbitalParallaxProjector):
    t = jnp.asarray(t, dtype=P.tref.dtype) + P.time_add

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

    d_tau  = piEN * ds[:, 1] + piEE * ds[:, 0]
    d_beta = -piEE * ds[:, 1] + piEN * ds[:, 0]
    return d_tau, d_beta

earth_orbital_parallax_offsets_jit = jax.jit(earth_orbital_parallax_offsets)


def earth_observer_lighttravel_delay(t, P: EarthOrbitalParallaxProjector):
    """
    Observer light-travel delay in days for JD-style inputs.

    VBMicrolensing evaluates the source trajectory using
    ``t + lighttravel`` when the input time is JD. For HJD-style inputs the
    trajectory time is already corrected, so this function returns zero.
    """
    t = jnp.asarray(t, dtype=P.tref.dtype) + P.time_add
    if P.use_HJD:
        return jnp.zeros_like(t)
    rv_t = interp_uniform_linear(t, P.t0, P.dt, P.rv)
    r_t = rv_t[:, :3]
    return jnp.sum(r_t * P.n_hat, axis=-1) * P.au_c_day


earth_observer_lighttravel_delay_jit = jax.jit(earth_observer_lighttravel_delay)

@jax.tree_util.register_pytree_node_class
class SpaceOrbitalParallaxProjector:
    """
    Annual parallax plus a geocentric spacecraft ephemeris.

    This mirrors the VBMicrolensing satellite-table convention: the spacecraft
    table gives geocentric RA/Dec/distance. VBMicrolensing adds the
    instantaneous geocentric spacecraft projection directly to the annual
    parallax displacement; it does not subtract the spacecraft position or
    velocity at the parallax reference epoch.
    """
    def __init__(self, earth: EarthOrbitalParallaxProjector, satellite: SatelliteEphemeris):
        self.earth = earth
        self.sat_t = satellite.t
        self.sat_r = satellite.r
        self.sat_v = satellite.v

    def tree_flatten(self):
        children = (self.earth, self.sat_t, self.sat_r, self.sat_v)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.earth, obj.sat_t, obj.sat_r, obj.sat_v = children
        return obj


def space_orbital_parallax_offsets(t, piEN, piEE, P: SpaceOrbitalParallaxProjector):
    d_tau_earth, d_beta_earth = earth_orbital_parallax_offsets(t, piEN, piEE, P.earth)

    t = jnp.asarray(t, dtype=P.earth.tref.dtype) + P.earth.time_add
    sat_r = interp_linear(t, P.sat_t, P.sat_r)
    # Components are [west, south], matching VB's [Et[1], Et[0]] convention.
    ds = -jnp.stack([sat_r @ P.earth.sky_east, sat_r @ P.earth.sky_north], axis=-1)

    d_tau_sat = piEN * ds[:, 1] + piEE * ds[:, 0]
    d_beta_sat = -piEE * ds[:, 1] + piEN * ds[:, 0]
    return d_tau_earth + d_tau_sat, d_beta_earth + d_beta_sat


space_orbital_parallax_offsets_jit = jax.jit(space_orbital_parallax_offsets)
