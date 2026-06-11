from pathlib import Path

import numpy as np
import pytest
import jax.numpy as jnp

from jacscanomaly import Finder, FinderConfig
from jacscanomaly.singlelens_model import A_fspl_logrho_func, A_pspl_space_parallax_func
from jacscanomaly import parallax
from jacscanomaly.trajectory import (
    make_space_parallax_projector,
    u_space_parallax_tau_beta,
)


ROMAN_SATELLITE1 = Path("/rogue1_8/nunota/sample_rtmodel_v2.4/satellitedir/satellite1.txt")


def test_vbm_satellite_loader_reads_roman_table():
    if not ROMAN_SATELLITE1.exists():
        pytest.skip("Roman satellite sample file is not available.")

    table = parallax.load_vbm_satellite_file(str(ROMAN_SATELLITE1))

    assert table.shape[1] == 4
    assert table.shape[0] > 1000
    np.testing.assert_allclose(
        table[0],
        [2458346.501847300, 325.424660, -14.099340, 0.01511615888369],
    )


def test_satellite_ephemeris_converts_radec_distance_to_cartesian_au():
    table = np.array(
        [
            [1.0, 0.0, 0.0, 2.0],
            [2.0, 90.0, 0.0, 2.0],
            [3.0, 0.0, 90.0, 2.0],
        ]
    )

    eph = parallax.SatelliteEphemeris.from_radec_distance_table(table)

    np.testing.assert_allclose(np.asarray(eph.r[0]), [2.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(np.asarray(eph.r[1]), [0.0, 2.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(np.asarray(eph.r[2]), [0.0, 0.0, 2.0], atol=1e-12)


def test_space_parallax_projector_adds_satellite_offsets():
    if not ROMAN_SATELLITE1.exists():
        pytest.skip("Roman satellite sample file is not available.")

    ra_deg = 267.623337808
    dec_deg = -29.1164180355
    tref = 2459000.0
    projector = make_space_parallax_projector(
        ra_deg,
        dec_deg,
        tref,
        str(ROMAN_SATELLITE1),
    )

    t = jnp.asarray([2458990.0, 2459000.0, 2459010.0])
    tau, beta = u_space_parallax_tau_beta(t, tref, 100.0, 0.1, 0.02, 0.03, projector)

    assert tau.shape == (3,)
    assert beta.shape == (3,)
    assert np.all(np.isfinite(np.asarray(tau)))
    assert np.all(np.isfinite(np.asarray(beta)))
    assert not np.allclose(np.asarray(tau), np.asarray((t - tref) / 100.0))


def test_gulls_space_parallax_uses_reference_frame_subtraction(tmp_path):
    # RA=0, Dec=0 gives sky east=(0,1,0), sky north=(0,0,1).
    vectors = np.array(
        [
            [2450000.0, 1.0, 0.0, 0.0],
            [2450010.0, 1.0, 0.1, 0.2],
            [2450020.0, 1.0, 0.4, 0.8],
        ]
    )
    xyz = vectors[:, 1:]
    dist = np.linalg.norm(xyz, axis=1)
    ra = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    dec = np.degrees(np.arcsin(xyz[:, 2] / dist))
    table = np.column_stack([vectors[:, 0], ra, dec, dist])
    path = tmp_path / "gulls_observer.dat"
    np.savetxt(path, table)

    tref = 2450010.0
    projector = make_space_parallax_projector(
        0.0,
        0.0,
        tref,
        str(path),
        convention="gulls",
    )
    t = jnp.asarray([2450000.0, 2450010.0, 2450020.0])
    tau, beta = u_space_parallax_tau_beta(t, tref, 100.0, 0.1, 0.5, 0.25, projector)

    expected_tau = np.asarray([-0.225, 0.0, -0.025])
    expected_beta = np.asarray([0.1, 0.1, 0.1])
    np.testing.assert_allclose(np.asarray(tau), expected_tau, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(beta), expected_beta, rtol=1e-12, atol=1e-12)


def test_space_parallax_matches_vbm_source_coordinate_formula():
    if not ROMAN_SATELLITE1.exists():
        pytest.skip("Roman satellite sample file is not available.")

    ra_deg = 267.623337808
    dec_deg = -29.1164180355
    tref = 2459000.0
    t0 = 2459001.0
    tE = 80.0
    u0 = 0.12
    piN = 0.02
    piE = 0.03
    t = jnp.asarray([2458990.0, 2459000.0, 2459010.0])
    projector = make_space_parallax_projector(
        ra_deg,
        dec_deg,
        tref,
        str(ROMAN_SATELLITE1),
    )

    tau, beta = u_space_parallax_tau_beta(t, t0, tE, u0, piN, piE, projector)

    d_tau_earth, d_beta_earth = parallax.earth_orbital_parallax_offsets_jit(
        t, piN, piE, projector.earth
    )
    sat_r = parallax.interp_linear(t + projector.earth.time_add, projector.sat_t, projector.sat_r)
    sat_west_south = -jnp.stack(
        [
            sat_r @ projector.earth.sky_east,
            sat_r @ projector.earth.sky_north,
        ],
        axis=-1,
    )
    sat_west = sat_west_south[:, 0]
    sat_south = sat_west_south[:, 1]

    # VBMicrolensing source coordinates are:
    # tn = tau0 + piN * Et_south + piE * Et_west
    # u1 = u0   + piN * Et_west  - piE * Et_south
    # y1 = -tn, y2 = -u1.
    expected_tau = (t - t0) / tE + d_tau_earth + piN * sat_south + piE * sat_west
    expected_beta = u0 + d_beta_earth + piN * sat_west - piE * sat_south

    np.testing.assert_allclose(np.asarray(tau), np.asarray(expected_tau), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(beta), np.asarray(expected_beta), rtol=1e-12, atol=1e-12)

    vbm_y1 = -np.asarray(expected_tau)
    vbm_y2 = -np.asarray(expected_beta)
    np.testing.assert_allclose(-np.asarray(tau), vbm_y1, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(-np.asarray(beta), vbm_y2, rtol=1e-12, atol=1e-12)


def test_space_parallax_is_consistent_with_vbmicrolensing_runtime():
    if not ROMAN_SATELLITE1.exists():
        pytest.skip("Roman satellite sample file is not available.")
    vb = pytest.importorskip("VBMicrolensing")

    coord_path = Path("/rogue1_8/nunota/sample_rtmodel_v2.4/event_2_675_639/Data/event.coordinates")
    if not coord_path.exists():
        pytest.skip("Roman coordinate sample file is not available.")

    ra_deg = 267.623337808
    dec_deg = -29.1164180355
    tref = 9000.0
    tE = 80.0
    u0 = 0.12
    piN = 0.50
    piE = 0.60
    time = np.linspace(tref - 100.0, tref + 100.0, 41)

    projector = make_space_parallax_projector(
        ra_deg,
        dec_deg,
        tref,
        str(ROMAN_SATELLITE1),
    )
    t_j = jnp.asarray(time)
    tau_jac, beta_jac = u_space_parallax_tau_beta(t_j, tref, tE, u0, piN, piE, projector)

    vbm = vb.VBMicrolensing()
    sun_table = Path(vb.__file__).parent / "data" / "SunEphemeris.txt"
    vbm.LoadSunTable(str(sun_table))
    vbm.SetObjectCoordinates(str(coord_path), str(ROMAN_SATELLITE1.parent))
    vbm.parallaxsystem = 1
    vbm.t0_par_fixed = 1
    vbm.t0_par = tref
    vbm.t_in_HJD = 1
    vbm.satellite = 1

    _, y1, y2 = vbm.PSPLLightCurveParallax(
        [u0, float(np.log(tE)), tref, piN, piE],
        time.tolist(),
    )
    tau_vbm = -np.asarray(y1, dtype=float)
    beta_vbm = -np.asarray(y2, dtype=float)

    u_vbm = np.sqrt(tau_vbm * tau_vbm + beta_vbm * beta_vbm)
    traj_err = np.sqrt((np.asarray(tau_jac) - tau_vbm) ** 2 + (np.asarray(beta_jac) - beta_vbm) ** 2)
    assert np.max(traj_err / np.maximum(u_vbm, np.finfo(float).eps)) < 2.0e-4


def test_finder_builds_pspl_space_parallax_fitter():
    if not ROMAN_SATELLITE1.exists():
        pytest.skip("Roman satellite sample file is not available.")

    finder = Finder(
        FinderConfig(
            fitter_kind="pspl_space_parallax",
            ra_deg=267.623337808,
            dec_deg=-29.1164180355,
            tref=2459000.0,
            satellite_ephemeris_path=str(ROMAN_SATELLITE1),
        )
    )

    finder._ensure_fitter(2459000.0)

    assert finder.fitter.__class__.__name__ == "PSPLSpaceParallaxFitter"


def test_finder_supports_vbm_finite_difference_fspl():
    pytest.importorskip("VBMicrolensing")

    finder = Finder(FinderConfig(fitter_kind="fspl_vbm_fd", grid_backend="cpp"))
    time = jnp.asarray(np.linspace(-5.0, 5.0, 21))
    q = jnp.asarray([0.0, 30.0, 0.2, np.log(0.01)])
    amp = A_fspl_logrho_func(q, time)
    flux = 1.7 * amp + 0.2
    ferr = jnp.full_like(time, 0.01)

    fit = finder.fit_single_lens(time, flux, ferr, x0=q)

    assert fit.param_names == ("t0", "tE", "u0", "rho")
    assert np.isfinite(np.asarray(fit.params)).all()
    assert float(fit.chi2_dof) < 1.0e-2


def test_finder_supports_gulls_vbm_finite_difference_fspl_space_parallax(tmp_path):
    pytest.importorskip("VBMicrolensing")

    vectors = np.array(
        [
            [2450000.0, 1.0, 0.0, 0.0],
            [2450010.0, 1.0, 0.1, 0.2],
            [2450020.0, 1.0, 0.4, 0.8],
        ]
    )
    xyz = vectors[:, 1:]
    dist = np.linalg.norm(xyz, axis=1)
    ra = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    dec = np.degrees(np.arcsin(xyz[:, 2] / dist))
    path = tmp_path / "gulls_observer.dat"
    np.savetxt(path, np.column_stack([vectors[:, 0], ra, dec, dist]))

    finder = Finder(
        FinderConfig(
            fitter_kind="fspl_space_parallax_gulls_vbm_fd",
            grid_backend="cpp",
            ra_deg=0.0,
            dec_deg=0.0,
            tref=2450010.0,
            satellite_ephemeris_path=str(path),
        )
    )
    finder._ensure_fitter(2450010.0)
    assert finder.fitter.__class__.__name__ == "VBMFiniteDiffGullsFSPLSpaceParallaxFitter"

    time = np.linspace(2450006.0, 2450014.0, 15)
    q = np.asarray([2450010.0, 30.0, 0.2, np.log(0.01), 0.3, -0.2])
    u = finder.fitter._gulls_u_numpy(time, q)
    amp = finder.fitter._magnification(u, np.exp(q[3]))
    flux = 1.7 * amp + 0.2
    ferr = np.full_like(time, 0.01)

    fit = finder.fit_single_lens(time, flux, ferr, x0=q)

    assert fit.param_names == ("t0", "tE", "u0", "rho", "piEN", "piEE")
    assert np.isfinite(np.asarray(fit.params)).all()
    assert float(fit.chi2) < 1.0e-8


def test_gulls_vbm_fitter_handles_offset_time_coordinates(tmp_path):
    pytest.importorskip("VBMicrolensing")

    vectors = np.array(
        [
            [2450000.0, 1.0, 0.0, 0.0],
            [2450010.0, 1.0, 0.1, 0.2],
            [2450020.0, 1.0, 0.4, 0.8],
        ]
    )
    xyz = vectors[:, 1:]
    dist = np.linalg.norm(xyz, axis=1)
    ra = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    dec = np.degrees(np.arcsin(xyz[:, 2] / dist))
    path = tmp_path / "gulls_observer.dat"
    np.savetxt(path, np.column_stack([vectors[:, 0], ra, dec, dist]))

    finder = Finder(
        FinderConfig(
            fitter_kind="fspl_space_parallax_gulls_vbm_fd",
            grid_backend="cpp",
            ra_deg=0.0,
            dec_deg=0.0,
            tref=10.0,
            satellite_ephemeris_path=str(path),
        )
    )
    finder._ensure_fitter(10.0)

    time = np.asarray([6.0, 10.0, 14.0])
    q = np.asarray([10.0, 30.0, 0.2, np.log(0.01), 0.3, -0.2])
    u_numpy = finder.fitter._gulls_u_numpy(time, q)
    tau, beta = u_space_parallax_tau_beta(
        jnp.asarray(time),
        q[0],
        q[1],
        q[2],
        q[4],
        q[5],
        finder.fitter._P,
    )
    u_jax = np.sqrt(np.asarray(tau) ** 2 + np.asarray(beta) ** 2)

    np.testing.assert_allclose(u_numpy, u_jax, rtol=1e-10, atol=1e-10)
    assert np.max(u_numpy) < 1.0


def test_finder_fit_single_lens_supports_pspl_space_parallax():
    if not ROMAN_SATELLITE1.exists():
        pytest.skip("Roman satellite sample file is not available.")

    tref = 9000.0
    finder = Finder(
        FinderConfig(
            fitter_kind="pspl_space_parallax",
            ra_deg=267.623337808,
            dec_deg=-29.1164180355,
            tref=tref,
            satellite_ephemeris_path=str(ROMAN_SATELLITE1),
        )
    )
    finder._ensure_fitter(tref)
    finder.fitter.maxiter = 40

    time = jnp.asarray(np.linspace(tref - 15.0, tref + 15.0, 24))
    params = jnp.asarray([tref, 80.0, 0.12, 0.10, 0.15])
    amp = A_pspl_space_parallax_func(params, time, finder.fitter._P)
    flux = 1.5 * amp + 0.2
    ferr = jnp.full_like(time, 0.01)

    fit = finder.fit_single_lens(time, flux, ferr, x0=params)

    assert fit.param_names == ("t0", "tE", "u0", "piEN", "piEE")
    assert np.isfinite(np.asarray(fit.params)).all()
    assert float(fit.chi2) < 1.0e-8


def test_space_parallax_requires_satellite_path():
    finder = Finder(
        FinderConfig(
            fitter_kind="pspl_space_parallax",
            ra_deg=267.623337808,
            dec_deg=-29.1164180355,
        )
    )

    with pytest.raises(ValueError, match="satellite_ephemeris_path"):
        finder._ensure_fitter(2459000.0)


def test_gulls_vbm_fd_space_parallax_requires_sky_and_satellite_path():
    finder = Finder(FinderConfig(fitter_kind="fspl_space_parallax_gulls_vbm_fd"))

    with pytest.raises(ValueError, match="ra_deg and dec_deg"):
        finder._ensure_fitter(2459000.0)

    finder = Finder(
        FinderConfig(
            fitter_kind="fspl_space_parallax_gulls_vbm_fd",
            ra_deg=267.623337808,
            dec_deg=-29.1164180355,
        )
    )

    with pytest.raises(ValueError, match="satellite_ephemeris_path"):
        finder._ensure_fitter(2459000.0)
