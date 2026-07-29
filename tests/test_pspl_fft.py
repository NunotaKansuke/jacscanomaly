import numpy as np
import pytest
import jax.numpy as jnp

import jacscanomaly
from jacscanomaly import Finder, FinderConfig, PSPLFFTScanner, pspl_excess_magnification


def test_public_package_exports():
    assert jacscanomaly.PSPLFFTScanner is PSPLFFTScanner
    assert callable(jacscanomaly.pspl_excess_magnification)
    assert "PSPLFFTProfile" in jacscanomaly.__all__
    assert "PSPLFFTCandidate" in jacscanomaly.__all__
    assert "PSPLFFTSearchResult" in jacscanomaly.__all__


def _direct_profile(time, flux, ferr, t0_grid, u0, teff, positive_source=True):
    w = 1.0 / ferr**2
    ybar = np.sum(w * flux) / np.sum(w)
    yc = flux - ybar
    syy = np.sum(w * yc**2)

    delta = np.empty_like(t0_grid)
    fs = np.empty_like(t0_grid)
    f0 = np.empty_like(t0_grid)
    fb = np.empty_like(t0_grid)
    for i, t0 in enumerate(t0_grid):
        x = pspl_excess_magnification(time - t0, u0, teff)
        xbar = np.sum(w * x) / np.sum(w)
        xc = x - xbar
        sxx = np.sum(w * xc**2)
        sxy = np.sum(w * xc * yc)
        numerator = max(sxy, 0.0) if positive_source else sxy
        fs[i] = numerator / sxx
        f0[i] = ybar - fs[i] * xbar
        fb[i] = f0[i] - fs[i]
        delta[i] = numerator**2 / sxx
    return delta, fs, f0, fb, syy


def test_stable_excess_matches_direct_pspl_formula():
    lag = np.r_[0.0, np.geomspace(1.0e-4, 1.0e5, 300)]
    u0 = 0.03
    teff = 2.5
    u = u0 * np.sqrt(1.0 + (lag / teff) ** 2)
    direct = (u * u + 2.0) / (u * np.sqrt(u * u + 4.0)) - 1.0
    stable = pspl_excess_magnification(lag, u0, teff)

    # The direct subtraction loses relative precision in the far wings, so use
    # absolute tolerance there while retaining tight agreement near the event.
    np.testing.assert_allclose(stable, direct, rtol=2.0e-7, atol=3.0e-16)
    assert np.all(stable >= 0.0)


def test_fft_profile_matches_direct_weighted_regression_with_gaps():
    rng = np.random.default_rng(12)
    dt = 0.25
    full_time = np.arange(0.0, 30.0 + 0.5 * dt, dt)
    keep = rng.random(full_time.size) > 0.3
    time = full_time[keep]
    ferr = rng.uniform(0.03, 0.15, time.size)

    u0 = 0.2
    teff = 1.5
    true_t0 = 13.0
    x = pspl_excess_magnification(time - true_t0, u0, teff)
    flux = 1.7 + 2.4 * x + rng.normal(0.0, ferr)

    profile = PSPLFFTScanner(grid_dt=dt).scan_template(
        time,
        flux,
        ferr,
        u0=u0,
        teff=teff,
    )
    direct = _direct_profile(time, flux, ferr, profile.t0, u0, teff)
    delta, fs, f0, fb, syy = direct

    np.testing.assert_allclose(profile.delta_chi2, delta, rtol=2.0e-11, atol=2.0e-9)
    np.testing.assert_allclose(profile.fs, fs, rtol=2.0e-11, atol=2.0e-11)
    np.testing.assert_allclose(profile.f0, f0, rtol=2.0e-11, atol=2.0e-11)
    np.testing.assert_allclose(profile.fb, fb, rtol=2.0e-11, atol=2.0e-11)
    np.testing.assert_allclose(profile.chi2, syy - delta, rtol=2.0e-11, atol=2.0e-9)


def test_search_recovers_seed_from_irregular_sampling():
    rng = np.random.default_rng(8)
    dt = 0.04
    time = 2_450_000.0 + np.sort(rng.uniform(0.0, 40.0, 500))

    true_t0 = 2_450_018.37
    true_u0 = 0.2
    true_teff = 0.9
    true_tE = true_teff / true_u0
    ferr = rng.uniform(0.02, 0.06, time.size)
    x = pspl_excess_magnification(time - true_t0, true_u0, true_teff)
    flux = 0.4 + 1.8 * (x + 1.0) + rng.normal(0.0, ferr)

    result = PSPLFFTScanner(grid_dt=dt).search(
        time,
        flux,
        ferr,
        u0_grid=[0.1, 0.2, 0.4],
        teff_grid=[0.75, 0.9, 1.1],
        top_k=5,
    )

    assert result.best is not None
    assert abs(result.best.t0 - true_t0) <= dt
    assert result.best_profile is not None
    assert result.best.u0 == pytest.approx(true_u0)
    assert result.best.teff == pytest.approx(true_teff)
    assert result.best.tE == pytest.approx(true_tE)
    np.testing.assert_allclose(result.initial_guesses()[0], result.best.as_pspl_params())


def test_positive_source_constraint_projects_negative_event_to_baseline():
    time = np.linspace(0.0, 20.0, 201)
    ferr = np.full_like(time, 0.05)
    x = pspl_excess_magnification(time - 10.0, 0.2, 1.0)
    flux = 2.0 - 0.8 * x

    constrained = PSPLFFTScanner(grid_dt=0.1, positive_source=True).scan_template(
        time, flux, ferr, u0=0.2, teff=1.0
    )
    unconstrained = PSPLFFTScanner(grid_dt=0.1, positive_source=False).scan_template(
        time, flux, ferr, u0=0.2, teff=1.0
    )

    center = int(np.argmin(np.abs(constrained.t0 - 10.0)))
    assert constrained.fs[center] == pytest.approx(0.0)
    assert constrained.delta_chi2[center] == pytest.approx(0.0)
    assert unconstrained.fs[center] < 0.0
    assert unconstrained.delta_chi2[center] > 0.0


def test_flat_light_curve_has_no_positive_source_candidate():
    time = np.linspace(0.0, 10.0, 101)
    flux = np.ones_like(time)
    ferr = np.full_like(time, 0.1)

    result = PSPLFFTScanner(grid_dt=0.1).search(
        time,
        flux,
        ferr,
        u0_grid=[0.1, 0.3],
        teff_grid=[0.5, 1.0],
    )

    assert result.best is None
    assert result.candidates == ()


def test_validation_and_grid_guard():
    scanner = PSPLFFTScanner(grid_dt=0.001, max_grid_points=100)
    time = np.array([0.0, 1.0])
    flux = np.array([1.0, 1.0])
    ferr = np.array([0.1, 0.1])

    with pytest.raises(ValueError, match="exceeding max_grid_points"):
        scanner.scan_template(time, flux, ferr, u0=0.2, teff=1.0)
    with pytest.raises(ValueError, match="ferr must be positive"):
        PSPLFFTScanner(grid_dt=0.1).scan_template(
            time, flux, np.array([0.1, 0.0]), u0=0.2, teff=1.0
        )
    with pytest.raises(ValueError, match="u0_grid"):
        PSPLFFTScanner(grid_dt=0.1).search(
            time, flux, ferr, u0_grid=[], teff_grid=[1.0]
        )


def test_finder_uses_two_dimensional_pspl_fft_initialization():
    rng = np.random.default_rng(3)
    time = np.linspace(0.0, 20.0, 401)
    true_t0 = 10.0
    true_u0 = 0.2
    true_teff = 0.5
    ferr = np.full_like(time, 0.02)
    flux = 1.2 + 2.5 * pspl_excess_magnification(time - true_t0, true_u0, true_teff)
    flux += rng.normal(0.0, ferr)

    finder = Finder(
        FinderConfig(
            fitter_kind="pspl",
            auto_init_teff_min=0.3,
            auto_init_teff_max=1.0,
            auto_init_teff_grid_n=5,
            auto_init_u0_min=0.1,
            auto_init_u0_max=0.4,
            auto_init_u0_grid_n=4,
            auto_init_fft_grid_dt=0.05,
            auto_init_fft_max_grid_points=1_000,
            auto_init_fft_top_k=4,
        )
    )
    guesses = finder._estimate_single_lens_initial_guesses(
        time_j=jnp.asarray(time),
        flux_j=jnp.asarray(flux),
        ferr_j=jnp.asarray(ferr),
        time_np=time,
    )

    assert guesses.shape == (4, 3)
    teff = guesses[:, 1] * guesses[:, 2]
    assert np.min(np.abs(guesses[:, 0] - true_t0)) <= 0.05
    assert np.min(np.abs(teff - true_teff)) <= 0.1
    assert np.all(guesses[:, 1] > 0.0)
    assert np.all(guesses[:, 2] > 0.0)
