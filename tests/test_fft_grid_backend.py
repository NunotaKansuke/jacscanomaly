import numpy as np
import pytest
import jax.numpy as jnp

from jacscanomaly.config import FinderConfig
from jacscanomaly.extract import ResultExtractor
from jacscanomaly.fft_grid import FFTAnomalyGridRunner
from jacscanomaly.runner import GridRunner, SeasonGridRunner
from jacscanomaly.seasons import SeasonSplitter


def _direct_flat_chi2(time, flux, weight, t0_grid, teff, teff_coeff):
    result = np.zeros_like(t0_grid, dtype=float)
    for index, t0 in enumerate(t0_grid):
        mask = (time > t0 - teff_coeff * teff) & (time < t0 + teff_coeff * teff)
        if not np.any(mask):
            continue
        w = weight[mask]
        y = flux[mask]
        mean = np.sum(w * y) / np.sum(w)
        result[index] = np.sum(w * (y - mean) ** 2)
    return result


@pytest.mark.parametrize("template_index", [0, 1])
def test_fft_evaluates_both_templates_and_local_constant_fit(template_index):
    rng = np.random.default_rng(11 + template_index)
    teff = 1.0
    t0_step = 0.2
    oversample = 4
    calc_dt = t0_step / oversample
    teff_coeff = 2.73  # avoid putting regular samples exactly on window edges

    full_time = np.arange(0.0, 20.0 + 0.5 * calc_dt, calc_dt)
    keep = rng.random(full_time.size) > 0.25
    time = full_time[keep]
    ferr = rng.uniform(0.04, 0.12, time.size)
    weight = 1.0 / ferr**2
    true_t0 = 10.0

    template_fn = (
        FFTAnomalyGridRunner.template_high_magnification
        if template_index == 0
        else FFTAnomalyGridRunner.template_low_magnification
    )
    amplitude = 2.0 if template_index == 0 else 30.0
    flux = 0.7 + amplitude * template_fn(time - true_t0, teff)
    t0_grid = np.arange(0.0, 20.0, t0_step)

    runner = FFTAnomalyGridRunner(oversample=oversample)
    fft = runner.run(
        time=time,
        flux=flux,
        weight=weight,
        t0_grids=[t0_grid],
        teff_values=[teff],
        t0_steps=[t0_step],
        teff_coeff=teff_coeff,
        min_pts=4,
    )
    exact = runner.refine_points(
        time=time,
        flux=flux,
        weight=weight,
        t0=t0_grid,
        teff=np.full_like(t0_grid, teff),
        teff_coeff=teff_coeff,
        min_pts=4,
    )

    np.testing.assert_allclose(fft.dchi2, exact.dchi2, rtol=2.0e-11, atol=3.0e-8)
    np.testing.assert_array_equal(fft.n_window, exact.n_window)
    np.testing.assert_allclose(
        fft.flat_chi2,
        _direct_flat_chi2(time, flux, weight, t0_grid, teff, teff_coeff),
        rtol=2.0e-11,
        atol=3.0e-8,
    )

    best = int(np.argmax(fft.dchi2))
    assert t0_grid[best] == pytest.approx(true_t0)
    assert fft.template_index[best] == template_index
    assert exact.template_index[best] == template_index


def test_fft_constant_fit_is_stable_under_large_flux_offset_and_weight_range():
    rng = np.random.default_rng(23)
    teff = 1.0
    t0_step = 0.2
    oversample = 4
    calc_dt = t0_step / oversample
    teff_coeff = 2.73
    time = np.arange(0.0, 20.0 + 0.5 * calc_dt, calc_dt)
    weight = np.exp(rng.uniform(-8.0, 8.0, time.size))
    true_t0 = 10.0
    signal = 3.0 * FFTAnomalyGridRunner.template_high_magnification(time - true_t0, teff)
    flux = 1.0e6 + signal + rng.normal(0.0, 1.0 / np.sqrt(weight))
    t0_grid = np.arange(0.0, 20.0, t0_step)

    runner = FFTAnomalyGridRunner(oversample=oversample)
    fft = runner.run(
        time=time,
        flux=flux,
        weight=weight,
        t0_grids=[t0_grid],
        teff_values=[teff],
        t0_steps=[t0_step],
        teff_coeff=teff_coeff,
        min_pts=4,
    )
    exact = runner.refine_points(
        time=time,
        flux=flux,
        weight=weight,
        t0=t0_grid,
        teff=np.full_like(t0_grid, teff),
        teff_coeff=teff_coeff,
        min_pts=4,
    )

    assert t0_grid[int(np.argmax(fft.dchi2))] == pytest.approx(true_t0)
    np.testing.assert_allclose(fft.dchi2, exact.dchi2, rtol=2.0e-8, atol=2.0e-5)


@pytest.mark.parametrize("template_index", [0, 1])
def test_fft_recovers_irregularly_sampled_candidate(template_index):
    rng = np.random.default_rng(30 + template_index)
    time = 2_450_000.0 + np.sort(rng.uniform(0.0, 35.0, 700))
    ferr = rng.uniform(0.02, 0.06, time.size)
    weight = 1.0 / ferr**2
    true_t0 = 2_450_017.43
    teff = 0.8
    t0_step = 0.08

    template_fn = (
        FFTAnomalyGridRunner.template_high_magnification
        if template_index == 0
        else FFTAnomalyGridRunner.template_low_magnification
    )
    amplitude = 1.5 if template_index == 0 else 35.0
    flux = 0.2 + amplitude * template_fn(time - true_t0, teff)
    flux += rng.normal(0.0, ferr)
    t0_grid = np.arange(time.min(), time.max(), t0_step)

    result = FFTAnomalyGridRunner(oversample=8).run(
        time=time,
        flux=flux,
        weight=weight,
        t0_grids=[t0_grid],
        teff_values=[teff],
        t0_steps=[t0_step],
        teff_coeff=3.0,
        min_pts=4,
    )

    best = int(np.argmax(result.dchi2))
    assert abs(result.t0[best] - true_t0) <= t0_step
    assert result.template_index[best] == template_index


def test_exact_cluster_refinement_matches_existing_jax_definitions():
    rng = np.random.default_rng(41)
    time = np.sort(rng.uniform(0.0, 20.0, 240))
    flux = rng.normal(0.0, 1.0, time.size)
    ferr = rng.uniform(0.1, 0.3, time.size)
    weight = 1.0 / ferr**2
    t0 = np.array([3.1, 7.3, 11.4, 17.2])
    teff = np.array([0.5, 1.0, 2.0, 0.7])

    exact = FFTAnomalyGridRunner().refine_points(
        time=time,
        flux=flux,
        weight=weight,
        t0=t0,
        teff=teff,
        sigma=3.0,
        teff_coeff=2.73,
        min_pts=4,
    )
    jax_result = GridRunner.run(
        jnp.asarray(time),
        jnp.asarray(flux),
        jnp.asarray(weight),
        jnp.asarray(t0),
        jnp.asarray(teff),
        sigma=3.0,
        teff_coeff=2.73,
        min_pts=4,
    )
    jax_metrics = [np.asarray(value) for value in jax_result[2:]]

    np.testing.assert_allclose(exact.dchi2, jax_metrics[0], rtol=2.0e-6, atol=2.0e-5)
    np.testing.assert_array_equal(exact.n_window, jax_metrics[1])
    np.testing.assert_array_equal(exact.n_contrib, jax_metrics[2])
    np.testing.assert_allclose(exact.n_eff, jax_metrics[3], rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_allclose(exact.peak_frac, jax_metrics[4], rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_allclose(exact.rho1, jax_metrics[5], rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_array_equal(exact.longest_run, jax_metrics[6])


def test_season_runner_accepts_fft_backend_and_refines_cluster_metrics():
    rng = np.random.default_rng(52)
    time = np.sort(rng.uniform(0.0, 40.0, 800))
    ferr = rng.uniform(0.04, 0.08, time.size)
    true_t0 = 20.2
    teff = 1.0
    residual = 1.5 * FFTAnomalyGridRunner.template_high_magnification(time - true_t0, teff)
    residual += rng.normal(0.0, ferr)

    config = FinderConfig(
        grid_backend="fft",
        teff_init=teff,
        teff_grid_n=1,
        dt0_coeff=0.2,
        teff_coeff=2.73,
        min_cluster_points=1,
        fft_oversample=8,
    )
    assert config.grid_backend == "fft"
    assert config.fft_oversample == 8

    runner = SeasonGridRunner(
        splitter=SeasonSplitter(gap=100.0),
        extractor=ResultExtractor(sigma_overlap=3.0, min_points=1),
        config=config,
    )
    seasons, clusters, metrics = runner.run(
        time_j=jnp.asarray(time),
        residual_j=jnp.asarray(residual),
        ferr_j=jnp.asarray(ferr),
        time_np=time,
        verbose=False,
    )

    assert len(seasons) == 1
    assert clusters.shape[1] == 3
    best = clusters[int(np.argmax(clusters[:, 2]))]
    assert abs(best[0] - true_t0) <= config.dt0_coeff * teff

    row_index = int(np.argmin(np.abs(metrics[:, 0] - best[0]) + np.abs(metrics[:, 1] - best[1])))
    row = metrics[row_index]
    assert row[2] == pytest.approx(best[2])
    assert row[3] >= config.min_pts_in_window
    assert row[5] > 0.0

    direct = FFTAnomalyGridRunner().refine_points(
        time=time,
        flux=residual,
        weight=1.0 / ferr**2,
        t0=[best[0]],
        teff=[best[1]],
        sigma=config.sigma,
        teff_coeff=config.teff_coeff,
        min_pts=config.min_pts_in_window,
    )
    assert best[2] == pytest.approx(direct.dchi2[0])
    assert int(row[3]) == int(direct.n_window[0])
    assert row[5] == pytest.approx(direct.n_eff[0])


def test_fft_grid_size_guard():
    runner = FFTAnomalyGridRunner(oversample=16, max_grid_points=100)
    time = np.array([0.0, 10.0])
    flux = np.ones_like(time)
    weight = np.ones_like(time)
    t0_grid = np.arange(0.0, 10.0, 0.1)

    with pytest.raises(ValueError, match="exceeding max_grid_points"):
        runner.run(
            time=time,
            flux=flux,
            weight=weight,
            t0_grids=[t0_grid],
            teff_values=[1.0],
            t0_steps=[0.1],
        )
