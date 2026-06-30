import numpy as np

from jacscanomaly import Finder, FinderConfig, PlanetSignalConfig, PlanetSignalExtractor
from jacscanomaly.singlelens_model import A_pspl_func


def test_planet_signal_extractor_masks_local_unexplained_signal():
    time = np.linspace(0.0, 20.0, 240)
    params = np.array([10.0, 5.0, 0.2])
    flux = 2.0 * np.asarray(A_pspl_func(params, time)) + 0.1
    ferr = np.full_like(time, 0.02)

    anomaly_center = 9.0
    anomaly = np.exp(-0.5 * ((time - anomaly_center) / 0.12) ** 2)
    flux = flux + 0.3 * anomaly

    finder = Finder(
        FinderConfig(
            grid_backend="jax",
            single_fit_backend="jax",
            teff_init=0.08,
            common_ratio=1.5,
            teff_grid_n=6,
            dt0_coeff=0.5,
            min_pts_in_window=3,
        )
    )
    extractor = PlanetSignalExtractor(
        finder,
        PlanetSignalConfig(
            max_iter=2,
            seed_min_dchi2=20.0,
            mask_teff_coeff=4.0,
            candidate_min_points=2,
        ),
    )

    result = extractor.run(time, flux, ferr, x0=params, refit=False)

    assert len(result.iterations) >= 1
    assert result.best is not None
    assert result.signal_mask.any()
    assert result.best.t_start <= anomaly_center <= result.best.t_end
    assert result.best.max_abs_z > 5.0


def test_planet_signal_extractor_robust_mode_downweights_connected_structure():
    time = np.linspace(0.0, 20.0, 360)
    params = np.array([10.0, 4.0, 0.25])
    flux = 1.5 * np.asarray(A_pspl_func(params, time)) + 0.05
    ferr = np.full_like(time, 0.02)

    left_peak = np.exp(-0.5 * ((time - 8.9) / 0.08) ** 2)
    right_peak = np.exp(-0.5 * ((time - 10.9) / 0.08) ** 2)
    bridge_region = (time > 9.1) & (time < 10.7)
    bridge_phase = (time - 9.1) / (10.7 - 9.1)
    bridge = bridge_region * (0.12 * np.sin(2.0 * np.pi * bridge_phase))
    flux = flux + 0.35 * left_peak + 0.35 * right_peak + bridge

    finder = Finder(
        FinderConfig(
            grid_backend="jax",
            single_fit_backend="jax",
            teff_init=0.08,
            common_ratio=1.5,
            teff_grid_n=6,
            dt0_coeff=0.5,
            min_pts_in_window=3,
        )
    )
    extractor = PlanetSignalExtractor(
        finder,
        PlanetSignalConfig(
            baseline_mode="robust",
            seed_min_dchi2=20.0,
            robust_max_iter=3,
            robust_eta=0.6,
            robust_z_soft=2.5,
            robust_z_hard=8.0,
            robust_smooth_time=0.2,
            signal_weight_threshold=0.65,
            signal_min_abs_z=2.5,
            candidate_min_points=2,
        ),
    )

    result = extractor.run(time, flux, ferr, x0=params, refit=False)
    bridge_core = (time > 9.9) & (time < 10.55)

    assert len(result.iterations) >= 1
    assert result.signal_mask.any()
    assert np.median(result.point_weight[bridge_core]) < 0.8
    assert np.any(result.signal_mask[bridge_core])


def test_planet_signal_extractor_beam_interval_selects_connected_interval():
    time = np.linspace(0.0, 20.0, 360)
    params = np.array([10.0, 4.0, 0.25])
    flux = 1.5 * np.asarray(A_pspl_func(params, time)) + 0.05
    ferr = np.full_like(time, 0.02)

    left_peak = np.exp(-0.5 * ((time - 8.9) / 0.08) ** 2)
    right_peak = np.exp(-0.5 * ((time - 10.9) / 0.08) ** 2)
    bridge = ((time > 9.1) & (time < 10.7)).astype(float)
    flux = flux + 0.35 * left_peak + 0.35 * right_peak + 0.10 * bridge

    finder = Finder(
        FinderConfig(
            grid_backend="jax",
            single_fit_backend="jax",
            teff_init=0.08,
            common_ratio=1.5,
            teff_grid_n=6,
            dt0_coeff=0.5,
            min_pts_in_window=3,
        )
    )
    extractor = PlanetSignalExtractor(
        finder,
        PlanetSignalConfig(
            baseline_mode="beam_interval",
            seed_min_dchi2=20.0,
            beam_max_iter=2,
            beam_width=3,
            beam_candidates_per_iter=8,
            beam_point_penalty=4.0,
            beam_interval_penalty=20.0,
            candidate_min_points=2,
        ),
    )

    result = extractor.run(time, flux, ferr, x0=params, refit=False)

    assert result.best is not None
    assert result.best.t_start < 9.0
    assert result.best.t_end > 10.9
    assert np.all(result.signal_mask[(time > 9.4) & (time < 10.4)])


def test_planet_signal_extractor_is_pspl_only_for_now():
    finder = Finder(FinderConfig(fitter_kind="fspl", grid_backend="jax"))
    extractor = PlanetSignalExtractor(finder)

    try:
        extractor.run([0, 1, 2, 3], [1, 1, 1, 1], [0.1, 0.1, 0.1, 0.1], x0=[1, 1, 0.1, -7])
    except NotImplementedError as exc:
        assert "pspl" in str(exc)
    else:
        raise AssertionError("Expected NotImplementedError.")
