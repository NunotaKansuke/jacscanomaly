from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from jacscanomaly import (
    Finder,
    FinderConfig,
    FlatBaselineDiagnostic,
    PlanetFeatureConfig,
    PlanetSignalCandidate,
    PlanetSignalConfig,
    PlanetSignalExtractor,
    PlanetSignalResult,
)
from jacscanomaly.singlelens_fit import SingleLensFitResult
from jacscanomaly.singlelens_model import A_pspl_func
from jacscanomaly.models import BestCandidate, CandidateQuality


def _flat_diagnostic(use_flat_baseline=False):
    return FlatBaselineDiagnostic(
        use_flat_baseline=use_flat_baseline,
        peak_in_mask=False,
        u0=0.2,
        n_peak_support=0,
        n_unmasked_peak_support=0,
        masked_peak_fraction=0.0,
        dchi2_flat_minus_pspl=0.0,
        improvement_n_eff=0.0,
        improvement_peak_frac=0.0,
    )


def _feature_result(time, flux, ferr, residual, mask, *, use_flat_baseline=False):
    params = np.array([10.0, 3.0, 0.2])
    A = np.asarray(A_pspl_func(params, time))
    fb = 1.0
    fs = 2.0
    model_flux = fs * A + fb
    fit = SingleLensFitResult(
        time=time,
        flux=flux,
        ferr=ferr,
        params=params,
        param_names=("t0", "tE", "u0"),
        chi2=np.array(float(np.sum((residual / ferr) ** 2))),
        chi2_dof=np.array(1.0),
        fs=np.array(fs),
        fb=np.array(fb),
        model_flux=model_flux,
        residual=residual,
    )
    candidate = PlanetSignalCandidate(
        start_index=int(np.flatnonzero(mask)[0]),
        end_index=int(np.flatnonzero(mask)[-1]) + 1,
        t_start=float(time[np.flatnonzero(mask)[0]]),
        t_end=float(time[np.flatnonzero(mask)[-1]]),
        t_center=float(np.mean(time[mask])),
        n_points=int(np.sum(mask)),
        chi2=float(np.sum((residual[mask] / ferr[mask]) ** 2)),
        reduced_chi2=1.0,
        max_abs_z=float(np.max(np.abs(residual[mask] / ferr[mask]))),
        peak_time=float(time[np.argmax(np.abs(residual / ferr))]),
        peak_z=float(np.max(residual / ferr)),
        signed_sum_z=float(np.sum(residual[mask] / ferr[mask])),
    )
    return PlanetSignalResult(
        time=time,
        flux=flux,
        ferr=ferr,
        initial_fit=fit,
        refined_fit=fit,
        initial_residual=residual,
        refined_residual=residual,
        signal_mask=mask,
        point_weight=np.where(mask, 0.0, 1.0),
        flat_baseline_diagnostic=_flat_diagnostic(use_flat_baseline=use_flat_baseline),
        iterations=(),
        candidates=(candidate,),
        best=candidate,
    )


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


def test_planet_signal_extractor_frozen_baseline_never_refits(monkeypatch):
    time = np.linspace(0.0, 20.0, 240)
    params = np.array([10.0, 5.0, 0.2])
    ferr = np.full_like(time, 0.02)
    flux = 2.0 * np.asarray(A_pspl_func(params, time)) + 0.1
    flux += 0.3 * np.exp(-0.5 * ((time - 9.0) / 0.12) ** 2)
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
    initial_fit = finder.fit_single_lens(time, flux, ferr, x0=params)
    extractor = PlanetSignalExtractor(
        finder,
        PlanetSignalConfig(max_iter=2, seed_min_dchi2=20.0, candidate_min_points=2),
    )

    def fail_refit(**_kwargs):
        raise AssertionError("a frozen adopted model must not be refit")

    monkeypatch.setattr(extractor, "_fit_masked_single_lens_and_evaluate_full", fail_refit)
    result = extractor.run(
        time,
        flux,
        ferr,
        initial_fit=initial_fit,
        refit=False,
        freeze_baseline=True,
        prior_signal_windows=((9.0, 0.25),),
    )

    assert result.refined_fit is initial_fit
    assert result.signal_mask.any()
    assert result.best is not None


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


def test_planet_signal_extractor_keeps_beam_intervals_compact_relative_to_tE():
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
    assert any(
        candidate.t_start < 8.9 < candidate.t_end
        for candidate in result.candidates
    )
    assert any(
        candidate.t_start < 10.9 < candidate.t_end
        for candidate in result.candidates
    )
    assert all(
        candidate.t_end - candidate.t_start
        <= 0.25 * abs(float(result.refined_fit.params[1]))
        for candidate in result.candidates
    )
    assert not np.all(result.signal_mask[(time > 9.4) & (time < 10.4)])


def test_prior_signal_window_does_not_mask_a_broad_smooth_residual(monkeypatch):
    time = np.linspace(0.0, 20.0, 401)
    params = np.array([10.0, 4.0, 0.2])
    baseline_flux = 1.5 * np.asarray(A_pspl_func(params, time)) + 0.1
    ferr = np.full_like(time, 0.02)
    finder = Finder(FinderConfig(grid_backend="jax"))
    fit = finder.fit_single_lens(time, baseline_flux, ferr, x0=params)
    flux = baseline_flux + 0.4 * np.exp(-0.5 * ((time - 10.0) / 2.0) ** 2)
    residual = flux - np.asarray(fit.model_flux)
    fit = replace(
        fit,
        flux=flux,
        residual=residual,
        chi2=np.asarray(np.sum(np.square(residual / ferr))),
    )
    extractor = PlanetSignalExtractor(
        finder,
        PlanetSignalConfig.fast(
            beam_max_iter=0,
            max_signal_span_over_tE=0.25,
        ),
    )

    result = extractor.run(
        time,
        flux,
        ferr,
        initial_fit=fit,
        refit=False,
        freeze_baseline=True,
        prior_signal_windows=((10.0, 8.0),),
    )

    assert not np.any(result.signal_mask)


def test_beam_interval_stops_after_first_weak_scan(monkeypatch):
    """Routine events must not repeat an unchanged beam grid scan."""
    time = np.linspace(0.0, 20.0, 80)
    params = np.array([10.0, 4.0, 0.2])
    flux = 1.5 * np.asarray(A_pspl_func(params, time)) + 0.1
    ferr = np.full_like(time, 0.02)
    extractor = PlanetSignalExtractor(
        Finder(FinderConfig(grid_backend="jax")),
        PlanetSignalConfig(beam_max_iter=3, seed_min_dchi2=20.0),
    )
    calls = []

    def weak_scan(*args, **kwargs):
        calls.append(None)
        return None

    monkeypatch.setattr(extractor, "_scan_best", weak_scan)
    result = extractor.run(time, flux, ferr, x0=params, refit=False)

    assert len(calls) == 1
    assert not result.signal_mask.any()


def test_default_beam_uses_three_adaptive_single_branch_iterations():
    config = PlanetSignalConfig()

    assert config.beam_max_iter == 3
    assert config.beam_width == 1
    assert config.beam_candidates_per_iter == 1


def test_beam_rejects_a_new_pspl_u0_boundary_solution():
    extractor = PlanetSignalExtractor(Finder())
    reference = SimpleNamespace(
        params=np.asarray([100.0, 2.0, 0.0047]),
        model_kind="pspl",
    )
    candidate = SimpleNamespace(
        params=np.asarray([100.0, 2.4, -1.0e-4]),
        model_kind="pspl",
    )

    assert extractor._continuation_hit_new_pspl_u0_bound(
        reference,
        candidate,
    )


def test_beam_keeps_a_pspl_solution_that_started_near_the_u0_bound():
    extractor = PlanetSignalExtractor(Finder())
    reference = SimpleNamespace(
        params=np.asarray([100.0, 2.0, 1.5e-4]),
        model_kind="pspl",
    )
    candidate = SimpleNamespace(
        params=np.asarray([100.0, 2.1, 1.0e-4]),
        model_kind="pspl",
    )

    assert not extractor._continuation_hit_new_pspl_u0_bound(
        reference,
        candidate,
    )


def test_beam_interval_stops_when_no_interval_is_adopted(monkeypatch):
    time = np.linspace(0.0, 20.0, 80)
    params = np.array([10.0, 4.0, 0.2])
    flux = 1.5 * np.asarray(A_pspl_func(params, time)) + 0.1
    ferr = np.full_like(time, 0.02)
    extractor = PlanetSignalExtractor(
        Finder(FinderConfig(grid_backend="jax")),
        PlanetSignalConfig(beam_max_iter=3, seed_min_dchi2=20.0),
    )
    calls = []
    seed = BestCandidate(
        t0=10.0,
        teff=0.1,
        dchi2=100.0,
        med_others=0.0,
        std_others=1.0,
        score=100.0,
        quality=CandidateQuality(10, 5, 5.0, 0.2, 0.0, 5),
    )

    def strong_scan(*args, **kwargs):
        calls.append(None)
        return seed

    monkeypatch.setattr(extractor, "_scan_best", strong_scan)
    monkeypatch.setattr(extractor, "_beam_interval_masks_from_seed", lambda **kwargs: ())
    result = extractor.run(time, flux, ferr, x0=params, refit=False)

    assert len(calls) == 1
    assert not result.signal_mask.any()


def test_beam_interval_reuses_cached_initial_seed_without_rescan(monkeypatch):
    time = np.linspace(0.0, 20.0, 80)
    params = np.array([10.0, 4.0, 0.2])
    flux = 1.5 * np.asarray(A_pspl_func(params, time)) + 0.1
    ferr = np.full_like(time, 0.02)
    extractor = PlanetSignalExtractor(
        Finder(FinderConfig(grid_backend="jax")),
        PlanetSignalConfig(beam_max_iter=1, seed_min_dchi2=20.0),
    )
    seed = BestCandidate(
        t0=10.0,
        teff=0.1,
        dchi2=100.0,
        med_others=0.0,
        std_others=1.0,
        score=100.0,
        quality=CandidateQuality(10, 5, 5.0, 0.2, 0.0, 5),
    )

    monkeypatch.setattr(
        extractor,
        "_scan_best",
        lambda *args, **kwargs: pytest.fail("cached first seed should avoid a grid scan"),
    )
    monkeypatch.setattr(extractor, "_beam_interval_masks_from_seed", lambda **kwargs: ())
    result = extractor.run(
        time, flux, ferr, x0=params, refit=False, initial_seed=seed
    )

    assert result.initial_seed is seed
    assert result.timing.n_scans == 0


def test_probe_returns_seed_without_masked_fit(monkeypatch):
    time = np.linspace(0.0, 20.0, 80)
    params = np.array([10.0, 4.0, 0.2])
    flux = 1.5 * np.asarray(A_pspl_func(params, time)) + 0.1
    ferr = np.full_like(time, 0.02)
    extractor = PlanetSignalExtractor(
        Finder(FinderConfig(grid_backend="jax")),
        PlanetSignalConfig.probe(seed_min_dchi2=20.0),
    )
    seed = BestCandidate(
        t0=10.0,
        teff=0.1,
        dchi2=100.0,
        med_others=0.0,
        std_others=1.0,
        score=100.0,
        quality=CandidateQuality(10, 5, 5.0, 0.2, 0.0, 5),
    )
    monkeypatch.setattr(extractor, "_scan_best", lambda *args, **kwargs: seed)
    monkeypatch.setattr(
        extractor,
        "_fit_masked_single_lens_and_evaluate_full",
        lambda **kwargs: pytest.fail("probe must not fit a masked baseline"),
    )

    result = extractor.run(time, flux, ferr, x0=params, refit=False)

    assert result.initial_seed is seed
    assert not result.signal_mask.any()


def test_planet_signal_extractor_uses_flat_baseline_for_masked_tiny_u0_peak():
    time = np.linspace(0.0, 20.0, 120)
    flux = np.ones_like(time) * 2.0
    ferr = np.full_like(time, 0.1)
    params = np.array([10.0, 1.0, 1.0e-4])
    fit = SingleLensFitResult(
        time=time,
        flux=flux,
        ferr=ferr,
        params=params,
        param_names=("t0", "tE", "u0"),
        chi2=np.array(0.0),
        chi2_dof=np.array(0.0),
        fs=np.array(1.0),
        fb=np.array(1.0),
        model_flux=np.ones_like(time),
        residual=flux - 1.0,
    )
    mask = np.abs(time - 10.0) < 2.0
    extractor = PlanetSignalExtractor(Finder(FinderConfig()))

    diagnostic = extractor._flat_baseline_diagnostic(fit, mask)
    assert diagnostic.use_flat_baseline
    assert diagnostic.peak_in_mask
    assert diagnostic.n_unmasked_peak_support == 0
    assert diagnostic.masked_peak_fraction == 1.0

    flat = extractor._fit_flat_baseline_and_evaluate_full(
        time_j=time,
        flux_j=flux,
        ferr_j=ferr,
        keep_mask_np=~mask,
        fit=fit,
    )

    assert flat.fs == 0.0
    assert np.allclose(flat.model_flux, 2.0)
    assert np.allclose(np.asarray(flat.params), params)


def test_planet_signal_refit_preserves_selected_model_family():
    time = np.linspace(0.0, 20.0, 40)
    ferr = np.full_like(time, 0.1)
    params = np.array([10.0, 3.0, 0.2])
    model = 2.0 * np.asarray(A_pspl_func(params, time)) + 1.0

    fit = SingleLensFitResult(
        time=time,
        flux=model,
        ferr=ferr,
        params=params,
        param_names=("t0", "tE", "u0"),
        chi2=np.array(0.0),
        chi2_dof=np.array(0.0),
        fs=np.array(2.0),
        fb=np.array(1.0),
        model_flux=model,
        residual=np.zeros_like(time),
    )
    object.__setattr__(fit, "model_kind", "fspl_vbm_fd")

    class FixedModelFitter:
        def __init__(self):
            self.called = None

        def fit_fixed_model(self, time, flux, ferr, q0, model_kind):
            self.called = model_kind
            return fit

        def fit(self, *args, **kwargs):  # pragma: no cover - must not be used
            raise AssertionError("masked refit changed model-selection path")

    fitter = FixedModelFitter()
    extractor = PlanetSignalExtractor(Finder(FinderConfig(), fitter=fitter))
    extractor._fit_masked_single_lens_and_evaluate_full(
        time_j=time,
        flux_j=model,
        ferr_j=ferr,
        keep_mask_np=np.ones(time.shape, dtype=bool),
        x0_j=params,
        model_kind="fspl_vbm_fd",
    )

    assert fitter.called == "fspl_vbm_fd"


def test_planet_signal_measurement_identifies_single_peak():
    time = np.linspace(0.0, 20.0, 401)
    ferr = np.full_like(time, 0.02)
    params = np.array([10.0, 3.0, 0.2])
    model = 2.0 * np.asarray(A_pspl_func(params, time)) + 1.0
    residual = 0.35 * np.exp(-0.5 * ((time - 9.2) / 0.12) ** 2)
    flux = model + residual
    mask = np.abs(time - 9.2) < 0.45
    result = _feature_result(time, flux, ferr, residual, mask)

    features = result.measure_features()

    assert features.n_peaks == 1
    assert features.n_dips == 0
    assert abs(features.peaks[0].time - 9.2) < 0.1
    assert features.peaks[0].magnification_ratio > 1.0


def test_planet_signal_peak_timescale_uses_interpolated_half_width():
    time = np.arange(9.0, 11.0001, 0.2)
    ferr = np.full_like(time, 0.02)
    params = np.array([10.0, 3.0, 0.2])
    model = 2.0 * np.asarray(A_pspl_func(params, time)) + 1.0
    sigma = 0.2
    residual = 0.2 * np.exp(-0.5 * ((time - 10.0) / sigma) ** 2)
    flux = model + residual
    mask = np.abs(time - 10.0) < 0.8
    result = _feature_result(time, flux, ferr, residual, mask)

    features = result.measure_features(PlanetFeatureConfig(smooth_points=1))

    peak = features.peaks[0]
    assert 9.70 < peak.t_start < 9.80
    assert 10.20 < peak.t_end < 10.30
    assert peak.timescale > 0.45


def test_planet_signal_measurement_identifies_dip():
    time = np.linspace(0.0, 20.0, 401)
    ferr = np.full_like(time, 0.02)
    params = np.array([10.0, 3.0, 0.2])
    model = 2.0 * np.asarray(A_pspl_func(params, time)) + 1.0
    residual = -0.25 * np.exp(-0.5 * ((time - 10.6) / 0.16) ** 2)
    flux = model + residual
    mask = np.abs(time - 10.6) < 0.5
    result = _feature_result(time, flux, ferr, residual, mask)

    features = result.measure_features()

    assert features.n_peaks == 0
    assert features.n_dips == 1
    assert abs(features.dips[0].time - 10.6) < 0.1
    assert features.dips[0].magnification_ratio < 1.0


def test_planet_signal_measurement_rejects_open_negative_tail():
    time = np.linspace(0.0, 20.0, 401)
    ferr = np.full_like(time, 0.02)
    params = np.array([10.0, 3.0, 0.2])
    model = 2.0 * np.asarray(A_pspl_func(params, time)) + 1.0
    residual = np.where(time >= 10.0, -0.20 * np.exp(-(time - 10.0) / 2.0), 0.0)
    flux = model + residual
    mask = (time >= 10.0) & (time < 12.0)
    result = _feature_result(time, flux, ferr, residual, mask)

    features = result.measure_features()

    assert features.n_peaks == 0
    assert features.n_dips == 0


def test_planet_signal_measurement_identifies_two_peaks():
    time = np.linspace(0.0, 20.0, 401)
    ferr = np.full_like(time, 0.02)
    params = np.array([10.0, 3.0, 0.2])
    model = 2.0 * np.asarray(A_pspl_func(params, time)) + 1.0
    residual = (
        0.32 * np.exp(-0.5 * ((time - 9.0) / 0.10) ** 2)
        + 0.30 * np.exp(-0.5 * ((time - 11.0) / 0.10) ** 2)
        + 0.05 * ((time > 9.2) & (time < 10.8))
    )
    flux = model + residual
    mask = (time > 8.6) & (time < 11.4)
    result = _feature_result(time, flux, ferr, residual, mask)

    features = result.measure_features(PlanetFeatureConfig(min_abs_z=4.0))

    assert features.n_peaks == 2
    assert abs(features.peaks[0].time - 9.0) < 0.1
    assert abs(features.peaks[1].time - 11.0) < 0.1
    assert all(peak.timescale > 0.0 for peak in features.peaks)


def test_planet_feature_measurement_prefers_positive_features_over_dips():
    time = np.linspace(0.0, 20.0, 801)
    ferr = np.full_like(time, 0.02)
    params = np.array([10.0, 3.0, 0.2])
    model = 2.0 * np.asarray(A_pspl_func(params, time)) + 1.0
    residual = (
        0.30 * np.exp(-0.5 * ((time - 9.0) / 0.10) ** 2)
        - 0.24 * np.exp(-0.5 * ((time - 10.0) / 0.14) ** 2)
        + 0.28 * np.exp(-0.5 * ((time - 11.0) / 0.12) ** 2)
    )
    flux = model + residual
    mask = (time > 8.5) & (time < 11.5)
    result = _feature_result(time, flux, ferr, residual, mask)

    features = result.measure_features(
        PlanetFeatureConfig(smooth_points=3, min_abs_z=4.0)
    )

    assert features.n_peaks == 2
    assert features.n_dips == 0
    assert [feature.kind for feature in features.features] == ["peak", "peak"]
    assert np.allclose([feature.time for feature in features.features], [9.0, 11.0], atol=0.05)
    assert all(feature.timescale > 0.0 for feature in features.features)
    assert all(feature.strength >= 10.0 for feature in features.features)
    assert features.peaks[0].fractional_deviation > 0.0
    assert features.summary_dict()["n_features"] == 2
    assert len(features.feature_dicts()) == 2
    assert "peaks=2, dips=0" in features.summary_text()


def test_planet_feature_measurement_keeps_deep_bracketed_dip():
    time = np.linspace(8.0, 12.0, 801)
    ferr = np.full_like(time, 0.01)
    params = np.array([10.0, 3.0, 0.2])
    model = 2.0 * np.asarray(A_pspl_func(params, time)) + 1.0
    residual = (
        0.08 * np.exp(-0.5 * ((time - 9.4) / 0.08) ** 2)
        - 0.20 * np.exp(-0.5 * ((time - 10.0) / 0.10) ** 2)
        + 0.07 * np.exp(-0.5 * ((time - 10.6) / 0.08) ** 2)
    )
    flux = model + residual
    mask = (time > 9.0) & (time < 11.0)
    result = _feature_result(time, flux, ferr, residual, mask)

    features = result.measure_features(PlanetFeatureConfig(smooth_points=3))

    assert features.n_peaks == 2
    assert features.n_dips == 1
    assert abs(features.dips[0].time - 10.0) < 0.05


def test_planet_feature_measurement_keeps_locally_prominent_smaller_peak():
    time = np.linspace(8.0, 12.0, 801)
    ferr = np.full_like(time, 0.01)
    params = np.array([10.0, 3.0, 0.2])
    model = 2.0 * np.asarray(A_pspl_func(params, time)) + 1.0
    residual = (
        0.08 * np.exp(-0.5 * ((time - 9.4) / 0.08) ** 2)
        + 0.32 * np.exp(-0.5 * ((time - 10.0) / 0.12) ** 2)
        - 0.06 * np.exp(-0.5 * ((time - 10.5) / 0.40) ** 2)
    )
    flux = model + residual
    mask = (time > 9.0) & (time < 11.5)
    result = _feature_result(time, flux, ferr, residual, mask)

    features = result.measure_features(PlanetFeatureConfig(smooth_points=3))

    assert features.n_peaks == 2
    assert features.n_dips == 0
    assert np.allclose([feature.time for feature in features.peaks], [9.4, 10.0], atol=0.05)


def test_planet_feature_measurement_returns_empty_result_without_signal():
    time = np.linspace(0.0, 20.0, 101)
    ferr = np.full_like(time, 0.02)
    residual = np.zeros_like(time)
    params = np.array([10.0, 3.0, 0.2])
    model = 2.0 * np.asarray(A_pspl_func(params, time)) + 1.0
    seed_mask = np.zeros_like(time, dtype=bool)
    seed_mask[len(time) // 2] = True
    result = _feature_result(
        time,
        model,
        ferr,
        residual,
        seed_mask,
    )
    # The helper requires a candidate when a mask exists, but an empty mask is
    # valid for a completed extraction with no measured signal.
    result = PlanetSignalResult(
        **{
            **result.__dict__,
            "signal_mask": np.zeros_like(time, dtype=bool),
            "point_weight": np.ones_like(time),
            "candidates": (),
            "best": None,
        }
    )

    features = result.measure_features()
    assert features.n_peaks == 0
    assert features.n_dips == 0
    assert features.features == ()
    assert features.strongest is None


def test_planet_signal_measurement_does_not_label_flat_baseline_shape():
    time = np.linspace(0.0, 20.0, 401)
    ferr = np.full_like(time, 0.02)
    params = np.array([10.0, 3.0, 0.2])
    model = 2.0 * np.asarray(A_pspl_func(params, time)) + 1.0
    residual = 0.3 * np.exp(-0.5 * ((time - 10.0) / 1.0) ** 2)
    flux = model + residual
    mask = np.abs(time - 10.0) < 2.0
    result = _feature_result(
        time,
        flux,
        ferr,
        residual,
        mask,
        use_flat_baseline=True,
    )

    features = result.measure_features()

    assert features.n_peaks == 1
    assert features.n_dips == 0


def test_planet_signal_extractor_accepts_non_pspl_fixed_fit():
    pytest.importorskip("microjax.fastlens")

    finder = Finder(FinderConfig(fitter_kind="fspl", grid_backend="jax"))
    extractor = PlanetSignalExtractor(finder)

    time = np.linspace(0.0, 4.0, 20)
    result = extractor.run(
        time,
        np.ones_like(time),
        np.full_like(time, 0.1),
        x0=[2.0, 1.0, 0.1, -7.0],
        refit=False,
    )

    assert tuple(result.refined_fit.param_names) == ("t0", "tE", "u0", "rho")
