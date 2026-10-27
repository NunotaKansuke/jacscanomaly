import numpy as np

from jacscanomaly import Candidate, Feature, FeatureConfig, measure_features
from jacscanomaly.singlelens_fit import SingleLensFitResult
from jacscanomaly.singlelens_model import A_pspl_func


def _fit(time, residual, ferr, params=(10.0, 3.0, 0.2)):
    params = np.asarray(params, dtype=float)
    model_flux = 2.0 * np.asarray(A_pspl_func(params, time)) + 1.0
    return SingleLensFitResult(
        time=time,
        flux=model_flux + residual,
        ferr=ferr,
        params=params,
        param_names=("t0", "tE", "u0"),
        chi2=np.array(float(np.sum((residual / ferr) ** 2))),
        chi2_dof=np.array(1.0),
        fs=np.array(2.0),
        fb=np.array(1.0),
        model_flux=model_flux,
        residual=residual,
    )


def _gauss(time, center, width, amplitude):
    return amplitude * np.exp(-0.5 * ((time - center) / width) ** 2)


def test_single_bump():
    time = np.linspace(0.0, 20.0, 401)
    candidate = Candidate(9.2 - 0.45, 9.2 + 0.45)
    fit = _fit(time, _gauss(time, 9.2, 0.12, 0.35), np.full_like(time, 0.02))

    features = measure_features(fit, candidate)

    assert len(features.bumps) == 1 and features.dips == ()
    assert abs(features.bumps[0].time - 9.2) < 0.1
    assert features.bumps[0].magnification_ratio > 1.0
    assert features.bumps[0].candidate is candidate


def test_bump_timescale_uses_interpolated_half_width():
    time = np.arange(9.0, 11.0001, 0.2)
    fit = _fit(time, _gauss(time, 10.0, 0.2, 0.2), np.full_like(time, 0.02))

    bump = measure_features(
        fit, Candidate(9.2001, 10.7999), config=FeatureConfig(smooth_points=1)
    ).bumps[0]

    assert 9.70 < bump.t_start < 9.80
    assert 10.20 < bump.t_end < 10.30
    assert bump.timescale > 0.45


def test_single_dip():
    time = np.linspace(0.0, 20.0, 401)
    fit = _fit(time, _gauss(time, 10.6, 0.16, -0.25), np.full_like(time, 0.02))

    features = measure_features(fit, Candidate(10.1, 11.1))

    assert features.bumps == () and len(features.dips) == 1
    assert abs(features.dips[0].time - 10.6) < 0.1
    assert features.dips[0].magnification_ratio < 1.0


def test_open_negative_tail_is_not_a_dip():
    time = np.linspace(0.0, 20.0, 401)
    residual = np.where(time >= 10.0, -0.20 * np.exp(-(time - 10.0) / 2.0), 0.0)

    features = measure_features(
        _fit(time, residual, np.full_like(time, 0.02)), Candidate(10.0, 11.99)
    )

    assert features.features == ()


def test_two_bumps():
    time = np.linspace(0.0, 20.0, 401)
    residual = (
        _gauss(time, 9.0, 0.10, 0.32)
        + _gauss(time, 11.0, 0.10, 0.30)
        + 0.05 * ((time > 9.2) & (time < 10.8))
    )

    features = measure_features(
        _fit(time, residual, np.full_like(time, 0.02)),
        Candidate(8.61, 11.39),
        config=FeatureConfig(min_snr=4.0),
    )

    assert len(features.bumps) == 2
    assert np.allclose([f.time for f in features.bumps], [9.0, 11.0], atol=0.1)
    assert all(f.timescale > 0.0 for f in features.bumps)


def test_closed_dip_between_bumps_and_summaries():
    time = np.linspace(0.0, 20.0, 801)
    residual = (
        _gauss(time, 9.0, 0.10, 0.30)
        + _gauss(time, 10.0, 0.14, -0.24)
        + _gauss(time, 11.0, 0.12, 0.28)
    )

    features = measure_features(
        _fit(time, residual, np.full_like(time, 0.02)),
        Candidate(8.51, 11.49),
        config=FeatureConfig(smooth_points=3, min_snr=4.0),
    )

    assert [f.kind for f in features.features] == ["bump", "dip", "bump"]
    assert np.allclose([f.time for f in features.features], [9.0, 10.0, 11.0], atol=0.05)
    assert all(f.strength >= 10.0 for f in features.features)
    assert features.bumps[0].fractional_deviation > 0.0
    assert features.summary_dict()["n_features"] == 3
    assert len(features.feature_dicts()) == 3
    assert "bumps=2, dips=1" in features.summary_text()


def test_deep_bracketed_dip():
    time = np.linspace(8.0, 12.0, 801)
    residual = (
        _gauss(time, 9.4, 0.08, 0.08)
        + _gauss(time, 10.0, 0.10, -0.20)
        + _gauss(time, 10.6, 0.08, 0.07)
    )

    features = measure_features(
        _fit(time, residual, np.full_like(time, 0.01)),
        Candidate(9.001, 10.999),
        config=FeatureConfig(smooth_points=3),
    )

    assert [f.kind for f in features.features] == ["bump", "dip", "bump"]
    assert abs(features.dips[0].time - 10.0) < 0.05


def test_dip_with_only_one_adjacent_bump():
    time = np.linspace(8.0, 12.0, 801)
    residual = _gauss(time, 10.0, 0.12, -0.22) + _gauss(time, 10.7, 0.08, 0.10)

    features = measure_features(
        _fit(time, residual, np.full_like(time, 0.01)),
        Candidate(9.501, 11.099),
        config=FeatureConfig(smooth_points=3),
    )

    assert [f.kind for f in features.features] == ["dip", "bump"]
    assert features.strongest is features.dips[0]


def test_locally_prominent_smaller_bump():
    time = np.linspace(8.0, 12.0, 801)
    residual = (
        _gauss(time, 9.4, 0.08, 0.08)
        + _gauss(time, 10.0, 0.12, 0.32)
        + _gauss(time, 10.5, 0.40, -0.06)
    )

    features = measure_features(
        _fit(time, residual, np.full_like(time, 0.01)),
        Candidate(9.001, 11.499),
        config=FeatureConfig(smooth_points=3),
    )

    assert len(features.bumps) == 2 and len(features.dips) == 1
    assert np.allclose([f.time for f in features.bumps], [9.4, 10.0], atol=0.05)


def test_features_split_at_large_time_gap_and_keep_their_candidate():
    time = np.concatenate((np.linspace(0.0, 1.0, 101), np.linspace(100.0, 101.0, 101)))
    residual = _gauss(time, 0.5, 0.08, 0.20) + _gauss(time, 100.5, 0.08, 0.18)
    first, second = Candidate(0.0, 1.0, label="a"), Candidate(100.0, 101.0, label="b")

    features = measure_features(
        _fit(time, residual, np.full_like(time, 0.01), params=(50.0, 20.0, 0.2)),
        [first, second],
        config=FeatureConfig(smooth_points=3),
    )

    assert len(features.bumps) == 2
    assert all(f.timescale < 1.0 for f in features.bumps)
    assert [f.candidate for f in features.bumps] == [first, second]


def test_no_signal_gives_empty_result():
    time = np.linspace(0.0, 20.0, 101)

    features = measure_features(
        _fit(time, np.zeros_like(time), np.full_like(time, 0.02)), Candidate(9.0, 11.0)
    )

    assert features.features == ()
    assert features.strongest is None


def test_signal_outside_every_candidate_is_ignored():
    time = np.linspace(0.0, 20.0, 401)
    fit = _fit(time, _gauss(time, 9.2, 0.12, 0.35), np.full_like(time, 0.02))

    assert measure_features(fit, Candidate(14.0, 16.0)).features == ()


def _edge(time, center, rise, decay, amplitude):
    width = np.where(time < center, rise, decay)
    return amplitude * np.exp(-0.5 * ((time - center) / width) ** 2)


def test_asymmetric_bump_is_a_bump():
    # asymmetry alone is not a caustic signature (a smooth skewed bump at high
    # S/N is as asymmetric), so no shape label is given
    rng = np.random.default_rng(1)
    time = np.linspace(8.0, 12.0, 801)
    residual = _edge(time, 9.5, 0.02, 0.15, 0.3) + 0.01 * rng.standard_normal(time.size)
    candidate = Candidate(9.0, 11.0)

    features = measure_features(_fit(time, residual, np.full_like(time, 0.01)), candidate)

    assert [f.kind for f in features.features] == ["bump"]
    assert abs(features.bumps[0].time - 9.5) < 0.05
    assert features.bumps[0].candidate is candidate
