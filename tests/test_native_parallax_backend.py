import numpy as np
import pytest
from types import SimpleNamespace

pytest.importorskip("jacscanomaly._parallax_cpp")

from jacscanomaly import (
    Ephemeris,
    NativePSPLAnnualParallaxFitter,
    ParallaxEvaluator,
    TimeSpec,
)
from jacscanomaly.parallax_backend import native_parallax_effect_score
from jacscanomaly.plot import _adaptive_single_lens_curve


def _linear_ephemeris(origin="sun", time_spec=TimeSpec()):
    time = np.asarray([0.0, 1.0, 2.0])
    position = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.1, 0.2], [0.0, 0.4, 0.8]])
    return Ephemeris(time, position, origin=origin, time_spec=time_spec)


def test_native_basis_and_canonical_sign_at_reference_epoch():
    earth = _linear_ephemeris(time_spec=TimeSpec("hjd"))
    evaluator = ParallaxEvaluator(
        np.asarray([0.0, 1.0, 2.0]), np.ones(3), np.ones(3),
        ra_deg=0.0, dec_deg=0.0, tref=1.0, time_spec=TimeSpec("hjd"),
        earth_ephemeris=earth,
    )
    debug = evaluator.trajectory(np.asarray([1.0, 0.0, 0.1, 0.0, 0.0]), components=True)
    np.testing.assert_allclose(debug["observer_n"][1], 0.0, atol=1e-14)
    np.testing.assert_allclose(debug["observer_e"][1], 0.0, atol=1e-14)
    np.testing.assert_allclose(debug["tau"], [-1.0, 0.0, 1.0], atol=1e-14)
    np.testing.assert_allclose(debug["beta"], 0.1, atol=1e-14)


def test_geocentric_satellite_offset_is_not_reference_subtracted():
    earth = Ephemeris(np.asarray([0.0, 1.0, 2.0]), np.zeros((3, 3)), origin="sun", time_spec=TimeSpec("hjd"))
    satellite = Ephemeris(
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([[0.0, 0.0, 0.2], [0.0, 0.0, 0.3], [0.0, 0.0, 0.4]]),
        origin="earth", time_spec=TimeSpec("hjd"),
    )
    evaluator = ParallaxEvaluator(
        np.asarray([1.0]), np.ones(1), np.ones(1), ra_deg=0.0, dec_deg=0.0, tref=1.0,
        time_spec=TimeSpec("hjd"), earth_ephemeris=earth,
        satellite_or_observer_ephemeris=satellite,
    )
    debug = evaluator.trajectory(np.asarray([1.0, 0.0, 0.1, 0.0, 1.0]), components=True)
    np.testing.assert_allclose(debug["satellite_n"], [0.3], atol=1e-14)
    np.testing.assert_allclose(debug["beta"], [0.4], atol=1e-14)


def test_strict_gulls_requires_explicit_reference_and_keeps_reference_zero():
    observer = _linear_ephemeris(origin="explicit_reference", time_spec=TimeSpec("jd"))
    reference = Ephemeris(
        observer.time,
        observer.position_au * 0.5,
        origin="explicit_reference", time_spec=TimeSpec("jd"),
    )
    evaluator = ParallaxEvaluator(
        np.asarray([1.0]), np.ones(1), np.ones(1), ra_deg=0.0, dec_deg=0.0, tref=1.0,
        time_spec=TimeSpec("jd"), observer_convention="gulls",
        satellite_or_observer_ephemeris=observer, reference_ephemeris=reference,
    )
    debug = evaluator.trajectory(np.asarray([1.0, 0.0, 0.1, 0.0, 1.0]), components=True)
    np.testing.assert_allclose(debug["observer_n"], [0.1], atol=1e-14)
    with pytest.raises(ValueError, match="reference"):
        ParallaxEvaluator(
            np.asarray([1.0]), np.ones(1), np.ones(1), ra_deg=0.0, dec_deg=0.0, tref=1.0,
            time_spec=TimeSpec("jd"), observer_convention="gulls",
            satellite_or_observer_ephemeris=observer,
        )


def test_origin_metadata_is_not_guessed_from_distance():
    earth_like = Ephemeris(np.asarray([0.0, 1.0]), np.zeros((2, 3)), origin="earth")
    with pytest.raises(ValueError, match="origin"):
        ParallaxEvaluator(
            np.asarray([0.0, 1.0]), np.ones(2), np.ones(2), ra_deg=0.0, dec_deg=0.0, tref=0.5,
            time_spec=TimeSpec("hjd"), earth_ephemeris=earth_like,
        )


def test_time_spec_is_required_in_public_evaluator():
    with pytest.raises(ValueError, match="TimeSpec"):
        ParallaxEvaluator(
            np.asarray([0.0, 1.0]), np.ones(2), np.ones(2),
            ra_deg=0.0, dec_deg=0.0, tref=0.5,
            earth_ephemeris=Ephemeris(np.asarray([0.0, 1.0]), np.zeros((2, 3)), origin="sun"),
        )


def test_native_module_has_no_jax_import():
    from pathlib import Path
    source = Path(__file__).parents[1] / "src" / "jacscanomaly" / "parallax_backend.py"
    assert "import jax" not in source.read_text(encoding="utf-8")


def test_native_continuation_seed_does_not_apply_log_twice():
    time_spec = TimeSpec("hjd", offset=2450000.0)
    earth = Ephemeris(
        np.asarray([2458990.0, 2459000.0, 2459010.0]),
        np.zeros((3, 3)),
        origin="sun",
        time_spec=time_spec,
    )
    fitter = NativePSPLAnnualParallaxFitter(
        0.0, 0.0, 9000.0, time_spec=time_spec, earth_ephemeris=earth
    )
    raw = fitter._raw_seed(np.asarray([9000.0, 80.0, 0.1, 0.2, -0.1]))
    np.testing.assert_allclose(
        raw,
        [2459000.0, np.log(80.0), 0.1, 0.2, -0.1],
        atol=1.0e-14,
    )


def test_native_annual_fit_recovers_synthetic_and_removes_effect_score():
    time = np.linspace(0.0, 20.0, 201)
    ephemeris_time = np.linspace(-5.0, 25.0, 61)
    earth = Ephemeris(
        ephemeris_time,
        np.column_stack(
            [
                np.zeros_like(ephemeris_time),
                np.zeros_like(ephemeris_time),
                0.002 * (ephemeris_time - 10.0) ** 2,
            ]
        ),
        origin="sun",
        time_spec=TimeSpec("hjd"),
    )
    truth_raw = np.asarray([10.0, np.log(4.0), 0.12, 0.25, -0.15])
    geometry = ParallaxEvaluator(
        time,
        np.ones_like(time),
        np.ones_like(time),
        ra_deg=0.0,
        dec_deg=0.0,
        tref=10.0,
        time_spec=TimeSpec("hjd"),
        earth_ephemeris=earth,
    )
    magnification = geometry.magnification(truth_raw)
    flux = 1.7 * magnification + 0.3
    ferr = np.full_like(time, 0.01)
    fitter = NativePSPLAnnualParallaxFitter(
        0.0,
        0.0,
        10.0,
        time_spec=TimeSpec("hjd"),
        earth_ephemeris=earth,
        maxiter=500,
        tol=1.0e-10,
    )
    fit = fitter.fit(time, flux, ferr, [10.2, 3.7, 0.14, 0.1, -0.05])

    np.testing.assert_allclose(
        np.asarray(fit.params),
        [10.0, 4.0, 0.12, 0.25, -0.15],
        atol=1.0e-7,
    )
    assert float(fit.chi2) < 1.0e-12
    assert native_parallax_effect_score(fit) < 1.0e-12


def test_native_fixed_evaluation_preserves_public_seed_coordinates():
    time = np.linspace(0.0, 2.0, 21)
    earth = Ephemeris(
        np.asarray([-1.0, 0.0, 1.0, 2.0, 3.0]),
        np.zeros((5, 3)),
        origin="sun",
        time_spec=TimeSpec("hjd"),
    )
    fitter = NativePSPLAnnualParallaxFitter(
        0.0, 0.0, 1.0, time_spec=TimeSpec("hjd"), earth_ephemeris=earth
    )
    seed = np.asarray([1.0, 0.4, 0.1, 0.2, -0.3])
    geometry = ParallaxEvaluator(
        time, np.ones_like(time), np.ones_like(time),
        ra_deg=0.0, dec_deg=0.0, tref=1.0, time_spec=TimeSpec("hjd"),
        earth_ephemeris=earth,
    )
    magnification = geometry.magnification(
        np.asarray([seed[0], np.log(seed[1]), *seed[2:]])
    )
    flux = 2.0 * magnification + 0.5
    fit = fitter.evaluate_fixed(time, flux, np.full_like(time, 0.01), seed)

    np.testing.assert_allclose(np.asarray(fit.params), seed, atol=1.0e-12)
    assert float(fit.chi2) < 1.0e-20
    assert fit.optimizer_status == "native_cpp_fixed_evaluation"


def test_adaptive_parallax_plot_is_clipped_to_ephemeris_support():
    calls = []
    ephemeris = SimpleNamespace(
        time=np.asarray([100.0, 110.0]),
        extrapolation="reject",
    )

    class Projector:
        time_spec = TimeSpec("jd", offset=100.0)
        earth_ephemeris = ephemeris
        satellite_or_observer_ephemeris = ephemeris
        reference_ephemeris = None

        def magnification_at(self, time, raw_params):
            values = np.asarray(time, dtype=float)
            calls.append(values)
            assert np.min(values) >= 0.0
            assert np.max(values) <= 10.0
            return np.ones_like(values)

    fit = SimpleNamespace(
        param_names=("t0", "tE", "u0", "piEN", "piEE"),
        params=np.asarray([5.0, 2.0, 0.1, 0.0, 0.0]),
        raw_params=np.asarray([105.0, np.log(2.0), 0.1, 0.0, 0.0]),
        parallax_projector=Projector(),
        fs=2.0,
        fb=1.0,
        ferr=np.ones(11),
        flux=np.full(11, 3.0),
        time=np.linspace(0.0, 10.0, 11),
    )

    time, flux = _adaptive_single_lens_curve(fit, (-5.0, 15.0))

    assert calls
    assert time[0] == pytest.approx(0.0)
    assert time[-1] == pytest.approx(10.0)
    np.testing.assert_allclose(flux, 3.0)
