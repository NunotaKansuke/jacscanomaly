import numpy as np

from jacscanomaly import (
    ContaminationConfig,
    FallbackConfig,
    Finder,
    FinderConfig,
    ParallaxEvaluator,
    RoutingThresholds,
    TimeSpec,
)
from jacscanomaly.parallax_backend import default_earth_ephemeris


def test_native_fallback_preserves_planet_candidate_through_after_scan(
    monkeypatch,
):
    ra_deg = 267.6
    dec_deg = -29.1
    tref = 2459000.0
    time = np.linspace(tref - 40.0, tref + 40.0, 241)
    time_spec = TimeSpec("hjd")
    earth = default_earth_ephemeris(time_spec=time_spec)
    geometry = ParallaxEvaluator(
        time,
        np.ones_like(time),
        np.ones_like(time),
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        tref=tref,
        time_spec=time_spec,
        earth_ephemeris=earth,
    )
    magnification = geometry.magnification(
        np.asarray([tref, np.log(20.0), 0.1, 0.35, -0.2])
    )
    planet = 0.15 * np.exp(-0.5 * ((time - (tref + 3.0)) / 0.55) ** 2)
    flux = 1.5 * magnification + 0.2 + planet
    ferr = np.full_like(time, 0.01)
    finder = Finder(
        FinderConfig(
            fitter_kind="pspl",
            single_fit_backend="cpp",
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            tref=tref,
            parallax_time_scale="hjd",
        )
    )
    fallback_masks = {}
    original_fallback = finder.robust_fallback

    def capture_fallback(*args, **kwargs):
        for name in (
            "known_anomaly_mask",
            "soft_anomaly_mask",
            "selection_exclusion_mask",
        ):
            fallback_masks[name] = np.asarray(kwargs[name], dtype=bool)
        return original_fallback(*args, **kwargs)

    monkeypatch.setattr(finder, "robust_fallback", capture_fallback)

    result = finder.run_effect_aware(
        time,
        flux,
        ferr,
        x0=[tref, 20.0, 0.1],
        routing_thresholds=RoutingThresholds(
            exact_probe_score=1.0,
            fallback_score=4.0,
        ),
        fallback_config=FallbackConfig(
            max_seeds=6,
            contamination=ContaminationConfig(max_iter=6),
        ),
    )

    assert result.fallback_result is not None
    assert result.fallback_result.success
    assert result.fallback_result.model_spec["backend"] == "native_cpp_scipy_trf"
    hard = fallback_masks["known_anomaly_mask"]
    soft = fallback_masks["soft_anomaly_mask"]
    selection = fallback_masks["selection_exclusion_mask"]
    assert np.any(hard)
    assert np.any(soft)
    assert not np.any(hard & soft)
    np.testing.assert_array_equal(selection, hard | soft)
    assert result.planet_before is not None and result.planet_before.candidates
    assert result.planet_after is not None and result.planet_after.candidates
    assert result.planet_after.iterations
    assert result.planet_after.refined_fit.param_names == (
        "t0",
        "tE",
        "u0",
        "piEN",
        "piEE",
    )
    assert all(
        iteration.fit.param_names
        == ("t0", "tE", "u0", "piEN", "piEE")
        for iteration in result.planet_after.iterations
    )
    reconstructed = finder.evaluate_saved_physical_solution(
        time,
        flux,
        ferr,
        effect=result.fallback_result.effect,
        params=result.fallback_result.fit.params,
        fs=float(result.fallback_result.fit.fs),
        fb=float(result.fallback_result.fit.fb),
    )
    assert reconstructed.param_names == (
        "t0",
        "tE",
        "u0",
        "piEN",
        "piEE",
    )
    assert np.allclose(
        reconstructed.model_flux,
        result.fallback_result.fit.model_flux,
        rtol=1.0e-8,
        atol=1.0e-8,
    )
    assert "planet_after_fixed_family_warm_start" in result.reason_codes
    assert any(match.category == "survived" for match in result.candidate_matches)
