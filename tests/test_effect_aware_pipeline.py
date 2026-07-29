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


def test_native_fallback_preserves_planet_candidate_through_after_scan():
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
    assert result.planet_before is not None and result.planet_before.candidates
    assert result.planet_after is not None and result.planet_after.candidates
    assert any(match.category == "survived" for match in result.candidate_matches)
