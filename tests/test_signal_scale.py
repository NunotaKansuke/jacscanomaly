import numpy as np
import pytest

from jacscanomaly.signal_scale import (
    measure_observed_magnification_fwhm,
    measure_observed_magnification_scale,
)


def test_direct_magnification_fwhm_uses_raw_points_and_local_component():
    time = np.arange(-5.0, 5.001, 0.01)
    sigma = 0.4
    flux = 1.0 + 4.0 * np.exp(-0.5 * (time / sigma) ** 2)
    # The remote excursions are deliberately stronger than the central peak;
    # they must not set the central event's FWHM.
    flux += 8.0 * np.exp(-0.5 * ((time - 3.0) / 0.08) ** 2)
    flux += 8.0 * np.exp(-0.5 * ((time + 3.0) / 0.08) ** 2)

    scale = measure_observed_magnification_fwhm(
        time,
        flux,
        center=0.0,
        search_half_width=5.0,
    )

    expected = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    assert scale.valid
    assert scale.source == "observed_magnification_fwhm"
    assert scale.width == pytest.approx(expected, rel=0.02)
    assert abs(scale.t_center) < 0.01
    assert scale.n_weighted_points >= 3


def test_direct_magnification_fwhm_marks_single_point_peak_unresolved():
    time = np.arange(-5.0, 5.001, 0.01)
    flux = np.ones_like(time)
    flux[time == 0.0] = 10.0

    scale = measure_observed_magnification_fwhm(
        time,
        flux,
        center=0.0,
        search_half_width=5.0,
    )

    assert not scale.valid
    assert scale.censored


def test_magnification_scale_uses_local_raw_run_for_unresolved_peak():
    time = np.arange(-5.0, 5.001, 0.01)
    flux = np.ones_like(time)
    flux[np.abs(time) <= 0.02] = 10.0
    # These broad, weaker excursions are deliberately far from the event.
    flux[np.abs(time + 3.0) <= 0.08] = 1.3
    flux[np.abs(time - 3.0) <= 0.08] = 1.3

    scale = measure_observed_magnification_scale(
        time,
        flux,
        center=0.0,
        search_half_width=5.0,
        n_bins=64,
        allow_point_fallback=True,
    )

    assert scale.valid
    assert scale.t_center == pytest.approx(0.0, abs=0.02)
    assert scale.width < 0.1
    assert scale.t_left > -0.1
    assert scale.t_right < 0.1


def test_magnification_scale_does_not_join_remote_profile_components():
    time = np.linspace(-10.0, 10.0, 4001)
    flux = 1.0 + 0.8 * np.exp(-0.5 * (time / 0.5) ** 2)
    flux += 0.5 * np.exp(-0.5 * ((time - 6.0) / 0.12) ** 2)
    flux += 0.5 * np.exp(-0.5 * ((time + 6.0) / 0.12) ** 2)

    scale = measure_observed_magnification_scale(
        time,
        flux,
        center=0.0,
        search_half_width=10.0,
        n_bins=256,
    )

    assert scale.valid
    assert abs(scale.t_center) < 0.1
    assert 0.5 < scale.width < 4.0
