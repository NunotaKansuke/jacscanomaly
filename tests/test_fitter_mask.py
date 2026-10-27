import numpy as np
import pytest

from jacscanomaly import FSPLFitter, PSPLFitter, PSPLParallaxFitter
from jacscanomaly.fitters import _pspl_magnification
from jacscanomaly.parallax_backend import TimeSpec


def _bumped_event(t0=0.3):
    rng = np.random.default_rng(1)
    time = np.linspace(-40.0, 40.0, 1500) + (t0 - 0.3)
    magnification = _pspl_magnification(time, t0, 12.0, 0.15)
    bump = 0.6 * np.exp(-0.5 * ((time - t0 - 3.7) / 0.3) ** 2)
    flux = magnification + bump + 0.2 + rng.normal(0.0, 0.01, time.size)
    ferr = np.full_like(time, 0.01)
    return time, flux, ferr, np.abs(time - t0 - 3.7) < 1.5


@pytest.mark.parametrize("fitter", [PSPLFitter(), FSPLFitter()])
def test_masked_fit_is_the_kept_point_fit_evaluated_on_all_points(fitter):
    time, flux, ferr, mask = _bumped_event()
    kept = ~mask

    masked = fitter.fit(time, flux, ferr, mask=mask)
    subset = fitter.fit(time[kept], flux[kept], ferr[kept])

    assert np.allclose(np.asarray(masked.params), np.asarray(subset.params))
    assert float(masked.chi2) == pytest.approx(float(subset.chi2), rel=1e-6)
    assert float(masked.chi2_dof) == pytest.approx(float(subset.chi2_dof), rel=1e-6)
    assert np.array_equal(masked.mask, mask)
    assert np.asarray(masked.residual).shape == time.shape
    assert float(np.max(np.asarray(masked.residual)[mask])) > 0.4


def test_masked_pspl_fit_recovers_the_baseline():
    time, flux, ferr, mask = _bumped_event()
    fitter = PSPLFitter()

    plain = fitter.fit(time, flux, ferr)
    masked = fitter.fit(time, flux, ferr, mask=mask)

    assert np.asarray(masked.params) == pytest.approx([0.3, 12.0, 0.15], rel=0.03, abs=0.02)
    assert float(masked.chi2_dof) < 1.3 < float(plain.chi2_dof)


def test_all_false_mask_equals_no_mask():
    time, flux, ferr, mask = _bumped_event()
    fitter = PSPLFitter()

    plain = fitter.fit(time, flux, ferr)
    empty = fitter.fit(time, flux, ferr, mask=np.zeros_like(mask))

    assert empty.mask is None
    assert np.allclose(np.asarray(plain.params), np.asarray(empty.params))
    assert float(plain.chi2) == pytest.approx(float(empty.chi2))


def test_evaluate_fixed_mask_profiles_fluxes_on_kept_points():
    time, flux, ferr, mask = _bumped_event()
    fitter = PSPLFitter()
    p0 = np.asarray([0.3, 12.0, 0.15])

    masked = fitter.evaluate_fixed(time, flux, ferr, p0, mask=mask)

    assert float(masked.fs) == pytest.approx(1.0, abs=0.01)
    assert float(masked.fb) == pytest.approx(0.2, abs=0.01)


def test_mask_length_and_kept_points_are_validated():
    time, flux, ferr, mask = _bumped_event()
    fitter = PSPLFitter()

    with pytest.raises(ValueError):
        fitter.fit(time, flux, ferr, mask=mask[:-1])
    with pytest.raises(ValueError):
        fitter.fit(time, flux, ferr, mask=np.arange(time.size) > 3)


def test_parallax_fitter_accepts_mask():
    t0 = 9000.3
    time, flux, ferr, mask = _bumped_event(t0)
    fitter = PSPLParallaxFitter(268.0, -29.0, t0, time_spec=TimeSpec("jd", 2450000.0))

    masked = fitter.fit(time, flux, ferr, mask=mask)

    assert np.array_equal(masked.mask, mask)
    assert float(masked.chi2_dof) < 1.3
    assert float(np.max(np.asarray(masked.residual)[mask])) > 0.5
