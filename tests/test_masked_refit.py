import numpy as np

from jacscanomaly import (
    Candidate, FSPLFitter, PSPLFitter, constant_fit, host_detected, masked_refit, window_mask,
)
from jacscanomaly.fitters import _pspl_magnification


def _event(anomaly_t0=1.0):
    rng = np.random.default_rng(5)
    time = np.linspace(-30.0, 30.0, 3000)
    flux = _pspl_magnification(time, 0.0, 20.0, 0.3) + 0.5
    flux = flux + 0.5 * np.exp(-0.5 * ((time - anomaly_t0) / 0.1) ** 2)
    ferr = np.full_like(time, 0.01)
    return time, flux + rng.normal(0.0, 0.01, time.size), ferr


def test_masked_refit_masks_the_window_and_recovers_the_host():
    time, flux, ferr = _event()
    fitter = PSPLFitter()
    fit0 = fitter.fit(time, flux, ferr, [0.0, 20.0, 0.3])
    window = Candidate.from_t0_teff(1.0, 0.15)  # a scan window over the bump

    m = masked_refit(fitter, fit0, window)

    assert m.n_fit == 2
    assert np.array_equal(m.mask, window.support(time))
    t0, tE, u0 = np.asarray(m.fit.params)
    assert abs(t0) < 0.05 and abs(tE - 20.0) < 0.5 and abs(abs(u0) - 0.3) < 0.01
    assert np.isclose(m.criterion, m.chi2 + 2 * m.n_masked + 3 * np.log(m.mask.size))


def test_masked_refit_leaves_a_wrong_basin_the_seed_is_stuck_in():
    # a fit off the event (t0 = 25, u0 >> 1): a refit from its seed stays
    # there; the fitter's own start under the mask finds the host
    time, flux, ferr = _event()
    fitter = PSPLFitter()
    fit0 = fitter.evaluate_fixed(time, flux, ferr, np.array([25.0, 1.0, 5.0]))
    window = Candidate.from_t0_teff(1.0, 0.15)
    stuck = fitter.fit(time, flux, ferr, fitter.seed_from_fit(fit0), mask=window.support(time))
    assert abs(np.asarray(stuck.params)[0]) > 10.0

    m = masked_refit(fitter, fit0, window)

    t0, tE, u0 = np.asarray(m.fit.params)
    assert abs(t0) < 0.05 and abs(tE - 20.0) < 0.5 and abs(abs(u0) - 0.3) < 0.01


def test_nested_start_brings_an_fspl_down_to_the_pspl_chi2():
    # an FSPL fit at rho ~ u0 on point-source data: with the PSPL as the
    # nested start the masked refit fits the kept samples as well as the PSPL
    time, flux, ferr = _event()
    window = Candidate.from_t0_teff(1.0, 0.15)
    pspl = masked_refit(PSPLFitter(), PSPLFitter().fit(time, flux, ferr, [0.0, 20.0, 0.3]), window)
    fitter = FSPLFitter()
    fit0 = fitter.evaluate_fixed(time, flux, ferr, np.array([0.0, 20.0, 0.3, np.log(0.3)]))

    m = masked_refit(fitter, fit0, window, nested=pspl.fit)

    assert np.array_equal(m.mask, pspl.mask)
    assert m.chi2 <= pspl.chi2 + 1.0


def test_masked_refit_keeps_the_mask_at_the_window_over_the_peak():
    # an anomaly on the peak: the host fitted with it is off over the whole
    # peak, but only the window is masked, so u0 stays constrained
    time, flux, ferr = _event(anomaly_t0=0.0)
    fitter = PSPLFitter()
    fit0 = fitter.fit(time, flux, ferr, [0.0, 20.0, 0.3])

    m = masked_refit(fitter, fit0, Candidate(-0.5, 0.5))

    assert m.n_masked == np.count_nonzero(np.abs(time) <= 0.5)
    assert abs(abs(np.asarray(m.fit.params)[2]) - 0.3) < 0.01


def test_masked_refit_does_not_refit_an_already_masked_fit():
    time, flux, ferr = _event()
    fitter = PSPLFitter()
    window = Candidate(0.5, 1.5)
    fit0 = fitter.fit(time, flux, ferr, [0.0, 20.0, 0.3], mask=window_mask(time, window))

    m = masked_refit(fitter, fit0, window)

    assert m.n_fit == 0 and m.fit is fit0


def test_masked_refit_with_an_empty_window_fits_unmasked():
    time, flux, ferr = _event()
    fitter = PSPLFitter()
    fit0 = fitter.fit(time, flux, ferr, [0.0, 20.0, 0.3])

    m = masked_refit(fitter, fit0, Candidate(40.0, 41.0))

    assert m.n_masked == 0 and m.n_fit == 0 and m.fit is fit0


def test_host_detected_when_the_mask_leaves_the_event():
    time, flux, ferr = _event()
    fitter = PSPLFitter()
    fit = fitter.fit(time, flux, ferr, [0.0, 20.0, 0.3], mask=window_mask(time, Candidate(0.5, 1.5)))

    assert host_detected(fit)


def test_host_not_detected_when_the_mask_covers_the_event():
    # only the baseline is kept: a constant explains it as well as any host
    rng = np.random.default_rng(5)
    time = np.linspace(-30.0, 30.0, 3000)
    ferr = np.full_like(time, 0.01)
    flux = _pspl_magnification(time, 0.0, 2.0, 0.3) + 0.5 + rng.normal(0.0, 0.01, time.size)
    fitter = PSPLFitter()
    mask = window_mask(time, Candidate(-15.0, 15.0))
    fit = fitter.fit(time, flux, ferr, [0.0, 2.0, 0.3], mask=mask)

    assert not host_detected(fit)
    flat = constant_fit(time, flux, ferr, mask)
    assert np.allclose(flat.residual, flux - flat.model_flux)
