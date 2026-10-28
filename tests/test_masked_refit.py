import numpy as np

from jacscanomaly import (
    Candidate, FSPLFitter, PSPLFitter, constant_fit, host_detected, masked_refit, window_mask,
    window_refit,
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

    # the bump is masked, the window's flat ends are not
    assert m.mask[np.abs(time - 1.0) < 0.15].all()
    assert 0 < m.n_masked < np.count_nonzero(window.support(time))
    assert len(m.candidates) == 1 and m.windows == (window,)
    t0, tE, u0 = np.asarray(m.fit.params)
    assert abs(t0) < 0.05 and abs(tE - 20.0) < 0.5 and abs(abs(u0) - 0.3) < 0.01
    ln_n = np.log(m.mask.size)
    assert np.isclose(m.criterion, m.chi2 + 2 * m.n_masked + 2 * ln_n + 3 * ln_n)


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

    # J = criterion - k ln n: no worse than the PSPL's
    ln_n = np.log(m.mask.size)
    assert m.criterion - 4 * ln_n <= pspl.criterion - 3 * ln_n + 1.0


def test_masked_refit_masks_the_anomaly_not_the_host_misfit_over_the_peak():
    # an anomaly on the peak: the host fitted with it is off over the whole
    # peak; the refit host fits the peak, so only the bump stays masked
    time, flux, ferr = _event(anomaly_t0=0.0)
    fitter = PSPLFitter()
    fit0 = fitter.fit(time, flux, ferr, [0.0, 20.0, 0.3])

    m = masked_refit(fitter, fit0, Candidate(-0.5, 0.5))

    assert m.mask[np.abs(time) < 0.15].all() and not m.mask[np.abs(time) > 0.6].any()
    assert abs(abs(np.asarray(m.fit.params)[2]) - 0.3) < 0.01


def test_a_broad_window_without_an_anomaly_masks_nothing():
    # a weak, broad candidate over the host peak: masking it would only free
    # the peak from the fit; no segment pays its price
    time, flux, ferr = _event()
    flux = flux - 0.5 * np.exp(-0.5 * ((time - 1.0) / 0.1) ** 2)  # remove the bump
    fitter = PSPLFitter()
    fit0 = fitter.fit(time, flux, ferr, [0.0, 20.0, 0.3])

    m = masked_refit(fitter, fit0, Candidate(-8.0, 8.0))

    assert m.n_masked == 0 and m.candidates == ()
    assert abs(abs(np.asarray(m.fit.params)[2]) - 0.3) < 0.01


def test_masked_refit_of_its_own_result_keeps_the_mask():
    time, flux, ferr = _event()
    fitter = PSPLFitter()
    window = Candidate(0.5, 1.5)
    m = masked_refit(fitter, fitter.fit(time, flux, ferr, [0.0, 20.0, 0.3]), window)

    again = masked_refit(fitter, m.fit, window)

    assert np.array_equal(again.mask, m.mask) and again.criterion <= m.criterion + 1e-6


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


def test_window_refit_masks_the_whole_window_and_starts_from_the_nested_pspl():
    time, flux, ferr = _event()
    window = Candidate(0.5, 1.5)
    pspl = window_refit(PSPLFitter(), PSPLFitter().fit(time, flux, ferr, [0.0, 20.0, 0.3]), window)
    fitter = FSPLFitter()
    fit0 = fitter.evaluate_fixed(time, flux, ferr, np.array([0.0, 20.0, 0.3, np.log(0.3)]))

    m = window_refit(fitter, fit0, window, nested=pspl.fit)

    assert np.array_equal(pspl.mask, window.support(time)) and np.array_equal(m.mask, pspl.mask)
    ln_n = np.log(m.mask.size)
    assert np.isclose(pspl.criterion, pspl.chi2 + 2 * pspl.n_masked + 2 * ln_n + 3 * ln_n)
    assert m.chi2 <= pspl.chi2 + 1.0
