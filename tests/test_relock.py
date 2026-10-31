import numpy as np

from jacscanomaly import PSPLFitter, relock_host
from jacscanomaly.fitters import _pspl_magnification
from jacscanomaly.relock import _peak_unmasked


def _event(spike: bool):
    rng = np.random.default_rng(3)
    time = np.linspace(-30.0, 30.0, 3000)
    flux = _pspl_magnification(time, 0.0, 20.0, 0.3) + 0.5
    if spike:
        flux = flux + 2.0 * (_pspl_magnification(time, 1.0, 0.02, 0.05) - 1.0)
    ferr = np.full_like(time, 0.01)
    return time, flux + rng.normal(0.0, 0.01, time.size), ferr


def test_fit_on_the_spike_moves_to_the_host_event():
    time, flux, ferr = _event(spike=True)
    fitter = PSPLFitter()
    fit0 = fitter.evaluate_fixed(time, flux, ferr, [1.0, 0.02, 0.05])

    r = relock_host(fitter, fit0)

    assert r.relocked and r.fit is r.host and r.dJ >= 0
    t0, tE, u0 = np.asarray(r.fit.params)
    assert abs(t0) < 0.2 and abs(tE - 20.0) < 1.0 and abs(abs(u0) - 0.3) < 0.02
    assert np.all(np.abs(time[r.mask] - 1.0) < 0.5)


def test_fit_on_the_host_event_is_kept():
    time, flux, ferr = _event(spike=False)
    fitter = PSPLFitter()
    fit0 = fitter.evaluate_fixed(time, flux, ferr, [0.0, 20.0, 0.3])

    r = relock_host(fitter, fit0)

    assert not r.relocked and r.fit is fit0 and r.dJ < 0


def test_fit_on_the_spike_is_left_when_both_hypotheses_reach_the_host():
    # weak host: H0 slides from the spike to the host, so J0 = J1 up to round-off
    rng = np.random.default_rng(3)
    time = np.linspace(-30.0, 30.0, 3000)
    flux = (_pspl_magnification(time, 0.0, 20.0, 3.0) + 0.5
            + 2.0 * (_pspl_magnification(time, 1.0, 0.02, 0.05) - 1.0))
    ferr = np.full_like(time, 0.003)
    fitter = PSPLFitter()
    fit0 = fitter.evaluate_fixed(time, flux + rng.normal(0.0, 0.003, time.size), ferr,
                                 [1.0, 0.004, 8.6])

    r = relock_host(fitter, fit0)

    assert r.relocked and r.fit is r.host
    assert abs(float(np.asarray(r.fit.params)[0])) < 1.0


def test_host_peak_freed_by_the_mask_is_rejected():
    # two seasons; the host peaks in the gap, just past the first season
    time = np.r_[np.linspace(-40.0, -5.0, 500), np.linspace(100.0, 140.0, 500)]
    flux = _pspl_magnification(time, 0.0, 20.0, 0.3) + 0.5
    ferr = np.full_like(time, 0.01)
    fit = PSPLFitter().evaluate_fixed(time, flux, ferr, [0.0, 20.0, 0.3])
    # the data next to the peak are kept: they constrain it
    assert _peak_unmasked(fit, np.zeros(time.size, bool))
    # masked up to the season's end: only the mask lets the peak sit there
    assert not _peak_unmasked(fit, (time > -25.0) & (time < 0.0))
    # a mask away from the peak does not matter
    assert _peak_unmasked(fit, (time > -35.0) & (time < -30.0))
    # a sampled peak with the anomaly masked on it
    on = PSPLFitter().evaluate_fixed(time, flux, ferr, [-20.0, 20.0, 0.3])
    assert _peak_unmasked(on, (time > -21.0) & (time < -19.0))
