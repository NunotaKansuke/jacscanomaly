import numpy as np

from jacscanomaly import PSPLFitter, relock_host
from jacscanomaly.fitters import _pspl_magnification


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

    assert r.relocked and r.fit is r.host
    t0, tE, u0 = np.asarray(r.fit.params)
    assert abs(t0) < 0.2 and abs(tE - 20.0) < 1.0 and abs(abs(u0) - 0.3) < 0.02
    assert np.all(np.abs(time[r.mask] - 1.0) < 0.5)


def test_fit_on_the_host_event_is_kept():
    time, flux, ferr = _event(spike=False)
    fitter = PSPLFitter()
    fit0 = fitter.evaluate_fixed(time, flux, ferr, [0.0, 20.0, 0.3])

    r = relock_host(fitter, fit0)

    assert not r.relocked and r.fit is fit0
