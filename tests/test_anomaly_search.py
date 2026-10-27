import numpy as np

from jacscanomaly import Finder, FinderConfig, search_anomalies
from jacscanomaly.fitters import _pspl_magnification


def _event(anomalies):
    rng = np.random.default_rng(7)
    time = np.linspace(-30.0, 30.0, 6000)
    flux = _pspl_magnification(time, 0.0, 20.0, 0.3) + 0.5
    for t0, amp in anomalies:
        flux = flux + amp * np.exp(-0.5 * ((time - t0) / 0.1) ** 2)
    ferr = np.full_like(time, 0.01)
    return time, flux + rng.normal(0.0, 0.01, time.size), ferr


def test_two_separate_anomalies_are_both_found():
    time, flux, ferr = _event([(-8.0, 0.3), (6.0, 0.1)])
    finder = Finder(FinderConfig(gap=100.0, grid_backend="cpp"))

    s = search_anomalies(finder, time, flux, ferr, [0.0, 20.0, 0.3], min_score=10.0)

    found = sorted(float(c.t0) for c in s.candidates)
    assert len(found) == 2
    assert abs(found[0] + 8.0) < 0.3 and abs(found[1] - 6.0) < 0.3
    assert s.refit is not None and s.refit.mask[np.argmin(np.abs(time - 6.0))]


def test_no_anomaly_finds_nothing():
    time, flux, ferr = _event([])
    finder = Finder(FinderConfig(gap=100.0, grid_backend="cpp"))

    s = search_anomalies(finder, time, flux, ferr, [0.0, 20.0, 0.3], min_score=10.0)

    assert s.candidates == () and s.refit.n_masked == 0 and len(s.scans) == 1


def test_a_given_first_scan_is_reused():
    time, flux, ferr = _event([(-8.0, 0.3)])
    finder = Finder(FinderConfig(gap=100.0, grid_backend="cpp"))
    scan = finder.run(time, flux, ferr, x0=np.array([0.0, 20.0, 0.3]), refit=False, verbose=False)

    s = search_anomalies(finder, time, flux, ferr, scan.fit.params, min_score=10.0, scan=scan)

    assert s.scans[0] is scan and len(s.candidates) == 1


def test_rescan_holds_the_refit_host_of_a_log_parameter_fitter():
    # FSPL seeds carry ln(rho), its params carry rho: the rescan must start from the seed
    from jacscanomaly import FSPLFitter

    time, flux, ferr = _event([(-8.0, 0.3)])
    finder = Finder(FinderConfig(gap=100.0, grid_backend="cpp"), fitter=FSPLFitter())

    s = search_anomalies(finder, time, flux, ferr, [0.0, 20.0, 0.3, np.log(1e-3)], min_score=10.0)

    assert s.refit is not None
    assert np.allclose(s.scans[1].fit.params, s.refit.fit.params, rtol=1e-6)
