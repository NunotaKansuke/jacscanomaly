import numpy as np
import pytest

from jacscanomaly import FSPLFitter, PSPLFitter
from jacscanomaly.fitters import _pspl_magnification


def _point_source_event():
    rng = np.random.default_rng(3)
    time = np.linspace(-30.0, 30.0, 3000)
    flux = 5.0 * _pspl_magnification(time, 0.0, 8.3, 0.13) + 20.0
    ferr = 0.02 * np.sqrt(flux)
    return time, flux + rng.normal(0.0, 1.0, time.size) * ferr, ferr


def test_fspl_starts_from_the_pspl_it_nests():
    time, flux, ferr = _point_source_event()
    pspl = PSPLFitter().fit(time, flux, ferr)
    fitter = FSPLFitter()

    seeds = fitter._automatic_initial_guesses(time, flux, ferr, pspl_params=pspl.params)
    fspl = fitter.fit(time, flux, ferr, pspl_params=pspl.params)

    assert np.allclose(seeds[-1], fitter.nested_seed(pspl.params))
    assert float(fspl.chi2) <= float(pspl.chi2) + 1e-3
