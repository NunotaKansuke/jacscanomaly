import numpy as np

from jacscanomaly import CandidateCriteria, Finder, FinderConfig
from jacscanomaly.singlelens_model import A_pspl_func


def test_run_can_scan_with_fixed_x0_without_refitting():
    time = np.linspace(0.0, 20.0, 60)
    x0 = np.array([10.0, 5.0, 0.2])
    magnification = np.asarray(A_pspl_func(x0, time))
    flux = 2.0 * magnification + 0.3
    ferr = np.full_like(time, 0.05)

    finder = Finder(
        FinderConfig(
            grid_backend="jax",
            teff_init=1.0,
            teff_grid_n=1,
            candidate_criteria=CandidateCriteria(min_dchi2=1.0e9),
        )
    )

    result = finder.run(time, flux, ferr, x0=x0, refit=False, verbose=False)

    np.testing.assert_allclose(np.asarray(result.fit.params), x0)
    np.testing.assert_allclose(result.model_flux, flux, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(result.residual, 0.0, atol=1e-10)


def test_run_refit_false_requires_x0():
    time = np.linspace(0.0, 20.0, 20)
    flux = np.ones_like(time)
    ferr = np.ones_like(time)
    finder = Finder(FinderConfig(grid_backend="jax"))

    try:
        finder.run(time, flux, ferr, refit=False, verbose=False)
    except ValueError as exc:
        assert "requires x0" in str(exc)
    else:
        raise AssertionError("Expected ValueError.")
