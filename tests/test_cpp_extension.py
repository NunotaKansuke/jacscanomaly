from __future__ import annotations

import numpy as np
import pytest


def test_cpp_grid_extension_imports():
    import jacscanomaly._cpp_grid as cpp_grid

    assert hasattr(cpp_grid, "run_grid")
    assert hasattr(cpp_grid, "extract_clusters")


def test_compiled_fspl_magnification_binding_is_available():
    native = pytest.importorskip("jacscanomaly._vbm_cpp")
    from jacscanomaly.parallax_backend import default_espl_table_path

    table_path = default_espl_table_path()
    if table_path is None:
        pytest.skip("The compiled finite-source table is not available.")
    values = np.asarray(
        native.fspl_magnification(
            np.asarray([0.01, 0.1, 1.0, 3.0]),
            0.05,
            espl_table=table_path,
        ),
        dtype=float,
    )

    assert values.shape == (4,)
    assert np.all(np.isfinite(values))
    assert np.all(values >= 1.0)


def test_canonical_pspl_fitter_uses_scipy_lm():
    from jacscanomaly import PSPLFitter

    time = np.linspace(-15.0, 15.0, 121)
    truth = np.asarray([1.5, 4.0, 0.2])
    u = np.sqrt(((time - truth[0]) / truth[1]) ** 2 + truth[2] ** 2)
    magnification = (u * u + 2.0) / (u * np.sqrt(u * u + 4.0))
    flux = 1.7 * magnification + 0.2
    fit = PSPLFitter(maxiter=500, tol=1.0e-10).fit(
        time, flux, np.full_like(time, 0.01), [1.0, 3.5, 0.25]
    )

    assert fit.model_kind == "pspl"
    assert fit.optimizer_success
    assert fit.optimizer_status.startswith("scipy_lm:")
    np.testing.assert_allclose(np.asarray(fit.params), truth, atol=1.0e-7)
    assert float(fit.chi2) < 1.0e-12


def test_canonical_fspl_fitter_uses_compiled_magnification():
    from jacscanomaly import FSPLFitter

    time = np.linspace(90.0, 110.0, 201)
    truth = np.asarray([100.0, 4.0, 0.15, 0.08])
    fitter = FSPLFitter(maxiter=500, tol=1.0e-10)
    magnification = fitter._magnification(
        time,
        np.asarray([truth[0], truth[1], truth[2], np.log(truth[3])]),
    )
    flux = 1.7 * magnification + 0.2
    fit = fitter.fit(
        time, flux, np.full_like(time, 0.01), [100.2, 3.5, 0.18, np.log(0.1)]
    )

    assert fit.model_kind == "fspl"
    assert fit.optimizer_success
    assert "compiled_magnification" in fit.optimizer_status
    np.testing.assert_allclose(np.asarray(fit.params), truth, atol=1.0e-6)
    assert float(fit.chi2) < 1.0e-10
