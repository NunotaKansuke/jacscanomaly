from __future__ import annotations


def test_cpp_grid_extension_imports():
    import jacscanomaly._cpp_grid as cpp_grid

    assert hasattr(cpp_grid, "run_grid")
    assert hasattr(cpp_grid, "fit_pspl")
    assert hasattr(cpp_grid, "extract_clusters")


def test_cpp_pspl_fitter_rejects_native_sentinel(monkeypatch):
    """A rejected C++ trial must not be exposed as a successful PSPL fit."""
    import numpy as np
    import jax.numpy as jnp
    import pytest
    from jacscanomaly import singlelens_fit

    class SentinelBackend:
        @staticmethod
        def fit_pspl(time, flux, ferr, p0, **_kwargs):
            n = len(time)
            return (
                np.asarray([1000.0, 10.0, 0.1]),
                0.0,
                0.0,
                1e200 * n,
                np.zeros(n),
                np.ones(n),
            )

    monkeypatch.setattr(singlelens_fit, "_cpp_grid", SentinelBackend())
    fitter = singlelens_fit.CPPPSPLFitter()
    with pytest.raises(RuntimeError, match="native residual sentinel"):
        fitter.fit(
            jnp.asarray([0.0, 1.0, 2.0, 3.0]),
            jnp.asarray([1.0, 1.0, 1.0, 1.0]),
            jnp.asarray([0.1, 0.1, 0.1, 0.1]),
            jnp.asarray([1.0, 1.0, 0.1]),
        )


def test_cpp_pspl_fitter_retries_nonnegative_only_for_flux_cancellation(
    monkeypatch,
):
    import numpy as np
    from jacscanomaly import singlelens_fit

    calls = []

    class CancellationBackend:
        @staticmethod
        def fit_pspl(time, flux, ferr, p0, **kwargs):
            constrained = bool(kwargs["nonnegative_fluxes"])
            calls.append(constrained)
            fs, fb = (1.0, 0.0) if constrained else (100.0, -99.0)
            model = np.full(len(time), fs + fb)
            residual = np.asarray(flux) - model
            return np.asarray(p0), fs, fb, 0.0, model, residual

    monkeypatch.setattr(singlelens_fit, "_cpp_grid", CancellationBackend())
    fitter = singlelens_fit.CPPPSPLFitter(
        nonnegative_on_cancellation=True,
        max_flux_cancellation_ratio=50.0,
    )
    fit = fitter.fit(
        np.arange(4.0),
        np.ones(4),
        np.ones(4),
        np.asarray([1.5, 1.0, 0.1]),
    )

    assert calls == [False, True]
    assert fit.fs == 1.0
    assert fit.fb == 0.0
    assert "nonnegative_flux_fallback" in fit.optimizer_status


def test_cpp_pspl_can_constrain_linear_fluxes_nonnegative():
    import numpy as np
    from jacscanomaly import _cpp_grid
    from jacscanomaly.singlelens_model import A_pspl_func

    time = np.linspace(-20.0, 20.0, 101)
    params = np.asarray([0.0, 4.0, 0.2])
    magnification = np.asarray(A_pspl_func(params, time))
    flux = 2.0 * magnification - 1.5
    ferr = np.full(time.shape, 0.01)

    unconstrained = _cpp_grid.fit_pspl(
        time, flux, ferr, params, maxiter=0
    )
    constrained = _cpp_grid.fit_pspl(
        time,
        flux,
        ferr,
        params,
        maxiter=0,
        nonnegative_fluxes=True,
    )

    assert unconstrained[2] < 0.0
    assert constrained[1] >= 0.0
    assert constrained[2] >= 0.0
