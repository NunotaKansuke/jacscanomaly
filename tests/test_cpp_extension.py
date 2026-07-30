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
