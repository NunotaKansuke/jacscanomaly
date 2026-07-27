import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("numpyro")
pytest.importorskip("microjax.fastlens")

import jax.numpy as jnp

from jacscanomaly.hmc import sample_fspl_parallax_hmc
from jacscanomaly.singlelens_fit import SingleLensFitResult
from jacscanomaly.singlelens_model import A_fspl_parallax_logrho_func
from jacscanomaly.trajectory import make_parallax_projector


def _optimized_fspl_parallax_fit():
    time = np.linspace(5999.5, 6000.5, 8)
    raw = jnp.array([6000.0, 10.0, 0.1, -4.0, 0.01, -0.01])
    projector = make_parallax_projector(268.1715, -29.279525, 6000.0)
    magnification = np.asarray(
        A_fspl_parallax_logrho_func(raw, jnp.asarray(time), projector)
    )
    flux = 2.0 * magnification + 0.1
    return SingleLensFitResult(
        time=time,
        flux=flux,
        ferr=np.full(time.shape, 0.1),
        params=jnp.array([6000.0, 10.0, 0.1, np.exp(-4.0), 0.01, -0.01]),
        param_names=("t0", "tE", "u0", "rho", "piEN", "piEE"),
        chi2=jnp.array(0.0),
        chi2_dof=jnp.array(0.0),
        fs=jnp.array(2.0),
        fb=jnp.array(0.1),
        model_flux=jnp.asarray(flux),
        residual=jnp.zeros_like(jnp.asarray(flux)),
        raw_params=raw,
        parallax_projector=projector,
    )


def test_fspl_parallax_hmc_samples_physical_parameters():
    result = sample_fspl_parallax_hmc(
        _optimized_fspl_parallax_fit(),
        rng_seed=1,
        num_warmup=1,
        num_samples=2,
        t0_bounds=(5999.0, 6001.0),
    )

    assert set(("t0", "tE", "u0", "rho", "piEN", "piEE", "fs", "fb")) <= set(result.samples)
    assert result.samples["t0"].shape == (2,)
    assert np.all(result.samples["tE"] > 0.0)
    assert np.all(result.samples["rho"] > 0.0)
    assert result.fspl_peak_window_days == 10.0
    assert result.n_fspl_points == 8


def test_fspl_parallax_hmc_limits_microjax_to_peak_window():
    result = sample_fspl_parallax_hmc(
        _optimized_fspl_parallax_fit(),
        rng_seed=2,
        num_warmup=0,
        num_samples=1,
        t0_bounds=(5999.0, 6001.0),
        peak_window_days=0.1,
    )
    assert result.fspl_peak_window_days == 0.1
    assert result.n_fspl_points == 2


def test_fspl_parallax_hmc_rejects_wrong_fit_type():
    fit = _optimized_fspl_parallax_fit()
    wrong_fit = SingleLensFitResult(
        **{**fit.__dict__, "param_names": ("t0", "tE", "u0")}
    )
    with pytest.raises(ValueError, match="FSPLParallaxFitter"):
        sample_fspl_parallax_hmc(wrong_fit, num_warmup=0, num_samples=1)
