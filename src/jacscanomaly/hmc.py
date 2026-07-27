"""Hamiltonian Monte Carlo for optimized FSPL annual-parallax fits.

The sampler deliberately reuses :class:`~jacscanomaly.singlelens_fit.SingleLensFitResult`
instead of reconstructing a trajectory from sky coordinates.  Consequently the
posterior uses exactly the same microjax finite-source magnification and
parallax projector as the preceding deterministic fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Optional, Sequence, Union

import numpy as np
import jax.numpy as jnp

from .singlelens_fit import SingleLensFitResult
from .photometry import solve_fs_fb
from .singlelens_model import (
    A_pspl_func,
    A_fspl_logrho_func,
    A_fspl_logrho_peak_func,
    A_fspl_parallax_logrho_func,
    A_fspl_parallax_logrho_peak_func,
)


@dataclass(frozen=True)
class FSPLParallaxHMCResult:
    """Posterior draws produced by :func:`sample_fspl_parallax_hmc`.

    ``samples`` contains physical ``tE`` and ``rho`` values as well as the
    sampled coordinates ``log_tE`` and ``logrho``.  Arrays have shape
    ``(num_chains * num_samples,)`` unless ``group_by_chain=True`` was
    requested from NumPyro internally (this API always returns flattened
    chains).
    """

    samples: Mapping[str, np.ndarray]
    chain_samples: Mapping[str, np.ndarray]
    diagnostics: Mapping[str, np.ndarray]
    optimized_fit: SingleLensFitResult
    num_warmup: int
    num_samples: int
    num_chains: int
    fspl_peak_window_days: Optional[float]
    n_fspl_points: int

    def median(self, name: str) -> float:
        """Return the posterior median for a sampled or deterministic site."""
        return float(np.median(self.samples[name]))


@dataclass(frozen=True)
class FSPLHMCResult:
    """Posterior draws for an FSPL (without parallax) HMC run."""

    samples: Mapping[str, np.ndarray]
    chain_samples: Mapping[str, np.ndarray]
    diagnostics: Mapping[str, np.ndarray]
    optimized_fit: SingleLensFitResult
    num_warmup: int
    num_samples: int
    num_chains: int
    fspl_peak_window_days: Optional[float]
    n_fspl_points: int

    def median(self, name: str) -> float:
        return float(np.median(self.samples[name]))


@dataclass(frozen=True)
class PSPLHMCResult:
    """Posterior draws for a point-source point-lens (PSPL) HMC run."""

    samples: Mapping[str, np.ndarray]
    chain_samples: Mapping[str, np.ndarray]
    diagnostics: Mapping[str, np.ndarray]
    optimized_fit: SingleLensFitResult
    num_warmup: int
    num_samples: int
    num_chains: int

    def median(self, name: str) -> float:
        return float(np.median(self.samples[name]))


def _require_numpyro():
    try:
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS, init_to_value
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "FSPL+parallax HMC requires the optional dependency numpyro. "
            "Install it with `pip install jacscanomaly[hmc]`."
        ) from exc
    return numpyro, dist, MCMC, NUTS, init_to_value


def _raw_parameters(fit: SingleLensFitResult) -> np.ndarray:
    """Get the optimization coordinates ``(t0, tE, u0, logrho, piEN, piEE)``."""
    if fit.parallax_projector is None:
        raise ValueError(
            "fit must be an FSPL+annual-parallax result; its parallax projector is missing."
        )
    if tuple(fit.param_names) != ("t0", "tE", "u0", "rho", "piEN", "piEE"):
        raise ValueError(
            "fit must come from FSPLParallaxFitter "
            "(expected t0, tE, u0, rho, piEN, piEE parameters)."
        )
    if fit.raw_params is not None:
        raw = np.asarray(fit.raw_params, dtype=float)
    else:
        params = np.asarray(fit.params, dtype=float)
        raw = np.array([params[0], params[1], params[2], np.log(params[3]), params[4], params[5]])
    if raw.shape != (6,) or not np.all(np.isfinite(raw)) or raw[1] <= 0.0:
        raise ValueError("fit contains invalid FSPL+parallax parameters.")
    return raw


def _raw_fspl_parameters(fit: SingleLensFitResult) -> np.ndarray:
    if tuple(fit.param_names) != ("t0", "tE", "u0", "rho"):
        raise ValueError("fit must be an FSPL result (t0, tE, u0, rho).")
    if fit.raw_params is not None:
        raw = np.asarray(fit.raw_params, dtype=float)
    else:
        params = np.asarray(fit.params, dtype=float)
        raw = np.array([params[0], params[1], params[2], np.log(params[3])])
    if raw.shape != (4,) or not np.all(np.isfinite(raw)) or raw[1] <= 0.0:
        raise ValueError("fit contains invalid FSPL parameters.")
    return raw


def _raw_pspl_parameters(fit: SingleLensFitResult) -> np.ndarray:
    if tuple(fit.param_names) != ("t0", "tE", "u0"):
        raise ValueError("fit must be a PSPL result (t0, tE, u0).")
    raw = np.asarray(fit.params, dtype=float)
    if raw.shape != (3,) or not np.all(np.isfinite(raw)) or raw[1] <= 0.0:
        raise ValueError("fit contains invalid PSPL parameters.")
    return raw


def sample_pspl_hmc(
    fit: SingleLensFitResult,
    *,
    rng_seed: int = 0,
    num_warmup: int = 500,
    num_samples: int = 1_000,
    num_chains: int = 1,
    target_accept_prob: float = 0.9,
    max_tree_depth: int = 10,
    t0_bounds: Optional[Sequence[float]] = None,
    tE_bounds: Sequence[float] = (1.0e-4, 1.0e6),
    u0_bounds: Sequence[float] = (-10.0, 10.0),
    dense_mass: bool = True,
    profile_linear_flux: bool = True,
    progress_bar: bool = False,
) -> PSPLHMCResult:
    """Sample a PSPL posterior with NumPyro NUTS, starting at ``fit``.

    By default the source and blend fluxes are solved by weighted linear least
    squares at every likelihood evaluation, leaving only ``t0``, ``log(tE)``,
    and ``u0`` as NUTS coordinates. This matches the deterministic fitter and
    avoids spending HMC trajectories on linear nuisance parameters. Set
    ``profile_linear_flux=False`` to sample ``fs`` and ``fb`` explicitly.
    NUTS adapts a full mass matrix during warmup by default.
    """
    if num_warmup < 0 or num_samples < 1 or num_chains < 1:
        raise ValueError("num_warmup must be non-negative; num_samples and num_chains must be positive.")
    if not 0.0 < target_accept_prob < 1.0 or max_tree_depth < 1:
        raise ValueError("target_accept_prob must lie in (0, 1) and max_tree_depth must be positive.")
    raw = _raw_pspl_parameters(fit)
    time, flux = jnp.asarray(fit.time), jnp.asarray(fit.flux)
    ferr = jnp.maximum(jnp.asarray(fit.ferr), 1.0e-12)

    def _bounds(name: str, values: Sequence[float]) -> tuple[float, float]:
        if len(values) != 2 or not np.all(np.isfinite(values)) or values[0] >= values[1]:
            raise ValueError(f"{name} must be two finite values in ascending order.")
        return float(values[0]), float(values[1])

    t0_lo, t0_hi = _bounds("t0_bounds", t0_bounds) if t0_bounds is not None else (float(np.min(fit.time)), float(np.max(fit.time)))
    tE_lo, tE_hi = _bounds("tE_bounds", tE_bounds)
    u0_lo, u0_hi = _bounds("u0_bounds", u0_bounds)
    if tE_lo <= 0.0:
        raise ValueError("tE_bounds must be strictly positive.")
    initial = {"t0": raw[0], "log_tE": np.log(raw[1]), "u0": raw[2], "fs": float(fit.fs), "fb": float(fit.fb)}
    bounded = {"t0": (t0_lo, t0_hi), "log_tE": (np.log(tE_lo), np.log(tE_hi)), "u0": (u0_lo, u0_hi)}
    for name, (lower, upper) in bounded.items():
        if not lower < initial[name] < upper:
            raise ValueError(f"optimized {name} lies outside its HMC prior.")
    flux_scale = max(float(np.std(np.asarray(fit.flux))), float(np.median(np.asarray(fit.ferr))), 1.0e-8)
    numpyro, dist, MCMC, NUTS, init_to_value = _require_numpyro()

    def model():
        t0 = numpyro.sample("t0", dist.Uniform(t0_lo, t0_hi))
        log_tE = numpyro.sample("log_tE", dist.Uniform(np.log(tE_lo), np.log(tE_hi)))
        u0 = numpyro.sample("u0", dist.Uniform(u0_lo, u0_hi))
        tE = jnp.exp(log_tE)
        magnification = A_pspl_func(jnp.array([t0, tE, u0]), time)
        if profile_linear_flux:
            fs, fb = solve_fs_fb(magnification, flux, ferr)
            numpyro.deterministic("fs", fs)
            numpyro.deterministic("fb", fb)
        else:
            fs = numpyro.sample("fs", dist.Normal(float(fit.fs), 100.0 * flux_scale))
            fb = numpyro.sample("fb", dist.Normal(float(fit.fb), 100.0 * flux_scale))
        numpyro.deterministic("tE", tE)
        numpyro.sample("flux", dist.Normal(fs * magnification + fb, ferr), obs=flux)

    init_values = initial if not profile_linear_flux else {key: initial[key] for key in ("t0", "log_tE", "u0")}
    kernel = NUTS(
        model,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        dense_mass=dense_mass,
        init_strategy=init_to_value(values=init_values),
    )
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, progress_bar=progress_bar)
    import jax
    mcmc.run(jax.random.PRNGKey(rng_seed))
    samples = {name: np.asarray(value) for name, value in mcmc.get_samples().items()}
    chain_samples = {name: np.asarray(value) for name, value in mcmc.get_samples(group_by_chain=True).items()}
    diagnostics = {name: np.asarray(value) for name, value in mcmc.get_extra_fields().items()}
    return PSPLHMCResult(samples, chain_samples, diagnostics, fit, num_warmup, num_samples, num_chains)


def sample_fspl_hmc(
    fit: SingleLensFitResult,
    *,
    rng_seed: int = 0,
    num_warmup: int = 500,
    num_samples: int = 1_000,
    num_chains: int = 1,
    target_accept_prob: float = 0.8,
    max_tree_depth: int = 10,
    t0_bounds: Optional[Sequence[float]] = None,
    tE_bounds: Sequence[float] = (1.0e-4, 1.0e6),
    u0_bounds: Sequence[float] = (-10.0, 10.0),
    logrho_bounds: Sequence[float] = (-20.0, 3.0),
    peak_window_days: Union[float, None, Literal["auto"]] = "auto",
    fspl_n_fft: int = 1024,
    progress_bar: bool = False,
) -> FSPLHMCResult:
    """Sample an optimized FSPL posterior with NumPyro NUTS.

    This is the parallax-free counterpart of
    :func:`sample_fspl_parallax_hmc`.  ``peak_window_days="auto"`` evaluates
    microjax FSPL within ``max(10, 5*rho*tE)`` days of the optimized peak and
    uses PSPL elsewhere while retaining all observations in the likelihood.
    """
    if num_warmup < 0 or num_samples < 1 or num_chains < 1:
        raise ValueError("num_warmup must be non-negative; num_samples and num_chains must be positive.")
    if not 0.0 < target_accept_prob < 1.0 or max_tree_depth < 1:
        raise ValueError("target_accept_prob must lie in (0, 1) and max_tree_depth must be positive.")
    if peak_window_days != "auto" and peak_window_days is not None and peak_window_days <= 0.0:
        raise ValueError("peak_window_days must be positive, None, or 'auto'.")

    raw = _raw_fspl_parameters(fit)
    time = jnp.asarray(fit.time)
    flux = jnp.asarray(fit.flux)
    ferr = jnp.maximum(jnp.asarray(fit.ferr), 1.0e-12)
    if time.ndim != 1 or time.size < 5:
        raise ValueError("FSPL HMC requires at least five one-dimensional observations.")

    def _bounds(name: str, values: Sequence[float]) -> tuple[float, float]:
        if len(values) != 2 or not np.all(np.isfinite(values)) or values[0] >= values[1]:
            raise ValueError(f"{name} must be two finite values in ascending order.")
        return float(values[0]), float(values[1])

    t0_lo, t0_hi = _bounds("t0_bounds", t0_bounds) if t0_bounds is not None else (float(np.min(fit.time)), float(np.max(fit.time)))
    tE_lo, tE_hi = _bounds("tE_bounds", tE_bounds)
    u0_lo, u0_hi = _bounds("u0_bounds", u0_bounds)
    logrho_lo, logrho_hi = _bounds("logrho_bounds", logrho_bounds)
    if tE_lo <= 0.0:
        raise ValueError("tE_bounds must be strictly positive.")
    if peak_window_days == "auto":
        peak_window_days = max(10.0, 5.0 * raw[1] * np.exp(raw[3]))
    if peak_window_days is None:
        peak_indices = None
        n_fspl_points = int(time.size)
    else:
        peak_indices = jnp.asarray(np.flatnonzero(np.abs(np.asarray(fit.time) - raw[0]) <= peak_window_days), dtype=jnp.int32)
        if peak_indices.size == 0:
            raise ValueError("peak_window_days selects no observations.")
        n_fspl_points = int(peak_indices.size)

    initial = {"t0": raw[0], "log_tE": np.log(raw[1]), "u0": raw[2], "logrho": raw[3], "fs": float(fit.fs), "fb": float(fit.fb)}
    bounded = {"t0": (t0_lo, t0_hi), "log_tE": (np.log(tE_lo), np.log(tE_hi)), "u0": (u0_lo, u0_hi), "logrho": (logrho_lo, logrho_hi)}
    for name, (lower, upper) in bounded.items():
        if not lower < initial[name] < upper:
            raise ValueError(f"optimized {name} lies outside its HMC prior.")
    flux_scale = max(float(np.std(np.asarray(fit.flux))), float(np.median(np.asarray(fit.ferr))), 1.0e-8)
    numpyro, dist, MCMC, NUTS, init_to_value = _require_numpyro()

    def model():
        t0 = numpyro.sample("t0", dist.Uniform(t0_lo, t0_hi))
        log_tE = numpyro.sample("log_tE", dist.Uniform(np.log(tE_lo), np.log(tE_hi)))
        u0 = numpyro.sample("u0", dist.Uniform(u0_lo, u0_hi))
        logrho = numpyro.sample("logrho", dist.Uniform(logrho_lo, logrho_hi))
        fs = numpyro.sample("fs", dist.Normal(float(fit.fs), 100.0 * flux_scale))
        fb = numpyro.sample("fb", dist.Normal(float(fit.fb), 100.0 * flux_scale))
        tE = jnp.exp(log_tE)
        rho = jnp.exp(logrho)
        q = jnp.array([t0, tE, u0, logrho])
        magnification = A_fspl_logrho_func(q, time) if peak_indices is None else A_fspl_logrho_peak_func(q, time, peak_indices, N_fft=fspl_n_fft)
        numpyro.deterministic("tE", tE)
        numpyro.deterministic("rho", rho)
        numpyro.sample("flux", dist.Normal(fs * magnification + fb, ferr), obs=flux)

    kernel = NUTS(model, target_accept_prob=target_accept_prob, max_tree_depth=max_tree_depth, init_strategy=init_to_value(values=initial))
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, progress_bar=progress_bar)
    import jax
    mcmc.run(jax.random.PRNGKey(rng_seed))
    samples = {name: np.asarray(value) for name, value in mcmc.get_samples().items()}
    chain_samples = {name: np.asarray(value) for name, value in mcmc.get_samples(group_by_chain=True).items()}
    diagnostics = {name: np.asarray(value) for name, value in mcmc.get_extra_fields().items()}
    return FSPLHMCResult(samples, chain_samples, diagnostics, fit, num_warmup, num_samples, num_chains, peak_window_days, n_fspl_points)


def sample_fspl_parallax_hmc(
    fit: SingleLensFitResult,
    *,
    rng_seed: int = 0,
    num_warmup: int = 500,
    num_samples: int = 1_000,
    num_chains: int = 1,
    target_accept_prob: float = 0.8,
    max_tree_depth: int = 10,
    t0_bounds: Optional[Sequence[float]] = None,
    tE_bounds: Sequence[float] = (1.0e-4, 1.0e6),
    u0_bounds: Sequence[float] = (-10.0, 10.0),
    logrho_bounds: Sequence[float] = (-20.0, 3.0),
    piE_max: float = 10.0,
    peak_window_days: Union[float, None, Literal["auto"]] = "auto",
    fspl_n_fft: int = 1024,
    progress_bar: bool = False,
) -> FSPLParallaxHMCResult:
    """Sample the FSPL + annual-parallax posterior with NumPyro NUTS.

    Pass the result of :class:`FSPLParallaxFitter` directly.  Its optimized
    nonlinear and linear-flux parameters initialize NUTS, while its cached
    parallax projector and microjax magnification define the likelihood.

    Priors are uniform in ``t0``, ``log(tE)``, ``u0``, ``logrho``, and the two
    parallax components.  The source and blend fluxes have deliberately broad
    normal priors centred on their optimized values. The defaults intentionally
    use broad bounds. Set scientifically motivated bounds for production
    inference. By default, microjax FSPL magnification is evaluated within
    ``max(10 days, 5 * rho * tE)`` of the optimized peak; all other observations
    remain in the likelihood with PSPL magnification. Pass
    ``peak_window_days=None`` only for full-curve FSPL.
    """
    if num_warmup < 0 or num_samples < 1 or num_chains < 1:
        raise ValueError("num_warmup must be non-negative; num_samples and num_chains must be positive.")
    if not 0.0 < target_accept_prob < 1.0:
        raise ValueError("target_accept_prob must lie strictly between 0 and 1.")
    if max_tree_depth < 1 or piE_max <= 0.0:
        raise ValueError("max_tree_depth and piE_max must be positive.")
    if peak_window_days != "auto" and peak_window_days is not None and peak_window_days <= 0.0:
        raise ValueError("peak_window_days must be positive, None, or 'auto'.")
    if fspl_n_fft < 16:
        raise ValueError("fspl_n_fft must be at least 16.")

    raw = _raw_parameters(fit)
    time = jnp.asarray(fit.time)
    flux = jnp.asarray(fit.flux)
    ferr = jnp.maximum(jnp.asarray(fit.ferr), 1.0e-12)
    if time.ndim != 1 or time.size < 7:
        raise ValueError("FSPL+parallax HMC requires at least seven one-dimensional observations.")
    if not bool(np.all(np.isfinite(np.asarray(time))) and np.all(np.isfinite(np.asarray(flux)))
                and np.all(np.isfinite(np.asarray(ferr)))):
        raise ValueError("fit data must be finite.")

    def _bounds(name: str, values: Sequence[float]) -> tuple[float, float]:
        if len(values) != 2 or not np.all(np.isfinite(values)) or values[0] >= values[1]:
            raise ValueError(f"{name} must be two finite values in ascending order.")
        return float(values[0]), float(values[1])

    if t0_bounds is None:
        t0_lo, t0_hi = float(np.min(fit.time)), float(np.max(fit.time))
    else:
        t0_lo, t0_hi = _bounds("t0_bounds", t0_bounds)
    tE_lo, tE_hi = _bounds("tE_bounds", tE_bounds)
    if tE_lo <= 0.0:
        raise ValueError("tE_bounds must be strictly positive.")
    u0_lo, u0_hi = _bounds("u0_bounds", u0_bounds)
    logrho_lo, logrho_hi = _bounds("logrho_bounds", logrho_bounds)

    initial = {
        "t0": raw[0], "log_tE": np.log(raw[1]), "u0": raw[2],
        "logrho": raw[3], "piEN": raw[4], "piEE": raw[5],
        "fs": float(fit.fs), "fb": float(fit.fb),
    }
    bounded = {
        "t0": (t0_lo, t0_hi), "log_tE": (np.log(tE_lo), np.log(tE_hi)),
        "u0": (u0_lo, u0_hi), "logrho": (logrho_lo, logrho_hi),
        "piEN": (-piE_max, piE_max), "piEE": (-piE_max, piE_max),
    }
    for name, (lower, upper) in bounded.items():
        if not lower < initial[name] < upper:
            raise ValueError(
                f"optimized {name}={initial[name]:.8g} lies outside its HMC prior "
                f"({lower:.8g}, {upper:.8g}); widen the corresponding bound."
            )

    # A scale based on the observed flux prevents an arbitrary magnitude unit
    # from making the linear-flux priors unintentionally informative.
    flux_scale = max(float(np.std(np.asarray(fit.flux))), float(np.median(np.asarray(fit.ferr))), 1.0e-8)
    flux_prior_scale = 100.0 * flux_scale
    P = fit.parallax_projector
    if peak_window_days == "auto":
        peak_window_days = max(10.0, 5.0 * raw[1] * np.exp(raw[3]))
    if peak_window_days is None:
        peak_indices = None
        n_fspl_points = int(time.size)
    else:
        peak_indices = jnp.asarray(
            np.flatnonzero(np.abs(np.asarray(fit.time) - raw[0]) <= peak_window_days),
            dtype=jnp.int32,
        )
        if peak_indices.size == 0:
            raise ValueError("peak_window_days selects no observations.")
        n_fspl_points = int(peak_indices.size)
    numpyro, dist, MCMC, NUTS, init_to_value = _require_numpyro()

    def model():
        t0 = numpyro.sample("t0", dist.Uniform(t0_lo, t0_hi))
        log_tE = numpyro.sample("log_tE", dist.Uniform(np.log(tE_lo), np.log(tE_hi)))
        u0 = numpyro.sample("u0", dist.Uniform(u0_lo, u0_hi))
        logrho = numpyro.sample("logrho", dist.Uniform(logrho_lo, logrho_hi))
        piEN = numpyro.sample("piEN", dist.Uniform(-piE_max, piE_max))
        piEE = numpyro.sample("piEE", dist.Uniform(-piE_max, piE_max))
        fs = numpyro.sample("fs", dist.Normal(float(fit.fs), flux_prior_scale))
        fb = numpyro.sample("fb", dist.Normal(float(fit.fb), flux_prior_scale))
        tE = jnp.exp(log_tE)
        rho = jnp.exp(logrho)
        q = jnp.array([t0, tE, u0, logrho, piEN, piEE])
        if peak_indices is None:
            magnification = A_fspl_parallax_logrho_func(q, time, P)
        else:
            magnification = A_fspl_parallax_logrho_peak_func(
                q, time, P, peak_indices, N_fft=fspl_n_fft
            )
        numpyro.deterministic("tE", tE)
        numpyro.deterministic("rho", rho)
        numpyro.sample("flux", dist.Normal(fs * magnification + fb, ferr), obs=flux)

    kernel = NUTS(
        model,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        init_strategy=init_to_value(values=initial),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    import jax

    mcmc.run(jax.random.PRNGKey(rng_seed))
    samples = {name: np.asarray(value) for name, value in mcmc.get_samples().items()}
    chain_samples = {
        name: np.asarray(value)
        for name, value in mcmc.get_samples(group_by_chain=True).items()
    }
    diagnostics = {name: np.asarray(value) for name, value in mcmc.get_extra_fields().items()}
    return FSPLParallaxHMCResult(
        samples=samples,
        chain_samples=chain_samples,
        diagnostics=diagnostics,
        optimized_fit=fit,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        fspl_peak_window_days=peak_window_days,
        n_fspl_points=n_fspl_points,
    )
