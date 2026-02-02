from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Callable

import numpy as np
import jax.numpy as jnp
from jaxopt import LevenbergMarquardt

from .photometry import solve_fs_fb
from .plot import SingleLensPlotter
from .objective import residual_norm_from_A, chi2_from_res
from .singlelens_model import (
    A_pspl_func,
    A_fspl_logrho_func,
    A_pspl_parallax_func,
    A_fspl_parallax_logrho_func,
)
from .trajectory import make_parallax_projector


@dataclass(frozen=True)
class SingleLensFitResult:
    """
    Result of a single-lens microlensing fit.

    Stores the input light curve on CPU (NumPy) for plotting convenience, while
    keeping fitted arrays as JAX arrays for downstream computation.
    """

    time: np.ndarray
    flux: np.ndarray
    ferr: np.ndarray

    params: jnp.ndarray
    param_names: Tuple[str, ...]
    chi2: jnp.ndarray
    chi2_dof: jnp.ndarray
    fs: jnp.ndarray
    fb: jnp.ndarray
    model_flux: jnp.ndarray
    residual: jnp.ndarray
    plot_flux: np.ndarray
    plot_time: np.ndarray

    # Optional: raw optimizer parameters (e.g. logrho), if different from `params`.
    raw_params: Optional[jnp.ndarray] = None


def _fit_single_lens(
    *,
    time: jnp.ndarray,
    flux: jnp.ndarray,
    ferr: jnp.ndarray,
    x0: jnp.ndarray,
    build_A: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    dof: int,
    param_names: Tuple[str, ...],
    x_to_params: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    maxiter: int = 1000,
    damping_parameter: float = 1e-6,
    tol: float = 1e-3,
    min_points: int = 4,
    store_raw_params: bool = False,
) -> SingleLensFitResult:
    """
    Shared fitting routine used by all single-lens fitters.

    This optimizes nonlinear parameters using Levenberg–Marquardt, while solving
    linear flux parameters (fs, fb) analytically at each evaluation via
    weighted linear regression.

    Notes
    -----
    `build_A` must be a pure function of (params, time). If it needs extra
    objects (e.g. a parallax projector), capture them via closure (do not pass
    them through JAX as arguments).
    """
    n = int(time.shape[0])
    if n < min_points:
        raise ValueError(f"Need at least {min_points} data points, got {n}.")

    eps = 1e-12
    ferr = jnp.maximum(ferr, eps)
    data = (time, flux, ferr)

    def residual_fun(x, data):
        t, f, fe = data
        A = build_A(x, t)
        return residual_norm_from_A(A, f, fe)

    solver = LevenbergMarquardt(
        residual_fun=residual_fun,
        maxiter=maxiter,
        damping_parameter=damping_parameter,
        tol=tol,
    )
    sol = solver.run(x0, data=data)
    x = sol.params

    A = build_A(x, time)
    fs, fb = solve_fs_fb(A, flux, ferr)
    model_flux = fs * A + fb
    residual = flux - model_flux

    plot_time = np.arange(np.min(time), np.max(time) + 0.5, 0.5)
    plot_A = build_A(x, plot_time)
    plot_flux = fs * plot_A + fb

    resn = residual_norm_from_A(A, flux, ferr)
    chi2 = chi2_from_res(resn)
    chi2_dof = chi2 / (n - dof)

    params_phys = x if x_to_params is None else x_to_params(x)
    raw = x if store_raw_params else None

    return SingleLensFitResult(
        time=np.asarray(time),
        flux=np.asarray(flux),
        ferr=np.asarray(ferr),
        params=params_phys,
        param_names=param_names,
        chi2=chi2,
        chi2_dof=chi2_dof,
        fs=fs,
        fb=fb,
        model_flux=model_flux,
        residual=residual,
        plot_time = plot_time,
        plot_flux = plot_flux,
        raw_params=raw,
    )


@dataclass
class PSPLFitter:
    """
    PSPL fitter (Point-Source Point-Lens).

    Nonlinear parameters: (t0, tE, u0)
    """

    maxiter: int = 1000
    damping_parameter: float = 1e-6
    tol: float = 1e-3

    def __post_init__(self):
        self.plotter = SingleLensPlotter()
        self._last_fit: Optional[SingleLensFitResult] = None

    def fit(self, time: jnp.ndarray, flux: jnp.ndarray, ferr: jnp.ndarray, p0: jnp.ndarray) -> SingleLensFitResult:
        """Fit PSPL to a light curve."""
        def build_A(p, t):
            return A_pspl_func(p, t)

        fit = _fit_single_lens(
            time=time, flux=flux, ferr=ferr, x0=p0,
            build_A=build_A,
            dof=3,
            param_names=("t0", "tE", "u0"),
            maxiter=self.maxiter,
            damping_parameter=self.damping_parameter,
            tol=self.tol,
            min_points=4,
        )
        self._last_fit = fit
        return fit

    def plot_lc(self, **kwargs):
        """Plot the light curve and best-fit model from the last fit."""
        if self._last_fit is None:
            raise RuntimeError("No fit has been run yet.")
        return self.plotter.plot_lc(self._last_fit, **kwargs)

    def plot_residual(self, **kwargs):
        """Plot residuals from the last fit."""
        if self._last_fit is None:
            raise RuntimeError("No fit has been run yet.")
        return self.plotter.plot_residual(self._last_fit, **kwargs)


@dataclass
class FSPLFitter:
    """
    FSPL fitter (Finite-Source Point-Lens).

    Optimizer parameters: (t0, tE, u0, logrho)
    Reported parameters:  (t0, tE, u0, rho)
    """

    maxiter: int = 1000
    damping_parameter: float = 1e-6
    tol: float = 1e-3

    def __post_init__(self):
        self.plotter = SingleLensPlotter()
        self._last_fit: Optional[SingleLensFitResult] = None

    def fit(self, time: jnp.ndarray, flux: jnp.ndarray, ferr: jnp.ndarray, q0: jnp.ndarray) -> SingleLensFitResult:
        """Fit FSPL to a light curve (uses logrho parameterization)."""
        def build_A(q, t):
            return A_fspl_logrho_func(q, t)

        def q_to_params(q):
            t0, tE, u0, logrho = q
            rho = jnp.exp(logrho)
            return jnp.array([t0, tE, u0, rho])

        fit = _fit_single_lens(
            time=time, flux=flux, ferr=ferr, x0=q0,
            build_A=build_A,
            dof=4,
            param_names=("t0", "tE", "u0", "rho"),
            x_to_params=q_to_params,
            maxiter=self.maxiter,
            damping_parameter=self.damping_parameter,
            tol=self.tol,
            min_points=4,
            store_raw_params=True,
        )
        self._last_fit = fit
        return fit

    def plot_lc(self, **kwargs):
        """Plot the light curve and best-fit model from the last fit."""
        if self._last_fit is None:
            raise RuntimeError("No fit has been run yet.")
        return self.plotter.plot_lc(self._last_fit, **kwargs)

    def plot_residual(self, **kwargs):
        """Plot residuals from the last fit."""
        if self._last_fit is None:
            raise RuntimeError("No fit has been run yet.")
        return self.plotter.plot_residual(self._last_fit, **kwargs)


@dataclass
class PSPLParallaxFitter:
    """
    PSPL + annual parallax fitter.

    Parameters: (t0, tE, u0, piEN, piEE)

    Notes
    -----
    The parallax projector is constructed once in `__post_init__`.
    """

    RA: float
    Dec: float
    tref: float
    maxiter: int = 1000
    damping_parameter: float = 1e-6
    tol: float = 1e-3

    def __post_init__(self):
        self.plotter = SingleLensPlotter()
        self._P = make_parallax_projector(self.RA, self.Dec, self.tref)
        self._last_fit: Optional[SingleLensFitResult] = None

    def fit(self, time: jnp.ndarray, flux: jnp.ndarray, ferr: jnp.ndarray, p0: jnp.ndarray) -> SingleLensFitResult:
        """Fit PSPL+parallax to a light curve."""
        P = self._P

        def build_A(p, t):
            return A_pspl_parallax_func(p, t, P)

        fit = _fit_single_lens(
            time=time, flux=flux, ferr=ferr, x0=p0,
            build_A=build_A,
            dof=5,
            param_names=("t0", "tE", "u0", "piEN", "piEE"),
            maxiter=self.maxiter,
            damping_parameter=self.damping_parameter,
            tol=self.tol,
            min_points=6,
        )
        self._last_fit = fit
        return fit

    def plot_lc(self, **kwargs):
        """Plot the light curve and best-fit model from the last fit."""
        if self._last_fit is None:
            raise RuntimeError("No fit has been run yet.")
        return self.plotter.plot_lc(self._last_fit, **kwargs)

    def plot_residual(self, **kwargs):
        """Plot residuals from the last fit."""
        if self._last_fit is None:
            raise RuntimeError("No fit has been run yet.")
        return self.plotter.plot_residual(self._last_fit, **kwargs)


@dataclass
class FSPLParallaxFitter:
    """
    FSPL + annual parallax fitter.

    Optimizer parameters: (t0, tE, u0, logrho, piEN, piEE)
    Reported parameters:  (t0, tE, u0, rho,  piEN, piEE)
    """

    RA: float
    Dec: float
    tref: float
    maxiter: int = 1000
    damping_parameter: float = 1e-6
    tol: float = 1e-3

    def __post_init__(self):
        self.plotter = SingleLensPlotter()
        self._P = make_parallax_projector(self.RA, self.Dec, self.tref)
        self._last_fit: Optional[SingleLensFitResult] = None

    def fit(self, time: jnp.ndarray, flux: jnp.ndarray, ferr: jnp.ndarray, q0: jnp.ndarray) -> SingleLensFitResult:
        """Fit FSPL+parallax to a light curve (uses logrho parameterization)."""
        P = self._P

        def build_A(q, t):
            return A_fspl_parallax_logrho_func(q, t, P)

        def q_to_params(q):
            t0, tE, u0, logrho, piEN, piEE = q
            rho = jnp.exp(logrho)
            return jnp.array([t0, tE, u0, rho, piEN, piEE])

        fit = _fit_single_lens(
            time=time, flux=flux, ferr=ferr, x0=q0,
            build_A=build_A,
            dof=6,
            param_names=("t0", "tE", "u0", "rho", "piEN", "piEE"),
            x_to_params=q_to_params,
            maxiter=self.maxiter,
            damping_parameter=self.damping_parameter,
            tol=self.tol,
            min_points=7,
            store_raw_params=True,
        )
        self._last_fit = fit
        return fit

    def plot_lc(self, **kwargs):
        """Plot the light curve and best-fit model from the last fit."""
        if self._last_fit is None:
            raise RuntimeError("No fit has been run yet.")
        return self.plotter.plot_lc(self._last_fit, **kwargs)

    def plot_residual(self, **kwargs):
        """Plot residuals from the last fit."""
        if self._last_fit is None:
            raise RuntimeError("No fit has been run yet.")
        return self.plotter.plot_residual(self._last_fit, **kwargs)
