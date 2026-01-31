from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
from typing import Tuple, Optional, Any

from .photometry import solve_fs_fb
from .plot import SingleLensPlotter
from jaxopt import LevenbergMarquardt
from .objective import residual_norm_from_A, chi2_from_res
from .singlelens_model import A_pspl_func, A_fspl_logrho_func, A_pspl_parallax_func, A_fspl_parallax_logrho_func
from .trajectory import make_parallax_projector

@dataclass(frozen=True)
class SingleLensFitResult:
    """
    Result container for single-lens model fitting.

    Attributes
    ----------
    time, flux, ferr : np.ndarray
        Input light-curve data (stored on CPU for plotting).
    params : jnp.ndarray
        Best-fit nonlinear parameters.
    param_names : tuple[str, ...]
        Names of nonlinear parameters corresponding to `params`.
    chi2 : jnp.ndarray
        Total chi-square of the best-fit model.
    chi2_dof : jnp.ndarray
        Reduced chi-square.
    fs, fb : jnp.ndarray
        Best-fit linear flux parameters.
    model_flux : jnp.ndarray
        Model flux evaluated at input times.
    residual : jnp.ndarray
        Flux residuals.
    """

    # input data (CPU, for plotting)
    time: np.ndarray
    flux: np.ndarray
    ferr: np.ndarray

    # fit results
    params: jnp.ndarray
    param_names: Tuple[str, ...]
    chi2: jnp.ndarray
    chi2_dof: jnp.ndarray
    fs: jnp.ndarray
    fb: jnp.ndarray
    model_flux: jnp.ndarray
    residual: jnp.ndarray
    raw_params: Optional[jnp.ndarray] = None

def _fit_single_lens(
    *,
    time: jnp.ndarray,
    flux: jnp.ndarray,
    ferr: jnp.ndarray,
    x0: jnp.ndarray,
    build_A,
    dof: int,
    param_names: Tuple[str, ...],
    x_to_params=None,
    maxiter: int = 1000,
    damping_parameter: float = 1e-6,
    tol: float = 1e-3,
    min_points: int = 4,
    store_raw_params: bool = False,
) -> SingleLensFitResult:
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

    # A, linear photometry, residuals
    A = build_A(x, time)
    fs, fb = solve_fs_fb(A, flux, ferr)
    model_flux = fs * A + fb
    residual = flux - model_flux

    resn = residual_norm_from_A(A, flux, ferr)
    chi2 = chi2_from_res(resn)
    chi2_dof = chi2 / (n - dof)

    if x_to_params is None:
        params_phys = x
        raw = x if store_raw_params else None
    else:
        params_phys = x_to_params(x)
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
        raw_params=raw,
    )


@dataclass
class PSPLFitter:
    """
    PSPL (Point-Source Point-Lens) light-curve fitter.

    Fits nonlinear PSPL parameters (t0, tE, u0) using
    Levenberg–Marquardt, while solving linear flux parameters
    (fs, fb) analytically by weighted linear regression.
    """
    maxiter: int = 1000
    damping_parameter: float = 1e-6
    tol: float = 1e-3

    def __post_init__(self):
        self.plotter = SingleLensPlotter()
        self._last_fit: SingleLensFitResult | None = None

    def fit(self, time, flux, ferr, p0) -> SingleLensFitResult:
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
        """
        Plot light curve with model using the last fit result.
        """
        if self._last_fit is None:
            raise RuntimeError("No PSPL fit has been run yet.")
        return self.plotter.plot_lc(self._last_fit, **kwargs)

    def plot_residual(self, **kwargs):
        """
        Plot residuals using the last fit result.
        """
        if self._last_fit is None:
            raise RuntimeError("No PSPL fit has been run yet.")
        return self.plotter.plot_residual(self._last_fit, **kwargs)

@dataclass
class FSPLFitter:
    maxiter: int = 1000
    damping_parameter: float = 1e-6
    tol: float = 1e-3

    def __post_init__(self):
        self.plotter = SingleLensPlotter()
        self._last_fit: SingleLensFitResult | None = None

    def fit(self, time, flux, ferr, q0) -> SingleLensFitResult:
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
        """
        Plot light curve with model using the last fit result.
        """
        if self._last_fit is None:
            raise RuntimeError("No PSPL fit has been run yet.")
        return self.plotter.plot_lc(self._last_fit, **kwargs)

    def plot_residual(self, **kwargs):
        """
        Plot residuals using the last fit result.
        """
        if self._last_fit is None:
            raise RuntimeError("No PSPL fit has been run yet.")
        return self.plotter.plot_residual(self._last_fit, **kwargs)

@dataclass
class PSPLParallaxFitter:
    RA: float
    Dec: float
    tref: float
    maxiter: int = 1000
    damping_parameter: float = 1e-6
    tol: float = 1e-3

    def __post_init__(self):
        self.plotter = SingleLensPlotter()
        self._P = make_parallax_projector(self.RA, self.Dec, self.tref)
        self._last_fit: SingleLensFitResult | None = None

    def fit(self, time, flux, ferr, p0) -> SingleLensFitResult:
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
        """
        Plot light curve with model using the last fit result.
        """
        if self._last_fit is None:
            raise RuntimeError("No PSPL fit has been run yet.")
        return self.plotter.plot_lc(self._last_fit, **kwargs)

    def plot_residual(self, **kwargs):
        """
        Plot residuals using the last fit result.
        """
        if self._last_fit is None:
            raise RuntimeError("No PSPL fit has been run yet.")
        return self.plotter.plot_residual(self._last_fit, **kwargs)

@dataclass
class FSPLParallaxFitter:
    RA: float
    Dec: float
    tref: float
    maxiter: int = 1000
    damping_parameter: float = 1e-6
    tol: float = 1e-3

    def __post_init__(self):
        self.plotter = SingleLensPlotter()
        self._P = make_parallax_projector(self.RA, self.Dec, self.tref)
        self._last_fit: SingleLensFitResult | None = None

    def fit(self, time, flux, ferr, q0) -> SingleLensFitResult:
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
        """
        Plot light curve with model using the last fit result.
        """
        if self._last_fit is None:
            raise RuntimeError("No PSPL fit has been run yet.")
        return self.plotter.plot_lc(self._last_fit, **kwargs)

    def plot_residual(self, **kwargs):
        """
        Plot residuals using the last fit result.
        """
        if self._last_fit is None:
            raise RuntimeError("No PSPL fit has been run yet.")
        return self.plotter.plot_residual(self._last_fit, **kwargs)
