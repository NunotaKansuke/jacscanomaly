from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Callable, Any

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
    A_cv_asymexp_logtau_func,
)
from .trajectory import make_parallax_projector

try:
    from . import _cpp_grid
except ImportError:  # pragma: no cover - optional compiled backend
    _cpp_grid = None


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

    # Optional: raw optimizer parameters (e.g. logrho), if different from `params`.
    raw_params: Optional[jnp.ndarray] = None
    parallax_projector: Optional[Any] = None


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
    parallax_projector: Optional[Any] = None,
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
        raw_params=raw,
        parallax_projector=parallax_projector,
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
class CPPPSPLFitter:
    """
    Experimental C++ PSPL fitter.

    Nonlinear parameters are optimized with a small finite-difference
    Levenberg-Marquardt implementation in the compiled extension. The linear
    flux parameters are solved analytically at each evaluation.
    """

    maxiter: int = 1000
    damping_parameter: float = 1e-6
    tol: float = 1e-3
    u0_min: float = 0.01
    min_t0_support_points: int = 3
    t0_support_tE_coeff: float = 3.0

    def __post_init__(self):
        self.plotter = SingleLensPlotter()
        self._last_fit: Optional[SingleLensFitResult] = None

    def fit(self, time: jnp.ndarray, flux: jnp.ndarray, ferr: jnp.ndarray, p0: jnp.ndarray) -> SingleLensFitResult:
        if _cpp_grid is None:
            raise RuntimeError("CPPPSPLFitter requires the compiled jacscanomaly._cpp_grid extension.")

        time_np = np.asarray(time, dtype=float)
        flux_np = np.asarray(flux, dtype=float)
        ferr_np = np.asarray(ferr, dtype=float)
        p0_np = np.asarray(p0, dtype=float)
        params, fs, fb, chi2, model_flux, residual = _cpp_grid.fit_pspl(
            time_np,
            flux_np,
            ferr_np,
            p0_np,
            maxiter=int(self.maxiter),
            damping_parameter=float(self.damping_parameter),
            tol=float(self.tol),
            u0_min=float(self.u0_min),
            min_t0_support_points=int(self.min_t0_support_points),
            t0_support_tE_coeff=float(self.t0_support_tE_coeff),
        )
        n = int(time_np.shape[0])
        chi2_dof = float(chi2) / max(n - 3, 1)
        fit = SingleLensFitResult(
            time=time_np,
            flux=flux_np,
            ferr=ferr_np,
            params=jnp.asarray(params),
            param_names=("t0", "tE", "u0"),
            chi2=jnp.asarray(float(chi2)),
            chi2_dof=jnp.asarray(chi2_dof),
            fs=jnp.asarray(float(fs)),
            fb=jnp.asarray(float(fb)),
            model_flux=jnp.asarray(model_flux),
            residual=jnp.asarray(residual),
        )
        self._last_fit = fit
        return fit

    def plot_lc(self, **kwargs):
        if self._last_fit is None:
            raise RuntimeError("No fit has been run yet.")
        return self.plotter.plot_lc(self._last_fit, **kwargs)

    def plot_residual(self, **kwargs):
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
            parallax_projector=P,
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
            parallax_projector=P,
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
class CVFitter:
    """
    Cataclysmic-variable-like transient fitter.

    Model shape is a linear rise:
    - linear ramp-up to peak over tau_rise
    - exponential decay over tau_decay

    Optimizer parameters: (t0, log_tau_rise, log_tau_decay)
    Reported parameters:  (t0, tau_rise, tau_decay)
    """

    maxiter: int = 1000
    damping_parameter: float = 1e-6
    tol: float = 1e-3

    def __post_init__(self):
        self.plotter = SingleLensPlotter()
        self._last_fit: Optional[SingleLensFitResult] = None

    def fit(self, time: jnp.ndarray, flux: jnp.ndarray, ferr: jnp.ndarray, q0: jnp.ndarray) -> SingleLensFitResult:
        """Fit a CV-shaped transient to a light curve."""
        def build_A(q, t):
            return A_cv_asymexp_logtau_func(q, t)

        def q_to_params(q):
            t0, log_tau_rise, log_tau_decay = q
            return jnp.array([t0, jnp.exp(log_tau_rise), jnp.exp(log_tau_decay)])

        fit = _fit_single_lens(
            time=time, flux=flux, ferr=ferr, x0=q0,
            build_A=build_A,
            dof=3,
            param_names=("t0", "tau_rise", "tau_decay"),
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
