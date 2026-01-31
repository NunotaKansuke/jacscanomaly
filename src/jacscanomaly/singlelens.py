from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp

from jaxopt import LevenbergMarquardt

from .utils import calc_A_pspl, calc_res_norm, calc_chi2
from .photometry import solve_fs_fb
from .plot import PSPLPlotter
from jaxopt import LevenbergMarquardt
from .objective import residual_norm_from_A, chi2_from_res
from .singlelens_model import A_pspl_func, A_fspl_logrho_func, A_pspl_parallax_func, A_fspl_parallax_logrho_func
from .trajectory import make_parallax_projector

@dataclass(frozen=True)
class PSPLFitResult:
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
        # PSPL-only plotter (Finder / anomaly not required)
        self.plotter = PSPLPlotter()
        self._last_fit = None

    def fit(
        self,
        time: jnp.ndarray,
        flux: jnp.ndarray,
        ferr: jnp.ndarray,
        p0: jnp.ndarray,
    ) -> PSPLFitResult:
        """
        Fit a PSPL model to a light curve.

        Parameters
        ----------
        time, flux, ferr : jnp.ndarray
            Input light-curve arrays (1D).
        p0 : jnp.ndarray
            Initial guess for (t0, tE, u0).

        Returns
        -------
        PSPLFitResult
            Best-fit result including data, model, and residuals.
        """
        n = int(time.shape[0])
        if n < 4:
            raise ValueError(f"Need at least 4 data points for PSPL fit, got {n}.")

        # avoid zero uncertainties
        eps = 1e-12
        ferr = jnp.maximum(ferr, eps)

        data = (time, flux, ferr)
        
        def residual_fun(params, data):
            t, f, fe = data
            A = A_pspl_func(params, t)
            return residual_norm_from_A(A, f, fe)
        
        solver = LevenbergMarquardt(
            residual_fun=residual_fun,
            maxiter=self.maxiter,
            damping_parameter=self.damping_parameter,
            tol=self.tol,
        )
        sol = solver.run(p0, data=data)
        params = sol.params

        # best-fit model
        A = calc_A_pspl(params[0], params[1], params[2], time)
        w = ferr ** (-2)
        fs, fb = solve_fs_fb(A, flux, ferr)
        model_flux = fs * A + fb
        residual = flux - model_flux

        t, f, fe = data
        A = A_pspl_func(params, t)
        res = residual_norm_from_A(A, f, fe)
        chi2 = chi2_from_res(res)
        dof = n - 3
        chi2_dof = chi2 / dof

        # store input data on CPU for plotting
        time_np = np.asarray(time)
        flux_np = np.asarray(flux)
        ferr_np = np.asarray(ferr)

        fit = PSPLFitResult(
            time=time_np,
            flux=flux_np,
            ferr=ferr_np,
            params=params,
            param_names=("t0", "tE", "u0"),
            chi2=chi2,
            chi2_dof=chi2_dof,
            fs=fs,
            fb=fb,
            model_flux=model_flux,
            residual=residual,
        )
        
        self._last_fit = fit
        return fit

    def plot_lc(self, **kwargs):
        """
        Plot light curve with PSPL model using the last fit result.
        """
        if self._last_fit is None:
            raise RuntimeError("No PSPL fit has been run yet.")
        return self.plotter.plot_lc(self._last_fit, **kwargs)

    def plot_residual(self, **kwargs):
        """
        Plot PSPL residuals using the last fit result.
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
        self._last_fit = None

    def fit(self, time: jnp.ndarray, flux: jnp.ndarray, ferr: jnp.ndarray, q0: jnp.ndarray):
        n = int(time.shape[0])
        if n < 4:
            raise ValueError(f"Need at least 4 data points, got {n}.")
        eps = 1e-12
        ferr = jnp.maximum(ferr, eps)

        data = (time, flux, ferr)

        def residual_fun(q, data):
            t, f, fe = data
            A = A_fspl_logrho_func(q, t)
            return residual_norm_from_A(A, f, fe)

        solver = LevenbergMarquardt(
            residual_fun=residual_fun,
            maxiter=self.maxiter,
            damping_parameter=self.damping_parameter,
            tol=self.tol,
        )
        sol = solver.run(q0, data=data)
        q = sol.params  # [t0, tE, u0, logrho]

        t0, tE, u0, logrho = q
        rho = jnp.exp(logrho)
        params_phys = jnp.array([t0, tE, u0, rho])

        A = A_fspl_logrho_func(q, time)
        fs, fb = solve_fs_fb(A, flux, ferr)
        model_flux = fs * A + fb
        residual = flux - model_flux

        resn = residual_norm_from_A(A, flux, ferr)
        chi2 = chi2_from_res(resn)
        dof = n - 4
        chi2_dof = chi2 / dof

        fit = PSPLFitResult(
            time=np.asarray(time),
            flux=np.asarray(flux),
            ferr=np.asarray(ferr),
            params=params_phys,
            param_names=("t0", "tE", "u0", "rho"),
            chi2=chi2,
            chi2_dof=chi2_dof,
            fs=fs,
            fb=fb,
            model_flux=model_flux,
            residual=residual,
        )
        self._last_fit = fit
        return fit

@dataclass
class PSPLParallaxFitter:
    RA: float
    Dec: float
    tref: float

    maxiter: int = 1000
    damping_parameter: float = 1e-6
    tol: float = 1e-3

    def __post_init__(self):
        self._P = make_parallax_projector(self.RA, self.Dec, self.tref)
        self._last_fit = None

    def fit(
        self,
        time: jnp.ndarray,
        flux: jnp.ndarray,
        ferr: jnp.ndarray,
        p0: jnp.ndarray,
    ) -> PSPLFitResult:
        """
        Fit PSPL + annual parallax.

        params = (t0, tE, u0, piEN, piEE)
        """
        n = int(time.shape[0])
        if n < 6:
            raise ValueError(f"Need at least 6 data points, got {n}.")

        eps = 1e-12
        ferr = jnp.maximum(ferr, eps)
        data = (time, flux, ferr)

        P = self._P

        def residual_fun(params, data):
            t, f, fe = data
            A = A_pspl_parallax_func(params, t, P)
            return residual_norm_from_A(A, f, fe)

        solver = LevenbergMarquardt(
            residual_fun=residual_fun,
            maxiter=self.maxiter,
            damping_parameter=self.damping_parameter,
            tol=self.tol,
        )
        sol = solver.run(p0, data=data)
        params = sol.params

        # best-fit model
        A = A_pspl_parallax_func(params, time, P)

        fs, fb = solve_fs_fb(A, flux, ferr)
        model_flux = fs * A + fb
        residual = flux - model_flux

        resn = residual_norm_from_A(A, flux, ferr)
        chi2 = chi2_from_res(resn)
        dof = n - 5
        chi2_dof = chi2 / dof

        fit = PSPLFitResult(
            time=np.asarray(time),
            flux=np.asarray(flux),
            ferr=np.asarray(ferr),
            params=params,
            param_names=("t0", "tE", "u0", "piEN", "piEE"),
            chi2=chi2,
            chi2_dof=chi2_dof,
            fs=fs,
            fb=fb,
            model_flux=model_flux,
            residual=residual,
        )
        self._last_fit = fit
        return fit

@dataclass
class FSPLParallaxFitter:
    RA: float
    Dec: float
    tref: float

    maxiter: int = 1000
    damping_parameter: float = 1e-6
    tol: float = 1e-3

    def __post_init__(self):
        self._P = make_parallax_projector(self.RA, self.Dec, self.tref)
        self._last_fit = None

    def fit(
        self,
        time: jnp.ndarray,
        flux: jnp.ndarray,
        ferr: jnp.ndarray,
        q0: jnp.ndarray,
    ) -> PSPLFitResult:
        """
        Fit FSPL + annual parallax.

        LM params q = (t0, tE, u0, logrho, piEN, piEE)
        Stored params = (t0, tE, u0, rho, piEN, piEE)
        """
        n = int(time.shape[0])
        if n < 7:
            raise ValueError(f"Need at least 7 data points, got {n}.")

        eps = 1e-12
        ferr = jnp.maximum(ferr, eps)
        data = (time, flux, ferr)

        P = self._P  # fixed

        def residual_fun(q, data):
            t, f, fe = data
            A = A_fspl_parallax_logrho_func(q, t, P)
            return residual_norm_from_A(A, f, fe)

        solver = LevenbergMarquardt(
            residual_fun=residual_fun,
            maxiter=self.maxiter,
            damping_parameter=self.damping_parameter,
            tol=self.tol,
        )
        sol = solver.run(q0, data=data)
        q = sol.params  # (t0, tE, u0, logrho, piEN, piEE)

        # convert to physical params
        t0, tE, u0, logrho, piEN, piEE = q
        rho = jnp.exp(logrho)
        params_phys = jnp.array([t0, tE, u0, rho, piEN, piEE])

        # evaluate best-fit model
        A = A_fspl_parallax_logrho_func(q, time, P)
        fs, fb = solve_fs_fb(A, flux, ferr)
        model_flux = fs * A + fb
        residual = flux - model_flux

        resn = residual_norm_from_A(A, flux, ferr)
        chi2 = chi2_from_res(resn)
        dof = n - 6  # nonlinear: t0,tE,u0,logrho,piEN,piEE
        chi2_dof = chi2 / dof

        fit = PSPLFitResult(
            time=np.asarray(time),
            flux=np.asarray(flux),
            ferr=np.asarray(ferr),
            params=params_phys,
            param_names=("t0", "tE", "u0", "rho", "piEN", "piEE"),
            chi2=chi2,
            chi2_dof=chi2_dof,
            fs=fs,
            fb=fb,
            model_flux=model_flux,
            residual=residual,
        )
        self._last_fit = fit
        return fit
