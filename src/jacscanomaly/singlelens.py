from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp

from jaxopt import LevenbergMarquardt

from .utils import calc_A_pspl, calc_res_norm, calc_chi2
from .photometry import solve_fs_fb
from .plot import PSPLPlotter

@dataclass(frozen=True)
class PSPLFitResult:
    """
    Result container for PSPL model fitting.

    This object intentionally stores the input data so that
    plotting can be performed directly via PSPLPlotter.

    Attributes
    ----------
    time, flux, ferr : np.ndarray
        Input light-curve data (stored on CPU for plotting).
    params : jnp.ndarray
        Best-fit nonlinear PSPL parameters (t0, tE, u0).
    chi2 : jnp.ndarray
        Total chi-square of the best-fit model.
    chi2_dof : jnp.ndarray
        Reduced chi-square (chi2 / degrees of freedom).
    fs, fb : jnp.ndarray
        Best-fit linear flux parameters.
    model_flux : jnp.ndarray
        Model flux evaluated at input times.
    residual : jnp.ndarray
        Flux residuals: (observed flux - model_flux).
    """

    # input data (CPU, for plotting)
    time: np.ndarray
    flux: np.ndarray
    ferr: np.ndarray

    # fit results
    params: jnp.ndarray
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

        data = jnp.stack([time, flux, ferr], axis=1)

        solver = LevenbergMarquardt(
            residual_fun=calc_res_norm,
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

        chi2 = calc_chi2(params, data)
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
