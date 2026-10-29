"""Canonical single-lens fitters.

There are only four model families in the public fitting API:

* :class:`PSPLFitter`
* :class:`FSPLFitter`
* :class:`PSPLParallaxFitter`
* :class:`FSPLParallaxFitter`

The model family owns the fit contract.  Observer geometry (annual or space)
and the space-observer convention (including GULLS) are options of the
parallax fitters, not separate fitter classes.  Continuous optimization is
performed by SciPy's Levenberg--Marquardt solver.  PSPL uses the analytic
point-source kernel; FSPL and parallax use compiled magnification/trajectory
evaluators.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Optional

import numpy as np
import jax.numpy as jnp

from .auto_init import automatic_seeds, best_fit_over_seeds
from .plot import SingleLensPlotter
from .singlelens_fit import SingleLensFitResult
from .parallax_backend import (
    NESTED_RHO,
    FSPLParallaxFitter,
    PSPLParallaxFitter,
    default_espl_table_path,
)

try:
    from scipy.optimize import least_squares
except ImportError:  # pragma: no cover - dependency is declared by the package
    least_squares = None

try:
    from . import _vbm_cpp
except ImportError:  # pragma: no cover - source-only installations
    _vbm_cpp = None

try:
    import VBMicrolensing
except ImportError:  # pragma: no cover - dependency is declared by the package
    VBMicrolensing = None


def _solve_fs_fb_numpy(
    magnification: np.ndarray,
    flux: np.ndarray,
    ferr: np.ndarray,
) -> tuple[float, float]:
    """Profile source/blend fluxes for one nonlinear model evaluation."""
    errors = np.maximum(np.asarray(ferr, dtype=float), 1.0e-12)
    weights = 1.0 / (errors * errors)
    weight_sum = np.sum(weights)
    x_mean = np.sum(weights * magnification) / weight_sum
    y_mean = np.sum(weights * flux) / weight_sum
    centered_magnification = magnification - x_mean
    centered_flux = flux - y_mean
    denominator = np.sum(weights * centered_magnification * centered_magnification)
    if not np.isfinite(denominator) or denominator <= 0.0:
        return np.nan, np.nan
    source_flux = np.sum(weights * centered_magnification * centered_flux) / denominator
    blend_flux = y_mean - source_flux * x_mean
    return float(source_flux), float(blend_flux)


def _fit_arrays(time, flux, ferr) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_np = np.asarray(time, dtype=float).reshape(-1)
    flux_np = np.asarray(flux, dtype=float).reshape(-1)
    ferr_np = np.asarray(ferr, dtype=float).reshape(-1)
    if not (time_np.shape == flux_np.shape == ferr_np.shape):
        raise ValueError("time, flux, and ferr must have equal one-dimensional shapes.")
    if not np.all(np.isfinite(time_np)) or not np.all(np.isfinite(flux_np)):
        raise ValueError("time and flux must be finite.")
    if not np.all(np.isfinite(ferr_np)) or np.any(ferr_np <= 0.0):
        raise ValueError("ferr must be finite and positive.")
    return time_np, flux_np, ferr_np


def _excluded_mask(mask, size: int, n_params: int) -> Optional[np.ndarray]:
    """Validate a fit mask (``True`` = excluded); ``None`` if nothing is excluded."""
    if mask is None:
        return None
    excluded = np.asarray(mask, dtype=bool).reshape(-1)
    if excluded.size != size:
        raise ValueError("mask must have the same length as time.")
    if not excluded.any():
        return None
    if np.count_nonzero(~excluded) < n_params + 3:
        raise ValueError("mask leaves too few points to fit the model.")
    return excluded


def masked_fit(fitter, fit_subset, time, flux, ferr, excluded: np.ndarray):
    """Fit the kept points, then evaluate the solution on every point.

    ``fit_subset(time, flux, ferr)`` runs the fitter's ordinary (unmasked)
    fit on the kept points.  The returned result is the fixed-parameter
    evaluation on the full arrays with ``fs``/``fb``/``chi2`` solved on the
    kept points, carrying the subset fit's optimizer status.
    """
    keep = ~excluded
    subset = fit_subset(time[keep], flux[keep], ferr[keep])
    full = fitter.evaluate_fixed(
        time, flux, ferr, fitter.seed_from_fit(subset), mask=excluded
    )
    full = replace(
        full,
        optimizer_success=subset.optimizer_success,
        optimizer_status=subset.optimizer_status,
        diagnostics=subset.diagnostics,
    )
    fitter._last_fit = full
    return full


def _pspl_magnification(time: np.ndarray, t0: float, tE: float, u0: float) -> np.ndarray:
    tE_safe = max(abs(float(tE)), 1.0e-12)
    tau = (np.asarray(time, dtype=float) - float(t0)) / tE_safe
    u = np.sqrt(tau * tau + float(u0) * float(u0))
    u_safe = np.maximum(u, 1.0e-12)
    return (u_safe * u_safe + 2.0) / (
        u_safe * np.sqrt(u_safe * u_safe + 4.0)
    )


def _profiled_residual_jacobian(
    magnification: np.ndarray,
    magnification_gradient: np.ndarray,
    flux: np.ndarray,
    ferr: np.ndarray,
) -> np.ndarray:
    """Jacobian of the profiled-flux residual (variable projection).

    ``magnification_gradient`` is dA/dz with shape (n, k).  With
    r = (y - ym - fs (A - xm)) / sigma and fs = Sxy / Sxx the exact derivative
    including the dependence of the profiled fs and fb on the parameters is
    dr = -(dfs (A - xm) + fs (dA - dxm)) / sigma.
    """
    errors = np.maximum(np.asarray(ferr, dtype=float), 1.0e-12)
    weights = 1.0 / (errors * errors)
    weight_sum = np.sum(weights)
    x_mean = np.sum(weights * magnification) / weight_sum
    y_mean = np.sum(weights * flux) / weight_sum
    centered = magnification - x_mean
    denominator = np.sum(weights * centered * centered)
    source_flux = np.sum(weights * centered * (flux - y_mean)) / denominator
    projection = weights * (flux - y_mean - 2.0 * source_flux * centered)
    d_source_flux = (projection @ magnification_gradient) / denominator
    d_x_mean = (weights @ magnification_gradient) / weight_sum
    return -(
        np.outer(centered, d_source_flux)
        + source_flux * (magnification_gradient - d_x_mean)
    ) / errors[:, None]


def _trajectory_gradient(
    time: np.ndarray, t0: float, tE: float, u0: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return u(t) and du/d(t0, ln tE, u0) for the rectilinear trajectory."""
    tE_safe = max(abs(float(tE)), 1.0e-12)
    tau = (np.asarray(time, dtype=float) - float(t0)) / tE_safe
    u = np.sqrt(tau * tau + float(u0) * float(u0))
    u_safe = np.maximum(u, 1.0e-12)
    du = np.column_stack(
        [-tau / (tE_safe * u_safe), -tau * tau / u_safe, np.full_like(u, float(u0)) / u_safe]
    )
    return u, du


def _pspl_gradient(time: np.ndarray, t0: float, tE: float, u0: float) -> np.ndarray:
    """dA/d(t0, ln tE, u0) of the point-source point-lens magnification."""
    u, du = _trajectory_gradient(time, t0, tE, u0)
    u_safe = np.maximum(u, 1.0e-12)
    u2 = u_safe * u_safe
    dA_du = -8.0 / (u2 * (u2 + 4.0) * np.sqrt(u2 + 4.0))
    return dA_du[:, None] * du


def _run_scipy_lm(
    residual,
    initial: np.ndarray,
    *,
    maxiter: int,
    tol: float,
    diff_step: Optional[float] = None,
    jac="2-point",
):
    if least_squares is None:
        raise ImportError("scipy is required for single-lens fitting.")
    tolerance = max(float(tol), 1.0e-14)
    kwargs = {
        "jac": jac,
        "method": "lm",
        "max_nfev": max(1, int(maxiter)),
        "xtol": tolerance,
        "ftol": tolerance,
        "gtol": tolerance,
    }
    if diff_step is not None and jac == "2-point":
        kwargs["diff_step"] = float(diff_step)
    return least_squares(residual, np.asarray(initial, dtype=float), **kwargs)


def _optimizer_status(result, *, maxiter: int, detail: str = "") -> str:
    if bool(getattr(result, "fixed", False)):
        return "fixed_parameters"
    suffix = f";{detail}" if detail else ""
    return (
        f"scipy_lm:status={int(result.status)};"
        f"nfev={int(result.nfev)};max_nfev={int(maxiter)};"
        f"message={result.message}{suffix}"
    )


def _cpp_fspl_magnification(
    u: np.ndarray,
    rho: float,
    *,
    espl_table_path: Optional[str],
    tol: float,
    reltol: float,
) -> np.ndarray:
    """Evaluate FSPL magnification in the compiled VBM-backed extension."""
    u_np = np.asarray(u, dtype=float)
    rho_safe = max(float(rho), 1.0e-12)
    if _vbm_cpp is not None and hasattr(_vbm_cpp, "fspl_magnification"):
        values = _vbm_cpp.fspl_magnification(
            u_np,
            rho_safe,
            espl_table=espl_table_path,
            tol=float(tol),
            reltol=float(reltol),
        )
        return np.asarray(values, dtype=float)

    # This fallback still calls the C++ VBMicrolensing library through its
    # Python binding.  It only exists for an already-installed extension built
    # before the vectorized binding was added; it never falls back to JAX.
    if VBMicrolensing is None:
        raise ImportError(
            "FSPL fitting requires the compiled jacscanomaly._vbm_cpp backend "
            "or VBMicrolensing. Rebuild jacscanomaly after installation."
        )
    vbm = VBMicrolensing.VBMicrolensing()
    vbm.Tol = float(tol)
    vbm.RelTol = float(reltol)
    if espl_table_path:
        vbm.LoadESPLTable(str(espl_table_path))
    values = np.asarray(
        [vbm.ESPLMag(float(abs(value)), rho_safe) for value in u_np],
        dtype=float,
    )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("VBMicrolensing returned an invalid FSPL magnification.")
    return values


def _cpp_fspl_magnification_gradient(
    u: np.ndarray,
    rho: float,
    *,
    espl_table_path: Optional[str],
    tol: float,
    reltol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FSPL magnification with analytic dA/du and dA/drho from the C++ kernel."""
    if _vbm_cpp is None or not hasattr(_vbm_cpp, "fspl_magnification_gradient"):
        raise ImportError(
            "Analytic FSPL Jacobian requires the compiled jacscanomaly._vbm_cpp "
            "backend; rebuild jacscanomaly."
        )
    values, d_du, d_drho = _vbm_cpp.fspl_magnification_gradient(
        np.asarray(u, dtype=float),
        max(float(rho), 1.0e-12),
        espl_table=espl_table_path,
        tol=float(tol),
        reltol=float(reltol),
    )
    return (
        np.asarray(values, dtype=float),
        np.asarray(d_du, dtype=float),
        np.asarray(d_drho, dtype=float),
    )


@dataclass
class PSPLFitter:
    """Point-source point-lens fitter using SciPy LM and a native model."""

    maxiter: int = 1000
    tol: float = 1.0e-6

    def __post_init__(self) -> None:
        self.plotter = SingleLensPlotter()
        self._last_fit: Optional[SingleLensFitResult] = None

    @property
    def parameter_dimension(self) -> int:
        return 3

    @staticmethod
    def seed_from_fit(fit) -> np.ndarray:
        return np.asarray(fit.params, dtype=float).reshape(-1).copy()

    @staticmethod
    def nested_seed(pspl_params) -> np.ndarray:
        """The point of this model that is the PSPL ``(t0, tE, u0)``: itself."""
        return np.asarray(pspl_params, dtype=float).reshape(-1)[:3].copy()

    @staticmethod
    def _decode(z: np.ndarray) -> np.ndarray:
        value = np.asarray(z, dtype=float)
        return np.asarray(
            [value[0], np.exp(np.clip(value[1], -50.0, 50.0)), value[2]],
            dtype=float,
        )

    def _magnification(self, time: np.ndarray, q: np.ndarray) -> np.ndarray:
        return _pspl_magnification(time, q[0], q[1], q[2])

    def magnification_model(self, time):
        """``p -> A(time; p)`` for seed parameters ``p = (t0, tE, u0)``."""
        time = np.asarray(time, dtype=float)
        return lambda p: self._magnification(time, np.asarray(p, dtype=float))

    def _make_fit(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        ferr: np.ndarray,
        q: np.ndarray,
        result,
        excluded: Optional[np.ndarray] = None,
    ) -> SingleLensFitResult:
        magnification = self._magnification(time, q)
        keep = slice(None) if excluded is None else ~excluded
        fs, fb = _solve_fs_fb_numpy(magnification[keep], flux[keep], ferr[keep])
        if not np.isfinite(fs) or not np.isfinite(fb):
            raise RuntimeError("PSPL profiled flux solve is singular.")
        model = fs * magnification + fb
        residual = flux - model
        normalized = residual[keep] / ferr[keep]
        chi2 = float(np.dot(normalized, normalized))
        fit = SingleLensFitResult(
            time=time,
            flux=flux,
            ferr=ferr,
            params=jnp.asarray(q),
            param_names=("t0", "tE", "u0"),
            chi2=jnp.asarray(chi2),
            chi2_dof=jnp.asarray(chi2 / max(normalized.size - 3, 1)),
            fs=jnp.asarray(fs),
            fb=jnp.asarray(fb),
            model_flux=jnp.asarray(model),
            residual=jnp.asarray(residual),
            model_kind="pspl",
            optimizer_success=bool(result.success),
            optimizer_status=_optimizer_status(result, maxiter=self.maxiter),
            mask=excluded,
        )
        self._last_fit = fit
        return fit

    def fit(
        self, time, flux, ferr, p0=None, *, pspl_params=None, mask=None
    ) -> SingleLensFitResult:
        """Fit PSPL, using the shared automatic multistart when ``p0`` is absent.

        ``pspl_params`` (only its ``t0``) roughly locates the brightening.
        ``mask`` (``True`` = excluded) removes points from the fit; the result
        is still evaluated on every point.
        """
        time_np, flux_np, ferr_np = _fit_arrays(time, flux, ferr)
        excluded = _excluded_mask(mask, time_np.size, 3)
        if excluded is not None:
            return masked_fit(
                self,
                lambda t, f, e: self.fit(t, f, e, p0, pspl_params=pspl_params),
                time_np, flux_np, ferr_np, excluded,
            )
        if p0 is None:
            seeds = automatic_seeds(
                time_np, flux_np, ferr_np, finite_source=False, pspl_params=pspl_params
            )
            best_fit = best_fit_over_seeds(
                lambda seed: self.fit(time_np, flux_np, ferr_np, seed), seeds, "PSPL"
            )
            self._last_fit = best_fit
            return best_fit
        q0 = np.asarray(p0, dtype=float).reshape(-1)
        if q0.size != 3:
            raise ValueError("PSPL p0 must contain (t0, tE, u0).")
        if not np.all(np.isfinite(q0)) or q0[1] == 0.0:
            raise ValueError("PSPL p0 must be finite and have nonzero tE.")
        # Fit t0 relative to the seed: LM's xtol is relative to |x|, so an
        # absolute t0 (~1e4 d) stopped the fit while u0 was still moving by
        # ~1e-3 (2371 0011).  Same translation as the FSPL fitter.
        time_origin = float(q0[0])
        fit_time = time_np - time_origin
        z0 = np.asarray([0.0, np.log(abs(q0[1])), q0[2]], dtype=float)

        def residual(z):
            q = self._decode(z)
            magnification = self._magnification(fit_time, q)
            fs, fb = _solve_fs_fb_numpy(magnification, flux_np, ferr_np)
            if not np.isfinite(fs) or not np.isfinite(fb):
                return np.full(time_np.size, 1.0e50, dtype=float)
            values = (flux_np - (fs * magnification + fb)) / ferr_np
            return values if np.all(np.isfinite(values)) else np.full_like(values, 1.0e50)

        def jacobian(z):
            q = self._decode(z)
            magnification = self._magnification(fit_time, q)
            gradient = _pspl_gradient(fit_time, q[0], q[1], q[2])
            if np.isfinite(gradient).all():
                try:
                    value = _profiled_residual_jacobian(
                        magnification, gradient, flux_np, ferr_np
                    )
                    if np.isfinite(value).all():
                        return value
                except FloatingPointError:
                    pass
            return np.zeros((time_np.size, 3), dtype=float)

        result = _run_scipy_lm(
            residual,
            z0,
            maxiter=self.maxiter,
            tol=self.tol,
            jac=jacobian,
        )
        q = self._decode(np.asarray(result.x, dtype=float))
        q[0] += time_origin
        return self._make_fit(time_np, flux_np, ferr_np, q, result)

    def evaluate_fixed(self, time, flux, ferr, p0, *, mask=None) -> SingleLensFitResult:
        time_np, flux_np, ferr_np = _fit_arrays(time, flux, ferr)
        excluded = _excluded_mask(mask, time_np.size, 3)
        q = np.asarray(p0, dtype=float).reshape(-1)
        if q.size != 3 or not np.all(np.isfinite(q)) or q[1] == 0.0:
            raise ValueError("PSPL fixed parameters must contain finite (t0, tE, u0).")
        return self._make_fit(
            time_np,
            flux_np,
            ferr_np,
            q,
            SimpleNamespace(
                fixed=True,
                success=True,
                status=0,
                nfev=0,
                message="fixed_parameters",
            ),
            excluded,
        )

    def fit_fixed_model(self, time, flux, ferr, p0, *, model_kind=None, mask=None):
        """Evaluate fixed nonlinear parameters through the common fitter API."""
        if model_kind not in {None, "pspl"}:
            raise ValueError(f"PSPLFitter cannot evaluate model_kind={model_kind!r}.")
        return self.evaluate_fixed(time, flux, ferr, p0, mask=mask)

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
    """Finite-source point-lens fitter using C++ magnification and SciPy LM."""

    maxiter: int = 1000
    tol: float = 1.0e-6
    magnification_tol: float = 1.0e-4
    magnification_reltol: float = 1.0e-4
    espl_table_path: Optional[str] = None

    def __post_init__(self) -> None:
        self.plotter = SingleLensPlotter()
        self._last_fit: Optional[SingleLensFitResult] = None
        if self.espl_table_path is None:
            self.espl_table_path = default_espl_table_path()

    @property
    def parameter_dimension(self) -> int:
        return 4

    @staticmethod
    def seed_from_fit(fit) -> np.ndarray:
        params = np.asarray(fit.params, dtype=float).reshape(-1)
        if params.size != 4:
            raise ValueError("FSPL fit must contain (t0, tE, u0, rho).")
        return np.asarray(
            [params[0], params[1], params[2], np.log(max(abs(params[3]), 1.0e-12))],
            dtype=float,
        )

    @staticmethod
    def nested_seed(pspl_params) -> np.ndarray:
        """The FSPL start that is the PSPL ``(t0, tE, u0)``: ``rho -> 0``.

        ``rho = NESTED_RHO`` is far below any ``u0`` a host fit resolves, so
        the start reproduces the PSPL magnification and its chi2.
        """
        p = np.asarray(pspl_params, dtype=float).reshape(-1)
        return np.asarray([p[0], p[1], p[2], np.log(NESTED_RHO)], dtype=float)

    @staticmethod
    def _decode(z: np.ndarray) -> np.ndarray:
        value = np.asarray(z, dtype=float)
        return np.asarray(
            [
                value[0],
                np.exp(np.clip(value[1], -50.0, 50.0)),
                value[2],
                value[3],
            ],
            dtype=float,
        )

    def magnification_model(self, time):
        """``p -> A(time; p)`` for seed parameters ``p = (t0, tE, u0, log rho)``."""
        time = np.asarray(time, dtype=float)
        return lambda p: self._magnification(time, np.asarray(p, dtype=float))

    def _magnification(self, time: np.ndarray, q: np.ndarray) -> np.ndarray:
        t0, tE, u0, logrho = map(float, q)
        u = np.sqrt(((time - t0) / max(abs(tE), 1.0e-12)) ** 2 + u0 * u0)
        rho = np.exp(np.clip(logrho, -50.0, 10.0))
        return _cpp_fspl_magnification(
            u,
            rho,
            espl_table_path=self.espl_table_path,
            tol=self.magnification_tol,
            reltol=self.magnification_reltol,
        )

    def _make_fit(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        ferr: np.ndarray,
        q: np.ndarray,
        result,
        excluded: Optional[np.ndarray] = None,
    ) -> SingleLensFitResult:
        magnification = self._magnification(time, q)
        keep = slice(None) if excluded is None else ~excluded
        fs, fb = _solve_fs_fb_numpy(magnification[keep], flux[keep], ferr[keep])
        if not np.isfinite(fs) or not np.isfinite(fb):
            raise RuntimeError("FSPL profiled flux solve is singular.")
        model = fs * magnification + fb
        residual = flux - model
        normalized = residual[keep] / ferr[keep]
        chi2 = float(np.dot(normalized, normalized))
        params = np.asarray([q[0], q[1], q[2], np.exp(np.clip(q[3], -50.0, 10.0))])
        fit = SingleLensFitResult(
            time=time,
            flux=flux,
            ferr=ferr,
            params=jnp.asarray(params),
            param_names=("t0", "tE", "u0", "rho"),
            chi2=jnp.asarray(chi2),
            chi2_dof=jnp.asarray(chi2 / max(normalized.size - 4, 1)),
            fs=jnp.asarray(fs),
            fb=jnp.asarray(fb),
            model_flux=jnp.asarray(model),
            residual=jnp.asarray(residual),
            raw_params=jnp.asarray(q),
            model_kind="fspl",
            model_evaluator=lambda values, q=q.copy(): self._magnification(
                np.asarray(values, dtype=float), q
            ),
            optimizer_success=bool(result.success),
            optimizer_status=_optimizer_status(
                result, maxiter=self.maxiter, detail="compiled_magnification"
            ),
            mask=excluded,
        )
        self._last_fit = fit
        return fit

    def fit_fixed_model(self, time, flux, ferr, p0, *, model_kind=None, mask=None):
        """Evaluate fixed nonlinear parameters through the common fitter API."""
        if model_kind not in {None, "fspl"}:
            raise ValueError(f"FSPLFitter cannot evaluate model_kind={model_kind!r}.")
        return self.evaluate_fixed(time, flux, ferr, p0, mask=mask)

    def _fit_from_seed(
        self,
        time_np: np.ndarray,
        flux_np: np.ndarray,
        ferr_np: np.ndarray,
        p0,
    ) -> SingleLensFitResult:
        q0 = np.asarray(p0, dtype=float).reshape(-1)
        if q0.size != 4:
            raise ValueError("FSPL p0 must contain (t0, tE, u0, logrho).")
        if not np.all(np.isfinite(q0)) or q0[1] == 0.0:
            raise ValueError("FSPL p0 must be finite and have nonzero tE.")

        # Keep the optimizer's t0 coordinate close to zero.  SciPy's default
        # finite-difference step is proportional to max(1, abs(x0)); using an
        # absolute BJD/HJD here can therefore make the first t0 derivative
        # many days wide.  The old explicit relative step had the opposite
        # problem for coordinates at zero (notably logrho=0), where it fell
        # back to a machine-scale perturbation.  Evaluate the model in the
        # translated time frame and restore the public t0 below.
        time_origin = float(q0[0])
        fit_time = time_np - time_origin
        z0 = np.asarray([0.0, np.log(abs(q0[1])), q0[2], q0[3]], dtype=float)

        def residual(z):
            q = self._decode(z)
            try:
                magnification = self._magnification(fit_time, q)
                fs, fb = _solve_fs_fb_numpy(magnification, flux_np, ferr_np)
                if not np.isfinite(fs) or not np.isfinite(fb):
                    raise ValueError
                values = (flux_np - (fs * magnification + fb)) / ferr_np
                if not np.all(np.isfinite(values)):
                    raise ValueError
                return values
            except Exception:
                return np.full(time_np.size, 1.0e50, dtype=float)

        def jacobian(z):
            q = self._decode(z)
            try:
                t0, tE, u0, logrho = map(float, q)
                u, du = _trajectory_gradient(fit_time, t0, tE, u0)
                rho = np.exp(np.clip(logrho, -50.0, 10.0))
                magnification, dA_du, dA_drho = _cpp_fspl_magnification_gradient(
                    u,
                    rho,
                    espl_table_path=self.espl_table_path,
                    tol=self.magnification_tol,
                    reltol=self.magnification_reltol,
                )
                in_range = -50.0 < logrho < 10.0
                dA_dlogrho = dA_drho * rho if in_range else np.zeros_like(dA_drho)
                gradient = np.column_stack([dA_du[:, None] * du, dA_dlogrho])
                value = _profiled_residual_jacobian(
                    magnification, gradient, flux_np, ferr_np
                )
                if not np.isfinite(value).all():
                    raise ValueError
                return value
            except Exception:
                return np.zeros((time_np.size, 4), dtype=float)

        def run_once(start: np.ndarray):
            result = _run_scipy_lm(
                residual,
                np.asarray(start, dtype=float),
                maxiter=self.maxiter,
                tol=self.tol,
                jac=jacobian,
            )
            q = self._decode(np.asarray(result.x, dtype=float))
            q[0] += time_origin
            values = residual(np.asarray(result.x, dtype=float))
            return result, q, float(np.dot(values, values))

        result, q, objective = run_once(z0)
        retry_count = 0
        initial_rho = np.exp(np.clip(q0[3], -50.0, 10.0))
        fitted_rho = np.exp(np.clip(q[3], -50.0, 10.0))
        # A finite-source fit can fall into the nearly-point-source basin when
        # the starting radius is on the large side.  Keep one canonical LM
        # implementation, but retry from a smaller radius when that diagnostic
        # is present; this is a basin rescue, not a second fitter/backend.
        if fitted_rho < 0.5 * initial_rho and np.isfinite(objective):
            retry = np.asarray(z0, dtype=float).copy()
            retry[3] -= np.log(10.0 / 3.0)
            retry_result, retry_q, retry_objective = run_once(retry)
            if np.isfinite(retry_objective) and retry_objective < objective:
                result, q, objective = retry_result, retry_q, retry_objective
                retry_count = 1
        if retry_count:
            result.message = f"{result.message}; basin_retry={retry_count}"
        return self._make_fit(time_np, flux_np, ferr_np, q, result)

    def _automatic_initial_guesses(
        self,
        time_np: np.ndarray,
        flux_np: np.ndarray,
        ferr_np: np.ndarray,
        *,
        pspl_params=None,
        observed_scale=None,
    ) -> tuple[np.ndarray, ...]:
        """Shared multistart set (duration-grid anchors + degeneracy family)."""
        return automatic_seeds(
            time_np,
            flux_np,
            ferr_np,
            finite_source=True,
            pspl_params=pspl_params,
            observed_scale=observed_scale,
            magnification_tol=self.magnification_tol,
            magnification_reltol=self.magnification_reltol,
            espl_table_path=self.espl_table_path,
        )

    def fit(
        self,
        time,
        flux,
        ferr,
        p0=None,
        *,
        pspl_params=None,
        observed_scale=None,
        mask=None,
    ) -> SingleLensFitResult:
        """Fit FSPL, using the observed-duration seed grid when ``p0`` is absent.

        ``mask`` (``True`` = excluded) removes points from the fit; the result
        is still evaluated on every point.
        """
        time_np, flux_np, ferr_np = _fit_arrays(time, flux, ferr)
        excluded = _excluded_mask(mask, time_np.size, 4)
        if excluded is not None:
            return masked_fit(
                self,
                lambda t, f, e: self.fit(
                    t, f, e, p0, pspl_params=pspl_params, observed_scale=observed_scale
                ),
                time_np, flux_np, ferr_np, excluded,
            )
        if p0 is None:
            seeds = self._automatic_initial_guesses(
                time_np,
                flux_np,
                ferr_np,
                pspl_params=pspl_params,
                observed_scale=observed_scale,
            )
            best_fit = best_fit_over_seeds(
                lambda seed: self._fit_from_seed(time_np, flux_np, ferr_np, seed),
                seeds,
                "FSPL",
            )
            self._last_fit = best_fit
            return best_fit
        return self._fit_from_seed(time_np, flux_np, ferr_np, p0)

    def evaluate_fixed(self, time, flux, ferr, p0, *, mask=None) -> SingleLensFitResult:
        time_np, flux_np, ferr_np = _fit_arrays(time, flux, ferr)
        excluded = _excluded_mask(mask, time_np.size, 4)
        q = np.asarray(p0, dtype=float).reshape(-1)
        if q.size != 4 or not np.all(np.isfinite(q)) or q[1] == 0.0:
            raise ValueError("FSPL fixed parameters must contain finite (t0, tE, u0, logrho).")
        fit = self._make_fit(
            time_np,
            flux_np,
            ferr_np,
            q,
            SimpleNamespace(
                fixed=True,
                success=True,
                status=0,
                nfev=0,
                message="fixed_parameters",
            ),
            excluded,
        )
        return fit

    def plot_lc(self, **kwargs):
        if self._last_fit is None:
            raise RuntimeError("No fit has been run yet.")
        return self.plotter.plot_lc(self._last_fit, **kwargs)

    def plot_residual(self, **kwargs):
        if self._last_fit is None:
            raise RuntimeError("No fit has been run yet.")
        return self.plotter.plot_residual(self._last_fit, **kwargs)


__all__ = [
    "PSPLFitter",
    "FSPLFitter",
    "PSPLParallaxFitter",
    "FSPLParallaxFitter",
]
