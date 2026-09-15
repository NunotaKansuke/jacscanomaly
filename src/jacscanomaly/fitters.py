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

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import numpy as np
import jax.numpy as jnp

from .plot import SingleLensPlotter
from .singlelens_fit import SingleLensFitResult
from .parallax_backend import (
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


def _pspl_magnification(time: np.ndarray, t0: float, tE: float, u0: float) -> np.ndarray:
    tE_safe = max(abs(float(tE)), 1.0e-12)
    tau = (np.asarray(time, dtype=float) - float(t0)) / tE_safe
    u = np.sqrt(tau * tau + float(u0) * float(u0))
    u_safe = np.maximum(u, 1.0e-12)
    return (u_safe * u_safe + 2.0) / (
        u_safe * np.sqrt(u_safe * u_safe + 4.0)
    )


def _run_scipy_lm(
    residual,
    initial: np.ndarray,
    *,
    maxiter: int,
    tol: float,
    diff_step: Optional[float] = None,
):
    if least_squares is None:
        raise ImportError("scipy is required for single-lens fitting.")
    tolerance = max(float(tol), 1.0e-14)
    kwargs = {
        "jac": "2-point",
        "method": "lm",
        "max_nfev": max(1, int(maxiter)),
        "xtol": tolerance,
        "ftol": tolerance,
        "gtol": tolerance,
    }
    if diff_step is not None:
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
    def _decode(z: np.ndarray) -> np.ndarray:
        value = np.asarray(z, dtype=float)
        return np.asarray(
            [value[0], np.exp(np.clip(value[1], -50.0, 50.0)), value[2]],
            dtype=float,
        )

    def _magnification(self, time: np.ndarray, q: np.ndarray) -> np.ndarray:
        return _pspl_magnification(time, q[0], q[1], q[2])

    def _make_fit(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        ferr: np.ndarray,
        q: np.ndarray,
        result,
    ) -> SingleLensFitResult:
        magnification = self._magnification(time, q)
        fs, fb = _solve_fs_fb_numpy(magnification, flux, ferr)
        if not np.isfinite(fs) or not np.isfinite(fb):
            raise RuntimeError("PSPL profiled flux solve is singular.")
        model = fs * magnification + fb
        residual = flux - model
        normalized = residual / ferr
        chi2 = float(np.dot(normalized, normalized))
        fit = SingleLensFitResult(
            time=time,
            flux=flux,
            ferr=ferr,
            params=jnp.asarray(q),
            param_names=("t0", "tE", "u0"),
            chi2=jnp.asarray(chi2),
            chi2_dof=jnp.asarray(chi2 / max(time.size - 3, 1)),
            fs=jnp.asarray(fs),
            fb=jnp.asarray(fb),
            model_flux=jnp.asarray(model),
            residual=jnp.asarray(residual),
            model_kind="pspl",
            optimizer_success=bool(result.success),
            optimizer_status=_optimizer_status(result, maxiter=self.maxiter),
        )
        self._last_fit = fit
        return fit

    def fit(self, time, flux, ferr, p0) -> SingleLensFitResult:
        time_np, flux_np, ferr_np = _fit_arrays(time, flux, ferr)
        q0 = np.asarray(p0, dtype=float).reshape(-1)
        if q0.size != 3:
            raise ValueError("PSPL p0 must contain (t0, tE, u0).")
        if not np.all(np.isfinite(q0)) or q0[1] == 0.0:
            raise ValueError("PSPL p0 must be finite and have nonzero tE.")
        z0 = np.asarray([q0[0], np.log(abs(q0[1])), q0[2]], dtype=float)

        def residual(z):
            q = self._decode(z)
            magnification = self._magnification(time_np, q)
            fs, fb = _solve_fs_fb_numpy(magnification, flux_np, ferr_np)
            if not np.isfinite(fs) or not np.isfinite(fb):
                return np.full(time_np.size, 1.0e50, dtype=float)
            values = (flux_np - (fs * magnification + fb)) / ferr_np
            return values if np.all(np.isfinite(values)) else np.full_like(values, 1.0e50)

        result = _run_scipy_lm(
            residual,
            z0,
            maxiter=self.maxiter,
            tol=self.tol,
        )
        q = self._decode(np.asarray(result.x, dtype=float))
        return self._make_fit(time_np, flux_np, ferr_np, q, result)

    def evaluate_fixed(self, time, flux, ferr, p0) -> SingleLensFitResult:
        time_np, flux_np, ferr_np = _fit_arrays(time, flux, ferr)
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
        )

    def fit_fixed_model(self, time, flux, ferr, p0, *, model_kind=None):
        """Evaluate fixed nonlinear parameters through the common fitter API."""
        if model_kind not in {None, "pspl"}:
            raise ValueError(f"PSPLFitter cannot evaluate model_kind={model_kind!r}.")
        return self.evaluate_fixed(time, flux, ferr, p0)

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
    ) -> SingleLensFitResult:
        magnification = self._magnification(time, q)
        fs, fb = _solve_fs_fb_numpy(magnification, flux, ferr)
        if not np.isfinite(fs) or not np.isfinite(fb):
            raise RuntimeError("FSPL profiled flux solve is singular.")
        model = fs * magnification + fb
        residual = flux - model
        normalized = residual / ferr
        chi2 = float(np.dot(normalized, normalized))
        params = np.asarray([q[0], q[1], q[2], np.exp(np.clip(q[3], -50.0, 10.0))])
        fit = SingleLensFitResult(
            time=time,
            flux=flux,
            ferr=ferr,
            params=jnp.asarray(params),
            param_names=("t0", "tE", "u0", "rho"),
            chi2=jnp.asarray(chi2),
            chi2_dof=jnp.asarray(chi2 / max(time.size - 4, 1)),
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
        )
        self._last_fit = fit
        return fit

    def fit_fixed_model(self, time, flux, ferr, p0, *, model_kind=None):
        """Evaluate fixed nonlinear parameters through the common fitter API."""
        if model_kind not in {None, "fspl"}:
            raise ValueError(f"FSPLFitter cannot evaluate model_kind={model_kind!r}.")
        return self.evaluate_fixed(time, flux, ferr, p0)

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
        z0 = np.asarray([q0[0], np.log(abs(q0[1])), q0[2], q0[3]], dtype=float)

        def residual(z):
            q = self._decode(z)
            try:
                magnification = self._magnification(time_np, q)
                fs, fb = _solve_fs_fb_numpy(magnification, flux_np, ferr_np)
                if not np.isfinite(fs) or not np.isfinite(fb):
                    raise ValueError
                values = (flux_np - (fs * magnification + fb)) / ferr_np
                if not np.all(np.isfinite(values)):
                    raise ValueError
                return values
            except Exception:
                return np.full(time_np.size, 1.0e50, dtype=float)

        def run_once(start: np.ndarray):
            result = _run_scipy_lm(
                residual,
                np.asarray(start, dtype=float),
                maxiter=self.maxiter,
                tol=self.tol,
                # ESPL table interpolation is accurate but not smooth at the
                # scale of SciPy's default finite-difference step.  A
                # deliberate relative step keeps LM derivatives informative
                # near a source crossing while retaining the same optimizer.
                diff_step=1.0e-3,
            )
            q = self._decode(np.asarray(result.x, dtype=float))
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
    ) -> tuple[np.ndarray, ...]:
        """Build duration-grid FSPL seeds for a fit without an explicit p0."""
        from .fspl_initialization import fspl_template_initial_guesses

        proxy = SimpleNamespace(
            time=time_np,
            flux=flux_np,
            ferr=ferr_np,
            params=pspl_params,
        )
        return fspl_template_initial_guesses(
            proxy,
            top_k=4,
            pspl_params=pspl_params,
            magnification_tol=self.magnification_tol,
            magnification_reltol=self.magnification_reltol,
            espl_table_path=self.espl_table_path,
        )

    def fit(self, time, flux, ferr, p0=None, *, pspl_params=None) -> SingleLensFitResult:
        """Fit FSPL, using the observed-duration seed grid when ``p0`` is absent."""
        time_np, flux_np, ferr_np = _fit_arrays(time, flux, ferr)
        if p0 is None:
            seeds = self._automatic_initial_guesses(
                time_np,
                flux_np,
                ferr_np,
                pspl_params=pspl_params,
            )
            best_fit = None
            best_chi2 = np.inf
            errors = []
            for seed in seeds:
                try:
                    candidate = self._fit_from_seed(
                        time_np,
                        flux_np,
                        ferr_np,
                        seed,
                    )
                    chi2 = float(np.asarray(candidate.chi2))
                except Exception as exc:
                    errors.append(exc)
                    continue
                if np.isfinite(chi2) and chi2 < best_chi2:
                    best_fit = candidate
                    best_chi2 = chi2
            if best_fit is None:
                message = "All automatic FSPL initial guesses failed."
                if errors:
                    message += f" First error: {errors[0]}"
                raise RuntimeError(message)
            self._last_fit = best_fit
            return best_fit
        return self._fit_from_seed(time_np, flux_np, ferr_np, p0)

    def evaluate_fixed(self, time, flux, ferr, p0) -> SingleLensFitResult:
        time_np, flux_np, ferr_np = _fit_arrays(time, flux, ferr)
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
