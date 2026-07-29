from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Callable, Any

import numpy as np
import jax.numpy as jnp

from .photometry import solve_fs_fb
from .plot import SingleLensPlotter
from .objective import residual_norm_from_A, chi2_from_res
from .singlelens_model import (
    A_pspl_func,
    A_fspl_logrho_func,
    A_fspl_logrho_peak_func,
    A_pspl_parallax_func,
    A_fspl_parallax_logrho_func,
    A_fspl_parallax_logrho_peak_func,
    A_pspl_space_parallax_func,
    A_fspl_space_parallax_logrho_func,
    A_cv_asymexp_logtau_func,
)
from .magnification import A_pspl_from_u, _get_fspl_disk
from .trajectory import make_parallax_projector, make_space_parallax_projector
from .parallax_backend import (
    NativePSPLAnnualParallaxFitter,
    NativeFSPLAnnualParallaxFitter,
    NativePSPLSpaceParallaxFitter,
    NativeFSPLSpaceParallaxFitter,
    Ephemeris,
    TimeSpec,
    default_earth_ephemeris,
    load_vbm_satellite_ephemeris,
)

try:
    from . import _cpp_grid
except ImportError:  # pragma: no cover - optional compiled backend
    _cpp_grid = None

try:
    from . import _vbm_cpp
except ImportError:  # pragma: no cover - optional VBMicrolensing compiled backend
    _vbm_cpp = None

try:
    from scipy.optimize import least_squares
except ImportError:  # pragma: no cover - optional VBM finite-difference backend
    least_squares = None

try:
    import VBMicrolensing
except ImportError:  # pragma: no cover - optional VBM finite-difference backend
    VBMicrolensing = None


def _make_vbm_magnifier(tol: float, reltol: float):
    if least_squares is None:
        raise ImportError("scipy is required for VBM finite-difference fitters.")
    if VBMicrolensing is None:
        raise ImportError("VBMicrolensing is required for VBM finite-difference fitters.")
    vbm = VBMicrolensing.VBMicrolensing()
    vbm.Tol = float(tol)
    vbm.RelTol = float(reltol)
    return vbm


def _vbm_espl_magnification(vbm, u: np.ndarray, rho: float) -> np.ndarray:
    rho_safe = max(float(rho), 1e-12)
    return np.asarray([vbm.ESPLMag(float(ui), rho_safe) for ui in u], dtype=float)


def _solve_fs_fb_numpy(A: np.ndarray, flux: np.ndarray, ferr: np.ndarray) -> tuple[float, float]:
    fe = np.maximum(ferr, 1e-12)
    w = 1.0 / (fe * fe)
    sw = np.sum(w)
    x_mean = np.sum(w * A) / sw
    y_mean = np.sum(w * flux) / sw
    xc = A - x_mean
    yc = flux - y_mean
    wxx = np.sum(w * xc * xc)
    if not np.isfinite(wxx) or wxx <= 0.0:
        return np.nan, np.nan
    fs = np.sum(w * xc * yc) / wxx
    fb = y_mean - fs * x_mean
    return float(fs), float(fb)


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
    optimizer_success: Optional[bool] = None
    optimizer_status: Optional[str] = None
    diagnostics: Optional[Any] = None


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
    optimizer_scale: Optional[jnp.ndarray] = None,
    positive_scale_indices: Tuple[int, ...] = (),
) -> SingleLensFitResult:
    """
    Shared fitting routine used by all single-lens fitters.

    This optimizes nonlinear parameters using SciPy's trust-region least
    squares solver, while solving
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
    if least_squares is None:
        raise ImportError("scipy is required for single-lens fitting.")

    def physical_residual_fun(x):
        A = build_A(jnp.asarray(x), time)
        return np.asarray(residual_norm_from_A(A, flux, ferr), dtype=float)

    locally_scaled = optimizer_scale is not None
    if locally_scaled:
        center = np.asarray(x0, dtype=float)
        scale = np.asarray(optimizer_scale, dtype=float)
        if scale.shape != center.shape or bool(np.any(scale <= 0.0)):
            raise ValueError("optimizer_scale must be positive and match x0.")
        positive_indices = tuple(int(index) for index in positive_scale_indices)
        if any(index < 0 or index >= center.size for index in positive_indices):
            raise ValueError("positive_scale_indices contains an invalid parameter index.")
        if any(float(center[index]) <= 0.0 for index in positive_indices):
            raise ValueError("Exponentially scaled parameters must start positive.")

        def unpack(z):
            value = center + scale * z
            for index in positive_indices:
                value[index] = center[index] * np.exp(z[index])
            return value

        def residual_fun(z):
            return physical_residual_fun(unpack(np.asarray(z, dtype=float)))

        result = least_squares(
            residual_fun,
            np.zeros_like(center),
            method="trf",
            jac="2-point",
            x_scale="jac",
            max_nfev=int(maxiter),
            xtol=float(tol),
            ftol=float(tol),
            gtol=float(tol),
        )
        x_np = unpack(np.asarray(result.x, dtype=float))
        optimizer_detail = "scaled_trf"
    else:
        result = least_squares(
            physical_residual_fun,
            np.asarray(x0, dtype=float),
            method="trf",
            jac="2-point",
            x_scale="jac",
            max_nfev=int(maxiter),
            xtol=float(tol),
            ftol=float(tol),
            gtol=float(tol),
        )
        x_np = np.asarray(result.x, dtype=float)
        optimizer_detail = "raw_trf"

    optimizer_success = bool(result.success and np.all(np.isfinite(x_np)))
    x = jnp.asarray(x_np, dtype=time.dtype)

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
        optimizer_success=optimizer_success,
        optimizer_status=(
            f"scipy_trf:status={int(result.status)};"
            f"nfev={int(result.nfev)};max_nfev={int(maxiter)};"
            f"{optimizer_detail};message={result.message}"
        ),
    )


def evaluate_single_lens_fixed(
    *,
    time: jnp.ndarray,
    flux: jnp.ndarray,
    ferr: jnp.ndarray,
    x0: jnp.ndarray,
    build_A: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    dof: int,
    param_names: Tuple[str, ...],
    x_to_params: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    min_points: int = 4,
    store_raw_params: bool = False,
    parallax_projector: Optional[Any] = None,
) -> SingleLensFitResult:
    """
    Evaluate a single-lens model at fixed nonlinear parameters.

    The nonlinear parameters in ``x0`` are not optimized. The linear flux
    parameters ``fs`` and ``fb`` are still solved analytically for the supplied
    light curve, matching the normal fitter convention.
    """
    n = int(time.shape[0])
    if n < min_points:
        raise ValueError(f"Need at least {min_points} data points, got {n}.")

    eps = 1e-12
    ferr = jnp.maximum(ferr, eps)

    x = jnp.asarray(x0, dtype=time.dtype)
    A = build_A(x, time)
    fs, fb = solve_fs_fb(A, flux, ferr)
    model_flux = fs * A + fb
    residual = flux - model_flux
    resn = residual / ferr
    chi2 = chi2_from_res(resn)
    chi2_dof = chi2 / max(n - dof, 1)

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
        optimizer_success=True,
        optimizer_status="fixed_parameters",
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
    C++ PSPL fitter.

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
    profile_peak_only: bool = False
    N_fft: int = 1024

    def __post_init__(self):
        self.plotter = SingleLensPlotter()
        self._last_fit: Optional[SingleLensFitResult] = None

    def fit(self, time: jnp.ndarray, flux: jnp.ndarray, ferr: jnp.ndarray, q0: jnp.ndarray) -> SingleLensFitResult:
        """Fit FSPL to a light curve (uses logrho parameterization)."""
        _get_fspl_disk()

        if self.profile_peak_only:
            q0_np = np.asarray(q0, dtype=float)
            rho0 = float(np.exp(np.clip(q0_np[3], -50.0, 10.0)))
            width = max(
                6.0 * rho0 * abs(float(q0_np[1])),
                2.0 * abs(float(q0_np[2])) * abs(float(q0_np[1])),
                0.5,
            )
            peak_indices = jnp.asarray(
                np.flatnonzero(np.abs(np.asarray(time, dtype=float) - q0_np[0]) <= width),
                dtype=jnp.int32,
            )
            if peak_indices.size == 0:
                peak_indices = jnp.asarray([int(np.argmin(np.abs(np.asarray(time, dtype=float) - q0_np[0])))])

            def build_A(q, t):
                return A_fspl_logrho_peak_func(q, t, peak_indices, N_fft=self.N_fft)
        else:
            def build_A(q, t):
                return A_fspl_logrho_func(q, t)

        def q_to_params(q):
            t0, tE, u0, logrho = q
            rho = jnp.exp(logrho)
            return jnp.array([t0, tE, u0, rho])

        q0_array = jnp.asarray(q0)
        optimizer_scale = jnp.asarray(
            [
                max(abs(float(np.asarray(q0_array[1]))), 1.0),
                1.0,
                max(
                    abs(float(np.asarray(q0_array[2]))),
                    float(np.exp(np.clip(float(np.asarray(q0_array[3])), -50.0, 10.0))),
                    1.0e-2,
                ),
                1.0,
            ],
            dtype=q0_array.dtype,
        )

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
            optimizer_scale=optimizer_scale,
            positive_scale_indices=(1,),
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
class CPPVBMFSPLFitter:
    """Native C++ non-parallax FSPL fitter.

    VBM is used only for finite-source magnification; its parallax coordinate
    conventions do not enter this model.
    """

    espl_table_path: Optional[str] = None
    maxiter: int = 300
    damping_parameter: float = 1.0e-4
    tol: float = 1.0e-5
    vbm_tol: float = 1.0e-4
    vbm_reltol: float = 1.0e-4

    def __post_init__(self):
        self.plotter = SingleLensPlotter()
        self._last_fit: Optional[SingleLensFitResult] = None
        if _vbm_cpp is None or not hasattr(_vbm_cpp, "fit_fspl"):
            raise ImportError(
                "CPPVBMFSPLFitter requires the native jacscanomaly._vbm_cpp "
                "extension with non-parallax FSPL support."
            )
        if VBMicrolensing is None:
            raise ImportError("CPPVBMFSPLFitter requires VBMicrolensing.")
        if self.espl_table_path is None:
            package_dir = Path(VBMicrolensing.__file__).resolve().parent
            self.espl_table_path = str(package_dir / "data" / "ESPL.tbl")

    def fit(
        self,
        time: jnp.ndarray,
        flux: jnp.ndarray,
        ferr: jnp.ndarray,
        q0: jnp.ndarray,
    ) -> SingleLensFitResult:
        time_np = np.asarray(time, dtype=float)
        flux_np = np.asarray(flux, dtype=float)
        ferr_np = np.maximum(np.asarray(ferr, dtype=float), 1.0e-12)
        q0_np = np.asarray(q0, dtype=float)
        if q0_np.shape != (4,):
            raise ValueError("q0 must be (t0, tE, u0, logrho).")
        (
            raw_params,
            fs,
            fb,
            chi2,
            model_flux,
            residual,
            converged,
            iterations,
        ) = _vbm_cpp.fit_fspl(
            time_np,
            flux_np,
            ferr_np,
            q0_np,
            espl_table=str(self.espl_table_path),
            maxiter=int(self.maxiter),
            damping_parameter=float(self.damping_parameter),
            tol=float(self.tol),
            vbm_tol=float(self.vbm_tol),
            vbm_reltol=float(self.vbm_reltol),
        )
        raw = np.asarray(raw_params, dtype=float)
        params = np.asarray([raw[0], raw[1], raw[2], np.exp(raw[3])], dtype=float)
        n = int(time_np.size)
        fit = SingleLensFitResult(
            time=time_np,
            flux=flux_np,
            ferr=ferr_np,
            params=jnp.asarray(params),
            param_names=("t0", "tE", "u0", "rho"),
            chi2=jnp.asarray(float(chi2)),
            chi2_dof=jnp.asarray(float(chi2) / max(n - 4, 1)),
            fs=jnp.asarray(float(fs)),
            fb=jnp.asarray(float(fb)),
            model_flux=jnp.asarray(model_flux),
            residual=jnp.asarray(residual),
            raw_params=jnp.asarray(raw),
            optimizer_success=bool(converged),
            optimizer_status=f"native_vbm_lm:iterations={int(iterations)}",
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
class VBMFiniteDiffFSPLFitter:
    """
    FSPL fitter with VBM magnification and finite-difference least squares.

    This is the non-parallax counterpart of
    ``VBMFiniteDiffGullsFSPLSpaceParallaxFitter``.
    """

    maxiter: int = 1000
    damping_parameter: float = 1e-6
    tol: float = 1e-6
    vbm_tol: float = 1e-4
    vbm_reltol: float = 1e-4

    def __post_init__(self):
        self.plotter = SingleLensPlotter()
        self._vbm = _make_vbm_magnifier(self.vbm_tol, self.vbm_reltol)
        self._last_fit: Optional[SingleLensFitResult] = None

    @staticmethod
    def _u_numpy(time: np.ndarray, q: np.ndarray) -> np.ndarray:
        t0, tE, u0, _logrho = q
        tE_safe = max(abs(float(tE)), 1e-12)
        tau = (time - t0) / tE_safe
        return np.sqrt(tau * tau + u0 * u0)

    def _magnification(self, u: np.ndarray, rho: float) -> np.ndarray:
        return _vbm_espl_magnification(self._vbm, u, rho)

    def _model_and_residual(
        self,
        q: np.ndarray,
        time: np.ndarray,
        flux: np.ndarray,
        ferr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        rho = np.exp(np.clip(float(q[3]), -50.0, 10.0))
        u = self._u_numpy(time, q)
        A = self._magnification(u, rho)
        fs, fb = _solve_fs_fb_numpy(A, flux, ferr)
        if not np.isfinite(fs) or not np.isfinite(fb):
            residual = np.full_like(flux, 1e100, dtype=float)
            model = np.full_like(flux, np.nan, dtype=float)
            return model, residual, np.inf, fs, fb
        model = fs * A + fb
        residual = flux - model
        resn = residual / np.maximum(ferr, 1e-12)
        chi2 = float(np.sum(resn * resn))
        return model, residual, chi2, fs, fb

    def fit(self, time: jnp.ndarray, flux: jnp.ndarray, ferr: jnp.ndarray, q0: jnp.ndarray) -> SingleLensFitResult:
        time_np = np.asarray(time, dtype=float)
        flux_np = np.asarray(flux, dtype=float)
        ferr_np = np.maximum(np.asarray(ferr, dtype=float), 1e-12)
        q0_np = np.asarray(q0, dtype=float)
        if time_np.size < 4:
            raise ValueError(f"Need at least 4 data points, got {time_np.size}.")

        def residual_fun(q):
            _model, residual, chi2, _fs, _fb = self._model_and_residual(q, time_np, flux_np, ferr_np)
            if not np.isfinite(chi2):
                return np.full_like(flux_np, 1e50, dtype=float)
            return residual / ferr_np

        lower = np.full(4, -np.inf, dtype=float)
        upper = np.full(4, np.inf, dtype=float)
        lower[1] = 1e-6
        lower[3] = -50.0
        upper[3] = 10.0
        result = least_squares(
            residual_fun,
            q0_np,
            jac="2-point",
            method="trf",
            bounds=(lower, upper),
            max_nfev=int(self.maxiter),
            xtol=float(self.tol),
            ftol=float(self.tol),
            gtol=float(self.tol),
        )
        q = np.asarray(result.x, dtype=float)
        model, residual, chi2, fs, fb = self._model_and_residual(q, time_np, flux_np, ferr_np)
        n = int(time_np.size)
        rho = float(np.exp(np.clip(q[3], -50.0, 10.0)))
        params = np.asarray([q[0], q[1], q[2], rho], dtype=float)
        fit = SingleLensFitResult(
            time=time_np,
            flux=flux_np,
            ferr=ferr_np,
            params=jnp.asarray(params),
            param_names=("t0", "tE", "u0", "rho"),
            chi2=jnp.asarray(chi2),
            chi2_dof=jnp.asarray(chi2 / max(n - 4, 1)),
            fs=jnp.asarray(fs),
            fb=jnp.asarray(fb),
            model_flux=jnp.asarray(model),
            residual=jnp.asarray(residual),
            raw_params=jnp.asarray(q),
            optimizer_success=bool(result.success),
            optimizer_status=str(result.message),
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
class CPPVBMFSPLParallaxFitter:
    """Native C++ LM fitter for finite-source annual-parallax point lenses.

    VBM evaluates the finite-source magnification at every original datum; no
    binning or peak-only data selection is used. The public parameter contract
    matches ``FSPLParallaxFitter``: ``(t0, tE, u0, rho, piEN, piEE)``.
    """

    coordinates: str
    sun_table_path: Optional[str] = None
    espl_table_path: Optional[str] = None
    maxiter: int = 200
    damping_parameter: float = 1e-4
    tol: float = 1e-5
    vbm_tol: float = 1e-4
    vbm_reltol: float = 1e-4
    max_piE: float = 5.0

    def __post_init__(self):
        self.plotter = SingleLensPlotter()
        self._last_fit: Optional[SingleLensFitResult] = None
        if VBMicrolensing is None:
            raise ImportError("CPPVBMFSPLParallaxFitter requires VBMicrolensing. Install jacscanomaly[vbm].")
        package_dir = Path(VBMicrolensing.__file__).resolve().parent
        if self.sun_table_path is None:
            self.sun_table_path = str(package_dir / "data" / "SunEphemeris.txt")
        if self.espl_table_path is None:
            self.espl_table_path = str(package_dir / "data" / "ESPL.tbl")

    def fit(self, time: jnp.ndarray, flux: jnp.ndarray, ferr: jnp.ndarray, p0: jnp.ndarray) -> SingleLensFitResult:
        if _vbm_cpp is None:
            raise RuntimeError(
                "CPPVBMFSPLParallaxFitter requires jacscanomaly._vbm_cpp. "
                "Reinstall jacscanomaly in an environment where VBMicrolensing is installed."
            )
        time_np = np.asarray(time, dtype=float)
        flux_np = np.asarray(flux, dtype=float)
        ferr_np = np.maximum(np.asarray(ferr, dtype=float), 1e-12)
        p0_np = np.asarray(p0, dtype=float)
        if p0_np.shape != (6,):
            raise ValueError("p0 must be (t0, tE, u0, logrho, piE_1, piE_2).")
        # The public Finder contract uses logrho.  The native wrapper accepts
        # physical rho and converts it to VBM's internal logrho coordinate.
        p0_vbm = p0_np.copy()
        p0_vbm[3] = np.exp(np.clip(p0_vbm[3], -50.0, 10.0))
        params, fs, fb, chi2, model_flux, residual = _vbm_cpp.fit_fspl_parallax(
            time_np, flux_np, ferr_np, p0_vbm,
            coordinates=str(self.coordinates),
            sun_table=str(self.sun_table_path),
            espl_table=str(self.espl_table_path),
            maxiter=int(self.maxiter),
            damping_parameter=float(self.damping_parameter),
            tol=float(self.tol),
            vbm_tol=float(self.vbm_tol),
            vbm_reltol=float(self.vbm_reltol),
            max_piE=float(self.max_piE),
        )
        n = int(time_np.size)
        fit = SingleLensFitResult(
            time=time_np, flux=flux_np, ferr=ferr_np,
            params=jnp.asarray(params),
            param_names=("t0", "tE", "u0", "rho", "piEN", "piEE"),
            chi2=jnp.asarray(float(chi2)),
            chi2_dof=jnp.asarray(float(chi2) / max(n - 6, 1)),
            fs=jnp.asarray(float(fs)), fb=jnp.asarray(float(fb)),
            model_flux=jnp.asarray(model_flux), residual=jnp.asarray(residual),
            raw_params=jnp.asarray(np.r_[np.asarray(params)[:3], np.log(max(float(params[3]), 1e-50)), np.asarray(params)[4:]]),
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
    use_HJD: bool = True
    maxiter: int = 1000
    damping_parameter: float = 1e-6
    tol: float = 1e-3

    def __post_init__(self):
        self.plotter = SingleLensPlotter()
        self._P = make_parallax_projector(self.RA, self.Dec, self.tref, use_HJD=self.use_HJD)
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
    use_HJD: bool = True
    maxiter: int = 1000
    damping_parameter: float = 1e-6
    tol: float = 1e-3
    peak_window_days: Optional[float] = None
    fspl_n_fft: int = 1024

    def __post_init__(self):
        self.plotter = SingleLensPlotter()
        self._P = make_parallax_projector(self.RA, self.Dec, self.tref, use_HJD=self.use_HJD)
        self._last_fit: Optional[SingleLensFitResult] = None

    def fit(self, time: jnp.ndarray, flux: jnp.ndarray, ferr: jnp.ndarray, q0: jnp.ndarray) -> SingleLensFitResult:
        """Fit FSPL+parallax to a light curve (uses logrho parameterization)."""
        _get_fspl_disk(self.fspl_n_fft)

        P = self._P

        if self.peak_window_days is None:
            def build_A(q, t):
                return A_fspl_parallax_logrho_func(q, t, P)
        else:
            if self.peak_window_days <= 0.:
                raise ValueError("peak_window_days must be positive when provided.")
            peak_indices = jnp.asarray(
                np.flatnonzero(np.abs(np.asarray(time) - float(q0[0])) <= self.peak_window_days),
                dtype=jnp.int32,
            )
            if peak_indices.size == 0:
                raise ValueError("peak_window_days selects no data points.")

            def build_A(q, t):
                return A_fspl_parallax_logrho_peak_func(
                    q, t, P, peak_indices, N_fft=self.fspl_n_fft)

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
class PSPLSpaceParallaxFitter:
    """
    PSPL + annual parallax + spacecraft parallax fitter.

    The spacecraft ephemeris is read from a VBMicrolensing/RTModel satellite
    table with columns ``JD RA_deg Dec_deg distance_AU``.
    """

    RA: float
    Dec: float
    tref: float
    satellite_ephemeris_path: str
    use_HJD: bool = True
    convention: str = "vbm"
    maxiter: int = 1000
    damping_parameter: float = 1e-6
    tol: float = 1e-3
    max_piE: float = 1.0

    def __post_init__(self):
        self.plotter = SingleLensPlotter()
        self._P = make_space_parallax_projector(
            self.RA, self.Dec, self.tref, self.satellite_ephemeris_path,
            use_HJD=self.use_HJD,
            convention=self.convention,
        )
        self._last_fit: Optional[SingleLensFitResult] = None

    def fit(self, time: jnp.ndarray, flux: jnp.ndarray, ferr: jnp.ndarray, p0: jnp.ndarray) -> SingleLensFitResult:
        P = self._P
        time_np = np.asarray(time, dtype=float)
        flux_np = np.asarray(flux, dtype=float)
        ferr_np = np.maximum(np.asarray(ferr, dtype=float), 1.0e-12)
        p0_np = np.asarray(p0, dtype=float)

        def model_and_residual(p):
            A = np.asarray(A_pspl_space_parallax_func(jnp.asarray(p), jnp.asarray(time_np), P), dtype=float)
            fs, fb = _solve_fs_fb_numpy(A, flux_np, ferr_np)
            residual = flux_np - (fs * A + fb)
            return residual / ferr_np, A, fs, fb

        if least_squares is None:
            raise RuntimeError("PSPLSpaceParallaxFitter requires scipy for bounded parallax fitting.")
        lower = np.full(5, -np.inf, dtype=float)
        upper = np.full(5, np.inf, dtype=float)
        lower[1] = 1.0e-6
        if np.isfinite(float(self.max_piE)) and float(self.max_piE) > 0.0:
            lower[3:5] = -float(self.max_piE)
            upper[3:5] = float(self.max_piE)
        result = least_squares(
            lambda p: model_and_residual(p)[0],
            p0_np,
            jac="2-point",
            method="trf",
            bounds=(lower, upper),
            max_nfev=int(self.maxiter),
            xtol=float(self.tol),
            ftol=float(self.tol),
            gtol=float(self.tol),
        )
        p = np.asarray(result.x, dtype=float)
        residual_norm, A, fs, fb = model_and_residual(p)
        residual = flux_np - (fs * A + fb)
        chi2 = float(np.sum(residual_norm * residual_norm))
        fit = SingleLensFitResult(
            time=time_np,
            flux=flux_np,
            ferr=ferr_np,
            params=jnp.asarray(p),
            param_names=("t0", "tE", "u0", "piEN", "piEE"),
            chi2=jnp.asarray(chi2),
            chi2_dof=jnp.asarray(chi2 / max(time_np.size - 5, 1)),
            fs=jnp.asarray(fs),
            fb=jnp.asarray(fb),
            model_flux=jnp.asarray(fs * A + fb),
            residual=jnp.asarray(residual),
            parallax_projector=P,
            optimizer_success=bool(result.success),
            optimizer_status=str(result.message),
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
class FSPLSpaceParallaxFitter:
    """
    FSPL + annual parallax + spacecraft parallax fitter.
    """

    RA: float
    Dec: float
    tref: float
    satellite_ephemeris_path: str
    use_HJD: bool = True
    convention: str = "vbm"
    maxiter: int = 1000
    damping_parameter: float = 1e-6
    tol: float = 1e-3

    def __post_init__(self):
        self.plotter = SingleLensPlotter()
        self._P = make_space_parallax_projector(
            self.RA, self.Dec, self.tref, self.satellite_ephemeris_path,
            use_HJD=self.use_HJD,
            convention=self.convention,
        )
        self._last_fit: Optional[SingleLensFitResult] = None

    def fit(self, time: jnp.ndarray, flux: jnp.ndarray, ferr: jnp.ndarray, q0: jnp.ndarray) -> SingleLensFitResult:
        _get_fspl_disk()

        P = self._P

        def build_A(q, t):
            return A_fspl_space_parallax_logrho_func(q, t, P)

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
        if self._last_fit is None:
            raise RuntimeError("No fit has been run yet.")
        return self.plotter.plot_lc(self._last_fit, **kwargs)

    def plot_residual(self, **kwargs):
        if self._last_fit is None:
            raise RuntimeError("No fit has been run yet.")
        return self.plotter.plot_residual(self._last_fit, **kwargs)


@dataclass
class VBMFiniteDiffGullsFSPLSpaceParallaxFitter:
    """
    GULLS-convention FSPL space-parallax fitter with VBM magnification.

    This keeps the GULLS trajectory calculation in NumPy, evaluates finite-source
    single-lens magnification with ``VBMicrolensing.ESPLMag(u, rho)``, and uses
    SciPy finite-difference least squares for the nonlinear parameters.
    """

    RA: float
    Dec: float
    tref: float
    satellite_ephemeris_path: str
    maxiter: int = 1000
    damping_parameter: float = 1e-6
    tol: float = 1e-6
    vbm_tol: float = 1e-4
    vbm_reltol: float = 1e-4
    max_piE: float = 1.0
    piE_prior_weight: float = 0.0
    piE_prior_eps: float = 1.0e-3

    def __post_init__(self):
        self.plotter = SingleLensPlotter()
        self._P = make_space_parallax_projector(
            self.RA,
            self.Dec,
            self.tref,
            self.satellite_ephemeris_path,
            use_HJD=False,
            convention="gulls",
        )
        self._vbm = _make_vbm_magnifier(self.vbm_tol, self.vbm_reltol)
        self._last_fit: Optional[SingleLensFitResult] = None

    def _gulls_u_numpy(self, time: np.ndarray, q: np.ndarray) -> np.ndarray:
        t0, tE, u0, _logrho, piEN, piEE = q
        P = self._P
        t = np.asarray(time, dtype=float) + float(np.asarray(P.time_add))
        t_grid = np.asarray(P.t, dtype=float)
        r_grid = np.asarray(P.r, dtype=float)
        r_t = np.column_stack([np.interp(t, t_grid, r_grid[:, j]) for j in range(3)])
        north = np.asarray(P.sky_north, dtype=float)
        east = np.asarray(P.sky_east, dtype=float)
        ne_t = np.column_stack([r_t @ north, r_t @ east])
        d_ne = (
            ne_t
            - np.asarray(P.NE_ref, dtype=float)[None, :]
            - (t - float(np.asarray(P.tref)))[:, None] * np.asarray(P.NE_vref, dtype=float)[None, :]
        )
        d_n = d_ne[:, 0]
        d_e = d_ne[:, 1]
        d_tau = -(piEN * d_n + piEE * d_e)
        d_beta = piEE * d_n - piEN * d_e
        tE_safe = max(abs(float(tE)), 1e-12)
        tau = (time - t0) / tE_safe + d_tau
        beta = u0 + d_beta
        return np.sqrt(tau * tau + beta * beta)

    def _magnification(self, u: np.ndarray, rho: float) -> np.ndarray:
        return _vbm_espl_magnification(self._vbm, u, rho)

    def _model_and_residual(
        self,
        q: np.ndarray,
        time: np.ndarray,
        flux: np.ndarray,
        ferr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        rho = np.exp(np.clip(float(q[3]), -50.0, 10.0))
        u = self._gulls_u_numpy(time, q)
        A = self._magnification(u, rho)
        fs, fb = _solve_fs_fb_numpy(A, flux, ferr)
        if not np.isfinite(fs) or not np.isfinite(fb):
            residual = np.full_like(flux, 1e100, dtype=float)
            model = np.full_like(flux, np.nan, dtype=float)
            return model, residual, np.inf, fs, fb
        model = fs * A + fb
        residual = flux - model
        resn = residual / np.maximum(ferr, 1e-12)
        chi2 = float(np.sum(resn * resn))
        return model, residual, chi2, fs, fb

    def _piE_prior_residual(self, q: np.ndarray) -> np.ndarray:
        weight = max(float(self.piE_prior_weight), 0.0)
        if weight <= 0.0:
            return np.empty(0, dtype=float)
        piE = float(np.hypot(float(q[4]), float(q[5])))
        eps = max(float(self.piE_prior_eps), 1e-12)
        return np.asarray([np.sqrt(weight * max(piE, eps))], dtype=float)

    def _penalized_chi2(self, chi2: float, q: np.ndarray) -> float:
        prior = self._piE_prior_residual(q)
        if prior.size == 0:
            return float(chi2)
        return float(chi2) + float(np.sum(prior * prior))

    def fit(self, time: jnp.ndarray, flux: jnp.ndarray, ferr: jnp.ndarray, q0: jnp.ndarray) -> SingleLensFitResult:
        time_np = np.asarray(time, dtype=float)
        flux_np = np.asarray(flux, dtype=float)
        ferr_np = np.maximum(np.asarray(ferr, dtype=float), 1e-12)
        q0_np = np.asarray(q0, dtype=float)
        if time_np.size < 7:
            raise ValueError(f"Need at least 7 data points, got {time_np.size}.")

        def residual_fun(q):
            _model, residual, chi2, _fs, _fb = self._model_and_residual(q, time_np, flux_np, ferr_np)
            if not np.isfinite(chi2):
                return np.full_like(flux_np, 1e50, dtype=float)
            return np.r_[residual / ferr_np, self._piE_prior_residual(q)]

        lower = np.full(6, -np.inf, dtype=float)
        upper = np.full(6, np.inf, dtype=float)
        lower[1] = 1e-6
        lower[3] = -50.0
        upper[3] = 10.0
        if np.isfinite(float(self.max_piE)) and float(self.max_piE) > 0.0:
            lower[4:6] = -float(self.max_piE)
            upper[4:6] = float(self.max_piE)
        result = least_squares(
            residual_fun,
            q0_np,
            jac="2-point",
            method="trf",
            bounds=(lower, upper),
            max_nfev=int(self.maxiter),
            xtol=float(self.tol),
            ftol=float(self.tol),
            gtol=float(self.tol),
        )
        q = np.asarray(result.x, dtype=float)
        model, residual, chi2, fs, fb = self._model_and_residual(q, time_np, flux_np, ferr_np)
        chi2_fit = self._penalized_chi2(chi2, q)
        n = int(time_np.size)
        rho = float(np.exp(np.clip(q[3], -50.0, 10.0)))
        params = np.asarray([q[0], q[1], q[2], rho, q[4], q[5]], dtype=float)
        fit = SingleLensFitResult(
            time=time_np,
            flux=flux_np,
            ferr=ferr_np,
            params=jnp.asarray(params),
            param_names=("t0", "tE", "u0", "rho", "piEN", "piEE"),
            chi2=jnp.asarray(chi2_fit),
            chi2_dof=jnp.asarray(chi2_fit / max(n - 6, 1)),
            fs=jnp.asarray(fs),
            fb=jnp.asarray(fb),
            model_flux=jnp.asarray(model),
            residual=jnp.asarray(residual),
            raw_params=jnp.asarray(q),
            parallax_projector=self._P,
            optimizer_success=bool(result.success),
            optimizer_status=str(result.message),
        )
        self._last_fit = fit
        return fit

    def plot_lc(self, **kwargs):
        if self._last_fit is None:
            raise RuntimeError("No fit has been run yet.")
        return self.plotter.plot_lc(self._last_fit, **kwargs)


@dataclass
class BICSingleLensFitter:
    """
    Select PSPL, FSPL, or native FSPL space-parallax by BIC.
    """

    RA: float
    Dec: float
    tref: float
    satellite_ephemeris_path: str
    max_piE: float = 1.0
    piE_prior_weight: float = 0.0
    piE_prior_eps: float = 1.0e-3
    include_space_parallax: bool = False
    observer_convention: str = "earth_geocentric_offset"
    time_scale: str = "jd"
    time_offset: float = 0.0
    ephemeris_extrapolation: str = "reject"

    def __post_init__(self):
        self.plotter = SingleLensPlotter()
        self._last_fit: Optional[SingleLensFitResult] = None

    @staticmethod
    def _annotate_fit(
        fit: SingleLensFitResult,
        *,
        model_kind: str,
        bic: float,
        model_selection: dict,
    ) -> SingleLensFitResult:
        object.__setattr__(fit, "model_kind", model_kind)
        object.__setattr__(fit, "bic", float(bic))
        object.__setattr__(fit, "model_selection", model_selection)
        return fit

    def _initial_values(self, q0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = np.asarray(q0, dtype=float).ravel()
        t0 = float(q[0])
        tE = float(q[1])
        u0 = float(q[2])
        if q.size >= 4:
            raw_rho = float(q[3])
            logrho = raw_rho if raw_rho <= 0.0 else float(np.log(max(raw_rho, 1e-12)))
        else:
            logrho = -7.0
        if q.size >= 6:
            piEN = float(q[4])
            piEE = float(q[5])
        else:
            piEN = 0.0
            piEE = 0.0
        return (
            np.asarray([t0, tE, u0], dtype=float),
            np.asarray([t0, tE, u0, logrho], dtype=float),
            np.asarray([t0, tE, u0, logrho, piEN, piEE], dtype=float),
        )

    def _candidate_specs(self, q0: np.ndarray):
        """Return the model candidates and their starting vectors.

        Keeping this construction in one place is important for masked
        refits: a BIC-selected model can be refit without silently selecting a
        different model family after the anomaly points have been removed.
        """
        q_pspl, q_fspl, q_space = self._initial_values(np.asarray(q0, dtype=float))
        q0_size = int(np.asarray(q0, dtype=float).size)
        if q0_size == 4:
            candidates = [("fspl_vbm_fd", VBMFiniteDiffFSPLFitter(), q_fspl)]
        else:
            candidates = [
                ("pspl", PSPLFitter(), q_pspl),
                ("fspl_vbm_fd", VBMFiniteDiffFSPLFitter(), q_fspl),
            ]
        if bool(self.include_space_parallax) and q0_size != 4:
            time_spec = TimeSpec(scale=self.time_scale, offset=float(self.time_offset))
            earth = default_earth_ephemeris(time_spec=time_spec)
            satellite = load_vbm_satellite_ephemeris(
                self.satellite_ephemeris_path,
                time_spec=time_spec,
                extrapolation=self.ephemeris_extrapolation,
            )
            observer = satellite
            reference = None
            if self.observer_convention != "earth_geocentric_offset":
                earth_r = np.column_stack(
                    [
                        np.interp(satellite.time, earth.time, earth.position_au[:, axis])
                        for axis in range(3)
                    ]
                )
                observer = Ephemeris(
                    satellite.time,
                    earth_r + satellite.position_au,
                    origin="explicit_reference",
                    time_spec=time_spec,
                    extrapolation=self.ephemeris_extrapolation,
                )
                reference = earth
            candidates.append(
                (
                    "fspl_space_parallax",
                    NativeFSPLSpaceParallaxFitter(
                        float(self.RA),
                        float(self.Dec),
                        float(self.tref),
                        observer,
                        convention=self.observer_convention,
                        time_spec=time_spec,
                        earth_ephemeris=earth,
                        reference_ephemeris=reference,
                        max_piE=float(self.max_piE),
                    ),
                    q_space,
                )
            )
        return candidates

    def fit_fixed_model(
        self,
        time: jnp.ndarray,
        flux: jnp.ndarray,
        ferr: jnp.ndarray,
        q0: jnp.ndarray,
        model_kind: str,
    ) -> SingleLensFitResult:
        """Refit a previously selected model without rerunning model selection.

        Planet-signal extraction masks the anomaly before refitting the
        single-lens baseline.  Running BIC again on that masked data can turn a
        well-supported FSPL baseline into PSPL simply because the finite-source
        information was inside the mask.  This method preserves the original
        model family for that refit.
        """
        time_np = np.asarray(time, dtype=float)
        candidates = {kind: (fitter, x0) for kind, fitter, x0 in self._candidate_specs(np.asarray(q0, dtype=float))}
        if model_kind not in candidates:
            raise ValueError(
                f"Model '{model_kind}' is not available for this parameter vector; "
                f"available models are {sorted(candidates)}."
            )
        fitter, x0 = candidates[model_kind]
        # Masked refits see a different effective light curve and can fall
        # into the same FSPL local basin as the initial full-data fit.  Use
        # the identical guarded candidate path here so a selected FSPL model
        # gets the small multi-start retry set as well.
        fit = self._fit_candidate(
            model_kind,
            fitter,
            x0,
            time,
            flux,
            ferr,
        )
        n = max(int(time_np.size), 1)
        k = len(tuple(fit.param_names)) + 2
        bic = float(np.asarray(fit.chi2)) + float(k) * float(np.log(n))
        selection = {"selected": model_kind, "bic": {model_kind: float(bic)}, "errors": {}}
        return self._annotate_fit(fit, model_kind=model_kind, bic=bic, model_selection=selection)

    @staticmethod
    def _fspl_retry_seeds(x0: np.ndarray) -> tuple[np.ndarray, ...]:
        """Generate a small set of FSPL starts for a poor local solution."""
        base = np.asarray(x0, dtype=float).copy()
        seeds = []
        # The source radius and tE are correlated near a central peak.  These
        # modest perturbations escape the common high-u0/small-rho local basin
        # without turning every survey fit into a large multistart search.
        for te_factor, rho_factor, u0_sign in (
            (0.70, 1.0, 1.0),
            (1.35, 1.0, 1.0),
            (1.0, 0.50, 1.0),
            (1.0, 2.00, 1.0),
            (1.0, 1.0, -1.0),
        ):
            seed = base.copy()
            seed[1] = max(abs(seed[1]) * te_factor, 1e-6)
            seed[2] = seed[2] * u0_sign
            seed[3] = seed[3] + np.log(rho_factor)
            seeds.append(seed)
        return tuple(seeds)

    def _fit_candidate(
        self,
        model_kind: str,
        fitter,
        x0: np.ndarray,
        time: jnp.ndarray,
        flux: jnp.ndarray,
        ferr: jnp.ndarray,
    ) -> SingleLensFitResult:
        """Fit one BIC candidate, retrying only a poor FSPL local solution."""
        fit = fitter.fit(time, flux, ferr, jnp.asarray(x0, dtype=float))
        if model_kind != "fspl_vbm_fd":
            return fit
        if not np.isfinite(float(np.asarray(fit.chi2_dof))) or float(np.asarray(fit.chi2_dof)) <= 10.0:
            return fit

        best = fit
        best_chi2 = float(np.asarray(fit.chi2))
        for seed in self._fspl_retry_seeds(x0):
            try:
                trial = VBMFiniteDiffFSPLFitter().fit(
                    time, flux, ferr, jnp.asarray(seed, dtype=float)
                )
                trial_chi2 = float(np.asarray(trial.chi2))
                if np.isfinite(trial_chi2) and trial_chi2 < best_chi2:
                    best = trial
                    best_chi2 = trial_chi2
            except Exception:
                continue
        return best

    def fit(self, time: jnp.ndarray, flux: jnp.ndarray, ferr: jnp.ndarray, q0: jnp.ndarray) -> SingleLensFitResult:
        time_np = np.asarray(time, dtype=float)
        n = max(int(time_np.size), 1)
        trials: list[tuple[str, SingleLensFitResult, float]] = []
        errors: dict[str, str] = {}

        candidates = self._candidate_specs(np.asarray(q0, dtype=float))

        for model_kind, fitter, x0 in candidates:
            try:
                fit = self._fit_candidate(model_kind, fitter, x0, time, flux, ferr)
                k = len(tuple(fit.param_names)) + 2
                bic = float(np.asarray(fit.chi2)) + float(k) * float(np.log(n))
                if np.isfinite(bic):
                    trials.append((model_kind, fit, bic))
            except Exception as exc:  # pragma: no cover - depends on optional backends
                errors[model_kind] = f"{type(exc).__name__}: {exc}"

        if not trials:
            raise RuntimeError(f"All BIC single-lens fits failed: {errors}")

        model_kind, fit, bic = min(trials, key=lambda item: item[2])
        selection = {
            "selected": model_kind,
            "bic": {kind: float(value) for kind, _fit, value in trials},
            "errors": errors,
        }
        fit = self._annotate_fit(fit, model_kind=model_kind, bic=bic, model_selection=selection)
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
class VBMFiniteDiffGullsPSPLSpaceParallaxFitter:
    """Bounded GULLS-convention PSPL space-parallax fitter.

    This is the point-source counterpart of the VBM finite-difference FSPL
    fitter.  It shares the GULLS observer-frame trajectory but avoids the
    unconstrained JAX LM path, which can otherwise run to a high-parallax
    degenerate basin on contaminated light curves.
    """

    RA: float
    Dec: float
    tref: float
    satellite_ephemeris_path: str
    maxiter: int = 500
    tol: float = 1.0e-6
    max_piE: float = 1.0

    def __post_init__(self):
        self.plotter = SingleLensPlotter()
        self._P = make_space_parallax_projector(
            self.RA, self.Dec, self.tref, self.satellite_ephemeris_path,
            use_HJD=False, convention="gulls",
        )
        self._last_fit: Optional[SingleLensFitResult] = None

    def _gulls_u_numpy(self, time: np.ndarray, p: np.ndarray) -> np.ndarray:
        t0, tE, u0, piEN, piEE = p
        P = self._P
        t = np.asarray(time, dtype=float) + float(np.asarray(P.time_add))
        t_grid = np.asarray(P.t, dtype=float)
        r_grid = np.asarray(P.r, dtype=float)
        r_t = np.column_stack([np.interp(t, t_grid, r_grid[:, j]) for j in range(3)])
        north = np.asarray(P.sky_north, dtype=float)
        east = np.asarray(P.sky_east, dtype=float)
        ne_t = np.column_stack([r_t @ north, r_t @ east])
        d_ne = (
            ne_t
            - np.asarray(P.NE_ref, dtype=float)[None, :]
            - (t - float(np.asarray(P.tref)))[:, None] * np.asarray(P.NE_vref, dtype=float)[None, :]
        )
        d_n, d_e = d_ne[:, 0], d_ne[:, 1]
        tau = (time - t0) / max(abs(float(tE)), 1.0e-12) - piEN * d_n - piEE * d_e
        beta = u0 + piEE * d_n - piEN * d_e
        return np.sqrt(tau * tau + beta * beta)

    def fit(self, time: jnp.ndarray, flux: jnp.ndarray, ferr: jnp.ndarray, p0: jnp.ndarray) -> SingleLensFitResult:
        if least_squares is None:
            raise RuntimeError("VBMFiniteDiffGullsPSPLSpaceParallaxFitter requires scipy.")
        time_np = np.asarray(time, dtype=float)
        flux_np = np.asarray(flux, dtype=float)
        ferr_np = np.maximum(np.asarray(ferr, dtype=float), 1.0e-12)

        def model_and_residual(p):
            u = self._gulls_u_numpy(time_np, p)
            A = np.asarray(A_pspl_from_u(jnp.asarray(u)), dtype=float)
            fs, fb = _solve_fs_fb_numpy(A, flux_np, ferr_np)
            residual = flux_np - (fs * A + fb)
            return residual / ferr_np, A, fs, fb

        lower = np.full(5, -np.inf, dtype=float)
        upper = np.full(5, np.inf, dtype=float)
        lower[1] = 1.0e-6
        if np.isfinite(float(self.max_piE)) and float(self.max_piE) > 0.0:
            lower[3:5] = -float(self.max_piE)
            upper[3:5] = float(self.max_piE)
        result = least_squares(
            lambda p: model_and_residual(p)[0], np.asarray(p0, dtype=float),
            jac="2-point", method="trf", bounds=(lower, upper),
            max_nfev=int(self.maxiter), xtol=float(self.tol),
            ftol=float(self.tol), gtol=float(self.tol),
        )
        p = np.asarray(result.x, dtype=float)
        residual_norm, A, fs, fb = model_and_residual(p)
        residual = flux_np - (fs * A + fb)
        chi2 = float(np.sum(residual_norm * residual_norm))
        fit = SingleLensFitResult(
            time=time_np, flux=flux_np, ferr=ferr_np,
            params=jnp.asarray(p),
            param_names=("t0", "tE", "u0", "piEN", "piEE"),
            chi2=jnp.asarray(chi2),
            chi2_dof=jnp.asarray(chi2 / max(time_np.size - 5, 1)),
            fs=jnp.asarray(fs), fb=jnp.asarray(fb),
            model_flux=jnp.asarray(fs * A + fb), residual=jnp.asarray(residual),
            parallax_projector=self._P,
            optimizer_success=bool(result.success),
            optimizer_status=str(result.message),
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
