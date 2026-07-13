"""
Local shape templates for anomaly measurement.

Each template is a residual model ``r(t) = P(t) + sum_j B_j K_j(t; theta)``
with a low-order nuisance polynomial ``P`` and linear amplitudes ``B_j``
profiled out at fixed nonlinear parameters ``theta``.  The template's role is
to measure the well-determined observables of the anomaly — center time,
duration, and (for caustic crossings) the source-radius crossing time — not
to model the full binary-lens light curve.

Templates:

* ``bump`` — PSPL-shaped positive perturbation
  ``K = A0(sqrt(u_p^2 + ((t - t_c)/t_p)^2)) - 1``;
* ``dip`` — smoothed-box trough (minor-image demagnification between the
  triangular planetary caustics);
* ``fold`` — one finite-source fold crossing ``G0(±(t - t_c)/t_*)``;
* ``caustic_crossing`` — entry fold + exit fold + interior plateau;
* ``null`` — nuisance polynomial only (reference for ``delta_chi2`` and the
  "no coherent shape" outcome).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

try:
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover - scipy is optional at runtime
    minimize = None

from .fold_kernel import fold_g0
from .linear import polynomial_design, weighted_linear_fit
from .pspl import pspl_magnification_from_u
from .types import AnomalyShapeFit, PlanetClassConfig, SegmentData


def pspl_peak_magnification_inverse(A: np.ndarray) -> np.ndarray:
    """
    Invert ``A0(u)`` for ``A > 1``: ``u = sqrt(2 (A/sqrt(A^2-1) - 1))``.
    """
    A_arr = np.asarray(A, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = 2.0 * (A_arr / np.sqrt(A_arr * A_arr - 1.0) - 1.0)
    return np.sqrt(np.maximum(v, 0.0))


def pspl_bump_fwhm(t_p: float, u_p: float) -> float:
    """
    FWHM of the ``bump`` template profile ``A0(sqrt(u_p^2 + x^2)) - 1``.
    """
    peak = float(pspl_magnification_from_u(np.asarray([max(u_p, 1e-12)]))[0])
    half = 0.5 * (peak + 1.0)
    u_half = float(pspl_peak_magnification_inverse(np.asarray([half]))[0])
    return 2.0 * float(t_p) * float(np.sqrt(max(u_half * u_half - u_p * u_p, 0.0)))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def fit_null(segment: SegmentData, config: PlanetClassConfig, *, center: float) -> AnomalyShapeFit:
    t = np.asarray(segment.time, dtype=float)
    y = np.asarray(segment.residual, dtype=float)
    ferr = np.maximum(np.asarray(segment.ferr, dtype=float), 1e-12)
    poly = polynomial_design(t, center=center, order=int(config.polynomial_order))
    _coeff, _model, chi2, ok = weighted_linear_fit(poly, y, ferr)
    n_params = poly.shape[1]
    n_data = int(t.size)
    return AnomalyShapeFit(
        name="null",
        params={},
        param_errors=None,
        chi2=float(chi2),
        chi2_null=float(chi2),
        delta_chi2=0.0,
        bic=float(chi2 + n_params * np.log(max(n_data, 1))),
        n_data=n_data,
        n_params=n_params,
        success=bool(ok),
    )


def fit_shape_template(
    *,
    name: str,
    segment: SegmentData,
    config: PlanetClassConfig,
    center: float,
    chi2_null: float,
    theta0_list: Sequence[np.ndarray],
    bounds: list[tuple[float, float]],
    shape_from_theta: Callable[[np.ndarray, np.ndarray], np.ndarray],
    params_from_theta: Callable[[np.ndarray], dict[str, float]],
    expected_amplitude_signs: Optional[Sequence[float]] = None,
) -> AnomalyShapeFit:
    """
    Fit one template with profiled linear amplitudes and multi-start L-BFGS-B.
    """
    t = np.asarray(segment.time, dtype=float)
    y = np.asarray(segment.residual, dtype=float)
    ferr = np.maximum(np.asarray(segment.ferr, dtype=float), 1e-12)
    poly = polynomial_design(t, center=center, order=int(config.polynomial_order))

    def evaluate(theta: np.ndarray) -> tuple[float, np.ndarray, bool]:
        shape = np.asarray(shape_from_theta(theta, t), dtype=float)
        if shape.ndim == 1:
            shape = shape[:, None]
        design = np.column_stack((poly, shape))
        coeff, _model, chi2, ok = weighted_linear_fit(design, y, ferr)
        return float(chi2), coeff, bool(ok)

    def objective(theta: np.ndarray) -> float:
        chi2, _coeff, ok = evaluate(theta)
        return chi2 if ok and np.isfinite(chi2) else 1e300

    best: Optional[tuple[float, np.ndarray, np.ndarray, bool]] = None
    for theta0 in theta0_list:
        theta0 = np.clip(
            np.asarray(theta0, dtype=float),
            [lo for lo, _hi in bounds],
            [hi for _lo, hi in bounds],
        )
        if minimize is not None:
            opt = minimize(
                objective,
                theta0,
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": int(config.optimizer_maxiter),
                    "ftol": float(config.optimizer_ftol),
                },
            )
            theta = np.asarray(opt.x, dtype=float)
        else:
            theta = theta0
        chi2, coeff, ok = evaluate(theta)
        if best is None or chi2 < best[0]:
            best = (chi2, theta, coeff, ok)

    if best is None:  # pragma: no cover - theta0_list is never empty
        raise ValueError(f"Template {name} received no initial guesses.")
    chi2, theta, coeff, success = best
    success = bool(success and np.isfinite(chi2))

    params = {key: float(value) for key, value in params_from_theta(theta).items()}
    n_poly = poly.shape[1]
    amplitudes = coeff[n_poly:] if coeff.size > n_poly else np.asarray([], dtype=float)
    for j, value in enumerate(amplitudes, start=1):
        key = "amplitude" if amplitudes.size == 1 else f"amplitude_{j}"
        params[key] = float(value)

    warnings: list[str] = []
    if expected_amplitude_signs is not None and amplitudes.size:
        for expected, value in zip(expected_amplitude_signs, amplitudes):
            if expected != 0.0 and float(expected) * float(value) <= 0.0:
                warnings.append("template amplitude has unexpected sign")
                success = False
                break

    n_params = int(theta.size + coeff.size)
    n_data = int(t.size)
    param_errors = None
    if bool(config.estimate_param_errors) and success:
        param_errors = _parameter_errors(
            theta=theta,
            bounds=bounds,
            objective=objective,
            params_from_theta=params_from_theta,
            config=config,
        )

    return AnomalyShapeFit(
        name=name,
        params=params,
        param_errors=param_errors,
        chi2=float(chi2),
        chi2_null=float(chi2_null),
        delta_chi2=float(chi2_null - chi2),
        bic=float(chi2 + n_params * np.log(max(n_data, 1))),
        n_data=n_data,
        n_params=n_params,
        success=success,
        warnings=tuple(warnings),
    )


def _parameter_errors(
    *,
    theta: np.ndarray,
    bounds: list[tuple[float, float]],
    objective: Callable[[np.ndarray], float],
    params_from_theta: Callable[[np.ndarray], dict[str, float]],
    config: PlanetClassConfig,
) -> Optional[dict[str, float]]:
    covariance = _theta_covariance(theta, objective, bounds, config)
    if covariance is None:
        return None
    base = {
        key: float(value)
        for key, value in params_from_theta(theta).items()
        if np.isfinite(float(value))
    }
    if not base:
        return None
    keys = list(base)
    jac = np.zeros((len(keys), theta.size), dtype=float)
    steps = np.maximum(np.abs(theta) * float(config.covariance_step), float(config.covariance_step))
    for j in range(theta.size):
        tp = theta.copy()
        tm = theta.copy()
        tp[j] = min(theta[j] + steps[j], bounds[j][1])
        tm[j] = max(theta[j] - steps[j], bounds[j][0])
        denom = tp[j] - tm[j]
        if denom <= 0.0:
            continue
        pp = params_from_theta(tp)
        pm = params_from_theta(tm)
        for i, key in enumerate(keys):
            vp = float(pp.get(key, np.nan))
            vm = float(pm.get(key, np.nan))
            if np.isfinite(vp) and np.isfinite(vm):
                jac[i, j] = (vp - vm) / denom
    cov_params = jac @ covariance @ jac.T
    sigmas = np.sqrt(np.maximum(np.diag(cov_params), 0.0))
    errors = {key: float(s) for key, s in zip(keys, sigmas) if np.isfinite(s) and s > 0.0}
    return errors or None


def _theta_covariance(
    theta: np.ndarray,
    objective: Callable[[np.ndarray], float],
    bounds: list[tuple[float, float]],
    config: PlanetClassConfig,
) -> Optional[np.ndarray]:
    theta = np.asarray(theta, dtype=float)
    n = theta.size
    f0 = float(objective(theta))
    if n == 0 or not np.isfinite(f0):
        return None
    steps = np.maximum(np.abs(theta) * float(config.covariance_step), float(config.covariance_step))

    def clipped(base: np.ndarray, index: int, delta: float) -> np.ndarray:
        out = base.copy()
        out[index] = min(max(base[index] + delta, bounds[index][0]), bounds[index][1])
        return out

    hess = np.zeros((n, n), dtype=float)
    for i in range(n):
        xp = clipped(theta, i, steps[i])
        xm = clipped(theta, i, -steps[i])
        denom = (xp[i] - theta[i]) * (theta[i] - xm[i])
        fp = float(objective(xp))
        fm = float(objective(xm))
        if denom <= 0.0 or not np.isfinite(fp + fm):
            return None
        hess[i, i] = (fp - 2.0 * f0 + fm) / denom
        for j in range(i + 1, n):
            xpp = clipped(clipped(theta, i, steps[i]), j, steps[j])
            xpm = clipped(clipped(theta, i, steps[i]), j, -steps[j])
            xmp = clipped(clipped(theta, i, -steps[i]), j, steps[j])
            xmm = clipped(clipped(theta, i, -steps[i]), j, -steps[j])
            denom2 = (xpp[i] - xmp[i]) * (xpp[j] - xpm[j])
            if denom2 <= 0.0:
                return None
            val = (objective(xpp) - objective(xpm) - objective(xmp) + objective(xmm)) / denom2
            hess[i, j] = hess[j, i] = float(val)
    try:
        cov = 2.0 * np.linalg.pinv(hess)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(cov)):
        return None
    cond = np.linalg.cond(cov)
    if not np.isfinite(cond) or cond > float(config.covariance_max_condition):
        return None
    return cov


def fit_bump(
    segment: SegmentData,
    config: PlanetClassConfig,
    features: dict[str, float],
    *,
    chi2_null: float,
) -> AnomalyShapeFit:
    t = np.asarray(segment.time, dtype=float)
    t_peak = float(features.get("t_positive_peak", features.get("t_peak", t[t.size // 2])))
    cadence = max(float(features.get("cadence", 0.0)), 1e-8)
    duration = max(float(features.get("duration", cadence)), cadence)
    width = max(float(features.get("fwhm", cadence)), cadence)
    lo_t, hi_t = float(t[0]), float(t[-1])
    min_tp = max(0.3 * cadence, 1e-8)
    max_tp = 3.0 * duration
    log_up_bounds = (np.log(1e-3), np.log(5.0))
    theta0_list = [
        np.asarray([t_peak, np.log(min(max(0.5 * width, min_tp), max_tp)), np.log(up)], dtype=float)
        for up in (0.3, 1.0)
    ]
    theta0_list.append(
        np.asarray([t_peak, np.log(min(max(1.5 * width, min_tp), max_tp)), np.log(0.1)], dtype=float)
    )

    def shape(theta: np.ndarray, time: np.ndarray) -> np.ndarray:
        t_c, log_tp, log_up = theta
        x = (time - t_c) / np.exp(log_tp)
        u = np.sqrt(np.exp(2.0 * log_up) + x * x)
        return pspl_magnification_from_u(u) - 1.0

    def params(theta: np.ndarray) -> dict[str, float]:
        t_c, log_tp, log_up = (float(v) for v in theta)
        t_p = float(np.exp(log_tp))
        u_p = float(np.exp(log_up))
        return {
            "t_anom": t_c,
            "t_p": t_p,
            "u_p": u_p,
            "fwhm": pspl_bump_fwhm(t_p, u_p),
        }

    return fit_shape_template(
        name="bump",
        segment=segment,
        config=config,
        center=t_peak,
        chi2_null=chi2_null,
        theta0_list=theta0_list,
        bounds=[(lo_t, hi_t), (np.log(min_tp), np.log(max_tp)), log_up_bounds],
        shape_from_theta=shape,
        params_from_theta=params,
        expected_amplitude_signs=(1.0,),
    )


def fit_dip(
    segment: SegmentData,
    config: PlanetClassConfig,
    features: dict[str, float],
    *,
    chi2_null: float,
) -> AnomalyShapeFit:
    t = np.asarray(segment.time, dtype=float)
    t_dip = float(features.get("t_negative_peak", features.get("t_peak", t[t.size // 2])))
    cadence = max(float(features.get("cadence", 0.0)), 1e-8)
    duration = max(float(features.get("duration", cadence)), cadence)
    width = max(float(features.get("fwhm", cadence)), cadence)
    lo_t, hi_t = float(t[0]), float(t[-1])
    min_dt = max(cadence, 1e-8)
    max_dt = 2.0 * duration
    min_edge = max(0.25 * cadence, 1e-8)
    max_edge = duration
    theta0_list = [
        np.asarray(
            [t_dip, np.log(min(max(scale * width, min_dt), max_dt)), np.log(min(max(0.2 * width, min_edge), max_edge))],
            dtype=float,
        )
        for scale in (0.7, 1.0, 1.5)
    ]

    def shape(theta: np.ndarray, time: np.ndarray) -> np.ndarray:
        t_c, log_dt, log_edge = theta
        half = 0.5 * np.exp(log_dt)
        edge = np.exp(log_edge)
        return -(_sigmoid((time - (t_c - half)) / edge) * _sigmoid(((t_c + half) - time) / edge))

    def params(theta: np.ndarray) -> dict[str, float]:
        t_c, log_dt, log_edge = (float(v) for v in theta)
        dt = float(np.exp(log_dt))
        return {
            "t_anom": t_c,
            "dt_dip": dt,
            "edge_width": float(np.exp(log_edge)),
            "t_start": t_c - 0.5 * dt,
            "t_end": t_c + 0.5 * dt,
        }

    return fit_shape_template(
        name="dip",
        segment=segment,
        config=config,
        center=t_dip,
        chi2_null=chi2_null,
        theta0_list=theta0_list,
        bounds=[
            (lo_t, hi_t),
            (np.log(min_dt), np.log(max_dt)),
            (np.log(min_edge), np.log(max_edge)),
        ],
        shape_from_theta=shape,
        params_from_theta=params,
        expected_amplitude_signs=(1.0,),
    )


def fit_fold(
    segment: SegmentData,
    config: PlanetClassConfig,
    features: dict[str, float],
    *,
    chi2_null: float,
) -> AnomalyShapeFit:
    t = np.asarray(segment.time, dtype=float)
    t_peak = float(features.get("t_peak", t[t.size // 2]))
    cadence = max(float(features.get("cadence", 0.0)), 1e-8)
    duration = max(float(features.get("duration", cadence)), cadence)
    width = max(float(features.get("fwhm", cadence)), cadence)
    lo_t, hi_t = float(t[0]), float(t[-1])
    min_tstar = max(0.5 * cadence, 1e-8)
    max_tstar = 2.0 * duration

    best: Optional[AnomalyShapeFit] = None
    for sign in (1.0, -1.0):
        theta0_list = [
            np.asarray([t_peak, np.log(min(max(scale * width, min_tstar), max_tstar))], dtype=float)
            for scale in (0.5, 1.0, 2.0)
        ]

        def shape(theta: np.ndarray, time: np.ndarray, s: float = sign) -> np.ndarray:
            return fold_g0(s * (time - theta[0]) / np.exp(theta[1]))

        def params(theta: np.ndarray, s: float = sign) -> dict[str, float]:
            return {
                "t_anom": float(theta[0]),
                "tstar": float(np.exp(theta[1])),
                "entry_exit_sign": float(s),
            }

        fit = fit_shape_template(
            name="fold",
            segment=segment,
            config=config,
            center=t_peak,
            chi2_null=chi2_null,
            theta0_list=theta0_list,
            bounds=[(lo_t, hi_t), (np.log(min_tstar), np.log(max_tstar))],
            shape_from_theta=shape,
            params_from_theta=params,
            expected_amplitude_signs=(1.0,),
        )
        if best is None or fit.bic < best.bic:
            best = fit
    return best


def fit_caustic_crossing(
    segment: SegmentData,
    config: PlanetClassConfig,
    features: dict[str, float],
    *,
    chi2_null: float,
) -> AnomalyShapeFit:
    t = np.asarray(segment.time, dtype=float)
    z = np.asarray(segment.residual, dtype=float) / np.maximum(
        np.asarray(segment.ferr, dtype=float), 1e-12
    )
    cadence = max(float(features.get("cadence", 0.0)), 1e-8)
    duration = max(float(features.get("duration", cadence)), cadence)
    width = max(float(features.get("fwhm", cadence)), cadence)
    lo_t, hi_t = float(t[0]), float(t[-1])
    min_tstar = max(0.5 * cadence, 1e-8)
    max_tstar = max(0.5 * duration, min_tstar * 2.0)
    min_gap = max(3.0 * cadence, 2.0 * min_tstar)
    max_gap = max(duration, min_gap * 1.01)

    peak_times = sorted(
        float(p.time)
        for p in (*segment.component.peaks, *segment.component.dips)
        if np.isfinite(p.time)
    )
    pairs: list[tuple[float, float]] = []
    if len(peak_times) >= 2 and peak_times[-1] - peak_times[0] >= min_gap:
        pairs.append((peak_times[0], peak_times[-1]))
    t_max = float(t[int(np.argmax(z))])
    pairs.append((max(lo_t, t_max - 0.5 * duration), min(hi_t, t_max + 0.5 * duration)))
    pairs.append((lo_t + 0.15 * duration, hi_t - 0.15 * duration))

    theta0_list = []
    for a, b in pairs:
        gap = min(max(b - a, min_gap), max_gap)
        for scale in (0.5, 1.0):
            tstar = min(max(scale * width, min_tstar), max_tstar)
            theta0_list.append(
                np.asarray([a, np.log(gap), np.log(tstar), np.log(tstar)], dtype=float)
            )

    def shape(theta: np.ndarray, time: np.ndarray) -> np.ndarray:
        t_entry = theta[0]
        t_exit = t_entry + np.exp(theta[1])
        tstar_entry = np.exp(theta[2])
        tstar_exit = np.exp(theta[3])
        edge = 0.5 * (tstar_entry + tstar_exit)
        window = _sigmoid((time - t_entry) / edge) * _sigmoid((t_exit - time) / edge)
        return np.column_stack(
            (
                fold_g0((time - t_entry) / tstar_entry),
                fold_g0(-(time - t_exit) / tstar_exit),
                window,
            )
        )

    def params(theta: np.ndarray) -> dict[str, float]:
        t_entry = float(theta[0])
        gap = float(np.exp(theta[1]))
        return {
            "t_anom": t_entry + 0.5 * gap,
            "t_entry": t_entry,
            "t_exit": t_entry + gap,
            "dt_cc": gap,
            "tstar_entry": float(np.exp(theta[2])),
            "tstar_exit": float(np.exp(theta[3])),
        }

    return fit_shape_template(
        name="caustic_crossing",
        segment=segment,
        config=config,
        center=float(features.get("t_peak", t[t.size // 2])),
        chi2_null=chi2_null,
        theta0_list=theta0_list,
        bounds=[
            (lo_t, hi_t),
            (np.log(min_gap), np.log(max_gap)),
            (np.log(min_tstar), np.log(max_tstar)),
            (np.log(min_tstar), np.log(max_tstar)),
        ],
        shape_from_theta=shape,
        params_from_theta=params,
        expected_amplitude_signs=(1.0, 1.0, 0.0),
    )
