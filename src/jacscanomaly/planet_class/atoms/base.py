from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover - scipy is optional at runtime
    minimize = None

from ..linear import polynomial_design, weighted_linear_fit
from ..types import AtomFitResult, PlanetClassConfig, SegmentData


class ResidualAtom:
    """
    Base class for residual-template atoms with profiled linear coefficients.
    """

    atom_name: str = "base"
    class_label: str = "diagnostic"

    def __init__(self, config: PlanetClassConfig):
        self.config = config

    def fit(self, segment: SegmentData, features: dict[str, float]) -> AtomFitResult:
        raise NotImplementedError

    def _poly_order(self, segment: SegmentData, features: dict[str, float]) -> int:
        if int(features.get("n_points", segment.time.size)) <= int(self.config.short_duration_points):
            return int(self.config.polynomial_order_short)
        if float(features.get("duration", 0.0)) >= float(self.config.wide_duration_tE_fraction) * segment.pspl.tE:
            return int(self.config.polynomial_order_wide)
        return int(self.config.polynomial_order_default)

    def _baseline_chi2(self, segment: SegmentData) -> float:
        z = np.asarray(segment.residual, dtype=float) / np.maximum(np.asarray(segment.ferr, dtype=float), 1e-12)
        return float(np.sum(z * z))

    def _fit_profiled(
        self,
        *,
        segment: SegmentData,
        features: dict[str, float],
        theta0_list: list[np.ndarray],
        bounds: list[tuple[float, float]],
        shape_from_theta,
        params_from_theta,
        expected_amplitude_sign: Optional[float] = None,
        extra_warnings: tuple[str, ...] = (),
    ) -> AtomFitResult:
        t = np.asarray(segment.time, dtype=float)
        y = np.asarray(segment.residual, dtype=float)
        ferr = np.maximum(np.asarray(segment.ferr, dtype=float), 1e-12)
        center = float(features.get("t_peak", np.mean(t) if t.size else 0.0))
        poly = polynomial_design(t, center=center, order=self._poly_order(segment, features))
        chi2_baseline = self._baseline_chi2(segment)

        best: Optional[tuple[float, np.ndarray, np.ndarray, bool, np.ndarray]] = None

        def design_for(theta: np.ndarray) -> np.ndarray:
            shape = np.asarray(shape_from_theta(theta, t), dtype=float)
            if shape.ndim == 1:
                shape = shape[:, None]
            return np.column_stack((poly, shape))

        def evaluate(theta: np.ndarray) -> tuple[float, np.ndarray, bool, np.ndarray]:
            design = design_for(theta)
            coeff, _model, chi2, ok = weighted_linear_fit(design, y, ferr)
            return float(chi2), coeff, bool(ok), design

        def objective(theta: np.ndarray) -> float:
            chi2, _coeff, ok, _design = evaluate(theta)
            return chi2 if ok and np.isfinite(chi2) else 1e300

        for theta0 in theta0_list:
            theta0 = np.asarray(theta0, dtype=float)
            if minimize is not None:
                opt = minimize(
                    objective,
                    theta0,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={
                        "maxiter": int(self.config.optimizer_maxiter),
                        "ftol": float(self.config.optimizer_ftol),
                    },
                )
                theta = np.asarray(opt.x, dtype=float)
                opt_success = bool(opt.success)
            else:
                theta = theta0
                opt_success = True
            chi2, coeff, ok, design = evaluate(theta)
            success = bool(ok and np.isfinite(chi2))
            if best is None or chi2 < best[0]:
                best = (chi2, theta, coeff, success and bool(opt_success), design)

        if best is None:
            best = (
                float("inf"),
                np.asarray(theta0_list[0], dtype=float),
                np.asarray([], dtype=float),
                False,
                np.zeros((t.size, 0), dtype=float),
            )

        chi2, theta, coeff, success, design = best
        params = dict(params_from_theta(theta))
        n_poly = poly.shape[1]
        atom_coeff = coeff[n_poly:] if coeff.size > n_poly else np.asarray([], dtype=float)
        if atom_coeff.size:
            params["amplitude"] = float(atom_coeff[0])
            for i, value in enumerate(atom_coeff, start=1):
                params[f"amplitude_{i}"] = float(value)
        warnings = list(extra_warnings)
        if success is False and np.isfinite(chi2):
            warnings.append("optimizer did not report convergence")
            success = True
        if expected_amplitude_sign is not None and atom_coeff.size:
            if float(expected_amplitude_sign) * float(atom_coeff[0]) <= 0.0:
                warnings.append("atom amplitude has unexpected sign")
                success = False
        n_params = int(theta.size + coeff.size)
        n_data = int(t.size)
        delta_chi2 = float(chi2_baseline - chi2)
        bic = float(chi2 + n_params * np.log(max(n_data, 1)))
        aic = float(chi2 + 2 * n_params)
        param_errors, diagnostics = self._estimate_errors(
            theta=theta,
            coeff=coeff,
            design=design,
            ferr=ferr,
            objective=objective,
            params_from_theta=params_from_theta,
            params=params,
            bounds=bounds,
            n_poly=n_poly,
        )
        validity_penalty, validity_warnings = self._validity_penalty(
            params=params,
            features=features,
            theta=theta,
            bounds=bounds,
            warnings=tuple(warnings),
        )
        warnings.extend(validity_warnings)
        score = float(delta_chi2 - n_params * np.log(max(n_data, 1)) - validity_penalty)
        return AtomFitResult(
            atom_name=self.atom_name,
            class_label=self.class_label,
            params=params,
            param_errors=param_errors,
            chi2=float(chi2),
            chi2_baseline=chi2_baseline,
            delta_chi2=delta_chi2,
            bic=bic,
            aic=aic,
            score=score,
            n_data=n_data,
            n_params=n_params,
            success=bool(success),
            warnings=tuple(warnings),
            validity_penalty=float(validity_penalty),
            fit_diagnostics=diagnostics,
        )

    def _estimate_errors(
        self,
        *,
        theta: np.ndarray,
        coeff: np.ndarray,
        design: np.ndarray,
        ferr: np.ndarray,
        objective,
        params_from_theta,
        params: dict[str, float],
        bounds: list[tuple[float, float]],
        n_poly: int,
    ) -> tuple[Optional[dict[str, float]], dict[str, float]]:
        diagnostics: dict[str, float] = {}
        if not bool(self.config.estimate_param_errors):
            return None, diagnostics

        errors: dict[str, float] = {}
        theta_cov = self._theta_covariance(theta, objective, bounds)
        if theta_cov is not None:
            diagnostics["theta_cov_condition"] = float(np.linalg.cond(theta_cov))
            errors.update(
                self._transformed_param_errors(
                    theta=theta,
                    covariance=theta_cov,
                    params_from_theta=params_from_theta,
                    bounds=bounds,
                )
            )
        else:
            diagnostics["theta_cov_condition"] = float("inf")

        coeff_cov = self._linear_covariance(design, ferr)
        if coeff_cov is not None and coeff.size:
            atom_sigmas = np.sqrt(np.maximum(np.diag(coeff_cov)[n_poly:], 0.0))
            for i, sigma in enumerate(atom_sigmas, start=1):
                key = "amplitude" if i == 1 else f"amplitude_{i}"
                if key in params:
                    errors[key] = float(sigma)
                alias = f"amplitude_{i}"
                if alias in params:
                    errors[alias] = float(sigma)
        return (errors if errors else None), diagnostics

    def _transformed_param_errors(
        self,
        *,
        theta: np.ndarray,
        covariance: np.ndarray,
        params_from_theta,
        bounds: list[tuple[float, float]],
    ) -> dict[str, float]:
        base_params = {
            key: float(value)
            for key, value in dict(params_from_theta(theta)).items()
            if np.isscalar(value) and np.isfinite(float(value))
        }
        if not base_params:
            return {}
        keys = list(base_params)
        jac = np.zeros((len(keys), theta.size), dtype=float)
        steps = np.maximum(np.abs(theta) * float(self.config.covariance_step), float(self.config.covariance_step))
        for j in range(theta.size):
            tp = theta.copy()
            tm = theta.copy()
            tp[j] = min(max(theta[j] + steps[j], bounds[j][0]), bounds[j][1])
            tm[j] = min(max(theta[j] - steps[j], bounds[j][0]), bounds[j][1])
            denom = tp[j] - tm[j]
            if denom <= 0.0:
                continue
            pp = dict(params_from_theta(tp))
            pm = dict(params_from_theta(tm))
            for i, key in enumerate(keys):
                vp = float(pp.get(key, np.nan))
                vm = float(pm.get(key, np.nan))
                if np.isfinite(vp) and np.isfinite(vm):
                    jac[i, j] = (vp - vm) / denom
        cov_params = jac @ covariance @ jac.T
        return {
            key: float(sigma)
            for key, sigma in zip(keys, np.sqrt(np.maximum(np.diag(cov_params), 0.0)))
            if np.isfinite(sigma)
        }

    def _theta_covariance(self, theta: np.ndarray, objective, bounds: list[tuple[float, float]]) -> Optional[np.ndarray]:
        theta = np.asarray(theta, dtype=float)
        n = theta.size
        if n == 0 or n > 6:
            return None
        hess = np.zeros((n, n), dtype=float)
        f0 = float(objective(theta))
        if not np.isfinite(f0):
            return None
        steps = np.maximum(np.abs(theta) * float(self.config.covariance_step), float(self.config.covariance_step))
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = steps[i]
            xp = np.clip(theta + ei, bounds[i][0] if n == 1 else -np.inf, bounds[i][1] if n == 1 else np.inf)
            xm = np.clip(theta - ei, bounds[i][0] if n == 1 else -np.inf, bounds[i][1] if n == 1 else np.inf)
            xp[i] = min(max(theta[i] + steps[i], bounds[i][0]), bounds[i][1])
            xm[i] = min(max(theta[i] - steps[i], bounds[i][0]), bounds[i][1])
            fp = float(objective(xp))
            fm = float(objective(xm))
            denom = (xp[i] - theta[i]) * (theta[i] - xm[i])
            if denom <= 0.0 or not np.isfinite(fp + fm):
                return None
            hess[i, i] = (fp - 2.0 * f0 + fm) / denom
            for j in range(i + 1, n):
                ej = np.zeros(n)
                ej[j] = steps[j]
                xpp = theta.copy()
                xpm = theta.copy()
                xmp = theta.copy()
                xmm = theta.copy()
                for arr, si, sj in ((xpp, 1, 1), (xpm, 1, -1), (xmp, -1, 1), (xmm, -1, -1)):
                    arr[i] = min(max(theta[i] + si * steps[i], bounds[i][0]), bounds[i][1])
                    arr[j] = min(max(theta[j] + sj * steps[j], bounds[j][0]), bounds[j][1])
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
        if not np.isfinite(cond) or cond > float(self.config.covariance_max_condition):
            return None
        return cov

    @staticmethod
    def _linear_covariance(design: np.ndarray, ferr: np.ndarray) -> Optional[np.ndarray]:
        if design.size == 0:
            return None
        w = 1.0 / np.maximum(np.asarray(ferr, dtype=float), 1e-12)
        xw = np.asarray(design, dtype=float) * w[:, None]
        fisher = xw.T @ xw
        try:
            cov = np.linalg.pinv(fisher)
        except np.linalg.LinAlgError:
            return None
        return cov if np.all(np.isfinite(cov)) else None

    def _validity_penalty(
        self,
        *,
        params: dict[str, float],
        features: dict[str, float],
        theta: np.ndarray,
        bounds: list[tuple[float, float]],
        warnings: tuple[str, ...],
    ) -> tuple[float, list[str]]:
        penalty = float(self.config.warning_penalty) * len(warnings)
        extra: list[str] = []
        cadence = float(features.get("cadence", 0.0))
        width = float(params.get("width", params.get("tstar", params.get("tE_2", np.nan))))
        if np.isfinite(width) and cadence > 0.0 and width < float(self.config.min_width_cadence_ratio) * cadence:
            penalty += float(self.config.cadence_width_penalty)
            extra.append("width is close to cadence")
        for value, (lo, hi) in zip(np.asarray(theta, dtype=float), bounds):
            span = max(float(hi) - float(lo), 1e-12)
            if abs(value - lo) / span < 1e-3 or abs(value - hi) / span < 1e-3:
                penalty += float(self.config.boundary_penalty)
                extra.append("optimizer parameter is near bound")
                break
        return float(penalty), extra
