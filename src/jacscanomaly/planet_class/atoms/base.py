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

        best: Optional[tuple[float, np.ndarray, np.ndarray, bool]] = None

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
            chi2, coeff, ok = evaluate(theta)
            success = bool(ok and np.isfinite(chi2))
            if best is None or chi2 < best[0]:
                best = (chi2, theta, coeff, success and bool(opt_success))

        if best is None:
            best = (float("inf"), np.asarray(theta0_list[0], dtype=float), np.asarray([], dtype=float), False)

        chi2, theta, coeff, success = best
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
        score = float(delta_chi2 - n_params * np.log(max(n_data, 1)))
        return AtomFitResult(
            atom_name=self.atom_name,
            class_label=self.class_label,
            params=params,
            param_errors=None,
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
        )
