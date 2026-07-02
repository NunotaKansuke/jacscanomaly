from __future__ import annotations

import numpy as np

from .base import ResidualAtom
from ..linear import weighted_linear_fit
from ..pspl import PSPLParams, pspl_flux
from ..types import AtomFitResult, SegmentData


class PSPLMisfitAtom(ResidualAtom):
    atom_name = "pspl_derivative_misfit"
    class_label = "pspl_misfit"

    def fit(self, segment: SegmentData, features: dict[str, float]) -> AtomFitResult:
        t = np.asarray(segment.time, dtype=float)
        y = np.asarray(segment.residual, dtype=float)
        ferr = np.maximum(np.asarray(segment.ferr, dtype=float), 1e-12)
        chi2_baseline = float(np.sum((y / ferr) ** 2))
        design = self._derivative_design(t, segment.pspl)
        coeff, _model, chi2, ok = weighted_linear_fit(design, y, ferr)
        n_params = int(design.shape[1])
        n_data = int(t.size)
        delta_chi2 = float(chi2_baseline - chi2)
        bic = float(chi2 + n_params * np.log(max(n_data, 1)))
        params = {f"coef_{i}": float(v) for i, v in enumerate(coeff)}
        return AtomFitResult(
            atom_name=self.atom_name,
            class_label=self.class_label,
            params=params,
            param_errors=None,
            chi2=float(chi2),
            chi2_baseline=chi2_baseline,
            delta_chi2=delta_chi2,
            bic=bic,
            aic=float(chi2 + 2 * n_params),
            score=float(delta_chi2 - n_params * np.log(max(n_data, 1))),
            n_data=n_data,
            n_params=n_params,
            success=bool(ok and np.isfinite(chi2)),
            warnings=(),
        )

    @staticmethod
    def _derivative_design(time: np.ndarray, pspl: PSPLParams) -> np.ndarray:
        cols = []
        base = pspl_flux(time, pspl)
        steps = {
            "t0": max(1e-5 * pspl.tE, 1e-6),
            "tE": max(1e-5 * pspl.tE, 1e-6),
            "u0": max(1e-5 * max(abs(pspl.u0), 1.0), 1e-6),
            "Fs": max(1e-5 * max(abs(pspl.Fs), 1.0), 1e-6),
            "Fb": max(1e-5 * max(abs(pspl.Fb), 1.0), 1e-6),
        }
        for name, step in steps.items():
            p_hi = PSPLParams(**{**pspl.__dict__, name: getattr(pspl, name) + step})
            p_lo = PSPLParams(**{**pspl.__dict__, name: getattr(pspl, name) - step})
            cols.append((pspl_flux(time, p_hi) - pspl_flux(time, p_lo)) / (2.0 * step))
        cols.append(np.ones_like(base))
        return np.column_stack(cols)
