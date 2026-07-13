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


class ShearQuadrupoleAtom(ResidualAtom):
    atom_name = "shear_quadrupole_smooth"
    class_label = "shear_quadrupole"

    def fit(self, segment: SegmentData, features: dict[str, float]) -> AtomFitResult:
        t = np.asarray(segment.time, dtype=float)
        y = np.asarray(segment.residual, dtype=float)
        ferr = np.maximum(np.asarray(segment.ferr, dtype=float), 1e-12)
        chi2_baseline = float(np.sum((y / ferr) ** 2))
        center = float(features.get("t_peak", np.mean(t) if t.size else 0.0))
        scale = max(float(features.get("duration", 0.0)), float(features.get("fwhm", 0.0)), 1e-6)
        tau = (t - center) / scale
        envelope = 1.0 / (1.0 + tau * tau)
        design = np.column_stack(
            (
                envelope * (tau * tau - np.mean(tau * tau)),
                envelope * tau,
                envelope,
                np.ones_like(tau),
            )
        )
        coeff, _model, chi2, ok = weighted_linear_fit(design, y, ferr)
        n_params = int(design.shape[1])
        n_data = int(t.size)
        delta_chi2 = float(chi2_baseline - chi2)
        bic = float(chi2 + n_params * np.log(max(n_data, 1)))
        gamma_c = float(coeff[0]) if coeff.size else float("nan")
        gamma_s = float(coeff[1]) if coeff.size > 1 else float("nan")
        flux_scale = max(abs(float(segment.pspl.Fs)), 1e-12)
        gamma_c_proxy = gamma_c / flux_scale
        gamma_s_proxy = gamma_s / flux_scale
        params = {
            "t_center": center,
            "width": scale,
            "shear_coeff_c_flux": gamma_c,
            "shear_coeff_s_flux": gamma_s,
            "gamma_c": gamma_c_proxy,
            "gamma_s": gamma_s_proxy,
            "gamma": float(np.hypot(gamma_c_proxy, gamma_s_proxy)),
            "shear_basis_angle": float(0.5 * np.arctan2(gamma_s_proxy, gamma_c_proxy)),
            "shear_width_over_tE": float(scale / max(segment.pspl.tE, 1e-12)),
        }
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
            warnings=("dimensionless shear proxy from a generic smooth basis; q,s grids are approximate",),
        )


class SystematicsArtifactAtom(ResidualAtom):
    atom_name = "sparse_systematics_artifact"
    class_label = "systematics_candidate"

    def fit(self, segment: SegmentData, features: dict[str, float]) -> AtomFitResult:
        t = np.asarray(segment.time, dtype=float)
        y = np.asarray(segment.residual, dtype=float)
        ferr = np.maximum(np.asarray(segment.ferr, dtype=float), 1e-12)
        z = y / ferr
        chi2_baseline = float(np.sum(z * z))
        n_data = int(t.size)
        if n_data == 0:
            raise ValueError("Cannot fit an empty segment.")

        cadence = max(float(features.get("cadence", 0.0)), 1e-12)
        width = max(0.65 * cadence, 1e-12)
        order = np.argsort(np.abs(z))[::-1]
        max_spikes = min(8, max(1, n_data // 2))
        selected = tuple(int(i) for i in order[:max_spikes] if abs(float(z[i])) >= 3.0)
        if not selected:
            selected = (int(order[0]),)

        cols = [np.ones_like(t)]
        for index in selected:
            cols.append(np.exp(-0.5 * ((t - t[index]) / width) ** 2))
        design = np.column_stack(cols)
        coeff, model, chi2, ok = weighted_linear_fit(design, y, ferr)
        n_params = int(design.shape[1])
        delta_chi2 = float(chi2_baseline - chi2)
        bic = float(chi2 + n_params * np.log(max(n_data, 1)))
        spike_times = [float(t[i]) for i in selected]
        spike_z = [float(z[i]) for i in selected]
        warnings = ["diagnostic artifact atom; no planet parameters inferred"]
        if float(features.get("fwhm", 0.0)) <= 1.5 * cadence:
            warnings.append("feature width is close to cadence")
        if len(selected) <= 2:
            warnings.append("fit dominated by one or two points")
        diagnostics = self._display_diagnostics(segment, features, model)
        return AtomFitResult(
            atom_name=self.atom_name,
            class_label=self.class_label,
            params={
                "n_spikes": float(len(selected)),
                "spike_width": float(width),
                "t_peak": float(spike_times[0]),
                "max_abs_z": float(np.max(np.abs(z))),
            },
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
            warnings=tuple(warnings),
            fit_diagnostics={
                **diagnostics,
                "spike_times": spike_times,
                "spike_z": spike_z,
            },
        )

    def _display_diagnostics(
        self,
        segment: SegmentData,
        features: dict[str, float],
        model_at_data: np.ndarray,
    ) -> dict[str, object]:
        t = np.asarray(segment.time, dtype=float)
        if t.size == 0:
            return {}
        return {
            "display_time": t.tolist(),
            "display_model_residual": np.asarray(model_at_data, dtype=float).tolist(),
            "display_atom_residual": np.asarray(model_at_data, dtype=float).tolist(),
        }
