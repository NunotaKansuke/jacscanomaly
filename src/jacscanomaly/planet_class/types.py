from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from ..planet_signal import PlanetSignalComponentClassification


_DIAGNOSTIC_CLASS_LABELS = {"second_pspl_like", "systematics_candidate", "pspl_misfit"}


def _planetary_reference_bic(atom_fits: Tuple["AtomFitResult", ...]) -> float:
    values = [
        atom.bic
        for atom in atom_fits
        if atom.class_label not in _DIAGNOSTIC_CLASS_LABELS and np.isfinite(atom.bic)
    ]
    return min(values) if values else float("inf")


@dataclass(frozen=True)
class PlanetClassConfig:
    """
    Configuration for residual-atom morphology fitting.
    """

    polynomial_order_short: int = 0
    polynomial_order_default: int = 1
    polynomial_order_wide: int = 2
    short_duration_points: int = 8
    wide_duration_tE_fraction: float = 0.2
    min_points_per_segment: int = 5
    min_delta_chi2_physical: float = 20.0
    # Deprecated compatibility alias. When set, it overrides the new name.
    min_delta_chi2_for_seed: Optional[float] = None
    physical_max_delta_bic: float = 50.0
    physical_window_max_delta_bic: float = 10.0
    keep_top_atom_fits: int = 14
    q_floor: float = 1e-7
    q_ceil: float = 1.0
    cusp_tail_powers: Tuple[float, ...] = (1.0, 2.0 / 3.0)
    central_window_factor: float = 3.0
    optimizer_maxiter: int = 300
    optimizer_ftol: float = 1e-8
    estimate_param_errors: bool = True
    covariance_step: float = 1e-4
    covariance_max_condition: float = 1e12
    warning_penalty: float = 5.0
    cadence_width_penalty: float = 20.0
    boundary_penalty: float = 10.0
    min_width_cadence_ratio: float = 1.5
    smooth_misfit_warning_delta_bic: float = 5.0
    cr_lookup_grid_size: int = 192
    cr_lookup_extent: float = 4.5
    cr_lookup_gamma_step: float = 0.025
    cr_lookup_source_radius_grid: Tuple[float, ...] = (0.0, 0.03, 0.1, 0.3, 1.0)
    cr_lookup_sqrt_q_factors: Tuple[float, ...] = (0.1, 0.3, 0.6, 1.0, 2.0, 4.0)
    cr_physical_q_max: float = 0.03
    cr_max_fwhm_tE_fraction: float = 0.2
    local_physical_min_points: int = 8
    local_physical_max_windows: int = 8
    local_physical_baseline_cadences: float = 8.0
    enable_positive_bump: bool = True
    enable_pspl_positive_bump: bool = True
    enable_negative_dip: bool = True
    enable_minor_image_box_trough: bool = True
    enable_central_perturbation: bool = True
    enable_central_double_cusp: bool = True
    enable_fold_caustic: bool = True
    enable_curved_fold_caustic: bool = True
    enable_full_caustic_crossing: bool = True
    enable_cusp_tail: bool = True
    enable_grazing_fold_caustic: bool = True
    enable_two_fold_caustic: bool = True
    enable_signed_two_fold_caustic: bool = False
    enable_rim_trough_caustic: bool = True
    enable_limb_darkened_fold_caustic: bool = True
    enable_canonical_cusp: bool = True
    enable_finite_source_cusp: bool = False
    enable_chang_refsdal: bool = True
    enable_second_pspl: bool = True
    enable_shear_quadrupole: bool = True
    enable_systematics_diagnostic: bool = True
    enable_pspl_misfit: bool = True

    @property
    def physical_delta_chi2_threshold(self) -> float:
        if self.min_delta_chi2_for_seed is not None:
            return float(self.min_delta_chi2_for_seed)
        return float(self.min_delta_chi2_physical)


@dataclass(frozen=True)
class PSPLParams:
    """
    Baseline point-lens parameters in the fitted trajectory frame.
    """

    t0: float
    tE: float
    u0: float
    Fs: float
    Fb: float


@dataclass(frozen=True)
class SegmentData:
    """
    Data slice for one connected anomaly component.
    """

    component: PlanetSignalComponentClassification
    time: np.ndarray
    flux: np.ndarray
    ferr: np.ndarray
    residual: np.ndarray
    model_flux: np.ndarray
    full_indices: np.ndarray
    pspl: PSPLParams


@dataclass(frozen=True)
class AtomFitResult:
    """
    Fit result for one residual-template atom.
    """

    atom_name: str
    class_label: str
    params: dict[str, float]
    param_errors: Optional[dict[str, float]]
    chi2: float
    chi2_baseline: float
    delta_chi2: float
    bic: float
    aic: float
    score: float
    n_data: int
    n_params: int
    success: bool
    warnings: Tuple[str, ...]
    validity_penalty: float = 0.0
    fit_diagnostics: Optional[dict[str, object]] = None
    estimation_role: str = "morphology"
    physical_params: dict[str, float] = field(default_factory=dict)
    constraint_relations: Tuple[str, ...] = ()
    physical_valid: bool = False
    physical_invalid_reasons: Tuple[str, ...] = ()

    def summary_dict(self, *, prefix: str = "") -> dict[str, object]:
        row: dict[str, object] = {
            f"{prefix}atom_name": self.atom_name,
            f"{prefix}class_label": self.class_label,
            f"{prefix}chi2": float(self.chi2),
            f"{prefix}delta_chi2": float(self.delta_chi2),
            f"{prefix}bic": float(self.bic),
            f"{prefix}aic": float(self.aic),
            f"{prefix}score": float(self.score),
            f"{prefix}validity_penalty": float(self.validity_penalty),
            f"{prefix}success": bool(self.success),
            f"{prefix}warnings": "; ".join(self.warnings),
            f"{prefix}estimation_role": self.estimation_role,
            f"{prefix}constraint_relations": "; ".join(self.constraint_relations),
            f"{prefix}physical_valid": bool(self.physical_valid),
            f"{prefix}physical_invalid_reasons": "; ".join(self.physical_invalid_reasons),
        }
        for key, value in self.params.items():
            row[f"{prefix}{key}"] = float(value) if np.isscalar(value) else value
        if self.param_errors:
            for key, value in self.param_errors.items():
                row[f"{prefix}{key}_err"] = float(value)
        for key, value in self.physical_params.items():
            row[f"{prefix}physical_{key}"] = float(value)
        return row


@dataclass(frozen=True)
class LocalPhysicalFitResult:
    """One physical atom fitted in a localized substructure window."""

    window_id: str
    locator_kind: str
    locator_time: float
    t_start: float
    t_end: float
    atom_fit: AtomFitResult

    def summary_dict(self, *, segment_index: int = 0, rank: int = 0) -> dict[str, object]:
        row: dict[str, object] = {
            "segment_index": int(segment_index),
            "window_id": self.window_id,
            "window_rank": int(rank),
            "locator_kind": self.locator_kind,
            "locator_time": float(self.locator_time),
            "window_t_start": float(self.t_start),
            "window_t_end": float(self.t_end),
        }
        row.update(self.atom_fit.summary_dict())
        if self.atom_fit.fit_diagnostics:
            for key, value in self.atom_fit.fit_diagnostics.items():
                if key.startswith("display_") or key == "physical_modes":
                    row[key] = value
        return row


@dataclass(frozen=True)
class SegmentModelResult:
    """
    Ranked atom fits and local physical fits for one anomaly component.
    """

    component: PlanetSignalComponentClassification
    features: dict[str, float]
    atom_fits: Tuple[AtomFitResult, ...]
    best_fit: Optional[AtomFitResult]
    class_probabilities: dict[str, float]
    warnings: Tuple[str, ...]
    local_physical_fits: Tuple[LocalPhysicalFitResult, ...] = ()

    def summary_dict(self, *, segment_index: int = 0) -> dict[str, object]:
        row: dict[str, object] = {
            "segment_index": int(segment_index),
            "signal_type": self.component.signal_type,
            "t_start": float(self.component.t_start),
            "t_end": float(self.component.t_end),
            "n_points": int(self.component.n_points),
            "positive_chi2": float(self.component.positive_chi2),
            "negative_chi2": float(self.component.negative_chi2),
            "n_atom_fits": len(self.atom_fits),
            "n_local_physical_fits": len(self.local_physical_fits),
            "warnings": "; ".join(self.warnings),
        }
        for key in ("t_peak", "fwhm", "duration", "snr", "edge_sharpness", "u_at_peak"):
            if key in self.features:
                row[key] = float(self.features[key])
        if self.best_fit is not None:
            row.update(self.best_fit.summary_dict(prefix="best_"))
        for label, probability in self.class_probabilities.items():
            row[f"p_{label}"] = float(probability)
        return row


@dataclass(frozen=True)
class PlanetAnomalyFitResult:
    """
    Event-level morphology-classification result.
    """

    pspl: PSPLParams
    segment_results: Tuple[SegmentModelResult, ...]
    best_label: str
    best_atom: Optional[AtomFitResult]
    class_probabilities: dict[str, float]
    warnings: Tuple[str, ...]
    physical_max_delta_bic: float = 50.0
    physical_window_max_delta_bic: float = 10.0

    @property
    def best_physical_fit(self) -> Optional[AtomFitResult]:
        candidates = []
        for segment in self.segment_results:
            by_window: dict[str, list[AtomFitResult]] = {}
            for local in segment.local_physical_fits:
                by_window.setdefault(local.window_id, []).append(local.atom_fit)
            for fits in by_window.values():
                finite = [
                    atom.bic
                    for atom in fits
                    if atom.physical_valid and np.isfinite(atom.bic)
                ]
                best_bic = min(finite) if finite else float("inf")
                candidates.extend(
                    atom
                    for atom in fits
                    if atom.physical_valid
                    and atom.estimation_role in {"physical_local", "physical_constraint"}
                    and np.isfinite(atom.bic)
                    and atom.bic - best_bic <= float(self.physical_window_max_delta_bic)
                )
        return min(candidates, key=lambda atom: atom.bic) if candidates else None

    def summary_dict(self) -> dict[str, object]:
        row: dict[str, object] = {
            "best_label": self.best_label,
            "n_segments": len(self.segment_results),
            "warnings": "; ".join(self.warnings),
            "pspl_t0": float(self.pspl.t0),
            "pspl_tE": float(self.pspl.tE),
            "pspl_u0": float(self.pspl.u0),
            "pspl_Fs": float(self.pspl.Fs),
            "pspl_Fb": float(self.pspl.Fb),
        }
        for label, probability in self.class_probabilities.items():
            row[f"p_{label}"] = float(probability)
        if self.best_atom is not None:
            row.update(self.best_atom.summary_dict(prefix="best_"))
        return row

    def segment_summary_dicts(self) -> tuple[dict[str, object], ...]:
        return tuple(
            segment.summary_dict(segment_index=i)
            for i, segment in enumerate(self.segment_results)
        )

    def atom_summary_dicts(self) -> tuple[dict[str, object], ...]:
        rows: list[dict[str, object]] = []
        for i, segment in enumerate(self.segment_results):
            for rank, atom in enumerate(segment.atom_fits):
                row = {"segment_index": i, "rank": rank}
                row.update(atom.summary_dict())
                if atom.fit_diagnostics:
                    for key, value in atom.fit_diagnostics.items():
                        if key.startswith("display_") or key == "physical_modes":
                            row[key] = value
                rows.append(row)
        return tuple(rows)

    def local_physical_summary_dicts(self) -> tuple[dict[str, object], ...]:
        rows: list[dict[str, object]] = []
        for segment_index, segment in enumerate(self.segment_results):
            ranks: dict[str, int] = {}
            for local in segment.local_physical_fits:
                rank = ranks.get(local.window_id, 0)
                rows.append(local.summary_dict(segment_index=segment_index, rank=rank))
                ranks[local.window_id] = rank + 1
        return tuple(rows)

    def physical_constraint_dicts(
        self,
        *,
        include_invalid: bool = False,
        max_delta_bic: Optional[float] = None,
    ) -> tuple[dict[str, object], ...]:
        rows: list[dict[str, object]] = []
        for segment_index, segment in enumerate(self.segment_results):
            by_window: dict[str, list[LocalPhysicalFitResult]] = {}
            for local in segment.local_physical_fits:
                by_window.setdefault(local.window_id, []).append(local)
            for window_id, local_fits in by_window.items():
                finite = [
                    local.atom_fit.bic
                    for local in local_fits
                    if local.atom_fit.physical_valid and np.isfinite(local.atom_fit.bic)
                ]
                best_bic = min(finite) if finite else float("inf")
                for local in local_fits:
                    atom = local.atom_fit
                    if atom.estimation_role not in {"physical_local", "physical_constraint"}:
                        continue
                    delta_bic = float(atom.bic - best_bic)
                    threshold = self.physical_window_max_delta_bic if max_delta_bic is None else float(max_delta_bic)
                    competitive = np.isfinite(delta_bic) and delta_bic <= threshold
                    if not include_invalid and (not atom.physical_valid or not competitive):
                        continue
                    row: dict[str, object] = {
                        "segment_index": segment_index,
                        "window_id": window_id,
                        "locator_kind": local.locator_kind,
                        "locator_time": float(local.locator_time),
                        "window_t_start": float(local.t_start),
                        "window_t_end": float(local.t_end),
                        "atom_name": atom.atom_name,
                        "class_label": atom.class_label,
                        "estimation_role": atom.estimation_role,
                        "bic": float(atom.bic),
                        "success": bool(atom.success),
                        "physical_valid": bool(atom.physical_valid),
                        "delta_bic_from_best": delta_bic,
                        "competitive": bool(competitive),
                        "constraint_relations": "; ".join(atom.constraint_relations),
                        "physical_invalid_reasons": "; ".join(atom.physical_invalid_reasons),
                        "warnings": "; ".join(atom.warnings),
                    }
                    row.update({key: float(value) for key, value in atom.physical_params.items()})
                    if atom.param_errors:
                        row.update(
                            {
                                f"{key}_err": float(atom.param_errors[key])
                                for key in atom.physical_params
                                if key in atom.param_errors and np.isfinite(atom.param_errors[key])
                            }
                        )
                    rows.append(row)
        return tuple(rows)

    def physical_relation_dicts(self) -> tuple[dict[str, object], ...]:
        """Combine independently fitted local edges without inventing a global caustic model."""
        constraints = self.physical_constraint_dicts()
        rows: list[dict[str, object]] = []
        for segment_index in range(len(self.segment_results)):
            segment_rows = [row for row in constraints if row["segment_index"] == segment_index]
            selected: dict[str, dict[str, object]] = {}
            for kind in ("entry_edge", "exit_edge"):
                candidates = [
                    row for row in segment_rows
                    if row["locator_kind"] == kind and "rho_over_abs_sin_psi" in row
                ]
                if candidates:
                    selected[kind] = min(candidates, key=lambda row: float(row["bic"]))
            if len(selected) != 2:
                continue
            entry = selected["entry_edge"]
            exit_ = selected["exit_edge"]
            r_entry = float(entry["rho_over_abs_sin_psi"])
            r_exit = float(exit_["rho_over_abs_sin_psi"])
            t_entry = float(entry.get("tc", entry["locator_time"]))
            t_exit = float(exit_.get("tc", exit_["locator_time"]))
            if not (r_entry > 0.0 and r_exit > 0.0 and t_exit > t_entry):
                continue
            row: dict[str, object] = {
                "segment_index": segment_index,
                "relation": "shared_source_radius_two_fold",
                "entry_window_id": entry["window_id"],
                "exit_window_id": exit_["window_id"],
                "t_entry": t_entry,
                "t_exit": t_exit,
                "caustic_center_crossing_duration": t_exit - t_entry,
                "rho_over_abs_sin_psi_entry": r_entry,
                "rho_over_abs_sin_psi_exit": r_exit,
                "abs_sin_psi_entry_over_exit": r_exit / r_entry,
                "constraint_relation": (
                    "assuming one source radius, R_i=tstar_i/tE=rho/abs(sin(psi_i)); "
                    "therefore abs(sin(psi_entry))/abs(sin(psi_exit))=R_exit/R_entry"
                ),
            }
            entry_err = float(entry.get("rho_over_abs_sin_psi_err", np.nan))
            exit_err = float(exit_.get("rho_over_abs_sin_psi_err", np.nan))
            if np.isfinite(entry_err) and np.isfinite(exit_err):
                ratio = float(row["abs_sin_psi_entry_over_exit"])
                row["abs_sin_psi_entry_over_exit_err"] = ratio * np.sqrt(
                    (entry_err / r_entry) ** 2 + (exit_err / r_exit) ** 2
                )
            t_entry_err = float(entry.get("tc_err", np.nan))
            t_exit_err = float(exit_.get("tc_err", np.nan))
            if np.isfinite(t_entry_err) and np.isfinite(t_exit_err):
                row["caustic_center_crossing_duration_err"] = np.hypot(t_entry_err, t_exit_err)
            rows.append(row)
        return tuple(rows)

    def summary_text(self) -> str:
        lines = [
            f"best_label: {self.best_label}",
            f"segments: {len(self.segment_results)}",
        ]
        if self.best_atom is not None:
            lines.extend(
                [
                    f"best_atom: {self.best_atom.atom_name}",
                    f"best_bic: {self.best_atom.bic:.3f}",
                    f"best_delta_chi2: {self.best_atom.delta_chi2:.3f}",
                ]
            )
        if self.best_physical_fit is not None:
            physical = ", ".join(f"{k}={v:.6g}" for k, v in self.best_physical_fit.physical_params.items())
            lines.append(f"best_physical_local: {self.best_physical_fit.atom_name} ({physical})")
        if self.class_probabilities:
            probs = ", ".join(f"{k}={v:.3f}" for k, v in sorted(self.class_probabilities.items()))
            lines.append(f"class_probabilities: {probs}")
        if self.warnings:
            lines.append("warnings: " + "; ".join(self.warnings))
        return "\n".join(lines)

    def summary_table(self):
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover - optional dependency
            return self.segment_summary_dicts()
        return pd.DataFrame(self.segment_summary_dicts())

    def atom_table(self):
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover - optional dependency
            return self.atom_summary_dicts()
        return pd.DataFrame(self.atom_summary_dicts())

    def physical_constraint_table(self):
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover - optional dependency
            return self.physical_constraint_dicts()
        return pd.DataFrame(self.physical_constraint_dicts())

    def physical_relation_table(self):
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover - optional dependency
            return self.physical_relation_dicts()
        return pd.DataFrame(self.physical_relation_dicts())

    def plot_summary(self, *, signal_result=None, show: bool = True):
        """
        Plot segment spans and best atom labels.

        If ``signal_result`` is provided, the residual light curve is shown in
        residual/error units.  Otherwise the method returns a compact text-only
        matplotlib panel.
        """
        if not show:
            import matplotlib

            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(9, 3.8))
        if signal_result is not None:
            time = np.asarray(signal_result.time, dtype=float)
            z = np.asarray(signal_result.refined_residual, dtype=float) / np.maximum(
                np.asarray(signal_result.ferr, dtype=float),
                1e-12,
            )
            ax.axhline(0.0, color="0.55", lw=1)
            ax.plot(time, z, "o", ms=2, color="C0", alpha=0.45)
            ax.set_xlabel("time")
            ax.set_ylabel("residual / error")
        else:
            ax.set_axis_off()

        for i, segment in enumerate(self.segment_results):
            color = f"C{(i + 1) % 10}"
            ax.axvspan(segment.component.t_start, segment.component.t_end, color=color, alpha=0.12)
            label = segment.best_fit.class_label if segment.best_fit is not None else "none"
            score = segment.best_fit.score if segment.best_fit is not None else float("nan")
            ypos = 0.96 - 0.12 * i
            ax.text(
                0.02,
                ypos,
                f"seg {i}: {label}, score={score:.1f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="black",
            )

        ax.set_title(f"Planet anomaly classification: {self.best_label}")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax
