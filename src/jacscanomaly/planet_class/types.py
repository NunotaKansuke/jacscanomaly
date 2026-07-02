from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..planet_signal import PlanetSignalComponentClassification


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
    min_delta_chi2_for_seed: float = 20.0
    keep_top_atom_fits: int = 14
    keep_top_seeds_per_segment: int = 120
    q_floor: float = 1e-7
    q_ceil: float = 1.0
    q_width_factors: Tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
    s_central_grid: Tuple[float, ...] = (
        0.55,
        0.65,
        0.75,
        0.85,
        0.92,
        1.08,
        1.18,
        1.35,
        1.6,
        2.0,
    )
    alpha_grid_size_central: int = 8
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
    enable_positive_bump: bool = True
    enable_negative_dip: bool = True
    enable_central_perturbation: bool = True
    enable_fold_caustic: bool = True
    enable_curved_fold_caustic: bool = True
    enable_cusp_tail: bool = True
    enable_grazing_fold_caustic: bool = True
    enable_two_fold_caustic: bool = True
    enable_limb_darkened_fold_caustic: bool = True
    enable_canonical_cusp: bool = True
    enable_finite_source_cusp: bool = False
    enable_chang_refsdal: bool = True
    enable_second_pspl: bool = True
    enable_pspl_misfit: bool = True


@dataclass(frozen=True)
class PSPLParams:
    """
    Baseline point-lens parameters in the trajectory frame used by seed rules.
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
    fit_diagnostics: Optional[dict[str, float]] = None

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
        }
        for key, value in self.params.items():
            row[f"{prefix}{key}"] = float(value) if np.isscalar(value) else value
        if self.param_errors:
            for key, value in self.param_errors.items():
                row[f"{prefix}{key}_err"] = float(value)
        return row


@dataclass(frozen=True)
class SeedCandidate:
    """
    Initial-value candidate for a downstream physical model.
    """

    model_type: str
    class_label: str
    params: dict[str, float]
    score: float
    source_atom: str
    degeneracy_tag: Optional[str]
    warnings: Tuple[str, ...]

    def summary_dict(self, *, prefix: str = "") -> dict[str, object]:
        row: dict[str, object] = {
            f"{prefix}model_type": self.model_type,
            f"{prefix}class_label": self.class_label,
            f"{prefix}score": float(self.score),
            f"{prefix}source_atom": self.source_atom,
            f"{prefix}degeneracy_tag": self.degeneracy_tag,
            f"{prefix}warnings": "; ".join(self.warnings),
        }
        for key, value in self.params.items():
            row[f"{prefix}{key}"] = float(value) if np.isscalar(value) else value
        return row


@dataclass(frozen=True)
class SegmentModelResult:
    """
    Ranked atom fits and derived seeds for one anomaly component.
    """

    component: PlanetSignalComponentClassification
    features: dict[str, float]
    atom_fits: Tuple[AtomFitResult, ...]
    best_fit: Optional[AtomFitResult]
    class_probabilities: dict[str, float]
    seeds: Tuple[SeedCandidate, ...]
    warnings: Tuple[str, ...]

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
            "n_seeds": len(self.seeds),
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
    event_seeds: Tuple[SeedCandidate, ...]
    best_label: str
    best_atom: Optional[AtomFitResult]
    class_probabilities: dict[str, float]
    warnings: Tuple[str, ...]

    def summary_dict(self) -> dict[str, object]:
        row: dict[str, object] = {
            "best_label": self.best_label,
            "n_segments": len(self.segment_results),
            "n_event_seeds": len(self.event_seeds),
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
                rows.append(row)
        return tuple(rows)

    def seed_summary_dicts(self, *, top_n: Optional[int] = None) -> tuple[dict[str, object], ...]:
        seeds = self.event_seeds if top_n is None else self.event_seeds[: int(top_n)]
        rows = []
        for rank, seed in enumerate(seeds):
            row = {"rank": rank}
            row.update(seed.summary_dict())
            rows.append(row)
        return tuple(rows)

    def summary_text(self, *, max_seeds: int = 5) -> str:
        lines = [
            f"best_label: {self.best_label}",
            f"segments: {len(self.segment_results)}",
            f"event_seeds: {len(self.event_seeds)}",
        ]
        if self.best_atom is not None:
            lines.extend(
                [
                    f"best_atom: {self.best_atom.atom_name}",
                    f"best_bic: {self.best_atom.bic:.3f}",
                    f"best_delta_chi2: {self.best_atom.delta_chi2:.3f}",
                ]
            )
        if self.class_probabilities:
            probs = ", ".join(f"{k}={v:.3f}" for k, v in sorted(self.class_probabilities.items()))
            lines.append(f"class_probabilities: {probs}")
        if self.event_seeds:
            lines.append("top_seeds:")
            for seed in self.event_seeds[: int(max_seeds)]:
                tag = seed.degeneracy_tag or "none"
                lines.append(
                    f"  - {seed.model_type}/{seed.class_label} "
                    f"score={seed.score:.3f} source={seed.source_atom} tag={tag}"
                )
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

    def seed_table(self, *, top_n: Optional[int] = None):
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover - optional dependency
            return self.seed_summary_dicts(top_n=top_n)
        return pd.DataFrame(self.seed_summary_dicts(top_n=top_n))

    def plot_summary(self, *, signal_result=None, show: bool = True, max_seeds: int = 5):
        """
        Plot segment spans, best atom labels, and top seed labels.

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
                f"seg {i}: {label}, score={score:.1f}, seeds={len(segment.seeds)}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="black",
            )

        if self.event_seeds:
            seed_text = "\n".join(
                f"{rank + 1}. {seed.model_type}/{seed.class_label} ({seed.degeneracy_tag or 'none'})"
                for rank, seed in enumerate(self.event_seeds[: int(max_seeds)])
            )
            ax.text(
                0.98,
                0.96,
                seed_text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85, "pad": 4},
            )

        ax.set_title(f"Planet anomaly classification: {self.best_label}")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax
