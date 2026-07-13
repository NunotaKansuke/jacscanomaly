from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..planet_signal import PlanetSignalComponentClassification
from .geometry import AnomalyGeometry


@dataclass(frozen=True)
class PlanetClassConfig:
    """
    Configuration for the heuristic planetary-anomaly estimator.
    """

    polynomial_order: int = 1
    min_points_per_segment: int = 5
    min_points_fold: int = 8
    min_points_crossing: int = 12
    min_delta_chi2: float = 20.0
    mixed_sign_power_fraction: float = 0.25
    central_u_anom_max: float = 0.2
    optimizer_maxiter: int = 300
    optimizer_ftol: float = 1e-8
    estimate_param_errors: bool = True
    covariance_step: float = 1e-4
    covariance_max_condition: float = 1e12
    # Grid-seed region widths, set to the ~84%-coverage values measured on
    # the Roman OMPLDG simulation sample (2026-07).  The remaining misses are
    # dominated by shape/component misidentification, not by these widths.
    seed_s_width_factor: float = 1.7
    seed_alpha_tolerance_deg: float = 20.0
    seed_q_width_factor_dip: float = 10.0
    seed_q_width_factor_bump: float = 100.0


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
class AnomalyShapeFit:
    """
    Fit of one local shape template to a component residual.
    """

    name: str
    params: dict[str, float]
    param_errors: Optional[dict[str, float]]
    chi2: float
    chi2_null: float
    delta_chi2: float
    bic: float
    n_data: int
    n_params: int
    success: bool
    warnings: Tuple[str, ...] = ()

    def summary_dict(self, *, prefix: str = "") -> dict[str, object]:
        row: dict[str, object] = {
            f"{prefix}shape": self.name,
            f"{prefix}chi2": float(self.chi2),
            f"{prefix}delta_chi2": float(self.delta_chi2),
            f"{prefix}bic": float(self.bic),
            f"{prefix}n_data": int(self.n_data),
            f"{prefix}n_params": int(self.n_params),
            f"{prefix}success": bool(self.success),
            f"{prefix}warnings": "; ".join(self.warnings),
        }
        for key, value in self.params.items():
            row[f"{prefix}{key}"] = float(value)
        if self.param_errors:
            for key, value in self.param_errors.items():
                row[f"{prefix}{key}_err"] = float(value)
        return row


@dataclass(frozen=True)
class AnomalyScales:
    """
    Timescale ratios and assumption-tagged mass-ratio estimates.

    ``dt`` is the anomaly duration whose definition depends on the shape:
    template FWHM for bumps, full trough duration for dips, and the
    entry-to-exit interval for caustic crossings.  ``q`` is only filled for
    bump and dip shapes; ``q_method`` names the assumed relation.  Fold-type
    shapes constrain ``tstar/tE = rho/|sin(psi)|`` per crossing (``psi`` is
    the local trajectory-fold angle, not ``alpha``).
    """

    dt: float = float("nan")
    dt_err: float = float("nan")
    dt_over_tE: float = float("nan")
    q: float = float("nan")
    q_err: float = float("nan")
    q_method: str = ""
    tstar_over_tE: float = float("nan")
    tstar_entry_over_tE: float = float("nan")
    tstar_exit_over_tE: float = float("nan")
    notes: Tuple[str, ...] = ()

    def summary_dict(self, *, prefix: str = "") -> dict[str, object]:
        row: dict[str, object] = {}
        for key in (
            "dt",
            "dt_err",
            "dt_over_tE",
            "q",
            "q_err",
            "tstar_over_tE",
            "tstar_entry_over_tE",
            "tstar_exit_over_tE",
        ):
            value = float(getattr(self, key))
            if np.isfinite(value):
                row[f"{prefix}{key}"] = value
        if self.q_method:
            row[f"{prefix}q_method"] = self.q_method
        if self.notes:
            row[f"{prefix}scale_notes"] = "; ".join(self.notes)
        return row


@dataclass(frozen=True)
class GridSeed:
    """
    Seed region for a downstream 2L1S grid search.

    ``s_candidates`` are the ``s_dagger`` branches (preferred branch first);
    the true separation is expected within ``s_width_factor`` of one of them
    (each branch stands for its degenerate inner/outer pair).
    ``alpha_candidates`` are the four mirror-degenerate trajectory angles in
    ``[0, 2*pi)``.  ``q_center``/``q_width_factor`` bound the mass ratio when
    an estimate exists; ``q_quality`` is ``"calibrated"`` (dip relation,
    scatter ~0.35 dex), ``"order_of_magnitude"`` (bump relation, scatter
    ~1.7 dex — use for search ordering rather than a hard cut), or
    ``"none"``.  Region widths are configured in
    :class:`PlanetClassConfig` from measured simulation coverage.
    """

    s_candidates: Tuple[float, ...]
    s_width_factor: float
    alpha_candidates: Tuple[float, ...]
    alpha_tolerance: float
    q_center: float = float("nan")
    q_width_factor: float = float("nan")
    q_quality: str = "none"

    def contains(
        self,
        *,
        s: Optional[float] = None,
        q: Optional[float] = None,
        alpha: Optional[float] = None,
    ) -> bool:
        """
        Check whether a parameter point falls inside the seed region.

        Only the supplied coordinates are tested; ``q`` is ignored when
        ``q_quality`` is ``"none"``.  ``alpha`` may use any of the four
        mirror-degenerate conventions.
        """
        if s is not None:
            s = float(s)
            if s <= 0.0 or not any(
                abs(np.log(s / candidate)) <= np.log(self.s_width_factor)
                for candidate in self.s_candidates
            ):
                return False
        if alpha is not None:
            alpha = float(alpha)
            deltas = [
                abs((alpha - candidate + np.pi) % (2.0 * np.pi) - np.pi)
                for candidate in self.alpha_candidates
            ]
            if min(deltas) > self.alpha_tolerance:
                return False
        if q is not None and self.q_quality != "none" and np.isfinite(self.q_center):
            q = float(q)
            if q <= 0.0 or abs(np.log10(q / self.q_center)) > np.log10(self.q_width_factor):
                return False
        return True

    def summary_dict(self, *, prefix: str = "seed_") -> dict[str, object]:
        row: dict[str, object] = {
            f"{prefix}s_candidates": ",".join(f"{s:.6g}" for s in self.s_candidates),
            f"{prefix}s_width_factor": float(self.s_width_factor),
            f"{prefix}alpha_candidates": ",".join(
                f"{a:.6g}" for a in self.alpha_candidates
            ),
            f"{prefix}alpha_tolerance": float(self.alpha_tolerance),
            f"{prefix}q_quality": self.q_quality,
        }
        if np.isfinite(self.q_center):
            row[f"{prefix}q_center"] = float(self.q_center)
            row[f"{prefix}q_width_factor"] = float(self.q_width_factor)
        return row


@dataclass(frozen=True)
class ComponentAnomalyResult:
    """
    Shape, geometry, and scale estimates for one anomaly component.
    """

    component: PlanetSignalComponentClassification
    features: dict[str, float]
    shape_fits: Tuple[AnomalyShapeFit, ...]
    best_fit: Optional[AnomalyShapeFit]
    shape: str
    geometry: Optional[AnomalyGeometry]
    scales: Optional[AnomalyScales]
    grid_seed: Optional["GridSeed"] = None
    warnings: Tuple[str, ...] = ()

    @property
    def significant(self) -> bool:
        return (
            self.best_fit is not None
            and self.best_fit.name != "null"
            and self.best_fit.success
        )

    def summary_dict(self, *, segment_index: int = 0) -> dict[str, object]:
        row: dict[str, object] = {
            "segment_index": int(segment_index),
            "signal_type": self.component.signal_type,
            "t_start": float(self.component.t_start),
            "t_end": float(self.component.t_end),
            "n_points": int(self.component.n_points),
            "shape": self.shape,
            "warnings": "; ".join(self.warnings),
        }
        if self.best_fit is not None:
            row.update(self.best_fit.summary_dict(prefix="best_"))
        if self.geometry is not None:
            row.update(self.geometry.summary_dict())
        if self.scales is not None:
            row.update(self.scales.summary_dict())
        if self.grid_seed is not None:
            row.update(self.grid_seed.summary_dict())
        return row

    def shape_fit_dicts(self, *, segment_index: int = 0) -> tuple[dict[str, object], ...]:
        rows = []
        for rank, fit in enumerate(self.shape_fits):
            row: dict[str, object] = {"segment_index": int(segment_index), "rank": rank}
            row.update(fit.summary_dict())
            rows.append(row)
        return tuple(rows)


@dataclass(frozen=True)
class PlanetAnomalyFitResult:
    """
    Event-level heuristic anomaly estimate.
    """

    pspl: PSPLParams
    components: Tuple[ComponentAnomalyResult, ...]
    best_component_index: Optional[int]
    best_shape: str
    warnings: Tuple[str, ...] = ()

    @property
    def best_component(self) -> Optional[ComponentAnomalyResult]:
        if self.best_component_index is None:
            return None
        return self.components[self.best_component_index]

    def summary_dict(self) -> dict[str, object]:
        row: dict[str, object] = {
            "best_shape": self.best_shape,
            "n_components": len(self.components),
            "warnings": "; ".join(self.warnings),
            "pspl_t0": float(self.pspl.t0),
            "pspl_tE": float(self.pspl.tE),
            "pspl_u0": float(self.pspl.u0),
            "pspl_Fs": float(self.pspl.Fs),
            "pspl_Fb": float(self.pspl.Fb),
        }
        best = self.best_component
        if best is not None:
            row.update(best.summary_dict(segment_index=self.best_component_index))
        return row

    def component_summary_dicts(self) -> tuple[dict[str, object], ...]:
        return tuple(
            component.summary_dict(segment_index=i)
            for i, component in enumerate(self.components)
        )

    def shape_fit_dicts(self) -> tuple[dict[str, object], ...]:
        rows: list[dict[str, object]] = []
        for i, component in enumerate(self.components):
            rows.extend(component.shape_fit_dicts(segment_index=i))
        return tuple(rows)

    def summary_text(self) -> str:
        lines = [
            f"best_shape: {self.best_shape}",
            f"components: {len(self.components)}",
        ]
        best = self.best_component
        if best is not None and best.best_fit is not None:
            lines.append(
                f"best_fit: {best.best_fit.name} "
                f"(delta_chi2={best.best_fit.delta_chi2:.1f}, bic={best.best_fit.bic:.1f})"
            )
        if best is not None and best.geometry is not None:
            g = best.geometry
            lines.append(
                "geometry: "
                f"tau_anom={g.tau_anom:.4f}, u_anom={g.u_anom:.4f}, "
                f"alpha={g.alpha:.4f} rad, "
                f"s_dagger(+)={g.s_dagger_plus:.4f}, s_dagger(-)={g.s_dagger_minus:.4f}, "
                f"branch={g.preferred_branch}, regime={g.regime}"
            )
        if best is not None and best.scales is not None:
            s = best.scales
            parts = []
            if np.isfinite(s.dt_over_tE):
                parts.append(f"dt/tE={s.dt_over_tE:.5f}")
            if np.isfinite(s.q):
                parts.append(f"q={s.q:.3e} ({s.q_method})")
            for label, value in (
                ("tstar/tE", s.tstar_over_tE),
                ("tstar_entry/tE", s.tstar_entry_over_tE),
                ("tstar_exit/tE", s.tstar_exit_over_tE),
            ):
                if np.isfinite(value):
                    parts.append(f"{label}={value:.5f}")
            if parts:
                lines.append("scales: " + ", ".join(parts))
        if self.warnings:
            lines.append("warnings: " + "; ".join(self.warnings))
        return "\n".join(lines)

    def summary_table(self):
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover - optional dependency
            return self.component_summary_dicts()
        return pd.DataFrame(self.component_summary_dicts())

    def shape_fit_table(self):
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover - optional dependency
            return self.shape_fit_dicts()
        return pd.DataFrame(self.shape_fit_dicts())

    def plot_summary(self, *, signal_result=None, show: bool = True):
        """
        Plot component spans, best shape labels, and key derived quantities.

        If ``signal_result`` is provided, the residual light curve is shown in
        residual/error units.
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

        for i, component in enumerate(self.components):
            color = f"C{(i + 1) % 10}"
            ax.axvspan(
                component.component.t_start,
                component.component.t_end,
                color=color,
                alpha=0.12,
            )
            label = f"seg {i}: {component.shape}"
            if component.geometry is not None:
                label += f", s†={component.geometry.s_dagger_preferred:.3f}"
            if component.scales is not None and np.isfinite(component.scales.q):
                label += f", q~{component.scales.q:.1e}"
            ax.text(
                0.02,
                0.96 - 0.12 * i,
                label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="black",
            )

        ax.set_title(f"Planet anomaly estimate: {self.best_shape}")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax
