from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

try:
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover - scipy is an optional runtime dependency
    minimize = None

from .finder import Finder
from .anomaly_models import get_chi2_anom_masked, get_chi2_flat_masked
from .models import BestCandidate
from .plot import _adaptive_single_lens_curve, _single_lens_model_flux
from .singlelens_fit import SingleLensFitResult


@dataclass(frozen=True)
class PlanetSignalConfig:
    """
    Configuration for iterative single-lens refinement and signal extraction.

    This first implementation is PSPL-only. It uses the existing template scan
    as a seed finder, masks the strongest unexplained local windows, and refits
    the PSPL baseline on the remaining data.
    """

    baseline_mode: str = "beam_interval"
    max_iter: int = 3
    seed_min_dchi2: float = 100.0
    mask_teff_coeff: float = 4.0
    mask_min_half_width: float = 0.0
    mask_core_peak_frac: float = 0.3
    mask_core_min_abs_z: float = 5.0
    mask_core_min_improvement: float = 9.0
    mask_core_improvement_peak_frac: float = 0.05
    mask_core_pad_teff: float = 0.25
    max_mask_fraction: float = 0.5
    max_unmasked_chi2_dof_increase: float = 0.05
    candidate_min_points: int = 1
    candidate_min_chi2: float = 0.0
    robust_max_iter: int = 6
    robust_eta: float = 0.35
    robust_z_soft: float = 3.0
    robust_z_hard: float = 10.0
    robust_min_weight: float = 0.02
    robust_smooth_time: float = 0.12
    robust_smooth_teff_coeff: float = 0.5
    robust_min_weight_change: float = 1e-3
    signal_weight_threshold: float = 0.35
    signal_min_abs_z: float = 3.0
    beam_max_iter: int = 3
    beam_width: int = 3
    beam_candidates_per_iter: int = 8
    beam_teff_factors: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 32.0)
    beam_min_half_width: float = 0.0
    beam_grow_min_abs_z: float = 3.0
    beam_point_penalty: float = 4.0
    beam_interval_penalty: float = 25.0
    beam_width_penalty: float = 0.0
    flat_baseline_on_masked_peak: bool = True
    flat_baseline_u0_threshold: float = 0.01
    flat_baseline_peak_support_frac: float = 0.01
    flat_baseline_min_unmasked_peak_points: int = 3
    flat_baseline_max_masked_peak_fraction: float = 0.8
    flat_baseline_min_dchi2_vs_flat: float = 25.0
    scan_unimodal_filter: bool = True
    scan_unimodal_top_n: int = 512
    scan_unimodal_min_improvement: float = 9.0
    scan_unimodal_peak_frac: float = 0.2
    scan_unimodal_smooth_points: int = 5
    scan_unimodal_max_lobes: int = 1


@dataclass(frozen=True)
class PlanetSignalCandidate:
    """
    A contiguous interval excluded from the refined single-lens baseline fit.
    """

    start_index: int
    end_index: int
    t_start: float
    t_end: float
    t_center: float
    n_points: int
    chi2: float
    reduced_chi2: float
    max_abs_z: float
    peak_time: float
    peak_z: float
    signed_sum_z: float


@dataclass(frozen=True)
class PlanetSignalIteration:
    """
    One mask/refit iteration.
    """

    iteration: int
    seed: Optional[BestCandidate]
    n_masked_before: int
    n_masked_after: int
    added_points: int
    fit: SingleLensFitResult


@dataclass(frozen=True)
class _BeamBranch:
    score: float
    fit: SingleLensFitResult
    mask: np.ndarray
    iterations: Tuple[PlanetSignalIteration, ...]


@dataclass(frozen=True)
class FlatBaselineDiagnostic:
    """
    Diagnostics for deciding whether a fitted PSPL peak is supported by
    unmasked data or should be replaced by a flat baseline.
    """

    use_flat_baseline: bool
    peak_in_mask: bool
    u0: float
    n_peak_support: int
    n_unmasked_peak_support: int
    masked_peak_fraction: float
    dchi2_flat_minus_pspl: float
    improvement_n_eff: float
    improvement_peak_frac: float


@dataclass(frozen=True)
class PlanetSignalClassificationConfig:
    """
    Configuration for shape-based classification of extracted planet signals.
    """

    smooth_points: int = 7
    min_peak_abs_z: float = 5.0
    peak_relative_height: float = 0.35
    min_peak_prominence: float = 3.0
    peak_prominence_frac: float = 0.15
    min_peak_separation: float = 0.15
    duration_min_abs_z: float = 3.0
    duration_relative_height: float = 0.5
    fit_template_timescale: bool = False
    fit_template_min_points: int = 6
    fit_template_min_teff: float = 0.01
    fit_template_max_teff: float = 10.0
    fit_template_half_width_scale: float = 1.0
    positive_dominance: float = 1.5
    negative_dominance: float = 1.5


@dataclass(frozen=True)
class PlanetSignalPeak:
    """
    A local extremum inside an extracted signal component.
    """

    index: int
    time: float
    z: float
    residual: float
    timescale: float
    t_start: float
    t_end: float
    pspl_magnification: float
    observed_magnification: float
    strength_ratio: float
    fitted_t0: float = np.nan
    fitted_teff: float = np.nan
    fitted_chi2: float = np.nan


@dataclass(frozen=True)
class PlanetSignalComponentClassification:
    """
    Shape classification for one contiguous extracted signal component.
    """

    signal_type: str
    start_index: int
    end_index: int
    t_start: float
    t_end: float
    n_points: int
    positive_chi2: float
    negative_chi2: float
    signed_chi2_balance: float
    peaks: Tuple[PlanetSignalPeak, ...]
    dips: Tuple[PlanetSignalPeak, ...]


@dataclass(frozen=True)
class PlanetSignalClassification:
    """
    Shape-classification summary for a :class:`PlanetSignalResult`.
    """

    signal_type: str
    components: Tuple[PlanetSignalComponentClassification, ...]
    best: Optional[PlanetSignalComponentClassification]


@dataclass(frozen=True)
class PlanetSignalResult:
    """
    Result of :class:`PlanetSignalExtractor`.
    """

    time: np.ndarray
    flux: np.ndarray
    ferr: np.ndarray
    initial_fit: SingleLensFitResult
    refined_fit: SingleLensFitResult
    initial_residual: np.ndarray
    refined_residual: np.ndarray
    signal_mask: np.ndarray
    point_weight: np.ndarray
    flat_baseline_diagnostic: FlatBaselineDiagnostic
    iterations: Tuple[PlanetSignalIteration, ...]
    candidates: Tuple[PlanetSignalCandidate, ...]
    best: Optional[PlanetSignalCandidate]

    def classify(
        self,
        config: PlanetSignalClassificationConfig = PlanetSignalClassificationConfig(),
    ) -> PlanetSignalClassification:
        """
        Classify the extracted signal morphology.
        """
        return PlanetSignalClassifier(config).classify(self)

    def plot_signal(
        self,
        *,
        show: bool = True,
        peak_xlim: Optional[tuple[float, float]] = None,
        signal_xlim: Optional[tuple[float, float]] = None,
        peak_tE_width: float = 1.5,
        signal_pad: float = 0.5,
        max_signal_width: float = 6.0,
        show_classification: bool = True,
        classification_config: PlanetSignalClassificationConfig = PlanetSignalClassificationConfig(),
    ):
        """
        Plot the refined baseline and highlight extracted signal points.

        Returns ``(fig, axes)`` with peak light curve, signal zoom, and residual
        zoom panels.
        """
        import matplotlib.pyplot as plt

        t = self.time
        signal = self.signal_mask
        normal = ~signal
        z = self.refined_residual / self.ferr

        if peak_xlim is None:
            peak_xlim = self.default_peak_xlim(
                peak_tE_width=float(peak_tE_width),
                signal_pad=float(signal_pad),
            )

        if signal_xlim is None:
            signal_xlim = self._default_signal_xlim(
                z=z,
                pad=float(signal_pad),
                max_width=float(max_signal_width),
            )

        fig, axes = plt.subplots(3, 1, figsize=(9, 8))
        ax_peak, ax_zoom, ax_res = axes
        t_peak_model, f_peak_model = _adaptive_single_lens_curve(self.refined_fit, peak_xlim)
        t_signal_model, f_signal_model = _adaptive_single_lens_curve(self.refined_fit, signal_xlim)

        signal_color = "C1"
        signal_alpha = 0.9

        ax_peak.errorbar(
            t[normal], self.flux[normal], yerr=self.ferr[normal],
            fmt="o", markersize=2, c="C0", ecolor="C0", alpha=0.45,
            label="baseline points", zorder=0,
        )
        ax_peak.errorbar(
            t[signal], self.flux[signal], yerr=self.ferr[signal],
            fmt="o", markersize=2.5, c=signal_color, ecolor=signal_color, alpha=signal_alpha,
            label="extracted signal", zorder=2,
        )
        ax_peak.plot(t_peak_model, f_peak_model, c="k", lw=2.0, label="refined PSPL", zorder=1)
        ax_peak.set_xlim(peak_xlim)
        ax_peak.set_ylabel("flux")

        ax_zoom.errorbar(
            t[normal], self.flux[normal], yerr=self.ferr[normal],
            fmt="o", markersize=2, c="C0", ecolor="C0", alpha=0.45,
            zorder=0,
        )
        ax_zoom.errorbar(
            t[signal], self.flux[signal], yerr=self.ferr[signal],
            fmt="o", markersize=2.5, c=signal_color, ecolor=signal_color, alpha=signal_alpha,
            zorder=2,
        )
        ax_zoom.plot(t_signal_model, f_signal_model, c="k", lw=2.0, zorder=1)
        ax_zoom.set_xlim(signal_xlim)
        ax_zoom.set_ylabel("flux")

        ax_res.axhline(0.0, c="0.5", lw=1)
        ax_res.plot(t[normal], z[normal], "o", ms=2, c="C0", alpha=0.45, zorder=0)
        ax_res.plot(t[signal], z[signal], "o", ms=2.5, c=signal_color, alpha=signal_alpha, zorder=2)
        ax_res.set_xlim(signal_xlim)
        ax_res.set_ylabel("residual / error")
        ax_res.set_xlabel("time")

        if self.best is not None:
            ax_peak.set_title(
                f"planet-signal candidate: t={self.best.t_center:.3f}, "
                f"chi2={self.best.chi2:.1f}, n={self.best.n_points}"
            )

        if show_classification:
            classification = self.classify(classification_config)
            self._draw_classification_overlay(
                axes=(ax_peak, ax_zoom, ax_res),
                classification=classification,
            )
        ax_peak.legend(loc="best")

        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes

    def default_peak_xlim(
        self,
        *,
        peak_tE_width: float = 1.5,
        signal_pad: float = 0.5,
        min_half_width: float = 1.0,
    ) -> tuple[float, float]:
        """
        Return the default full-event window used by :meth:`plot_signal`.
        """
        t0, tE, u0 = map(float, np.asarray(self.refined_fit.params)[:3])
        effective_width = abs(tE) * max(abs(u0), 1.0)
        hw = max(float(peak_tE_width) * effective_width, float(min_half_width))
        xlim = (t0 - hw, t0 + hw)
        if self.best is not None:
            xlim = (
                min(float(xlim[0]), self.best.t_start - float(signal_pad)),
                max(float(xlim[1]), self.best.t_end + float(signal_pad)),
            )
        return xlim

    def _default_signal_xlim(
        self,
        *,
        z: np.ndarray,
        pad: float,
        max_width: float,
    ) -> tuple[float, float]:
        if self.best is None:
            return (float(np.min(self.time)), float(np.max(self.time)))

        in_best = (
            self.signal_mask
            & (self.time >= self.best.t_start)
            & (self.time <= self.best.t_end)
        )
        if not np.any(in_best):
            return (self.best.t_start - pad, self.best.t_end + pad)

        abs_z = np.abs(z)
        peak = max(float(np.max(abs_z[in_best])), 0.0)
        threshold = max(5.0, 0.1 * peak)
        core = in_best & (abs_z >= threshold)
        if not np.any(core):
            core = in_best

        lo = float(np.min(self.time[core])) - pad
        hi = float(np.max(self.time[core])) + pad
        center = 0.5 * (lo + hi)
        width = hi - lo
        if max_width > 0 and width > max_width:
            half = 0.5 * max_width
            lo = center - half
            hi = center + half
        if hi <= lo:
            hi = lo + 1.0
        return (lo, hi)

    @staticmethod
    def _draw_classification_overlay(
        *,
        axes,
        classification: PlanetSignalClassification,
    ) -> None:
        ax_peak, ax_zoom, ax_res = axes
        signal_color = "tab:purple"
        signal_alpha = 0.14
        best = classification.best
        n_peaks = len(best.peaks) if best is not None else 0
        n_dips = len(best.dips) if best is not None else 0
        label = (
            f"type: {classification.signal_type}\n"
            f"components: {len(classification.components)}\n"
            f"peaks: {n_peaks}, dips: {n_dips}"
        )
        ax_peak.text(
            0.02,
            0.98,
            label,
            transform=ax_peak.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="k",
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.85, "pad": 4},
            zorder=10,
        )

        if best is None:
            return

        extrema: Tuple[PlanetSignalPeak, ...]
        if best.signal_type == "dip":
            raw_extrema = best.dips
        elif best.signal_type == "caustic_crossing":
            raw_extrema = best.peaks
        elif best.signal_type in {"single_peak", "whole_event_anomaly"}:
            raw_extrema = best.peaks
        else:
            raw_extrema = best.peaks + best.dips
        extrema = PlanetSignalResult._non_overlapping_extrema(raw_extrema)
        extrema = tuple(sorted(extrema, key=lambda p: p.time))

        first_line = True
        for peak in extrema:
            is_dip = peak.z < 0.0
            line_style = "--" if is_dip else "-"
            line_label = "signal peak" if first_line else None
            line_time = float(peak.time)
            first_line = False
            for ax in (ax_peak, ax_zoom, ax_res):
                if peak.t_end > peak.t_start:
                    ax.axvspan(
                        peak.t_start,
                        peak.t_end,
                        color=signal_color,
                        alpha=signal_alpha,
                        lw=0,
                        zorder=0.5,
                    )
                ax.axvline(
                    line_time,
                    color=signal_color,
                    ls=line_style,
                    lw=1.5,
                    alpha=0.9,
                    label=line_label if ax is ax_peak else None,
                    zorder=4,
                )

    @staticmethod
    def _non_overlapping_extrema(
        extrema: Tuple[PlanetSignalPeak, ...],
        *,
        max_overlap_fraction: float = 0.25,
    ) -> Tuple[PlanetSignalPeak, ...]:
        selected: list[PlanetSignalPeak] = []
        for peak in sorted(extrema, key=lambda p: abs(float(p.z)), reverse=True):
            width = max(float(peak.t_end) - float(peak.t_start), 0.0)
            keep = True
            for other in selected:
                other_width = max(float(other.t_end) - float(other.t_start), 0.0)
                overlap = max(
                    0.0,
                    min(float(peak.t_end), float(other.t_end))
                    - max(float(peak.t_start), float(other.t_start)),
                )
                denom = max(min(width, other_width), 1e-12)
                if overlap / denom > float(max_overlap_fraction):
                    keep = False
                    break
            if keep:
                selected.append(peak)
        return tuple(sorted(selected, key=lambda p: p.time))


@dataclass
class PlanetSignalClassifier:
    """
    Shape-based classifier for extracted planet-signal components.

    The classifier does not try to infer the physical caustic orientation.
    If two prominent positive peaks are found in one connected component, they
    are reported as the same kind of crossing peak in time order.
    """

    config: PlanetSignalClassificationConfig = PlanetSignalClassificationConfig()

    def classify(self, result: PlanetSignalResult) -> PlanetSignalClassification:
        time = np.asarray(result.time, dtype=float)
        flux = np.asarray(result.flux, dtype=float)
        ferr = np.asarray(result.ferr, dtype=float)
        residual = np.asarray(result.refined_residual, dtype=float)
        mask = np.asarray(result.signal_mask, dtype=bool)

        if not np.any(mask):
            return PlanetSignalClassification(
                signal_type="none",
                components=(),
                best=None,
            )

        z = residual / np.maximum(ferr, 1e-12)
        z_smooth = self._smooth(z, int(self.config.smooth_points))
        components: list[PlanetSignalComponentClassification] = []
        for start, end in self._mask_slices(mask):
            component = self._classify_component(
                result=result,
                time=time,
                flux=flux,
                ferr=ferr,
                residual=residual,
                z=z,
                z_smooth=z_smooth,
                start=start,
                end=end,
            )
            components.append(component)

        components_tuple = tuple(sorted(components, key=self._component_chi2, reverse=True))
        best = components_tuple[0] if components_tuple else None
        return PlanetSignalClassification(
            signal_type=best.signal_type if best is not None else "none",
            components=components_tuple,
            best=best,
        )

    def _classify_component(
        self,
        *,
        result: PlanetSignalResult,
        time: np.ndarray,
        flux: np.ndarray,
        ferr: np.ndarray,
        residual: np.ndarray,
        z: np.ndarray,
        z_smooth: np.ndarray,
        start: int,
        end: int,
    ) -> PlanetSignalComponentClassification:
        sl = slice(start, end)
        z_seg = z[sl]
        positive_chi2 = float(np.sum(np.where(z_seg > 0.0, z_seg * z_seg, 0.0)))
        negative_chi2 = float(np.sum(np.where(z_seg < 0.0, z_seg * z_seg, 0.0)))
        total_chi2 = positive_chi2 + negative_chi2
        signed_balance = (
            (positive_chi2 - negative_chi2) / total_chi2
            if total_chi2 > 0.0
            else 0.0
        )

        peak_indices = self._prominent_extrema(
            time,
            z_smooth,
            start=start,
            end=end,
            sign=1.0,
        )
        dip_indices = self._prominent_extrema(
            time,
            z_smooth,
            start=start,
            end=end,
            sign=-1.0,
        )
        peaks = tuple(
            self._peaks_from_indices(
                result=result,
                time=time,
                flux=flux,
                residual=residual,
                z=z,
                z_smooth=z_smooth,
                indices=peak_indices,
                component_start=start,
                component_end=end,
                sign=1.0,
            )
        )
        dips = tuple(
            self._peaks_from_indices(
                result=result,
                time=time,
                flux=flux,
                residual=residual,
                z=z,
                z_smooth=z_smooth,
                indices=dip_indices,
                component_start=start,
                component_end=end,
                sign=-1.0,
            )
        )

        signal_type = self._component_type(
            n_peaks=len(peaks),
            n_dips=len(dips),
            positive_chi2=positive_chi2,
            negative_chi2=negative_chi2,
            flat_baseline=result.flat_baseline_diagnostic.use_flat_baseline,
        )

        return PlanetSignalComponentClassification(
            signal_type=signal_type,
            start_index=int(start),
            end_index=int(end),
            t_start=float(time[start]),
            t_end=float(time[end - 1]),
            n_points=int(end - start),
            positive_chi2=positive_chi2,
            negative_chi2=negative_chi2,
            signed_chi2_balance=float(signed_balance),
            peaks=peaks,
            dips=dips,
        )

    def _component_type(
        self,
        *,
        n_peaks: int,
        n_dips: int,
        positive_chi2: float,
        negative_chi2: float,
        flat_baseline: bool,
    ) -> str:
        if flat_baseline:
            return "whole_event_anomaly"
        if n_dips > 0 and negative_chi2 >= float(self.config.negative_dominance) * max(positive_chi2, 1e-12):
            return "dip"
        if n_peaks >= 2 and positive_chi2 >= max(negative_chi2, 1e-12):
            return "caustic_crossing"
        if n_peaks == 1 and positive_chi2 >= float(self.config.positive_dominance) * max(negative_chi2, 1e-12):
            return "single_peak"
        if n_peaks == 0 and n_dips == 0:
            return "low_significance"
        return "complex"

    def _prominent_extrema(
        self,
        time: np.ndarray,
        values: np.ndarray,
        *,
        start: int,
        end: int,
        sign: float,
    ) -> list[int]:
        segment = sign * values[start:end]
        if segment.size == 0:
            return []

        min_height = float(self.config.min_peak_abs_z)
        peak_height = float(np.max(segment))
        threshold = max(min_height, float(self.config.peak_relative_height) * peak_height)
        if not np.isfinite(peak_height) or peak_height < threshold:
            return []

        extrema = self._monotonic_extrema(segment)
        if not extrema:
            return []

        local: list[int] = []
        selected_offsets: list[int] = []
        for offset in sorted(extrema, key=lambda i: segment[i], reverse=True):
            value = float(segment[offset])
            if value < threshold:
                continue

            separate = True
            for kept in selected_offsets:
                lo = min(int(offset), int(kept))
                hi = max(int(offset), int(kept)) + 1
                valley = float(np.min(segment[lo:hi]))
                weaker = min(value, float(segment[kept]))
                min_prominence = max(
                    float(self.config.min_peak_prominence),
                    float(self.config.peak_prominence_frac) * weaker,
                )
                if weaker - valley < min_prominence:
                    separate = False
                    break
            if separate:
                selected_offsets.append(int(offset))

        for offset in selected_offsets:
            local.append(start + int(offset))

        local.sort(key=lambda idx: sign * values[idx], reverse=True)
        return self._suppress_nearby_extrema(time, values, local, sign)

    @staticmethod
    def _monotonic_extrema(segment: np.ndarray) -> list[int]:
        y = np.asarray(segment, dtype=float)
        if y.size < 3:
            return [int(np.argmax(y))] if y.size else []

        dy = np.diff(y)
        signs = np.sign(dy)
        if np.all(signs == 0):
            return [int(np.argmax(y))]

        last = 0.0
        for i, sign in enumerate(signs):
            if sign == 0.0:
                signs[i] = last
            else:
                last = sign
        last = 0.0
        for i in range(signs.size - 1, -1, -1):
            if signs[i] == 0.0:
                signs[i] = last
            else:
                last = signs[i]

        extrema: list[int] = []
        for i in range(signs.size - 1):
            if signs[i] > 0.0 and signs[i + 1] < 0.0:
                extrema.append(i + 1)
        if not extrema:
            extrema.append(int(np.argmax(y)))
        return extrema

    def _suppress_nearby_extrema(
        self,
        time: np.ndarray,
        values: np.ndarray,
        extrema: list[int],
        sign: float,
    ) -> list[int]:
        selected: list[int] = []
        min_sep = float(self.config.min_peak_separation)
        for index in extrema:
            far_enough = all(
                abs(float(time[index]) - float(time[kept])) >= min_sep
                for kept in selected
            )
            if far_enough:
                selected.append(index)
        selected.sort()
        return selected

    def _peaks_from_indices(
        self,
        *,
        result: PlanetSignalResult,
        time: np.ndarray,
        flux: np.ndarray,
        residual: np.ndarray,
        z: np.ndarray,
        z_smooth: np.ndarray,
        indices: list[int],
        component_start: int,
        component_end: int,
        sign: float,
    ) -> list[PlanetSignalPeak]:
        if not indices:
            return []

        sorted_indices = sorted(int(index) for index in indices)
        values = sign * z_smooth
        cells: list[tuple[int, int]] = []
        prev_boundary = int(component_start)
        for left, right in zip(sorted_indices[:-1], sorted_indices[1:]):
            between = slice(int(left), int(right) + 1)
            valley = int(left + np.argmin(values[between]))
            cells.append((prev_boundary, valley + 1))
            prev_boundary = valley + 1
        cells.append((prev_boundary, int(component_end)))

        return [
            self._peak_from_index(
                result=result,
                time=time,
                flux=flux,
                residual=residual,
                z=z,
                z_smooth=z_smooth,
                index=self._raw_extremum_in_cell(
                    z=z,
                    cell_start=max(component_start, cell_start),
                    cell_end=min(component_end, cell_end),
                    sign=sign,
                ),
                cell_start=max(component_start, cell_start),
                cell_end=min(component_end, cell_end),
                sign=sign,
            )
            for index, (cell_start, cell_end) in zip(sorted_indices, cells)
        ]

    @staticmethod
    def _raw_extremum_in_cell(
        *,
        z: np.ndarray,
        cell_start: int,
        cell_end: int,
        sign: float,
    ) -> int:
        values = sign * np.asarray(z[cell_start:cell_end], dtype=float)
        if values.size == 0 or not np.any(np.isfinite(values)):
            return int(cell_start)
        values = np.where(np.isfinite(values), values, -np.inf)
        return int(cell_start + np.argmax(values))

    def _peak_from_index(
        self,
        *,
        result: PlanetSignalResult,
        time: np.ndarray,
        flux: np.ndarray,
        residual: np.ndarray,
        z: np.ndarray,
        z_smooth: np.ndarray,
        index: int,
        cell_start: int,
        cell_end: int,
        sign: float,
    ) -> PlanetSignalPeak:
        values = sign * z_smooth
        height = float(values[index])
        left_floor = float(np.min(values[cell_start : index + 1]))
        right_floor = float(np.min(values[index:cell_end]))
        floor = max(left_floor, right_floor)
        prominence = max(height - floor, 1e-12)
        threshold = max(
            float(self.config.duration_min_abs_z),
            floor + float(self.config.duration_relative_height) * prominence,
        )

        lo = int(index)
        while lo > cell_start and values[lo - 1] >= threshold:
            lo -= 1
        hi = int(index)
        while hi + 1 < cell_end and values[hi + 1] >= threshold:
            hi += 1

        pspl_mag, observed_mag, strength_ratio = self._magnification_strength(
            result=result,
            flux=float(flux[index]),
            model_flux=float(np.asarray(result.refined_fit.model_flux)[index]),
        )

        fitted_t0 = np.nan
        fitted_teff = np.nan
        fitted_chi2 = np.nan
        if bool(self.config.fit_template_timescale):
            fit = self._fit_template_peak(
                time=time,
                residual=residual,
                ferr=np.asarray(result.ferr, dtype=float),
                index=index,
                cell_start=cell_start,
                cell_end=cell_end,
                t_start=float(time[lo]),
                t_end=float(time[hi]),
            )
            if fit is not None:
                fitted_t0, fitted_teff, fitted_chi2 = fit
                half_width = float(self.config.fit_template_half_width_scale) * float(fitted_teff)
                if np.isfinite(half_width) and half_width > 0.0:
                    lo = int(np.searchsorted(time, float(fitted_t0) - half_width, side="left"))
                    hi = int(np.searchsorted(time, float(fitted_t0) + half_width, side="right") - 1)
                    lo = max(int(cell_start), min(lo, int(index)))
                    hi = min(int(cell_end) - 1, max(hi, int(index)))

        return PlanetSignalPeak(
            index=int(index),
            time=float(time[index]),
            z=float(z[index]),
            residual=float(residual[index]),
            timescale=float(time[hi] - time[lo]) if hi > lo else 0.0,
            t_start=float(time[lo]),
            t_end=float(time[hi]),
            pspl_magnification=float(pspl_mag),
            observed_magnification=float(observed_mag),
            strength_ratio=float(strength_ratio),
            fitted_t0=float(fitted_t0),
            fitted_teff=float(fitted_teff),
            fitted_chi2=float(fitted_chi2),
        )

    def _fit_template_peak(
        self,
        *,
        time: np.ndarray,
        residual: np.ndarray,
        ferr: np.ndarray,
        index: int,
        cell_start: int,
        cell_end: int,
        t_start: float,
        t_end: float,
    ) -> Optional[tuple[float, float, float]]:
        if minimize is None:
            return None

        time = np.asarray(time, dtype=float)
        residual = np.asarray(residual, dtype=float)
        ferr = np.maximum(np.asarray(ferr, dtype=float), 1e-12)
        cell_start = int(max(0, cell_start))
        cell_end = int(min(time.size, cell_end))
        if cell_end - cell_start < int(self.config.fit_template_min_points):
            return None

        t_peak = float(time[index])
        shape_width = max(float(t_end) - float(t_start), 0.0)
        if shape_width > 0.0:
            init_teff = max(0.5 * shape_width, float(self.config.fit_template_min_teff))
        else:
            local_dt = np.median(np.diff(time[cell_start:cell_end])) if cell_end - cell_start > 1 else 0.05
            init_teff = max(float(local_dt), float(self.config.fit_template_min_teff))
        init_teff = float(np.clip(init_teff, self.config.fit_template_min_teff, self.config.fit_template_max_teff))

        lo_t = float(time[cell_start])
        hi_t = float(time[cell_end - 1])
        use = np.zeros(time.shape, dtype=bool)
        use[cell_start:cell_end] = True
        if int(np.sum(use)) < int(self.config.fit_template_min_points):
            return None

        t_fit = time[use]
        r_fit = residual[use]
        fe_fit = ferr[use]
        w_fit = 1.0 / (fe_fit * fe_fit)

        def model_chi2(x: np.ndarray) -> float:
            t0 = float(x[0])
            teff = float(np.exp(x[1]))
            if not (lo_t <= t0 <= hi_t):
                return 1e300
            if not (
                float(self.config.fit_template_min_teff)
                <= teff
                <= float(self.config.fit_template_max_teff)
            ):
                return 1e300

            chi2_best = np.inf
            for kind in (0, 1):
                if kind == 0:
                    A = 1.0 / np.sqrt(1.0 + ((t_fit - t0) / teff) ** 2)
                else:
                    Q = 1.0 + ((t_fit - t0) / teff) ** 2
                    A = (Q + 2.0) / np.sqrt(Q * (Q + 4.0))
                sw = float(np.sum(w_fit))
                x_mean = float(np.sum(w_fit * A) / sw)
                y_mean = float(np.sum(w_fit * r_fit) / sw)
                xc = A - x_mean
                yc = r_fit - y_mean
                wxx = float(np.sum(w_fit * xc * xc))
                if not np.isfinite(wxx) or wxx <= 0.0:
                    continue
                fs = float(np.sum(w_fit * xc * yc) / wxx)
                fb = y_mean - fs * x_mean
                chi2 = float(np.sum(((r_fit - (fs * A + fb)) / fe_fit) ** 2))
                if chi2 < chi2_best:
                    chi2_best = chi2
            return chi2_best if np.isfinite(chi2_best) else 1e300

        bounds = ((lo_t, hi_t), (np.log(float(self.config.fit_template_min_teff)), np.log(float(self.config.fit_template_max_teff))))
        fixed_t0 = t_peak
        best = None
        for teff0 in (init_teff, max(0.5 * init_teff, float(self.config.fit_template_min_teff)), min(2.0 * init_teff, float(self.config.fit_template_max_teff))):
            def fixed_t0_chi2(log_teff: np.ndarray) -> float:
                return model_chi2(np.asarray([fixed_t0, float(log_teff[0])], dtype=float))

            opt = minimize(
                fixed_t0_chi2,
                x0=np.asarray([np.log(float(teff0))], dtype=float),
                method="Nelder-Mead",
                options={"maxiter": 200, "xatol": 1e-5, "fatol": 1e-3},
            )
            log_teff = float(np.clip(float(opt.x[0]), bounds[1][0], bounds[1][1]))
            x = np.asarray([fixed_t0, log_teff], dtype=float)
            chi2 = model_chi2(x)
            if best is None or chi2 < best[2]:
                best = (float(fixed_t0), float(np.exp(log_teff)), float(chi2))
        return best

    @staticmethod
    def _magnification_strength(
        *,
        result: PlanetSignalResult,
        flux: float,
        model_flux: float,
    ) -> tuple[float, float, float]:
        fs = float(np.asarray(result.refined_fit.fs))
        fb = float(np.asarray(result.refined_fit.fb))
        if np.isfinite(fs) and abs(fs) > 1e-12:
            pspl_mag = (model_flux - fb) / fs
            observed_mag = (flux - fb) / fs
            strength_ratio = observed_mag / pspl_mag if abs(pspl_mag) > 1e-12 else np.nan
        else:
            pspl_mag = np.nan
            observed_mag = np.nan
            strength_ratio = flux / model_flux if abs(model_flux) > 1e-12 else np.nan
        return float(pspl_mag), float(observed_mag), float(strength_ratio)

    @staticmethod
    def _mask_slices(mask: np.ndarray) -> list[tuple[int, int]]:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return []
        breaks = np.flatnonzero(np.diff(idx) > 1)
        starts = np.r_[idx[0], idx[breaks + 1]]
        ends = np.r_[idx[breaks], idx[-1]] + 1
        return [(int(start), int(end)) for start, end in zip(starts, ends)]

    @staticmethod
    def _smooth(values: np.ndarray, width: int) -> np.ndarray:
        width = max(1, int(width))
        if width <= 1 or values.size <= 2:
            return np.asarray(values, dtype=float)
        if width % 2 == 0:
            width += 1
        pad = width // 2
        padded = np.pad(np.asarray(values, dtype=float), pad, mode="edge")
        kernel = np.ones(width, dtype=float) / float(width)
        return np.convolve(padded, kernel, mode="valid")

    @staticmethod
    def _component_chi2(component: PlanetSignalComponentClassification) -> float:
        return float(component.positive_chi2 + component.negative_chi2)


@dataclass
class PlanetSignalExtractor:
    """
    Extract strong non-PSPL signal after iteratively refining the baseline.

    Parameters
    ----------
    finder
        Existing :class:`jacscanomaly.Finder` instance. Its fitter, grid runner,
        and template-scan configuration are reused, but the Finder's public
        ``run`` pipeline is not called.
    config
        Extractor-specific configuration.
    """

    finder: Finder
    config: PlanetSignalConfig = PlanetSignalConfig()

    def run(
        self,
        time,
        flux,
        ferr,
        x0=None,
        *,
        refit: bool = True,
        verbose: bool = False,
    ) -> PlanetSignalResult:
        time_j, flux_j, ferr_j, x0_j, time_np, flux_np, ferr_np = self.finder._to_arrays(
            time, flux, ferr, x0
        )
        self.finder._ensure_fitter(float(np.median(time_np)))

        if refit:
            initial_fit = self.finder.fit_single_lens(time_np, flux_np, ferr_np, x0)
        else:
            initial_fit = self.finder._fixed_single_lens_from_x0(time_j, flux_j, ferr_j, x0_j)

        mode = str(self.config.baseline_mode).lower()
        if mode == "mask":
            current_fit, signal_mask, point_weight, iterations = self._run_mask_baseline(
                initial_fit=initial_fit,
                time_j=time_j,
                flux_j=flux_j,
                ferr_j=ferr_j,
                time_np=time_np,
                verbose=verbose,
            )
        elif mode == "robust":
            current_fit, signal_mask, point_weight, iterations = self._run_robust_baseline(
                initial_fit=initial_fit,
                time_j=time_j,
                flux_j=flux_j,
                ferr_j=ferr_j,
                time_np=time_np,
                verbose=verbose,
            )
        elif mode == "beam_interval":
            current_fit, signal_mask, point_weight, iterations = self._run_beam_interval_baseline(
                initial_fit=initial_fit,
                time_j=time_j,
                flux_j=flux_j,
                ferr_j=ferr_j,
                time_np=time_np,
                verbose=verbose,
            )
        else:
            raise ValueError(
                "PlanetSignalConfig.baseline_mode must be 'mask', 'robust', or 'beam_interval'."
            )

        flat_diagnostic = self._flat_baseline_diagnostic(current_fit, signal_mask)
        if flat_diagnostic.use_flat_baseline:
            current_fit = self._fit_flat_baseline_and_evaluate_full(
                time_j=time_j,
                flux_j=flux_j,
                ferr_j=ferr_j,
                keep_mask_np=~signal_mask,
                fit=current_fit,
            )

        initial_residual = np.asarray(jax.device_get(initial_fit.residual), dtype=float)
        refined_residual = np.asarray(jax.device_get(current_fit.residual), dtype=float)
        candidates = self._candidates_from_mask(time_np, refined_residual, ferr_np, signal_mask)
        best = max(candidates, key=lambda c: c.chi2, default=None)

        return PlanetSignalResult(
            time=time_np,
            flux=flux_np,
            ferr=ferr_np,
            initial_fit=initial_fit,
            refined_fit=current_fit,
            initial_residual=initial_residual,
            refined_residual=refined_residual,
            signal_mask=signal_mask,
            point_weight=point_weight,
            flat_baseline_diagnostic=flat_diagnostic,
            iterations=tuple(iterations),
            candidates=tuple(candidates),
            best=best,
        )

    def _run_mask_baseline(
        self,
        *,
        initial_fit: SingleLensFitResult,
        time_j: jnp.ndarray,
        flux_j: jnp.ndarray,
        ferr_j: jnp.ndarray,
        time_np: np.ndarray,
        verbose: bool,
    ) -> tuple[SingleLensFitResult, np.ndarray, np.ndarray, list[PlanetSignalIteration]]:
        current_fit = initial_fit
        current_residual_j = current_fit.residual
        signal_mask = np.zeros(time_np.shape, dtype=bool)
        iterations: list[PlanetSignalIteration] = []
        current_unmasked_chi2_dof = self._masked_chi2_dof(current_fit, ~signal_mask)

        for iteration in range(max(0, int(self.config.max_iter))):
            residual_for_seed = self._suppress_masked_residual(current_residual_j, signal_mask)
            seed = self._scan_best(time_j, residual_for_seed, ferr_j, time_np, verbose=verbose)
            if seed is None or not np.isfinite(seed.dchi2) or seed.dchi2 < float(self.config.seed_min_dchi2):
                break

            before = int(np.sum(signal_mask))
            half_width = max(
                float(self.config.mask_teff_coeff) * float(seed.teff),
                float(self.config.mask_min_half_width),
            )
            seed_window = np.abs(time_np - float(seed.t0)) <= half_width
            new_mask = self._template_improvement_mask_from_seed(
                time_np=time_np,
                time_j=time_j,
                residual_j=current_residual_j,
                ferr_j=ferr_j,
                seed_window=seed_window,
                seed_t0=float(seed.t0),
                seed_teff=float(seed.teff),
            )
            if not np.any(new_mask):
                break
            combined = signal_mask | new_mask

            if np.mean(combined) > float(self.config.max_mask_fraction):
                break

            added = int(np.sum(combined) - before)
            if added <= 0:
                break

            candidate_fit = self._fit_masked_single_lens_and_evaluate_full(
                time_j=time_j,
                flux_j=flux_j,
                ferr_j=ferr_j,
                keep_mask_np=~combined,
                x0_j=self._raw_params_for_refit(current_fit),
            )
            new_unmasked_chi2_dof = self._masked_chi2_dof(candidate_fit, ~combined)
            allowed = current_unmasked_chi2_dof * (1.0 + float(self.config.max_unmasked_chi2_dof_increase))
            if np.isfinite(current_unmasked_chi2_dof) and new_unmasked_chi2_dof > allowed:
                signal_mask = combined
                iterations.append(
                    PlanetSignalIteration(
                        iteration=iteration,
                        seed=seed,
                        n_masked_before=before,
                        n_masked_after=int(np.sum(signal_mask)),
                        added_points=added,
                        fit=current_fit,
                    )
                )
                break

            signal_mask = combined
            current_fit = candidate_fit
            current_residual_j = current_fit.residual
            current_unmasked_chi2_dof = new_unmasked_chi2_dof
            iterations.append(
                PlanetSignalIteration(
                    iteration=iteration,
                    seed=seed,
                    n_masked_before=before,
                    n_masked_after=int(np.sum(signal_mask)),
                    added_points=added,
                    fit=current_fit,
                )
            )

        point_weight = np.where(signal_mask, 0.0, 1.0)
        return current_fit, signal_mask, point_weight, iterations

    def _run_robust_baseline(
        self,
        *,
        initial_fit: SingleLensFitResult,
        time_j: jnp.ndarray,
        flux_j: jnp.ndarray,
        ferr_j: jnp.ndarray,
        time_np: np.ndarray,
        verbose: bool,
    ) -> tuple[SingleLensFitResult, np.ndarray, np.ndarray, list[PlanetSignalIteration]]:
        current_fit = initial_fit
        point_weight = np.ones(time_np.shape, dtype=float)
        iterations: list[PlanetSignalIteration] = []
        last_seed: Optional[BestCandidate] = None

        for iteration in range(max(0, int(self.config.robust_max_iter))):
            residual_j = current_fit.residual
            seed = self._scan_best(time_j, residual_j, ferr_j, time_np, verbose=verbose)
            if seed is not None and np.isfinite(seed.dchi2) and seed.dchi2 >= float(self.config.seed_min_dchi2):
                last_seed = seed

            residual_np = np.asarray(jax.device_get(residual_j), dtype=float)
            ferr_np = np.asarray(jax.device_get(ferr_j), dtype=float)
            z = residual_np / ferr_np
            raw_target = self._robust_target_weight(np.abs(z))

            smooth_width = float(self.config.robust_smooth_time)
            if last_seed is not None:
                smooth_width = max(
                    smooth_width,
                    float(self.config.robust_smooth_teff_coeff) * abs(float(last_seed.teff)),
                )
            target_weight = self._smooth_target_weight(time_np, raw_target, smooth_width)
            target_weight = np.clip(
                target_weight,
                float(self.config.robust_min_weight),
                1.0,
            )

            eta = np.clip(float(self.config.robust_eta), 0.0, 1.0)
            updated_weight = (1.0 - eta) * point_weight + eta * target_weight
            updated_weight = np.clip(
                updated_weight,
                float(self.config.robust_min_weight),
                1.0,
            )
            max_change = float(np.max(np.abs(updated_weight - point_weight)))
            before = int(np.sum(point_weight <= float(self.config.signal_weight_threshold)))
            point_weight = updated_weight

            candidate_fit = self._fit_weighted_single_lens_and_evaluate_full(
                time_j=time_j,
                flux_j=flux_j,
                ferr_j=ferr_j,
                point_weight=point_weight,
                x0_j=self._raw_params_for_refit(current_fit),
            )
            current_fit = candidate_fit
            signal_mask = self._signal_mask_from_weight_and_residual(
                fit=current_fit,
                point_weight=point_weight,
            )
            after = int(np.sum(signal_mask))
            iterations.append(
                PlanetSignalIteration(
                    iteration=iteration,
                    seed=last_seed,
                    n_masked_before=before,
                    n_masked_after=after,
                    added_points=max(after - before, 0),
                    fit=current_fit,
                )
            )

            if max_change < float(self.config.robust_min_weight_change):
                break

        signal_mask = self._signal_mask_from_weight_and_residual(
            fit=current_fit,
            point_weight=point_weight,
        )
        if np.mean(signal_mask) > float(self.config.max_mask_fraction):
            order = np.argsort(point_weight)
            keep_n = int(float(self.config.max_mask_fraction) * signal_mask.size)
            clipped = np.zeros_like(signal_mask, dtype=bool)
            clipped[order[:keep_n]] = signal_mask[order[:keep_n]]
            signal_mask = clipped
        return current_fit, signal_mask, point_weight, iterations

    def _run_beam_interval_baseline(
        self,
        *,
        initial_fit: SingleLensFitResult,
        time_j: jnp.ndarray,
        flux_j: jnp.ndarray,
        ferr_j: jnp.ndarray,
        time_np: np.ndarray,
        verbose: bool,
    ) -> tuple[SingleLensFitResult, np.ndarray, np.ndarray, list[PlanetSignalIteration]]:
        empty_mask = np.zeros(time_np.shape, dtype=bool)
        branches = (
            _BeamBranch(
                score=self._beam_score(initial_fit, empty_mask),
                fit=initial_fit,
                mask=empty_mask,
                iterations=(),
            ),
        )

        for iteration in range(max(0, int(self.config.beam_max_iter))):
            next_branches: list[_BeamBranch] = list(branches)
            for branch in branches:
                residual_for_seed = self._suppress_masked_residual(branch.fit.residual, branch.mask)
                seed = self._scan_best(time_j, residual_for_seed, ferr_j, time_np, verbose=verbose)
                if seed is None or not np.isfinite(seed.dchi2) or seed.dchi2 < float(self.config.seed_min_dchi2):
                    continue

                interval_masks = self._beam_interval_masks_from_seed(
                    time_np=time_np,
                    fit=branch.fit,
                    current_mask=branch.mask,
                    seed=seed,
                )
                for interval_mask in interval_masks:
                    combined = branch.mask | interval_mask
                    if np.array_equal(combined, branch.mask):
                        continue
                    if np.mean(combined) > float(self.config.max_mask_fraction):
                        continue

                    try:
                        candidate_fit = self._fit_masked_single_lens_and_evaluate_full(
                            time_j=time_j,
                            flux_j=flux_j,
                            ferr_j=ferr_j,
                            keep_mask_np=~combined,
                            x0_j=self._raw_params_for_refit(branch.fit),
                        )
                    except ValueError:
                        continue

                    added = int(np.sum(combined) - np.sum(branch.mask))
                    score = self._beam_score(candidate_fit, combined)
                    next_branches.append(
                        _BeamBranch(
                            score=score,
                            fit=candidate_fit,
                            mask=combined,
                            iterations=branch.iterations
                            + (
                                PlanetSignalIteration(
                                    iteration=iteration,
                                    seed=seed,
                                    n_masked_before=int(np.sum(branch.mask)),
                                    n_masked_after=int(np.sum(combined)),
                                    added_points=added,
                                    fit=candidate_fit,
                                ),
                            ),
                        )
                    )

            branches = tuple(sorted(next_branches, key=lambda b: b.score)[: max(1, int(self.config.beam_width))])

        best_branch = min(branches, key=lambda b: b.score)
        point_weight = np.where(best_branch.mask, 0.0, 1.0)
        return best_branch.fit, best_branch.mask, point_weight, list(best_branch.iterations)

    def _beam_interval_masks_from_seed(
        self,
        *,
        time_np: np.ndarray,
        fit: SingleLensFitResult,
        current_mask: np.ndarray,
        seed: BestCandidate,
    ) -> tuple[np.ndarray, ...]:
        residual = np.asarray(jax.device_get(fit.residual), dtype=float)
        ferr = np.asarray(fit.ferr, dtype=float)
        z = residual / ferr
        abs_z = np.abs(z)
        masks: list[np.ndarray] = []

        for factor in tuple(self.config.beam_teff_factors):
            half_width = max(
                abs(float(factor)) * abs(float(seed.teff)),
                float(self.config.beam_min_half_width),
            )
            masks.append(np.abs(time_np - float(seed.t0)) <= half_width)

        max_factor = max([abs(float(v)) for v in tuple(self.config.beam_teff_factors)] + [1.0])
        search_half_width = max(
            max_factor * abs(float(seed.teff)),
            float(self.config.beam_min_half_width),
        )
        search = (~current_mask) & (np.abs(time_np - float(seed.t0)) <= search_half_width)
        core = search & (abs_z >= float(self.config.beam_grow_min_abs_z))
        if np.any(core):
            idx = np.flatnonzero(search)
            core_idx = np.flatnonzero(core)
            lo = int(core_idx[0])
            hi = int(core_idx[-1])
            grown = np.zeros_like(current_mask, dtype=bool)
            grown[lo : hi + 1] = True
            grown &= search
            masks.append(grown)

            for pad_factor in (1.0, 2.0, 4.0):
                pad = pad_factor * abs(float(seed.teff))
                padded = (
                    (time_np >= float(time_np[lo]) - pad)
                    & (time_np <= float(time_np[hi]) + pad)
                )
                masks.append(padded)

        unique: list[np.ndarray] = []
        seen: set[bytes] = set()
        for mask in masks:
            mask = np.asarray(mask, dtype=bool) & (~current_mask)
            if not np.any(mask):
                continue
            key = np.packbits(mask).tobytes()
            if key in seen:
                continue
            seen.add(key)
            unique.append(mask)

        def interval_priority(mask: np.ndarray) -> float:
            added = mask & (~current_mask)
            if not np.any(added):
                return -np.inf
            return float(np.sum(abs_z[added] ** 2) - float(self.config.beam_point_penalty) * np.sum(added))

        unique = sorted(unique, key=interval_priority, reverse=True)
        return tuple(unique[: max(1, int(self.config.beam_candidates_per_iter))])

    def _beam_score(self, fit: SingleLensFitResult, mask: np.ndarray) -> float:
        keep = ~np.asarray(mask, dtype=bool)
        residual = np.asarray(jax.device_get(fit.residual), dtype=float)
        ferr = np.asarray(fit.ferr, dtype=float)
        z = residual[keep] / ferr[keep]
        kept_chi2 = float(np.sum(z * z))
        n_masked = int(np.sum(mask))
        n_intervals = self._count_mask_intervals(mask)
        total_width = self._mask_total_width(np.asarray(fit.time), mask)
        return (
            kept_chi2
            + float(self.config.beam_point_penalty) * n_masked
            + float(self.config.beam_interval_penalty) * n_intervals
            + float(self.config.beam_width_penalty) * total_width
        )

    def _scan_best(
        self,
        time_j: jnp.ndarray,
        residual_j: jnp.ndarray,
        ferr_j: jnp.ndarray,
        time_np: np.ndarray,
        *,
        verbose: bool,
    ) -> Optional[BestCandidate]:
        _seasons, clusters_all, grid_metrics_all = self.finder.runner.run(
            time_j=time_j,
            residual_j=residual_j,
            ferr_j=ferr_j,
            time_np=time_np,
            verbose=verbose,
        )
        if bool(self.config.scan_unimodal_filter) and grid_metrics_all.size:
            clusters_all, grid_metrics_all = self._unimodal_scan_clusters(
                time_j=time_j,
                residual_j=residual_j,
                ferr_j=ferr_j,
                grid_metrics_all=grid_metrics_all,
            )
        return self.finder._pick_best_candidate(clusters_all, grid_metrics_all)

    def _unimodal_scan_clusters(
        self,
        *,
        time_j: jnp.ndarray,
        residual_j: jnp.ndarray,
        ferr_j: jnp.ndarray,
        grid_metrics_all: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        metrics = np.asarray(grid_metrics_all, dtype=float)
        if metrics.size == 0:
            return np.zeros((0, 3), dtype=float), metrics

        dchi2 = np.asarray(metrics[:, 2], dtype=float)
        valid = np.isfinite(dchi2) & (dchi2 > 0.0)
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size == 0:
            return np.zeros((0, 3), dtype=float), metrics[:0]

        top_n = max(1, int(self.config.scan_unimodal_top_n))
        if valid_idx.size > top_n:
            order = np.argsort(dchi2[valid_idx])[-top_n:]
            eval_idx = valid_idx[order]
        else:
            eval_idx = valid_idx

        keep = np.zeros(metrics.shape[0], dtype=bool)
        for idx in eval_idx:
            t0 = float(metrics[idx, 0])
            teff = float(metrics[idx, 1])
            if self._scan_grid_is_unimodal(
                time_j=time_j,
                residual_j=residual_j,
                ferr_j=ferr_j,
                t0=t0,
                teff=teff,
            ):
                keep[idx] = True

        filtered = metrics[keep]
        if filtered.size == 0:
            return np.zeros((0, 3), dtype=float), filtered

        clusters = self.finder.runner.extractor.iterative_anomaly_extraction(
            filtered[:, 0],
            filtered[:, 1],
            filtered[:, 2],
        )
        return clusters, filtered

    def _scan_grid_is_unimodal(
        self,
        *,
        time_j: jnp.ndarray,
        residual_j: jnp.ndarray,
        ferr_j: jnp.ndarray,
        t0: float,
        teff: float,
    ) -> bool:
        time_np = np.asarray(jax.device_get(time_j), dtype=float)
        residual_np = np.asarray(jax.device_get(residual_j), dtype=float)
        ferr_np = np.asarray(jax.device_get(ferr_j), dtype=float)
        half_width = float(self.finder.config.teff_coeff) * abs(float(teff))
        window = np.abs(time_np - float(t0)) < half_width
        if int(np.sum(window)) < int(self.finder.config.min_pts_in_window):
            return False

        w_j = 1.0 / (ferr_j ** 2)
        mask_j = jnp.asarray(window)
        chi2_anom, chi2s_anom = get_chi2_anom_masked(
            jnp.asarray(t0, dtype=time_j.dtype),
            jnp.asarray(teff, dtype=time_j.dtype),
            time_j,
            residual_j,
            w_j,
            mask_j,
        )
        chi2_flat, chi2s_flat = get_chi2_flat_masked(residual_j, w_j, mask_j)
        chi2s_flat_np = np.asarray(jax.device_get(chi2s_flat), dtype=float)
        chi2s_anom_np = np.asarray(jax.device_get(chi2s_anom), dtype=float)
        improvement = np.where(window, np.maximum(chi2s_flat_np - chi2s_anom_np, 0.0), 0.0)

        in_window = improvement[window]
        if in_window.size == 0:
            return False
        peak = float(np.max(in_window))
        if not np.isfinite(peak) or peak <= 0.0:
            return False
        threshold = max(
            float(self.config.scan_unimodal_min_improvement),
            float(self.config.scan_unimodal_peak_frac) * peak,
        )
        smooth = self._smooth_1d(improvement, int(self.config.scan_unimodal_smooth_points))
        lobes = self._count_positive_lobes(smooth, window, threshold)
        return lobes <= int(self.config.scan_unimodal_max_lobes)

    @staticmethod
    def _smooth_1d(values: np.ndarray, width: int) -> np.ndarray:
        width = max(1, int(width))
        values = np.asarray(values, dtype=float)
        if width <= 1 or values.size <= 2:
            return values
        if width % 2 == 0:
            width += 1
        pad = width // 2
        padded = np.pad(values, pad, mode="edge")
        kernel = np.ones(width, dtype=float) / float(width)
        return np.convolve(padded, kernel, mode="valid")

    @staticmethod
    def _count_positive_lobes(values: np.ndarray, window: np.ndarray, threshold: float) -> int:
        active = np.asarray(window, dtype=bool) & (np.asarray(values, dtype=float) >= float(threshold))
        idx = np.flatnonzero(active)
        if idx.size == 0:
            return 0
        return int(1 + np.sum(np.diff(idx) > 1))

    def _template_improvement_mask_from_seed(
        self,
        *,
        time_np: np.ndarray,
        time_j: jnp.ndarray,
        residual_j: jnp.ndarray,
        ferr_j: jnp.ndarray,
        seed_window: np.ndarray,
        seed_t0: float,
        seed_teff: float,
    ) -> np.ndarray:
        residual_np = np.asarray(jax.device_get(residual_j), dtype=float)
        ferr_np = np.asarray(jax.device_get(ferr_j), dtype=float)
        z = residual_np / ferr_np
        abs_z = np.abs(z)
        if not np.any(seed_window):
            return np.zeros_like(seed_window, dtype=bool)

        w_j = 1.0 / (ferr_j ** 2)
        mask_j = jnp.asarray(seed_window)
        chi2_anom, chi2s_anom = get_chi2_anom_masked(
            jnp.asarray(seed_t0, dtype=time_j.dtype),
            jnp.asarray(seed_teff, dtype=time_j.dtype),
            time_j,
            residual_j,
            w_j,
            mask_j,
        )
        chi2_flat, chi2s_flat = get_chi2_flat_masked(residual_j, w_j, mask_j)

        chi2s_flat_np = np.asarray(jax.device_get(chi2s_flat), dtype=float)
        chi2s_anom_np = np.asarray(jax.device_get(chi2s_anom), dtype=float)
        improvement = np.where(seed_window, np.maximum(chi2s_flat_np - chi2s_anom_np, 0.0), 0.0)

        peak_abs_z = float(np.max(abs_z[seed_window]))
        z_threshold = max(
            float(self.config.mask_core_min_abs_z),
            float(self.config.mask_core_peak_frac) * peak_abs_z,
        )
        peak_improvement = float(np.max(improvement[seed_window]))
        improvement_threshold = max(
            float(self.config.mask_core_min_improvement),
            float(self.config.mask_core_improvement_peak_frac) * peak_improvement,
        )

        core = seed_window & (abs_z >= z_threshold) & (improvement >= improvement_threshold)
        if not np.any(core):
            return np.zeros_like(seed_window, dtype=bool)

        pad = max(0.0, float(self.config.mask_core_pad_teff) * float(seed_teff))
        if pad <= 0.0:
            return core

        core_times = time_np[core]
        padded = np.zeros_like(core, dtype=bool)
        for t_core in core_times:
            padded |= np.abs(time_np - float(t_core)) <= pad
        return padded & seed_window

    @staticmethod
    def _suppress_masked_residual(residual_j: jnp.ndarray, mask_np: np.ndarray) -> jnp.ndarray:
        if not np.any(mask_np):
            return residual_j
        mask_j = jnp.asarray(mask_np)
        return jnp.where(mask_j, jnp.asarray(0.0, residual_j.dtype), residual_j)

    @staticmethod
    def _raw_params_for_refit(fit: SingleLensFitResult) -> jnp.ndarray:
        raw = fit.raw_params
        if raw is not None:
            return jnp.asarray(raw)
        return jnp.asarray(fit.params)

    def _fit_masked_single_lens_and_evaluate_full(
        self,
        *,
        time_j: jnp.ndarray,
        flux_j: jnp.ndarray,
        ferr_j: jnp.ndarray,
        keep_mask_np: np.ndarray,
        x0_j: jnp.ndarray,
    ) -> SingleLensFitResult:
        if int(np.sum(keep_mask_np)) < 4:
            raise ValueError("Not enough unmasked points to refit single-lens model.")

        keep_j = jnp.asarray(keep_mask_np)
        masked_fit = self.finder.fitter.fit(time_j[keep_j], flux_j[keep_j], ferr_j[keep_j], x0_j)
        return self._evaluate_single_lens_fit_on_full_data(
            time_j=time_j,
            flux_j=flux_j,
            ferr_j=ferr_j,
            fit=masked_fit,
            fs=masked_fit.fs,
            fb=masked_fit.fb,
        )

    def _fit_weighted_single_lens_and_evaluate_full(
        self,
        *,
        time_j: jnp.ndarray,
        flux_j: jnp.ndarray,
        ferr_j: jnp.ndarray,
        point_weight: np.ndarray,
        x0_j: jnp.ndarray,
    ) -> SingleLensFitResult:
        weight_j = jnp.asarray(np.clip(point_weight, float(self.config.robust_min_weight), 1.0))
        ferr_eff_j = ferr_j / jnp.sqrt(weight_j)
        weighted_fit = self.finder.fitter.fit(time_j, flux_j, ferr_eff_j, x0_j)
        return self._evaluate_single_lens_fit_on_full_data(
            time_j=time_j,
            flux_j=flux_j,
            ferr_j=ferr_j,
            fit=weighted_fit,
            fs=weighted_fit.fs,
            fb=weighted_fit.fb,
        )

    def _robust_target_weight(self, abs_z: np.ndarray) -> np.ndarray:
        soft = max(float(self.config.robust_z_soft), 0.0)
        hard = max(float(self.config.robust_z_hard), soft + np.finfo(float).eps)
        min_weight = float(self.config.robust_min_weight)

        target = np.ones_like(abs_z, dtype=float)
        transition = (abs_z > soft) & (abs_z < hard)
        x = (abs_z[transition] - soft) / (hard - soft)
        target[transition] = min_weight + (1.0 - min_weight) * (1.0 - x) ** 2
        target[abs_z >= hard] = min_weight
        return np.clip(target, min_weight, 1.0)

    def _smooth_target_weight(
        self,
        time: np.ndarray,
        target_weight: np.ndarray,
        half_width: float,
    ) -> np.ndarray:
        half_width = max(float(half_width), 0.0)
        if half_width == 0.0 or target_weight.size <= 1:
            return target_weight

        order = np.argsort(time)
        t = np.asarray(time[order], dtype=float)
        outlier_score = 1.0 - np.asarray(target_weight[order], dtype=float)
        csum = np.r_[0.0, np.cumsum(outlier_score)]
        smoothed_score = np.zeros_like(outlier_score)
        left = 0
        right = 0
        n = t.size
        for i in range(n):
            while left < n and t[left] < t[i] - half_width:
                left += 1
            while right < n and t[right] <= t[i] + half_width:
                right += 1
            count = max(right - left, 1)
            smoothed_score[i] = (csum[right] - csum[left]) / count

        smoothed = 1.0 - smoothed_score
        result = np.empty_like(smoothed)
        result[order] = smoothed
        return np.clip(result, float(self.config.robust_min_weight), 1.0)

    def _signal_mask_from_weight_and_residual(
        self,
        *,
        fit: SingleLensFitResult,
        point_weight: np.ndarray,
    ) -> np.ndarray:
        residual = np.asarray(jax.device_get(fit.residual), dtype=float)
        ferr = np.asarray(fit.ferr, dtype=float)
        z = residual / ferr
        return (
            (point_weight <= float(self.config.signal_weight_threshold))
            & (np.abs(z) >= float(self.config.signal_min_abs_z))
        )

    def _flat_baseline_diagnostic(
        self,
        fit: SingleLensFitResult,
        signal_mask: np.ndarray,
        *,
        forced: bool = False,
    ) -> FlatBaselineDiagnostic:
        params = np.asarray(fit.params, dtype=float)
        u0 = float(params[2]) if params.size >= 3 and np.isfinite(params[2]) else float("nan")

        empty = FlatBaselineDiagnostic(
            use_flat_baseline=bool(forced),
            peak_in_mask=False,
            u0=u0,
            n_peak_support=0,
            n_unmasked_peak_support=0,
            masked_peak_fraction=0.0,
            dchi2_flat_minus_pspl=0.0,
            improvement_n_eff=0.0,
            improvement_peak_frac=0.0,
        )
        if params.size < 3 or not np.any(signal_mask):
            return empty

        t0, _tE, u0 = params[:3]
        if not np.isfinite(t0) or not np.isfinite(u0):
            return empty

        time = np.asarray(fit.time, dtype=float)
        peak_index = int(np.argmin(np.abs(time - float(t0))))
        peak_in_mask = bool(signal_mask[peak_index])

        fs = float(np.asarray(fit.fs))
        fb = float(np.asarray(fit.fb))
        if abs(fs) > 1e-12:
            model_flux = _single_lens_model_flux(fit, np.asarray(fit.time, dtype=float))
            A = (model_flux - fb) / fs
            amp = np.maximum(A - 1.0, 0.0)
        else:
            amp = np.zeros_like(signal_mask, dtype=float)
        peak_amp = float(np.max(amp)) if amp.size else 0.0
        if peak_amp > 0.0:
            peak_support = amp >= float(self.config.flat_baseline_peak_support_frac) * peak_amp
        else:
            peak_support = np.zeros_like(signal_mask, dtype=bool)
        n_peak_support = int(np.sum(peak_support))
        n_unmasked_peak_support = int(np.sum(peak_support & (~signal_mask)))
        masked_peak_fraction = (
            float(np.sum(peak_support & signal_mask)) / float(n_peak_support)
            if n_peak_support > 0
            else 0.0
        )

        keep = ~np.asarray(signal_mask, dtype=bool)
        residual = np.asarray(jax.device_get(fit.residual), dtype=float)
        ferr = np.asarray(fit.ferr, dtype=float)
        flux = np.asarray(fit.flux, dtype=float)
        if np.any(keep):
            w = 1.0 / np.maximum(ferr[keep], 1e-12) ** 2
            flat = float(np.sum(w * flux[keep]) / np.sum(w))
            z_flat = (flux[keep] - flat) / ferr[keep]
            z_pspl = residual[keep] / ferr[keep]
            improvement = np.maximum(z_flat * z_flat - z_pspl * z_pspl, 0.0)
            dchi2_flat_minus_pspl = float(np.sum(z_flat * z_flat) - np.sum(z_pspl * z_pspl))
            sum_imp = float(np.sum(improvement))
            sum_imp2 = float(np.sum(improvement * improvement))
            improvement_n_eff = (sum_imp * sum_imp / sum_imp2) if sum_imp2 > 0.0 else 0.0
            improvement_peak_frac = float(np.max(improvement) / sum_imp) if sum_imp > 0.0 else 0.0
        else:
            dchi2_flat_minus_pspl = 0.0
            improvement_n_eff = 0.0
            improvement_peak_frac = 0.0

        tiny_masked_peak = (
            bool(self.config.flat_baseline_on_masked_peak)
            and peak_in_mask
            and abs(float(u0)) <= float(self.config.flat_baseline_u0_threshold)
        )
        unsupported_peak = (
            n_unmasked_peak_support < int(self.config.flat_baseline_min_unmasked_peak_points)
            or masked_peak_fraction > float(self.config.flat_baseline_max_masked_peak_fraction)
            or dchi2_flat_minus_pspl < float(self.config.flat_baseline_min_dchi2_vs_flat)
        )

        return FlatBaselineDiagnostic(
            use_flat_baseline=bool(forced or (tiny_masked_peak and unsupported_peak)),
            peak_in_mask=peak_in_mask,
            u0=float(u0),
            n_peak_support=n_peak_support,
            n_unmasked_peak_support=n_unmasked_peak_support,
            masked_peak_fraction=masked_peak_fraction,
            dchi2_flat_minus_pspl=dchi2_flat_minus_pspl,
            improvement_n_eff=float(improvement_n_eff),
            improvement_peak_frac=float(improvement_peak_frac),
        )

    @staticmethod
    def _fit_flat_baseline_and_evaluate_full(
        *,
        time_j: jnp.ndarray,
        flux_j: jnp.ndarray,
        ferr_j: jnp.ndarray,
        keep_mask_np: np.ndarray,
        fit: SingleLensFitResult,
    ) -> SingleLensFitResult:
        ferr_safe = jnp.maximum(ferr_j, 1e-12)
        keep = np.asarray(keep_mask_np, dtype=bool)
        if not np.any(keep):
            keep = np.ones(int(time_j.shape[0]), dtype=bool)

        keep_j = jnp.asarray(keep)
        w = 1.0 / (ferr_safe[keep_j] ** 2)
        fb = jnp.sum(w * flux_j[keep_j]) / jnp.sum(w)
        fs = jnp.asarray(0.0, dtype=flux_j.dtype)
        model_flux = jnp.full_like(flux_j, fb)
        residual = flux_j - model_flux
        z = residual / ferr_safe
        chi2 = jnp.sum(z * z)
        n = int(time_j.shape[0])

        return SingleLensFitResult(
            time=np.asarray(time_j),
            flux=np.asarray(flux_j),
            ferr=np.asarray(ferr_safe),
            params=jnp.asarray(fit.params, dtype=time_j.dtype),
            param_names=tuple(getattr(fit, "param_names", ("t0", "tE", "u0"))),
            chi2=chi2,
            chi2_dof=chi2 / max(n - 1, 1),
            fs=fs,
            fb=fb,
            model_flux=model_flux,
            residual=residual,
            raw_params=jnp.asarray(fit.raw_params, dtype=time_j.dtype) if fit.raw_params is not None else None,
            parallax_projector=getattr(fit, "parallax_projector", None),
        )

    @staticmethod
    def _count_mask_intervals(mask: np.ndarray) -> int:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return 0
        return int(1 + np.sum(np.diff(idx) > 1))

    @staticmethod
    def _mask_total_width(time: np.ndarray, mask: np.ndarray) -> float:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return 0.0
        breaks = np.flatnonzero(np.diff(idx) > 1)
        starts = np.r_[idx[0], idx[breaks + 1]]
        ends = np.r_[idx[breaks], idx[-1]]
        total = 0.0
        for start, end in zip(starts, ends):
            total += float(time[end] - time[start])
        return total

    @staticmethod
    def _masked_chi2_dof(fit: SingleLensFitResult, keep_mask_np: np.ndarray) -> float:
        if not np.any(keep_mask_np):
            return float("inf")
        residual = np.asarray(jax.device_get(fit.residual), dtype=float)
        ferr = np.asarray(fit.ferr, dtype=float)
        keep = np.asarray(keep_mask_np, dtype=bool)
        z = residual[keep] / ferr[keep]
        dof = max(int(np.sum(keep)) - len(tuple(getattr(fit, "param_names", ()))), 1)
        return float(np.sum(z * z) / dof)

    @staticmethod
    def _evaluate_single_lens_fit_on_full_data(
        *,
        time_j: jnp.ndarray,
        flux_j: jnp.ndarray,
        ferr_j: jnp.ndarray,
        fit: SingleLensFitResult,
        fs,
        fb,
    ) -> SingleLensFitResult:
        ferr_safe = jnp.maximum(ferr_j, 1e-12)
        time_np = np.asarray(jax.device_get(time_j), dtype=float)
        model_flux_np = _single_lens_model_flux(fit, time_np)
        model_flux = jnp.asarray(model_flux_np, dtype=time_j.dtype)
        residual = flux_j - model_flux
        z = residual / ferr_safe
        chi2 = jnp.sum(z * z)
        n = int(time_j.shape[0])
        param_names = tuple(getattr(fit, "param_names", ("t0", "tE", "u0")))

        return SingleLensFitResult(
            time=np.asarray(time_j),
            flux=np.asarray(flux_j),
            ferr=np.asarray(ferr_safe),
            params=jnp.asarray(fit.params, dtype=time_j.dtype),
            param_names=param_names,
            chi2=chi2,
            chi2_dof=chi2 / max(n - len(param_names), 1),
            fs=fs,
            fb=fb,
            model_flux=model_flux,
            residual=residual,
            raw_params=jnp.asarray(fit.raw_params, dtype=time_j.dtype) if fit.raw_params is not None else None,
            parallax_projector=getattr(fit, "parallax_projector", None),
        )

    def _candidates_from_mask(
        self,
        time: np.ndarray,
        residual: np.ndarray,
        ferr: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[PlanetSignalCandidate, ...]:
        if not np.any(mask):
            return ()

        z = residual / ferr
        candidates: list[PlanetSignalCandidate] = []
        idx = np.flatnonzero(mask)
        breaks = np.flatnonzero(np.diff(idx) > 1)
        starts = np.r_[idx[0], idx[breaks + 1]]
        ends = np.r_[idx[breaks], idx[-1]] + 1

        for start, end in zip(starts, ends):
            sl = slice(int(start), int(end))
            n_points = int(end - start)
            if n_points < int(self.config.candidate_min_points):
                continue

            z_seg = z[sl]
            chi2 = float(np.sum(z_seg * z_seg))
            if chi2 < float(self.config.candidate_min_chi2):
                continue

            peak_local = int(np.argmax(np.abs(z_seg)))
            peak_index = int(start + peak_local)
            candidates.append(
                PlanetSignalCandidate(
                    start_index=int(start),
                    end_index=int(end),
                    t_start=float(time[start]),
                    t_end=float(time[end - 1]),
                    t_center=float(0.5 * (time[start] + time[end - 1])),
                    n_points=n_points,
                    chi2=chi2,
                    reduced_chi2=chi2 / max(n_points, 1),
                    max_abs_z=float(np.max(np.abs(z_seg))),
                    peak_time=float(time[peak_index]),
                    peak_z=float(z[peak_index]),
                    signed_sum_z=float(np.sum(z_seg)),
                )
            )

        return tuple(sorted(candidates, key=lambda c: c.chi2, reverse=True))
