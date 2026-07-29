from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from time import perf_counter
from typing import Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from .finder import Finder
from .anomaly_models import get_chi2_anom_masked, get_chi2_flat_masked
from .models import BestCandidate
from .plot import _adaptive_single_lens_curve, _single_lens_model_flux
from .singlelens_fit import SingleLensFitResult


@dataclass(frozen=True)
class PlanetSignalConfig:
    """
    Configuration for iterative single-lens refinement and signal extraction.

    It uses the existing template scan as a seed finder, masks the strongest
    unexplained local windows, and refits the selected single-lens baseline on
    the remaining data while preserving its model family (PSPL or FSPL).
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
    max_refined_chi2_dof_ratio: float = 1000.0
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
    beam_probe_only: bool = False
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

    @classmethod
    def fast(cls, **overrides) -> "PlanetSignalConfig":
        """Return the inexpensive first-pass beam configuration.

        A fast pass performs one grid scan and evaluates only its best
        interval.  It is intended for routine events, where a full beam
        search is needlessly expensive.  Callers can rerun with the regular
        configuration when this pass finds a credible signal or another
        physical-effect stage warrants the extra scrutiny.
        """
        values = {
            "baseline_mode": "beam_interval",
            "beam_max_iter": 1,
            "beam_width": 1,
            "beam_candidates_per_iter": 1,
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def probe(cls, **overrides) -> "PlanetSignalConfig":
        """Return a one-scan routing probe without baseline refits.

        This is for orchestration paths that will immediately run a full beam
        pass when the seed is credible.  It exposes the cached seed while
        avoiding duplicate masked fits during that escalation.
        """
        values = {"beam_probe_only": True}
        values.update(overrides)
        return cls.fast(**values)


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
class PlanetSignalTiming:
    """Wall-clock accounting for one extraction, excluding caller setup."""

    total_seconds: float = 0.0
    scan_seconds: float = 0.0
    n_scans: int = 0


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
class PlanetFeatureConfig:
    """Controls peak and dip measurement on an extracted residual signal."""

    smooth_points: int = 7
    min_abs_z: float = 5.0
    min_relative_strength: float = 0.1
    min_prominence: float = 3.0
    prominence_fraction: float = 0.15
    min_separation: float = 0.15
    duration_min_abs_z: float = 3.0
    duration_relative_height: float = 0.5
    # Keep a negative feature only when it is a deep, closed trough bracketed
    # by positive recoveries.  This rejects one-sided caustic wings while
    # retaining bump--dip--bump structures.
    allow_bracketed_dips: bool = True
    dip_bracket_min_peak_frac: float = 0.1
    dip_bracket_min_depth_ratio: float = 1.5
    dip_bracket_max_gap_factor: float = 4.0
    bracketed_peak_min_separation: float = 0.05


@dataclass(frozen=True)
class PlanetFeature:
    """One measured peak or dip, without a physical-shape interpretation."""

    kind: str
    index: int
    time: float
    t_start: float
    t_end: float
    timescale: float
    strength: float
    signed_z: float
    residual: float
    fractional_deviation: float
    magnification_ratio: float

    def summary_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "index": int(self.index),
            "time": float(self.time),
            "t_start": float(self.t_start),
            "t_end": float(self.t_end),
            "timescale": float(self.timescale),
            "strength": float(self.strength),
            "signed_z": float(self.signed_z),
            "residual": float(self.residual),
            "fractional_deviation": float(self.fractional_deviation),
            "magnification_ratio": float(self.magnification_ratio),
        }


@dataclass(frozen=True)
class PlanetFeatureResult:
    """Flat peak/dip measurements for one extracted planet signal."""

    peaks: Tuple[PlanetFeature, ...]
    dips: Tuple[PlanetFeature, ...]

    @property
    def n_peaks(self) -> int:
        return len(self.peaks)

    @property
    def n_dips(self) -> int:
        return len(self.dips)

    @property
    def features(self) -> Tuple[PlanetFeature, ...]:
        return tuple(sorted((*self.peaks, *self.dips), key=lambda feature: feature.time))

    @property
    def strongest(self) -> Optional[PlanetFeature]:
        return max(self.features, key=lambda feature: feature.strength, default=None)

    def summary_dict(self) -> dict[str, object]:
        strongest = self.strongest
        row: dict[str, object] = {
            "n_peaks": self.n_peaks,
            "n_dips": self.n_dips,
            "n_features": self.n_peaks + self.n_dips,
        }
        if strongest is not None:
            row.update(
                {
                    "strongest_kind": strongest.kind,
                    "strongest_time": strongest.time,
                    "strongest_timescale": strongest.timescale,
                    "strongest_strength": strongest.strength,
                }
            )
        return row

    def feature_dicts(self) -> list[dict[str, object]]:
        return [feature.summary_dict() for feature in self.features]

    def summary_table(self):
        """Return one row per feature as a pandas table when pandas is installed."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - optional convenience
            raise ImportError("PlanetFeatureResult.summary_table() requires pandas.") from exc
        return pd.DataFrame(self.feature_dicts())

    def summary_text(self) -> str:
        lines = [f"peaks={self.n_peaks}, dips={self.n_dips}"]
        for feature in self.features:
            lines.append(
                f"{feature.kind}: t={feature.time:.6g}, "
                f"timescale={feature.timescale:.6g}, "
                f"strength={feature.strength:.3g} sigma"
            )
        return "\n".join(lines)


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
    initial_seed: Optional[BestCandidate] = None
    timing: PlanetSignalTiming = PlanetSignalTiming()

    def measure_features(
        self,
        config: Optional[PlanetFeatureConfig] = None,
    ) -> PlanetFeatureResult:
        """
        Measure prominent peaks and dips in the extracted residual signal.

        Parameters
        ----------
        config : PlanetFeatureConfig, optional
            Smoothing, prominence, separation, and duration thresholds.

        Returns
        -------
        PlanetFeatureResult
            Flat peak and dip lists. Each feature reports its time,
            threshold-crossing timescale, absolute z-score strength, signed
            residual, fractional deviation, and magnification ratio.
        """
        cfg = PlanetFeatureConfig() if config is None else config
        return _PlanetFeatureDetector(cfg).run(self)

    def plot_signal(
        self,
        *,
        show: bool = True,
        peak_xlim: Optional[tuple[float, float]] = None,
        signal_xlim: Optional[tuple[float, float]] = None,
        peak_tE_width: float = 1.5,
        signal_pad: float = 0.5,
        max_signal_width: float = 6.0,
        show_features: bool = True,
        feature_config: Optional[PlanetFeatureConfig] = None,
    ):
        """
        Plot the refined baseline and highlight extracted signal points.

        Parameters
        ----------
        show : bool, optional
            Display the figure immediately when True.
        peak_xlim, signal_xlim : tuple[float, float], optional
            Explicit time limits for the full-event and signal panels.
        peak_tE_width, signal_pad, max_signal_width : float, optional
            Controls used to derive plot limits when explicit limits are not
            supplied.
        show_features : bool, optional
            Overlay measured peak and dip positions and durations.
        feature_config : PlanetFeatureConfig, optional
            Configuration used for the measurements.

        Returns
        -------
        tuple
            ``(fig, axes)`` with peak light curve, signal zoom, and residual
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
        model_kind = str(getattr(self.refined_fit, "model_kind", "single_lens"))
        model_label = {
            "pspl": "refined PSPL",
            "fspl": "refined FSPL",
            "fspl_vbm_fd": "refined FSPL",
            "fspl_space_parallax": "refined FSPL+parallax",
        }.get(model_kind, "refined single-lens")
        ax_peak.plot(t_peak_model, f_peak_model, c="k", lw=2.0, label=model_label, zorder=1)
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

        if show_features:
            features = self.measure_features(feature_config)
            self._draw_feature_overlay(
                axes=(ax_peak, ax_zoom, ax_res),
                features=features,
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
    def _draw_feature_overlay(
        *,
        axes,
        features: PlanetFeatureResult,
    ) -> None:
        ax_peak, ax_zoom, ax_res = axes
        signal_alpha = 0.14
        label = f"peaks: {features.n_peaks}, dips: {features.n_dips}"
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

        used_labels: set[str] = set()
        for feature in features.features:
            is_dip = feature.kind == "dip"
            color = "tab:red" if is_dip else "tab:green"
            line_style = "--" if is_dip else "-"
            line_label = feature.kind if feature.kind not in used_labels else None
            used_labels.add(feature.kind)
            for ax in (ax_peak, ax_zoom, ax_res):
                if feature.t_end > feature.t_start:
                    ax.axvspan(
                        feature.t_start,
                        feature.t_end,
                        color=color,
                        alpha=signal_alpha,
                        lw=0,
                        zorder=0.5,
                    )
                ax.axvline(
                    feature.time,
                    color=color,
                    ls=line_style,
                    lw=1.5,
                    alpha=0.9,
                    label=line_label if ax is ax_peak else None,
                    zorder=4,
                )

@dataclass
class _PlanetFeatureDetector:
    """Measure prominent peaks and dips in an extracted residual signal."""

    config: PlanetFeatureConfig = PlanetFeatureConfig()

    def run(self, result: PlanetSignalResult) -> PlanetFeatureResult:
        time = np.asarray(result.time, dtype=float)
        flux = np.asarray(result.flux, dtype=float)
        ferr = np.asarray(result.ferr, dtype=float)
        residual = np.asarray(result.refined_residual, dtype=float)
        mask = np.asarray(result.signal_mask, dtype=bool)

        if not np.any(mask):
            return PlanetFeatureResult(peaks=(), dips=())

        z = residual / np.maximum(ferr, 1e-12)
        z_smooth = self._smooth(z, int(self.config.smooth_points))
        model_flux = np.asarray(result.refined_fit.model_flux, dtype=float)
        peaks: list[PlanetFeature] = []
        for start, end in self._mask_slices(mask):
            indices = self._prominent_extrema(
                time,
                z_smooth,
                start=start,
                end=end,
                sign=1.0,
                min_separation=(
                    min(
                        float(self.config.min_separation),
                        float(self.config.bracketed_peak_min_separation),
                    )
                    if bool(self.config.allow_bracketed_dips)
                    else None
                ),
            )
            peaks.extend(
                self._features_from_indices(
                    kind="peak",
                    time=time,
                    flux=flux,
                    residual=residual,
                    z=z,
                    z_smooth=z_smooth,
                    model_flux=model_flux,
                    indices=indices,
                    component_start=start,
                    component_end=end,
                    sign=1.0,
                    result=result,
                )
            )

        dips: list[PlanetFeature] = []
        for start, end in self._mask_slices(mask):
            indices = self._prominent_extrema(
                time,
                z_smooth,
                start=start,
                end=end,
                sign=-1.0,
            )
            dips.extend(
                self._features_from_indices(
                    kind="dip",
                    time=time,
                    flux=flux,
                    residual=residual,
                    z=z,
                    z_smooth=z_smooth,
                    model_flux=model_flux,
                    indices=indices,
                    component_start=start,
                    component_end=end,
                    sign=-1.0,
                    result=result,
                    require_closed=True,
                )
            )

        if peaks and dips and bool(self.config.allow_bracketed_dips):
            dips = [dip for dip in dips if self._is_bracketed_dip(dip, peaks, result)]

        # Positive caustic/bump features still take precedence over ordinary
        # negative wings.  The explicit bracketed-dip exception above keeps a
        # genuine bump--dip--bump structure without reopening every dip.
        if peaks or dips:
            return PlanetFeatureResult(
                peaks=tuple(sorted(peaks, key=lambda feature: feature.time)),
                dips=tuple(sorted(dips, key=lambda feature: feature.time)),
            )

        return PlanetFeatureResult(
            peaks=(),
            dips=tuple(sorted(dips, key=lambda feature: feature.time)),
        )

    def _is_bracketed_dip(
        self,
        dip: PlanetFeature,
        peaks: list[PlanetFeature],
        result: PlanetSignalResult,
    ) -> bool:
        """Keep only a deep trough with local positive recovery on both sides."""
        left = [peak for peak in peaks if peak.time < dip.time]
        right = [peak for peak in peaks if peak.time > dip.time]
        if not left or not right:
            return False
        left_peak = max(left, key=lambda peak: peak.time)
        right_peak = min(right, key=lambda peak: peak.time)

        peak_floor = max(
            float(self.config.min_abs_z),
            float(self.config.dip_bracket_min_peak_frac) * float(dip.strength),
        )
        if left_peak.strength < peak_floor or right_peak.strength < peak_floor:
            return False
        if dip.strength < float(self.config.dip_bracket_min_depth_ratio) * max(
            left_peak.strength, right_peak.strength
        ):
            return False

        dt = np.diff(np.asarray(result.time, dtype=float))
        positive_dt = dt[dt > 0.0]
        cadence = float(np.nanmedian(positive_dt)) if positive_dt.size else 0.0
        local_scale = max(
            float(dip.timescale),
            float(left_peak.timescale),
            float(right_peak.timescale),
            3.0 * cadence,
            np.finfo(float).eps,
        )
        max_gap = float(self.config.dip_bracket_max_gap_factor) * local_scale
        return (
            float(dip.time) - float(left_peak.time) <= max_gap
            and float(right_peak.time) - float(dip.time) <= max_gap
        )

    def _prominent_extrema(
        self,
        time: np.ndarray,
        values: np.ndarray,
        *,
        start: int,
        end: int,
        sign: float,
        min_separation: Optional[float] = None,
    ) -> list[int]:
        segment = sign * values[start:end]
        if segment.size == 0:
            return []

        min_height = float(self.config.min_abs_z)
        peak_height = float(np.max(segment))
        if not np.isfinite(peak_height) or peak_height < min_height:
            return []
        min_height = max(
            min_height,
            float(self.config.min_relative_strength) * peak_height,
        )

        extrema = self._monotonic_extrema(segment)
        if not extrema:
            return []

        local: list[int] = []
        for offset in extrema:
            value = float(segment[offset])
            if value < min_height:
                continue
            prominence = self._local_prominence(segment, int(offset))
            min_prominence = max(
                float(self.config.min_prominence),
                float(self.config.prominence_fraction) * value,
            )
            if prominence >= min_prominence:
                local.append(start + int(offset))

        local.sort(key=lambda idx: sign * values[idx], reverse=True)
        return self._suppress_nearby_extrema(
            time,
            values,
            local,
            sign,
            min_separation=min_separation,
        )

    @staticmethod
    def _local_prominence(segment: np.ndarray, offset: int) -> float:
        """Return prominence relative to the higher of the two local floors."""
        values = np.asarray(segment, dtype=float)
        height = float(values[offset])

        left_floor = height
        for i in range(int(offset) - 1, -1, -1):
            value = float(values[i])
            if value > height:
                break
            left_floor = min(left_floor, value)

        right_floor = height
        for i in range(int(offset) + 1, values.size):
            value = float(values[i])
            if value > height:
                break
            right_floor = min(right_floor, value)

        return max(height - max(left_floor, right_floor), 0.0)

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
        min_separation: Optional[float] = None,
    ) -> list[int]:
        selected: list[int] = []
        min_sep = (
            float(self.config.min_separation)
            if min_separation is None
            else max(float(min_separation), 0.0)
        )
        for index in extrema:
            far_enough = all(
                abs(float(time[index]) - float(time[kept])) >= min_sep
                for kept in selected
            )
            if far_enough:
                selected.append(index)
        selected.sort()
        return selected

    def _features_from_indices(
        self,
        *,
        kind: str,
        result: PlanetSignalResult,
        time: np.ndarray,
        flux: np.ndarray,
        residual: np.ndarray,
        z: np.ndarray,
        z_smooth: np.ndarray,
        model_flux: np.ndarray,
        indices: list[int],
        component_start: int,
        component_end: int,
        sign: float,
        require_closed: bool = False,
    ) -> list[PlanetFeature]:
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

        features: list[PlanetFeature] = []
        for cell_start, cell_end in cells:
            feature = self._feature_from_index(
                kind=kind,
                result=result,
                time=time,
                flux=flux,
                residual=residual,
                z=z,
                z_smooth=z_smooth,
                model_flux=model_flux,
                index=self._raw_extremum_in_cell(
                    z=z,
                    cell_start=max(component_start, cell_start),
                    cell_end=min(component_end, cell_end),
                    sign=sign,
                ),
                cell_start=max(component_start, cell_start),
                cell_end=min(component_end, cell_end),
                sign=sign,
                require_closed=require_closed,
            )
            if feature is not None:
                features.append(feature)
        return features

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

    def _feature_from_index(
        self,
        *,
        kind: str,
        result: PlanetSignalResult,
        time: np.ndarray,
        flux: np.ndarray,
        residual: np.ndarray,
        z: np.ndarray,
        z_smooth: np.ndarray,
        model_flux: np.ndarray,
        index: int,
        cell_start: int,
        cell_end: int,
        sign: float,
        require_closed: bool = False,
    ) -> Optional[PlanetFeature]:
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
        if require_closed and (lo == cell_start or hi + 1 == cell_end):
            return None
        t_start, t_end = self._interpolated_threshold_bounds(
            time=time,
            values=values,
            lo=lo,
            hi=hi,
            cell_start=cell_start,
            cell_end=cell_end,
            threshold=threshold,
        )

        magnification_ratio = self._magnification_ratio(
            result=result,
            flux=float(flux[index]),
            model_flux=float(model_flux[index]),
        )
        baseline = float(model_flux[index])
        fractional_deviation = (
            float(residual[index]) / baseline
            if np.isfinite(baseline) and abs(baseline) > 1e-30
            else float("nan")
        )

        return PlanetFeature(
            kind=kind,
            index=int(index),
            time=float(time[index]),
            t_start=float(t_start),
            t_end=float(t_end),
            timescale=max(float(t_end) - float(t_start), 0.0),
            strength=abs(float(z[index])),
            signed_z=float(z[index]),
            residual=float(residual[index]),
            fractional_deviation=float(fractional_deviation),
            magnification_ratio=float(magnification_ratio),
        )

    @staticmethod
    def _interpolated_threshold_bounds(
        *,
        time: np.ndarray,
        values: np.ndarray,
        lo: int,
        hi: int,
        cell_start: int,
        cell_end: int,
        threshold: float,
    ) -> tuple[float, float]:
        def crossing(left: int, right: int) -> float:
            t0 = float(time[left])
            t1 = float(time[right])
            y0 = float(values[left])
            y1 = float(values[right])
            if not (np.isfinite(t0) and np.isfinite(t1) and np.isfinite(y0) and np.isfinite(y1)):
                return t1
            denom = y1 - y0
            if denom == 0.0:
                return 0.5 * (t0 + t1)
            frac = (float(threshold) - y0) / denom
            frac = min(max(float(frac), 0.0), 1.0)
            return t0 + frac * (t1 - t0)

        start = float(time[lo])
        end = float(time[hi])
        if lo > cell_start:
            start = crossing(lo - 1, lo)
        if hi + 1 < cell_end:
            end = crossing(hi, hi + 1)
        if end < start:
            start, end = end, start
        return start, end

    @staticmethod
    def _magnification_ratio(
        *,
        result: PlanetSignalResult,
        flux: float,
        model_flux: float,
    ) -> float:
        fs = float(np.asarray(result.refined_fit.fs))
        fb = float(np.asarray(result.refined_fit.fb))
        if np.isfinite(fs) and abs(fs) > 1e-12:
            pspl_mag = (model_flux - fb) / fs
            observed_mag = (flux - fb) / fs
            ratio = observed_mag / pspl_mag if abs(pspl_mag) > 1e-12 else np.nan
        else:
            ratio = flux / model_flux if abs(model_flux) > 1e-12 else np.nan
        return float(ratio)

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
        prior_signal_windows: tuple[tuple[float, float], ...] = (),
        initial_fit: Optional[SingleLensFitResult] = None,
        initial_seed: Optional[BestCandidate] = None,
    ) -> PlanetSignalResult:
        """
        Separate localized residual signal while refining a baseline fit.

        Parameters
        ----------
        time, flux, ferr : array-like
            One-dimensional light-curve arrays with positive flux errors.
        x0 : array-like, optional
            Initial nonlinear single-lens parameters. Required when
            ``refit=False``.
        refit : bool, optional
            Optimize the initial single-lens fit when True. When False, keep
            ``x0`` fixed and solve only the linear flux terms.
        verbose : bool, optional
            Emit refinement progress through the finder's logging path.
        prior_signal_windows : tuple[tuple[float, float], ...], optional
            Known signal windows expressed as ``(center_time, half_width)``.
            They are added to the final mask and trigger one guarded refit.
        initial_seed : BestCandidate, optional
            Cached first grid-scan result for this exact initial fit and input
            data.  Supplying it lets a full beam pass continue after a fast
            pass without repeating the initial grid scan.

        Returns
        -------
        PlanetSignalResult
            Initial and refined fits, residuals, signal mask, selected
            intervals, and refinement history.

        Raises
        ------
        ValueError
            If ``baseline_mode`` is not ``"mask"``, ``"robust"``, or
            ``"beam_interval"``, or if fixed-baseline inputs are incomplete.

        Notes
        -----
        The extractor uses the finder's template grid to propose signal
        intervals, but does not call ``Finder.run``. Read
        :doc:`planet_features` for mode selection and interpretation.
        """
        started = perf_counter()
        self._scan_seconds = 0.0
        self._n_scans = 0
        time_j, flux_j, ferr_j, x0_j, time_np, flux_np, ferr_np = self.finder._to_arrays(
            time, flux, ferr, x0
        )
        self.finder._ensure_fitter(float(np.median(time_np)))

        if initial_fit is None:
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
            current_fit, signal_mask, point_weight, iterations, observed_initial_seed = self._run_beam_interval_baseline(
                initial_fit=initial_fit,
                time_j=time_j,
                flux_j=flux_j,
                ferr_j=ferr_j,
                time_np=time_np,
                verbose=verbose,
                initial_seed=initial_seed,
            )
        else:
            raise ValueError(
                "PlanetSignalConfig.baseline_mode must be 'mask', 'robust', or 'beam_interval'."
            )

        if mode != "beam_interval":
            observed_initial_seed = None

        if self._fit_is_catastrophically_worse(initial_fit, current_fit):
            current_fit = initial_fit
            signal_mask = np.zeros(time_np.shape, dtype=bool)
            point_weight = np.ones(time_np.shape, dtype=float)
            iterations = []

        prior_mask = self._prior_signal_window_mask(time_np, prior_signal_windows)
        if np.any(prior_mask & ~signal_mask):
            signal_mask = signal_mask | prior_mask
            point_weight = np.where(signal_mask, 0.0, point_weight)
            candidate_fit = self._fit_masked_single_lens_and_evaluate_full(
                time_j=time_j,
                flux_j=flux_j,
                ferr_j=ferr_j,
                keep_mask_np=~signal_mask,
                x0_j=self._raw_params_for_refit(current_fit),
                model_kind=getattr(current_fit, "model_kind", None),
            )
            if not self._fit_is_catastrophically_worse(current_fit, candidate_fit):
                current_fit = candidate_fit

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
            initial_seed=observed_initial_seed,
            timing=PlanetSignalTiming(
                total_seconds=float(perf_counter() - started),
                scan_seconds=float(self._scan_seconds),
                n_scans=int(self._n_scans),
            ),
        )

    @staticmethod
    def _prior_signal_window_mask(
        time: np.ndarray,
        prior_signal_windows: tuple[tuple[float, float], ...],
    ) -> np.ndarray:
        mask = np.zeros(np.asarray(time).shape, dtype=bool)
        for center, half_width in tuple(prior_signal_windows):
            center_f = float(center)
            half_width_f = max(float(half_width), 0.0)
            if not (np.isfinite(center_f) and np.isfinite(half_width_f)):
                continue
            mask |= np.abs(np.asarray(time, dtype=float) - center_f) <= half_width_f
        return mask

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
                model_kind=getattr(current_fit, "model_kind", None),
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
                model_kind=getattr(current_fit, "model_kind", None),
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
        initial_seed: Optional[BestCandidate] = None,
    ) -> tuple[SingleLensFitResult, np.ndarray, np.ndarray, list[PlanetSignalIteration], Optional[BestCandidate]]:
        empty_mask = np.zeros(time_np.shape, dtype=bool)
        branches = (
            _BeamBranch(
                score=self._beam_score(initial_fit, empty_mask),
                fit=initial_fit,
                mask=empty_mask,
                iterations=(),
            ),
        )
        observed_initial_seed = initial_seed

        for iteration in range(max(0, int(self.config.beam_max_iter))):
            next_branches: list[_BeamBranch] = list(branches)
            for branch in branches:
                if iteration == 0 and branch is branches[0] and initial_seed is not None:
                    seed = initial_seed
                else:
                    residual_for_seed = self._suppress_masked_residual(branch.fit.residual, branch.mask)
                    seed = self._scan_best(time_j, residual_for_seed, ferr_j, time_np, verbose=verbose)
                    if iteration == 0 and branch is branches[0]:
                        observed_initial_seed = seed
                if seed is None or not np.isfinite(seed.dchi2) or seed.dchi2 < float(self.config.seed_min_dchi2):
                    # At the first pass every branch has the same unmasked
                    # baseline, so another iteration would repeat the same
                    # expensive grid scan.  On later passes it is likewise
                    # safe to retire this branch; the loop terminates below
                    # when no branch made progress.
                    continue

                if self.config.beam_probe_only:
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
                            model_kind=getattr(branch.fit, "model_kind", None),
                        )
                    except ValueError:
                        continue
                    old_unmasked = self._masked_chi2_dof(branch.fit, ~combined)
                    new_unmasked = self._masked_chi2_dof(candidate_fit, ~combined)
                    allowed = old_unmasked * (1.0 + float(self.config.max_unmasked_chi2_dof_increase))
                    if np.isfinite(old_unmasked) and new_unmasked > allowed:
                        continue
                    if self._fit_is_catastrophically_worse(branch.fit, candidate_fit):
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

            retained = tuple(
                sorted(next_branches, key=lambda b: b.score)[: max(1, int(self.config.beam_width))]
            )
            # Do not rescan unchanged branches.  This is particularly
            # important for ordinary events whose first seed is below the
            # signal threshold, and for rejected interval proposals.
            if not any(
                all(branch is not existing for existing in branches)
                for branch in retained
            ):
                break
            branches = retained

        best_branch = min(branches, key=lambda b: b.score)
        point_weight = np.where(best_branch.mask, 0.0, 1.0)
        return best_branch.fit, best_branch.mask, point_weight, list(best_branch.iterations), observed_initial_seed

    def _fit_is_catastrophically_worse(
        self,
        reference_fit: SingleLensFitResult,
        candidate_fit: SingleLensFitResult,
    ) -> bool:
        ratio = float(self.config.max_refined_chi2_dof_ratio)
        if not np.isfinite(ratio) or ratio <= 0.0:
            return False
        reference = float(np.asarray(reference_fit.chi2_dof))
        candidate = float(np.asarray(candidate_fit.chi2_dof))
        if not (np.isfinite(reference) and np.isfinite(candidate)):
            return False
        return candidate > max(reference * ratio, reference + 1.0)

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
        started = perf_counter()
        try:
            seasons, clusters_all, grid_metrics_all = self.finder.runner.run(
                time_j=time_j,
                residual_j=residual_j,
                ferr_j=ferr_j,
                time_np=time_np,
                verbose=verbose,
            )
        finally:
            self._scan_seconds = getattr(self, "_scan_seconds", 0.0) + (perf_counter() - started)
            self._n_scans = getattr(self, "_n_scans", 0) + 1
        if bool(self.config.scan_unimodal_filter) and grid_metrics_all.size:
            clusters_all, grid_metrics_all = self._unimodal_scan_clusters(
                time_j=time_j,
                residual_j=residual_j,
                ferr_j=ferr_j,
                grid_metrics_all=grid_metrics_all,
            )
        return self.finder._pick_best_candidate(
            clusters_all,
            grid_metrics_all,
            seasons=seasons,
        )

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

        # Transfer the candidate decisions only once.  The former per-grid
        # loop performed several device_get calls for every one of the top-N
        # candidates, which dominated the C++ grid evaluation itself.
        is_unimodal = np.asarray(
            jax.device_get(
                self._scan_grids_are_unimodal(
                    time_j=time_j,
                    residual_j=residual_j,
                    ferr_j=ferr_j,
                    t0_j=jnp.asarray(metrics[eval_idx, 0], dtype=time_j.dtype),
                    teff_j=jnp.asarray(metrics[eval_idx, 1], dtype=time_j.dtype),
                    teff_coeff=float(self.finder.config.teff_coeff),
                    min_pts=int(self.finder.config.min_pts_in_window),
                    min_improvement=float(self.config.scan_unimodal_min_improvement),
                    peak_frac=float(self.config.scan_unimodal_peak_frac),
                    smooth_points=int(self.config.scan_unimodal_smooth_points),
                    max_lobes=int(self.config.scan_unimodal_max_lobes),
                )
            ),
            dtype=bool,
        )
        keep = np.zeros(metrics.shape[0], dtype=bool)
        keep[eval_idx] = is_unimodal

        filtered = metrics[keep]
        if filtered.size == 0:
            return np.zeros((0, 3), dtype=float), filtered

        clusters = self.finder.runner.extractor.iterative_anomaly_extraction(
            filtered[:, 0],
            filtered[:, 1],
            filtered[:, 2],
        )
        return clusters, filtered

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=(
            "teff_coeff",
            "min_pts",
            "min_improvement",
            "peak_frac",
            "smooth_points",
            "max_lobes",
        ),
    )
    def _scan_grids_are_unimodal(
        *,
        time_j: jnp.ndarray,
        residual_j: jnp.ndarray,
        ferr_j: jnp.ndarray,
        t0_j: jnp.ndarray,
        teff_j: jnp.ndarray,
        teff_coeff: float,
        min_pts: int,
        min_improvement: float,
        peak_frac: float,
        smooth_points: int,
        max_lobes: int,
    ) -> jnp.ndarray:
        """Evaluate all candidate lobe tests in one JAX dispatch."""
        width = max(1, int(smooth_points))
        if width % 2 == 0:
            width += 1
        pad = width // 2
        kernel = jnp.ones((width,), dtype=time_j.dtype) / float(width)
        weights = 1.0 / (ferr_j ** 2)

        def one(t0, teff):
            window = jnp.abs(time_j - t0) < float(teff_coeff) * jnp.abs(teff)
            chi2_anom, chi2s_anom = get_chi2_anom_masked(
                t0, teff, time_j, residual_j, weights, window
            )
            chi2_flat, chi2s_flat = get_chi2_flat_masked(residual_j, weights, window)
            improvement = jnp.where(window, jnp.maximum(chi2s_flat - chi2s_anom, 0.0), 0.0)
            peak = jnp.max(improvement)
            threshold = jnp.maximum(float(min_improvement), float(peak_frac) * peak)
            if width > 1:
                smoothed = jnp.convolve(jnp.pad(improvement, (pad, pad), mode="edge"), kernel, mode="valid")
            else:
                smoothed = improvement
            active = window & (smoothed >= threshold)
            starts = active & ~jnp.concatenate((jnp.zeros((1,), dtype=bool), active[:-1]))
            lobes = jnp.sum(starts)
            return (
                (jnp.sum(window) >= int(min_pts))
                & jnp.isfinite(peak)
                & (peak > 0.0)
                & (lobes <= int(max_lobes))
            )

        return jax.vmap(one)(t0_j, teff_j)

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
        model_kind: Optional[str] = None,
    ) -> SingleLensFitResult:
        if int(np.sum(keep_mask_np)) < 4:
            raise ValueError("Not enough unmasked points to refit single-lens model.")

        keep_j = jnp.asarray(keep_mask_np)
        fitter = self.finder.fitter
        if model_kind is not None and hasattr(fitter, "fit_fixed_model"):
            masked_fit = fitter.fit_fixed_model(
                time_j[keep_j],
                flux_j[keep_j],
                ferr_j[keep_j],
                x0_j,
                model_kind=model_kind,
            )
        else:
            masked_fit = fitter.fit(time_j[keep_j], flux_j[keep_j], ferr_j[keep_j], x0_j)
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
        model_kind: Optional[str] = None,
    ) -> SingleLensFitResult:
        weight_j = jnp.asarray(np.clip(point_weight, float(self.config.robust_min_weight), 1.0))
        ferr_eff_j = ferr_j / jnp.sqrt(weight_j)
        fitter = self.finder.fitter
        if model_kind is not None and hasattr(fitter, "fit_fixed_model"):
            weighted_fit = fitter.fit_fixed_model(
                time_j,
                flux_j,
                ferr_eff_j,
                x0_j,
                model_kind=model_kind,
            )
        else:
            weighted_fit = fitter.fit(time_j, flux_j, ferr_eff_j, x0_j)
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

        evaluated = SingleLensFitResult(
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
        return evaluated

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

        evaluated = SingleLensFitResult(
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
        # Keep the selected model family attached to the full-data evaluation.
        # This is used by subsequent masked/weighted refits to avoid silently
        # switching from FSPL to PSPL after the anomaly has been removed.
        for attr in ("model_kind", "bic", "model_selection"):
            if hasattr(fit, attr):
                object.__setattr__(evaluated, attr, getattr(fit, attr))
        return evaluated

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
