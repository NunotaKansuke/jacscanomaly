"""Bump / dip / caustic classification of residual structure.

:func:`measure_features` measures the residual of a single-lens fit inside
the windows of :class:`~jacscanomaly.Candidate` objects, or inside the
signal segments of a :class:`~jacscanomaly.SignalMask`, and classifies the
structure into three kinds:

* ``"bump"``    -- a prominent positive excursion,
* ``"dip"``     -- a closed negative excursion,
* ``"caustic"`` -- a bump that is significantly left/right asymmetric, the
  shape a source makes when it crosses or passes close to a caustic.

A caustic is a bump: it replaces the bump in the result, with its asymmetry
test in :attr:`Feature.caustic`.

Asymmetry test
--------------
Around each bump the raw z = residual / ferr is fitted by a Lorentzian
``A / (1 + ((t - tc) / w)**2)`` and by a split Lorentzian whose half-width is
``w_left`` before ``tc`` and ``w_right`` after it; both carry a local linear
baseline, and A and the baseline are fitted linearly. The split model adds
one parameter to a nested symmetric one, so for a symmetric bump
``delta_chi2 = chi2_sym - chi2_split`` follows chi-square with one degree of
freedom. A bump is a caustic when ``delta_chi2 >= min_asymmetry_dchi2`` and
the widths differ by at least ``min_width_ratio``. A fast rise and slow decay
(``w_left < w_right``) is labelled ``"entry"``, the reverse ``"exit"``; the
label is a shape description only (a near-caustic passage looks the same).

Assumptions: the bump's shape near its peak is captured by a (split)
Lorentzian; the noise is independent with the given ``ferr``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Optional, Sequence

import numpy as np

from .candidate import Candidate, CandidateLike, as_candidates
from .signal_mask import SignalMask


@dataclass(frozen=True)
class FeatureConfig:
    """Controls for extremum measurement and the caustic asymmetry test."""

    smooth_points: int = 7
    # An extremum's lobe (the run where the smoothed residual keeps its sign)
    # must reach S/N = sum(z) / sqrt(n) >= min_snr, i.e. a one-parameter
    # constant offset over the lobe improves chi-square by >= min_snr**2.
    min_snr: float = 5.0
    min_relative_strength: float = 0.1
    min_prominence: float = 3.0
    prominence_fraction: float = 0.15
    min_separation: float = 0.15
    # A feature's duration is where the smoothed residual stays above
    # floor + duration_relative_height * prominence (half prominence).
    duration_relative_height: float = 0.5
    # With ``allow_bracketed_dips`` bumps are separated by the smaller
    # ``bracketed_peak_min_separation`` so that the two bumps around a dip
    # (bump--dip--bump) are both kept.
    allow_bracketed_dips: bool = True
    bracketed_peak_min_separation: float = 0.05
    # Caustic = bump whose split-Lorentzian fit beats the symmetric one by
    # delta_chi2 >= min_asymmetry_dchi2 (chi2_1 under symmetry: 25 = 5 sigma)
    # with max(w)/min(w) >= min_width_ratio.
    min_asymmetry_dchi2: float = 25.0
    min_width_ratio: float = 1.5
    # Fit window: peak +- max(asymmetry_window_timescales * timescale,
    # asymmetry_window_cadences * cadence), cut half way to the neighbouring bump or dip.
    asymmetry_window_timescales: float = 3.0
    asymmetry_window_cadences: float = 10.0

FeatureKind = Literal["bump", "dip", "caustic"]


@dataclass(frozen=True)
class Feature:
    """One classified residual structure.

    For a bump or dip, time/index locate the extremum and
    t_start/t_end its duration at half prominence; snr is
    sum(z) / sqrt(n) over its lobe (the run where the smoothed residual keeps
    its sign), so it also holds for a broad, low-amplitude feature whose
    point-wise strength is small.  A caustic is measured like a bump and
    caustic holds its asymmetry test.
    """

    kind: FeatureKind
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
    snr: float = float("nan")
    candidate: Optional[Candidate] = None
    caustic: Optional["CausticDetail"] = None

    def summary_dict(self) -> dict[str, object]:
        row: dict[str, object] = {
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
            "snr": float(self.snr),
            "candidate": None if self.candidate is None else self.candidate.label,
        }
        if self.caustic is not None:
            row["caustic"] = self.caustic.summary_dict()
        return row


@dataclass(frozen=True)
class CausticDetail:
    """Asymmetry test of a caustic bump (see the module docstring)."""

    orientation: Literal["entry", "exit"]
    peak_time: float
    left_width: float
    right_width: float
    delta_chi2: float

    @property
    def width_ratio(self) -> float:
        return max(self.left_width, self.right_width) / max(
            min(self.left_width, self.right_width), 1.0e-12
        )

    def summary_dict(self) -> dict[str, object]:
        return {
            "orientation": self.orientation,
            "peak_time": float(self.peak_time),
            "left_width": float(self.left_width),
            "right_width": float(self.right_width),
            "width_ratio": float(self.width_ratio),
            "delta_chi2": float(self.delta_chi2),
        }


@dataclass(frozen=True)
class FeatureResult:
    """Time-ordered bumps, dips and caustics of one fit."""

    features: tuple[Feature, ...]
    candidates: tuple[Candidate, ...] = ()

    def _of(self, kind: str) -> tuple[Feature, ...]:
        return tuple(feature for feature in self.features if feature.kind == kind)

    @property
    def bumps(self) -> tuple[Feature, ...]:
        return self._of("bump")

    @property
    def dips(self) -> tuple[Feature, ...]:
        return self._of("dip")

    @property
    def caustics(self) -> tuple[Feature, ...]:
        return self._of("caustic")

    @property
    def strongest(self) -> Optional[Feature]:
        return max(self.features, key=lambda feature: feature.strength, default=None)

    def summary_dict(self) -> dict[str, object]:
        strongest = self.strongest
        row: dict[str, object] = {
            "n_bumps": len(self.bumps),
            "n_dips": len(self.dips),
            "n_caustics": len(self.caustics),
            "n_features": len(self.features),
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
            raise ImportError("FeatureResult.summary_table() requires pandas.") from exc
        return pd.DataFrame(self.feature_dicts())

    def summary_text(self) -> str:
        lines = [
            f"bumps={len(self.bumps)}, dips={len(self.dips)}, caustics={len(self.caustics)}"
        ]
        for feature in self.features:
            lines.append(
                f"{feature.kind}: t={feature.time:.6g}, "
                f"timescale={feature.timescale:.6g}, "
                f"strength={feature.strength:.3g} sigma, snr={feature.snr:.3g}"
            )
        return "\n".join(lines)


def _lorentzian_chi2(
    t: np.ndarray, z: np.ndarray, tc: np.ndarray, w_left: np.ndarray, w_right: np.ndarray
) -> np.ndarray:
    """Best chi2 of A * L(t) + b0 + b1 * t (A > 0) for each (tc, w_left, w_right)."""
    d = t - tc[..., None]
    shape = 1.0 / (1.0 + (d / np.where(d < 0.0, w_left[..., None], w_right[..., None])) ** 2)
    x = t - t.mean()
    basis = np.stack(np.broadcast_arrays(shape, 1.0, x), axis=-1)
    gram = np.einsum("...ni,...nj->...ij", basis, basis)
    coef = np.linalg.solve(gram + 1.0e-12 * np.eye(3), np.einsum("...ni,n->...i", basis, z)[..., None])[..., 0]
    chi2 = np.sum(np.square(z - np.einsum("...ni,...i->...n", basis, coef)), axis=-1)
    return np.where(coef[..., 0] > 0.0, chi2, np.inf)


def _asymmetry_test(t: np.ndarray, z: np.ndarray, cadence: float, half_window: float):
    """(delta_chi2, tc, w_left, w_right) of split vs symmetric Lorentzian."""
    peak = float(t[np.argmax(z)])
    tcs = peak + cadence * np.arange(-3.0, 3.01, 0.25)
    widths = np.geomspace(0.3 * cadence, max(half_window, 0.6 * cadence), 32)
    tc, w = np.meshgrid(tcs, widths, indexing="ij")
    chi2_sym = float(np.min(_lorentzian_chi2(t, z, tc, w, w)))
    best = (np.inf, peak, np.nan, np.nan)
    wl, wr = np.meshgrid(widths, widths, indexing="ij")
    for value in tcs:  # one tc at a time keeps memory at widths**2 * n
        chi2 = _lorentzian_chi2(t, z, np.full(wl.shape, value), wl, wr)
        k = np.unravel_index(int(np.argmin(chi2)), chi2.shape)
        if chi2[k] < best[0]:
            best = (float(chi2[k]), float(value), float(wl[k]), float(wr[k]))
    return chi2_sym - best[0], best[1], best[2], best[3]


def _caustic_details(
    time: np.ndarray,
    z: np.ndarray,
    bumps: Sequence[Feature],
    dips: Sequence[Feature],
    config: FeatureConfig,
) -> list[Optional[CausticDetail]]:
    """Per bump: its CausticDetail when it passes the asymmetry test, else None."""
    dt = np.diff(time)
    dt = dt[np.isfinite(dt) & (dt > 0.0)]
    if dt.size == 0:
        return [None] * len(bumps)
    cadence = float(np.median(dt))
    # A neighbouring extremum (bump or dip) bounds the window half way to it,
    # so the slope into a dip is not read as the bump's own sharp edge.
    peaks = sorted(float(f.time) for f in (*bumps, *dips))
    out: list[Optional[CausticDetail]] = []
    for bump in bumps:
        tp = float(bump.time)
        half = max(
            config.asymmetry_window_timescales * float(bump.timescale),
            config.asymmetry_window_cadences * cadence,
        )
        lo, hi = tp - half, tp + half
        for other in peaks:
            if other < tp:
                lo = max(lo, 0.5 * (other + tp))
            elif other > tp:
                hi = min(hi, 0.5 * (other + tp))
        sel = (time >= lo) & (time <= hi) & np.isfinite(z)
        if np.count_nonzero(sel) < 8:
            out.append(None)
            continue
        dchi2, tc, wl, wr = _asymmetry_test(time[sel], z[sel], cadence, half)
        ratio = max(wl, wr) / max(min(wl, wr), 1.0e-12)
        if dchi2 >= config.min_asymmetry_dchi2 and ratio >= config.min_width_ratio:
            out.append(
                CausticDetail(
                    orientation="entry" if wl < wr else "exit",
                    peak_time=tc,
                    left_width=wl,
                    right_width=wr,
                    delta_chi2=float(dchi2),
                )
            )
        else:
            out.append(None)
    return out


@dataclass(frozen=True)
class _ExtremumDetector:
    """Measure prominent positive (bump) and negative (dip) residual extrema."""

    config: FeatureConfig = FeatureConfig()

    def run_arrays(
        self,
        *,
        time: np.ndarray,
        flux: np.ndarray,
        ferr: np.ndarray,
        residual: np.ndarray,
        model_flux: np.ndarray,
        fit,
        mask: np.ndarray,
        require_closed_peaks: bool = False,
    ) -> tuple[list[Feature], list[Feature]]:
        """Return ``(bumps, dips)`` measured inside the support ``mask``."""
        time = np.asarray(time, dtype=float)
        flux = np.asarray(flux, dtype=float)
        ferr = np.asarray(ferr, dtype=float)
        residual = np.asarray(residual, dtype=float)
        model_flux = np.asarray(model_flux, dtype=float)
        mask = np.asarray(mask, dtype=bool)

        if not np.any(mask):
            return [], []

        z = residual / np.maximum(ferr, 1e-12)
        z_smooth = self._smooth(z, int(self.config.smooth_points))
        peaks: list[Feature] = []
        for start, end in self._mask_slices(mask, time):
            indices = self._prominent_extrema(
                time,
                z_smooth,
                z,
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
                    kind="bump",
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
                    fit=fit,
                    require_closed=bool(require_closed_peaks),
                )
            )

        dips: list[Feature] = []
        for start, end in self._mask_slices(mask, time):
            indices = self._prominent_extrema(
                time,
                z_smooth,
                z,
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
                    fit=fit,
                    require_closed=True,
                )
            )

        # Bump and dip extrema are independent measurements. Negative tails
        # that run into a mask boundary were already rejected by
        # ``require_closed=True``; the presence of an unrelated positive peak
        # must not erase an otherwise closed dip.
        return (
            sorted(peaks, key=lambda feature: feature.time),
            sorted(dips, key=lambda feature: feature.time),
        )

    def _prominent_extrema(
        self,
        time: np.ndarray,
        values: np.ndarray,
        raw: np.ndarray,
        *,
        start: int,
        end: int,
        sign: float,
        min_separation: Optional[float] = None,
    ) -> list[int]:
        segment = sign * values[start:end]
        if segment.size == 0:
            return []

        peak_height = float(np.max(segment))
        if not np.isfinite(peak_height) or peak_height <= 0.0:
            return []
        min_height = float(self.config.min_relative_strength) * peak_height

        extrema = self._monotonic_extrema(segment)
        if not extrema:
            return []

        signed_raw = sign * np.asarray(raw[start:end], dtype=float)
        local: list[int] = []
        for offset in extrema:
            value = float(segment[offset])
            if value < min_height:
                continue
            if self._lobe_snr(segment, signed_raw, int(offset)) < float(
                self.config.min_snr
            ):
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
    def _lobe_snr(segment: np.ndarray, raw: np.ndarray, offset: int) -> float:
        """S/N of a constant offset over the same-sign lobe holding ``offset``."""
        positive = np.asarray(segment) > 0.0
        if not positive[offset]:
            return 0.0
        lo = int(offset)
        while lo > 0 and positive[lo - 1]:
            lo -= 1
        hi = int(offset) + 1
        while hi < positive.size and positive[hi]:
            hi += 1
        lobe = raw[lo:hi]
        lobe = lobe[np.isfinite(lobe)]
        if lobe.size == 0:
            return 0.0
        return float(np.sum(lobe) / np.sqrt(lobe.size))

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
        fit,
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
    ) -> list[Feature]:
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

        features: list[Feature] = []
        for cell_start, cell_end in cells:
            feature = self._feature_from_index(
                kind=kind,
                fit=fit,
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
        fit,
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
    ) -> Optional[Feature]:
        values = sign * z_smooth
        height = float(values[index])
        left_floor = float(np.min(values[cell_start : index + 1]))
        right_floor = float(np.min(values[index:cell_end]))
        floor = max(left_floor, right_floor)
        prominence = max(height - floor, 1e-12)
        threshold = floor + float(self.config.duration_relative_height) * prominence

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
            fit=fit,
            flux=float(flux[index]),
            model_flux=float(model_flux[index]),
        )
        baseline = float(model_flux[index])
        fractional_deviation = (
            float(residual[index]) / baseline
            if np.isfinite(baseline) and abs(baseline) > 1e-30
            else float("nan")
        )

        return Feature(
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
            snr=self._lobe_snr(
                values[cell_start:cell_end],
                sign * z[cell_start:cell_end],
                int(index) - int(cell_start),
            ),
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
        fit,
        flux: float,
        model_flux: float,
    ) -> float:
        fs = float(np.asarray(fit.fs))
        fb = float(np.asarray(fit.fb))
        if np.isfinite(fs) and abs(fs) > 1e-12:
            pspl_mag = (model_flux - fb) / fs
            observed_mag = (flux - fb) / fs
            ratio = observed_mag / pspl_mag if abs(pspl_mag) > 1e-12 else np.nan
        else:
            ratio = flux / model_flux if abs(model_flux) > 1e-12 else np.nan
        return float(ratio)

    @staticmethod
    def _mask_slices(
        mask: np.ndarray,
        time: Optional[np.ndarray] = None,
        *,
        max_gap_cadences: float = 10.0,
    ) -> list[tuple[int, int]]:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return []
        disconnected = np.diff(idx) > 1
        if time is not None and idx.size > 1:
            t = np.asarray(time, dtype=float).reshape(-1)
            if t.size != np.asarray(mask).size:
                raise ValueError("time and mask must have the same length.")
            dt = np.diff(t)
            positive = dt[np.isfinite(dt) & (dt > 0.0)]
            if positive.size:
                cadence = float(np.median(positive))
                selected_dt = np.diff(t[idx])
                time_gap = (
                    ~np.isfinite(selected_dt)
                    | (selected_dt <= 0.0)
                    | (
                        selected_dt
                        > max(
                            float(max_gap_cadences) * cadence,
                            np.finfo(float).eps,
                        )
                    )
                )
                disconnected |= time_gap
        breaks = np.flatnonzero(disconnected)
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



def _signal_supports(
    signal: SignalMask, time: np.ndarray, pad: int
) -> list[np.ndarray]:
    """Per candidate: its window united with its signal segment, padded.

    The segment ends where the residual is back at the noise level, so a
    feature that runs to it is closed only once the smoothed residual also
    is; ``pad`` samples (one smoothing kernel) on each side let it be.
    """
    supports = []
    for candidate, segment in zip(signal.candidates, signal.per_candidate):
        support = candidate.support(time) | np.asarray(segment, dtype=bool)
        idx = np.flatnonzero(support)
        if idx.size and pad > 0:
            lo, hi = int(idx[0]), int(idx[-1])
            support[max(lo - pad, 0) : lo] = True
            support[hi + 1 : hi + 1 + pad] = True
        supports.append(support)
    return supports


def measure_features(
    fit,
    signal: SignalMask | CandidateLike,
    *,
    config: FeatureConfig = FeatureConfig(),
    require_closed_peaks: bool = False,
) -> FeatureResult:
    """Classify the residual of fit inside the candidate windows.

    fit is any single-lens fit result; its time, flux, ferr,
    model_flux, residual, fs and fb are used.  Typically this
    is a refit with the signal masked, so the residual holds the anomaly.

    signal is either candidates, measured inside their windows, or the
    :class:`~jacscanomaly.SignalMask` of those candidates, measured inside
    each window united with its signal segment (padded by
    ``config.smooth_points`` samples).  The segment can extend beyond the
    window; measuring only the window would cut a feature at its edge.

    Each feature records the first candidate whose support contains it.
    """
    time = np.asarray(fit.time, dtype=float).reshape(-1)
    flux = np.asarray(fit.flux, dtype=float).reshape(-1)
    ferr = np.asarray(fit.ferr, dtype=float).reshape(-1)
    residual = np.asarray(fit.residual, dtype=float).reshape(-1)
    model_flux = np.asarray(fit.model_flux, dtype=float).reshape(-1)
    if isinstance(signal, SignalMask):
        values = signal.candidates
        supports = _signal_supports(signal, time, int(config.smooth_points))
    else:
        values = as_candidates(signal)
        supports = [candidate.support(time) for candidate in values]
    support = np.zeros(time.shape, dtype=bool)
    for candidate_support in supports:
        support |= candidate_support

    bumps, dips = _ExtremumDetector(config).run_arrays(
        time=time,
        flux=flux,
        ferr=ferr,
        residual=residual,
        model_flux=model_flux,
        fit=fit,
        mask=support,
        require_closed_peaks=require_closed_peaks,
    )

    def owner(feature: Feature) -> Feature:
        match = next(
            (c for c, s in zip(values, supports) if s[feature.index]), None
        )
        return replace(feature, candidate=match)

    bumps = [owner(feature) for feature in bumps]
    dips = [owner(feature) for feature in dips]
    z = residual / np.maximum(ferr, 1.0e-12)
    bumps = [
        feature if detail is None else replace(feature, kind="caustic", caustic=detail)
        for feature, detail in zip(bumps, _caustic_details(time, z, bumps, dips, config))
    ]
    features = tuple(sorted((*bumps, *dips), key=lambda f: f.time))
    return FeatureResult(features=features, candidates=values)


__all__ = [
    "Feature",
    "FeatureConfig",
    "FeatureKind",
    "FeatureResult",
    "CausticDetail",
    "measure_features",
]
