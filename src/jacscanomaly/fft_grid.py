from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


__all__ = [
    "ExactGridDiagnostics",
    "FFTAnomalyGridRunner",
    "FFTGridResult",
]


@dataclass(frozen=True)
class FFTGridResult:
    """FFT approximation for one season's complete ``(t0, teff)`` grid."""

    t0: np.ndarray
    teff: np.ndarray
    dchi2: np.ndarray
    n_window: np.ndarray
    template_index: np.ndarray
    flat_chi2: np.ndarray


@dataclass(frozen=True)
class ExactGridDiagnostics:
    """Exact direct evaluations for selected grid points."""

    dchi2: np.ndarray
    n_window: np.ndarray
    n_contrib: np.ndarray
    n_eff: np.ndarray
    peak_frac: np.ndarray
    rho1: np.ndarray
    longest_run: np.ndarray
    template_index: np.ndarray


class FFTAnomalyGridRunner:
    """Internal FFT engine for the two jacscanomaly anomaly templates.

    The public entry point remains :class:`jacscanomaly.Finder` with
    ``FinderConfig(grid_backend="fft")``. This engine bins irregular
    observations onto a regular calculation grid, evaluates the local weighted
    constant model and both existing anomaly templates for every translated
    ``t0``, and supports exact re-evaluation of extracted representatives.
    """

    def __init__(
        self,
        *,
        oversample: int = 4,
        max_grid_points: int = 1_000_000,
        singular_rtol: float = 1.0e-12,
    ) -> None:
        oversample_value = int(oversample)
        if oversample_value < 1:
            raise ValueError("oversample must be at least 1.")
        max_points_value = int(max_grid_points)
        if max_points_value < 2:
            raise ValueError("max_grid_points must be at least 2.")
        singular_value = float(singular_rtol)
        if not np.isfinite(singular_value) or singular_value <= 0.0:
            raise ValueError("singular_rtol must be positive and finite.")

        self.oversample = oversample_value
        self.max_grid_points = max_points_value
        self.singular_rtol = singular_value

    @staticmethod
    def template_high_magnification(lag, teff: float) -> np.ndarray:
        """Return ``1 / sqrt(1 + ((t-t0)/teff)^2)``."""
        teff_value = FFTAnomalyGridRunner._positive_scalar(teff, "teff")
        lag_np = np.asarray(lag, dtype=float)
        if np.any(~np.isfinite(lag_np)):
            raise ValueError("lag must be finite.")
        tau = lag_np / teff_value
        return 1.0 / np.hypot(1.0, tau)

    @staticmethod
    def template_low_magnification(lag, teff: float) -> np.ndarray:
        """Return the existing low-magnification ``A1`` scan template."""
        teff_value = FFTAnomalyGridRunner._positive_scalar(teff, "teff")
        lag_np = np.asarray(lag, dtype=float)
        if np.any(~np.isfinite(lag_np)):
            raise ValueError("lag must be finite.")
        tau = lag_np / teff_value
        q = 1.0 + tau * tau
        return (q + 2.0) / np.sqrt(q * (q + 4.0))

    def run(
        self,
        *,
        time,
        flux,
        weight,
        t0_grids: Sequence[np.ndarray],
        teff_values: Sequence[float],
        t0_steps: Sequence[float],
        teff_coeff: float = 3.0,
        min_pts: int = 4,
    ) -> FFTGridResult:
        """Evaluate a bank of regular ``t0`` grids with FFT correlations."""
        time_np, flux_np, weight_np = self._validated_data(time, flux, weight)
        teff_np = np.asarray(teff_values, dtype=float)
        step_np = np.asarray(t0_steps, dtype=float)
        if teff_np.ndim != 1 or step_np.ndim != 1:
            raise ValueError("teff_values and t0_steps must be 1D arrays.")
        if not (len(t0_grids) == teff_np.size == step_np.size):
            raise ValueError("t0_grids, teff_values, and t0_steps must have the same length.")
        if np.any(~np.isfinite(teff_np)) or np.any(teff_np <= 0.0):
            raise ValueError("teff_values must contain positive finite values.")
        if np.any(~np.isfinite(step_np)) or np.any(step_np <= 0.0):
            raise ValueError("t0_steps must contain positive finite values.")

        coeff = self._positive_scalar(teff_coeff, "teff_coeff")
        min_pts_value = int(min_pts)
        if min_pts_value < 1:
            raise ValueError("min_pts must be at least 1.")

        parts: list[FFTGridResult] = []
        for t0_grid, teff, step in zip(t0_grids, teff_np, step_np):
            t0_np = np.asarray(t0_grid, dtype=float)
            if t0_np.ndim != 1:
                raise ValueError("every t0 grid must be one-dimensional.")
            if t0_np.size == 0:
                continue
            if np.any(~np.isfinite(t0_np)):
                raise ValueError("t0 grids must be finite.")
            if t0_np.size > 1:
                differences = np.diff(t0_np)
                spacing_atol = max(
                    2.0e-12 * max(1.0, abs(float(step))),
                    32.0 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(t0_np)))),
                )
                if not np.allclose(
                    differences,
                    float(step),
                    rtol=2.0e-11,
                    atol=spacing_atol,
                ):
                    raise ValueError("each t0 grid must be regular with its supplied t0_step.")
            parts.append(
                self._run_scale(
                    time=time_np,
                    flux=flux_np,
                    weight=weight_np,
                    t0_grid=t0_np,
                    teff=float(teff),
                    t0_step=float(step),
                    teff_coeff=coeff,
                    min_pts=min_pts_value,
                )
            )

        if not parts:
            empty_float = np.zeros(0, dtype=float)
            empty_int = np.zeros(0, dtype=np.int32)
            return FFTGridResult(
                t0=empty_float,
                teff=empty_float.copy(),
                dchi2=empty_float.copy(),
                n_window=empty_int,
                template_index=empty_int.copy(),
                flat_chi2=empty_float.copy(),
            )

        return FFTGridResult(
            t0=np.concatenate([part.t0 for part in parts]),
            teff=np.concatenate([part.teff for part in parts]),
            dchi2=np.concatenate([part.dchi2 for part in parts]),
            n_window=np.concatenate([part.n_window for part in parts]),
            template_index=np.concatenate([part.template_index for part in parts]),
            flat_chi2=np.concatenate([part.flat_chi2 for part in parts]),
        )

    def _run_scale(
        self,
        *,
        time: np.ndarray,
        flux: np.ndarray,
        weight: np.ndarray,
        t0_grid: np.ndarray,
        teff: float,
        t0_step: float,
        teff_coeff: float,
        min_pts: int,
    ) -> FFTGridResult:
        calc_dt = t0_step / float(self.oversample)
        first_t0 = float(t0_grid[0])
        left_bins = max(0, int(np.ceil((first_t0 - float(np.min(time))) / calc_dt)))
        grid_start = first_t0 - left_bins * calc_dt
        grid_end = max(float(np.max(time)), float(t0_grid[-1]))
        n_calc = int(np.ceil((grid_end - grid_start) / calc_dt)) + 1
        n_calc = max(n_calc, 2)
        if n_calc > self.max_grid_points:
            raise ValueError(
                "FFT calculation grid would contain "
                f"{n_calc} points, exceeding max_grid_points={self.max_grid_points}. "
                "Reduce fft_oversample, increase dt0_coeff, or increase fft_max_grid_points."
            )

        bin_index = np.floor((time - grid_start) / calc_dt + 0.5).astype(np.int64)
        bin_index = np.clip(bin_index, 0, n_calc - 1)
        count = np.bincount(bin_index, minlength=n_calc).astype(float, copy=False)
        total_weight = float(np.sum(weight))
        global_mean = float(np.sum(weight * flux) / total_weight)
        centered_flux = flux - global_mean
        w = np.bincount(bin_index, weights=weight, minlength=n_calc).astype(float, copy=False)
        wy = np.bincount(
            bin_index,
            weights=weight * centered_flux,
            minlength=n_calc,
        ).astype(float, copy=False)
        wy2 = np.bincount(
            bin_index,
            weights=weight * centered_flux * centered_flux,
            minlength=n_calc,
        ).astype(float, copy=False)

        convolution_size = 3 * n_calc - 2
        nfft = self._next_power_of_two(convolution_size)
        crop = slice(n_calc - 1, 2 * n_calc - 1)
        f_count = np.fft.rfft(count, n=nfft)
        f_w = np.fft.rfft(w, n=nfft)
        f_wy = np.fft.rfft(wy, n=nfft)
        f_wy2 = np.fft.rfft(wy2, n=nfft)

        lags = (np.arange(2 * n_calc - 1, dtype=float) - (n_calc - 1)) * calc_dt
        window = (np.abs(lags) < teff_coeff * teff).astype(float)

        def kernel_fft(kernel: np.ndarray) -> np.ndarray:
            return np.fft.rfft(kernel[::-1], n=nfft)

        def correlate(data_fft: np.ndarray, reversed_kernel_fft: np.ndarray) -> np.ndarray:
            return np.fft.irfft(data_fft * reversed_kernel_fft, n=nfft)[crop]

        f_window = kernel_fft(window)
        qn = correlate(f_count, f_window)
        qw = correlate(f_w, f_window)
        qy = correlate(f_wy, f_window)
        qy2 = correlate(f_wy2, f_window)

        n_window_all = np.maximum(np.rint(qn), 0.0).astype(np.int32)
        valid_window = (n_window_all >= min_pts) & np.isfinite(qw) & (qw > 0.0)
        flat_chi2_all = np.zeros(n_calc, dtype=float)
        flat_chi2_all[valid_window] = (
            qy2[valid_window] - np.square(qy[valid_window]) / qw[valid_window]
        )
        flat_chi2_all = np.maximum(flat_chi2_all, 0.0)

        high = self.template_high_magnification(lags, teff) * window
        low = self.template_low_magnification(lags, teff) * window

        dchi2_templates = []
        valid_templates = []
        for template in (high, low):
            f_template = kernel_fft(template)
            f_template2 = kernel_fft(template * template)
            qx = correlate(f_w, f_template)
            qxx = correlate(f_w, f_template2)
            qxy = correlate(f_wy, f_template)
            projection = np.zeros(n_calc, dtype=float)
            projection[valid_window] = np.square(qx[valid_window]) / qw[valid_window]
            sxx = qxx - projection
            sxy = np.zeros(n_calc, dtype=float)
            sxy[valid_window] = qxy[valid_window] - qx[valid_window] * qy[valid_window] / qw[valid_window]
            local_scale = np.maximum(np.abs(qxx), np.abs(projection))
            threshold = self.singular_rtol * np.maximum(local_scale, np.finfo(float).tiny)
            valid = valid_window & np.isfinite(sxx) & np.isfinite(sxy) & (sxx > threshold)
            dchi2 = np.zeros(n_calc, dtype=float)
            dchi2[valid] = np.square(sxy[valid]) / sxx[valid]
            dchi2 = np.minimum(np.maximum(dchi2, 0.0), flat_chi2_all)
            dchi2_templates.append(dchi2)
            valid_templates.append(valid)

        d0, d1 = dchi2_templates
        v0, v1 = valid_templates
        # Match the existing JAX/C++ rule: exact ties select A1.
        choose_high = v0 & (~v1 | (d0 > d1))
        choose_low = v1 & (~v0 | (d1 >= d0))
        valid_any = choose_high | choose_low
        dchi2_all = np.where(choose_high, d0, np.where(choose_low, d1, 0.0))
        template_index_all = np.where(choose_high, 0, np.where(choose_low, 1, -1)).astype(np.int32)
        n_window_all = np.where(valid_any, n_window_all, 0).astype(np.int32)
        flat_chi2_all = np.where(valid_any, flat_chi2_all, 0.0)

        t0_index = np.rint((t0_grid - grid_start) / calc_dt).astype(np.int64)
        if np.any(t0_index < 0) or np.any(t0_index >= n_calc):
            raise RuntimeError("internal FFT t0 index fell outside the calculation grid.")

        return FFTGridResult(
            t0=t0_grid.copy(),
            teff=np.full(t0_grid.size, teff, dtype=float),
            dchi2=dchi2_all[t0_index],
            n_window=n_window_all[t0_index],
            template_index=template_index_all[t0_index],
            flat_chi2=flat_chi2_all[t0_index],
        )

    def refine_points(
        self,
        *,
        time,
        flux,
        weight,
        t0,
        teff,
        sigma: float = 3.0,
        teff_coeff: float = 3.0,
        min_pts: int = 4,
    ) -> ExactGridDiagnostics:
        """Exactly re-evaluate selected points on the original timestamps."""
        time_np, flux_np, weight_np = self._validated_data(time, flux, weight)
        t0_np = np.asarray(t0, dtype=float)
        teff_np = np.asarray(teff, dtype=float)
        if t0_np.ndim != 1 or teff_np.ndim != 1 or t0_np.shape != teff_np.shape:
            raise ValueError("t0 and teff must be 1D arrays with the same shape.")
        if np.any(~np.isfinite(t0_np)):
            raise ValueError("t0 must be finite.")
        if np.any(~np.isfinite(teff_np)) or np.any(teff_np <= 0.0):
            raise ValueError("teff must be positive and finite.")
        sigma_value = self._nonnegative_scalar(sigma, "sigma")
        coeff = self._positive_scalar(teff_coeff, "teff_coeff")
        min_pts_value = int(min_pts)
        if min_pts_value < 1:
            raise ValueError("min_pts must be at least 1.")

        n = t0_np.size
        dchi2 = np.zeros(n, dtype=float)
        n_window = np.zeros(n, dtype=np.int32)
        n_contrib = np.zeros(n, dtype=np.int32)
        n_eff = np.zeros(n, dtype=float)
        peak_frac = np.zeros(n, dtype=float)
        rho1 = np.zeros(n, dtype=float)
        longest_run = np.zeros(n, dtype=np.int32)
        template_index = np.full(n, -1, dtype=np.int32)

        sigma2 = sigma_value * sigma_value
        for j, (t0_value, teff_value) in enumerate(zip(t0_np, teff_np)):
            mask = (time_np > t0_value - coeff * teff_value) & (time_np < t0_value + coeff * teff_value)
            count = int(np.count_nonzero(mask))
            if count < min_pts_value:
                continue

            y = flux_np[mask]
            w = weight_np[mask]
            t = time_np[mask]
            sw = float(np.sum(w))
            if not np.isfinite(sw) or sw <= 0.0:
                continue
            mean_y = float(np.sum(w * y) / sw)
            residual_flat = y - mean_y
            chi2_flat = float(np.sum(w * residual_flat * residual_flat))

            basis0 = self.template_high_magnification(t - t0_value, teff_value)
            basis1 = self.template_low_magnification(t - t0_value, teff_value)
            fit0 = self._weighted_line(basis0, y, w)
            fit1 = self._weighted_line(basis1, y, w)
            if fit0 is None and fit1 is None:
                continue
            if fit1 is None or (fit0 is not None and fit0[2] < fit1[2]):
                amplitude, intercept, chi2_anom = fit0
                basis = basis0
                template_index[j] = 0
            else:
                amplitude, intercept, chi2_anom = fit1
                basis = basis1
                template_index[j] = 1

            residual_anom = y - (amplitude * basis + intercept)
            delta = max(chi2_flat - float(chi2_anom), 0.0)
            diff = residual_flat * residual_flat - residual_anom * residual_anom
            improvement = np.maximum(diff, 0.0)
            contrib = improvement > sigma2
            sum_u = float(np.sum(improvement))
            sum_u2 = float(np.sum(improvement * improvement))
            centered = diff - float(np.mean(diff))
            variance = float(np.mean(centered * centered))
            covariance = float(np.mean(centered[:-1] * centered[1:])) if count > 1 else 0.0

            dchi2[j] = delta
            n_window[j] = count
            n_contrib[j] = int(np.count_nonzero(contrib))
            n_eff[j] = (sum_u * sum_u) / sum_u2 if sum_u2 > 0.0 else 0.0
            peak_frac[j] = float(np.max(improvement)) / sum_u if sum_u > 0.0 else 0.0
            rho1[j] = covariance / variance if count > 1 and variance > 0.0 else 0.0
            longest_run[j] = self._longest_true_run(contrib)

        return ExactGridDiagnostics(
            dchi2=dchi2,
            n_window=n_window,
            n_contrib=n_contrib,
            n_eff=n_eff,
            peak_frac=peak_frac,
            rho1=rho1,
            longest_run=longest_run,
            template_index=template_index,
        )

    @staticmethod
    def _weighted_line(x: np.ndarray, y: np.ndarray, w: np.ndarray):
        sw = float(np.sum(w))
        x_mean = float(np.sum(w * x) / sw)
        y_mean = float(np.sum(w * y) / sw)
        xc = x - x_mean
        yc = y - y_mean
        wxx = float(np.sum(w * xc * xc))
        if not np.isfinite(wxx) or wxx <= 0.0:
            return None
        amplitude = float(np.sum(w * xc * yc) / wxx)
        intercept = y_mean - amplitude * x_mean
        residual = y - (amplitude * x + intercept)
        chi2 = float(np.sum(w * residual * residual))
        return amplitude, intercept, chi2

    @staticmethod
    def _longest_true_run(values: np.ndarray) -> int:
        longest = 0
        current = 0
        for value in values:
            if bool(value):
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return int(longest)

    @staticmethod
    def _validated_data(time, flux, weight) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        time_np = np.asarray(time, dtype=float)
        flux_np = np.asarray(flux, dtype=float)
        weight_np = np.asarray(weight, dtype=float)
        if time_np.ndim != 1 or flux_np.ndim != 1 or weight_np.ndim != 1:
            raise ValueError("time/flux/weight must be 1D arrays.")
        if not (time_np.size == flux_np.size == weight_np.size):
            raise ValueError("time/flux/weight must have the same length.")
        if np.any(~np.isfinite(time_np)) or np.any(~np.isfinite(flux_np)) or np.any(~np.isfinite(weight_np)):
            raise ValueError("time/flux/weight must be finite.")
        if np.any(weight_np < 0.0):
            raise ValueError("weight must be non-negative.")
        if float(np.sum(weight_np)) <= 0.0:
            raise ValueError("total weight must be positive.")
        order = np.argsort(time_np, kind="stable")
        return time_np[order], flux_np[order], weight_np[order]

    @staticmethod
    def _positive_scalar(value: float, name: str) -> float:
        result = float(value)
        if not np.isfinite(result) or result <= 0.0:
            raise ValueError(f"{name} must be positive and finite.")
        return result

    @staticmethod
    def _nonnegative_scalar(value: float, name: str) -> float:
        result = float(value)
        if not np.isfinite(result) or result < 0.0:
            raise ValueError(f"{name} must be non-negative and finite.")
        return result

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        return 1 << (int(value) - 1).bit_length()
