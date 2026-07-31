from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy import fft as scipy_fft


__all__ = [
    "PSPLFFTCandidate",
    "PSPLFFTProfile",
    "PSPLFFTScanner",
    "PSPLFFTSearchResult",
    "pspl_excess_magnification",
]


def pspl_excess_magnification(lag, u0: float, teff: float) -> np.ndarray:
    """Return the PSPL excess magnification ``A - 1``.

    Parameters
    ----------
    lag : array-like
        Time offset ``t - t0``.
    u0 : float
        Positive impact parameter.
    teff : float
        Positive effective timescale ``u0 * tE``.

    Notes
    -----
    The implementation uses an algebraically rationalized expression for
    ``A - 1``.  It avoids subtracting two nearly equal numbers in the far
    wings, where the direct expression for ``A`` approaches one.
    """
    u0_value = float(u0)
    teff_value = float(teff)
    if not np.isfinite(u0_value) or u0_value <= 0.0:
        raise ValueError("u0 must be positive and finite.")
    if not np.isfinite(teff_value) or teff_value <= 0.0:
        raise ValueError("teff must be positive and finite.")

    lag_np = np.asarray(lag, dtype=float)
    if np.any(~np.isfinite(lag_np)):
        raise ValueError("lag must be finite.")

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        scaled = lag_np / teff_value
        u = u0_value * np.hypot(1.0, scaled)
        u2 = u * u
        root = np.hypot(u, 2.0)
        excess = 4.0 / (u * root * (u2 + 2.0 + u * root))
    return np.nan_to_num(excess, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass(frozen=True)
class PSPLFFTCandidate:
    """One PSPL seed selected from an FFT profile."""

    t0: float
    teff: float
    u0: float
    tE: float
    fs: float
    f0: float
    fb: float
    chi2: float
    delta_chi2: float
    grid_index: int

    def as_pspl_params(self, dtype=float) -> np.ndarray:
        """Return ``(t0, tE, u0)`` for :class:`PSPLFitter`."""
        return np.asarray((self.t0, self.tE, self.u0), dtype=dtype)


@dataclass(frozen=True)
class PSPLFFTProfile:
    """Profiled PSPL likelihood over every ``t0`` grid point."""

    t0: np.ndarray
    chi2: np.ndarray
    delta_chi2: np.ndarray
    fs: np.ndarray
    f0: np.ndarray
    fb: np.ndarray
    valid: np.ndarray
    u0: float
    teff: float
    tE: float
    null_chi2: float
    grid_dt: float

    @property
    def best(self) -> Optional[PSPLFFTCandidate]:
        """Best valid grid point, or ``None`` for a singular profile."""
        valid_indices = np.flatnonzero(self.valid & np.isfinite(self.delta_chi2))
        if valid_indices.size == 0:
            return None
        index = int(valid_indices[np.argmax(self.delta_chi2[valid_indices])])
        if self.delta_chi2[index] <= 0.0:
            return None
        return self.candidate(index)

    def candidate(self, index: int) -> PSPLFFTCandidate:
        """Build a scalar candidate from one profile index."""
        index_value = int(index)
        if index_value < 0 or index_value >= self.t0.size:
            raise IndexError("profile index is out of range.")
        if not bool(self.valid[index_value]):
            raise ValueError("profile index is numerically singular.")
        return PSPLFFTCandidate(
            t0=float(self.t0[index_value]),
            teff=float(self.teff),
            u0=float(self.u0),
            tE=float(self.tE),
            fs=float(self.fs[index_value]),
            f0=float(self.f0[index_value]),
            fb=float(self.fb[index_value]),
            chi2=float(self.chi2[index_value]),
            delta_chi2=float(self.delta_chi2[index_value]),
            grid_index=index_value,
        )


@dataclass(frozen=True)
class PSPLFFTSearchResult:
    """Ranked PSPL seeds from an FFT template-bank search."""

    candidates: tuple[PSPLFFTCandidate, ...]
    best: Optional[PSPLFFTCandidate]
    best_profile: Optional[PSPLFFTProfile]
    t0_grid: np.ndarray
    grid_dt: float
    null_chi2: float
    n_observations: int
    n_grid: int
    u0_grid: np.ndarray
    teff_grid: np.ndarray
    tE_grid: Optional[np.ndarray] = None

    def initial_guesses(self, dtype=float) -> np.ndarray:
        """Return ranked ``(t0, tE, u0)`` rows for multistart fitting."""
        if not self.candidates:
            return np.empty((0, 3), dtype=dtype)
        return np.vstack([candidate.as_pspl_params(dtype=dtype) for candidate in self.candidates])


@dataclass(frozen=True)
class _PreparedFFTData:
    t0_grid: np.ndarray
    grid_dt: float
    lags: np.ndarray
    total_weight: float
    mean_flux: float
    null_chi2: float
    fft_weights: np.ndarray
    fft_weighted_centered_flux: np.ndarray
    nfft: int
    crop: slice
    n_observations: int


class PSPLFFTScanner:
    """FFT matched-filter scanner for PSPL initial values.

    The input observations may be irregular.  They are accumulated as weighted
    sufficient statistics on a regular calculation grid.  The returned seed is
    therefore intended for subsequent continuous fitting on the original
    timestamps, not as a replacement for the final likelihood evaluation.

    Parameters
    ----------
    grid_dt : float, optional
        Regular calculation-grid spacing.  If omitted, ``search`` uses the
        smallest trial ``teff`` divided by ``samples_per_teff``.
    samples_per_teff : float, optional
        Grid samples across the smallest trial effective timescale when
        ``grid_dt`` is omitted.
    positive_source : bool, optional
        If true, project negative source-flux solutions to ``fs = 0``.
    max_grid_points : int, optional
        Guard against unexpectedly large FFT allocations.
    singular_rtol : float, optional
        Relative threshold used to reject nearly constant templates.
    fft_workers : int, optional
        SciPy FFT worker count for batched source-plane transforms.  Negative
        values count backward from the available CPU count; ``-1`` uses all
        available workers.
    """

    def __init__(
        self,
        *,
        grid_dt: Optional[float] = None,
        samples_per_teff: float = 8.0,
        positive_source: bool = True,
        max_grid_points: int = 1_000_000,
        singular_rtol: float = 1.0e-12,
        fft_workers: int = -1,
    ) -> None:
        if grid_dt is not None:
            grid_dt_value = float(grid_dt)
            if not np.isfinite(grid_dt_value) or grid_dt_value <= 0.0:
                raise ValueError("grid_dt must be positive and finite.")
        else:
            grid_dt_value = None

        samples_value = float(samples_per_teff)
        if not np.isfinite(samples_value) or samples_value <= 0.0:
            raise ValueError("samples_per_teff must be positive and finite.")

        max_points_value = int(max_grid_points)
        if max_points_value < 2:
            raise ValueError("max_grid_points must be at least 2.")

        singular_value = float(singular_rtol)
        if not np.isfinite(singular_value) or singular_value <= 0.0:
            raise ValueError("singular_rtol must be positive and finite.")
        workers_value = int(fft_workers)
        if workers_value == 0:
            raise ValueError("fft_workers must be non-zero.")

        self.grid_dt = grid_dt_value
        self.samples_per_teff = samples_value
        self.positive_source = bool(positive_source)
        self.max_grid_points = max_points_value
        self.singular_rtol = singular_value
        self.fft_workers = workers_value

    def scan_template(
        self,
        time,
        flux,
        ferr,
        *,
        u0: float,
        teff: float,
    ) -> PSPLFFTProfile:
        """Scan all calculation-grid ``t0`` values for one template shape."""
        u0_value = self._positive_scalar(u0, "u0")
        teff_value = self._positive_scalar(teff, "teff")
        grid_dt = self._resolve_grid_dt(teff_value)
        prepared = self._prepare(time, flux, ferr, grid_dt=grid_dt)
        return self._scan_prepared(prepared, u0=u0_value, teff=teff_value)

    def scan_tE(
        self,
        time,
        flux,
        ferr,
        *,
        u0_grid: Sequence[float],
        tE: float,
    ) -> tuple[PSPLFFTProfile, ...]:
        """Scan the full ``(u0, t0)`` IFFT plane for one exact PSPL ``tE``."""
        u0_values = self._positive_grid(u0_grid, "u0_grid")
        tE_value = self._positive_scalar(tE, "tE")
        min_teff = float(np.min(u0_values) * tE_value)
        prepared = self._prepare(
            time,
            flux,
            ferr,
            grid_dt=self._resolve_grid_dt(min_teff),
        )
        return self._scan_prepared_tE_batch(
            prepared,
            u0_values=u0_values,
            tE=tE_value,
        )

    def search(
        self,
        time,
        flux,
        ferr,
        *,
        u0_grid: Sequence[float],
        teff_grid: Sequence[float],
        top_k: int = 8,
        peaks_per_template: int = 1,
    ) -> PSPLFFTSearchResult:
        """Search a ``(u0, teff)`` bank and rank PSPL initial values.

        ``peaks_per_template=1`` is usually appropriate for initializing a
        single event.  Larger values retain additional local maxima from each
        template and can be useful for ambiguous or multi-event light curves.
        """
        u0_values = self._positive_grid(u0_grid, "u0_grid")
        teff_values = self._positive_grid(teff_grid, "teff_grid")

        top_k_value = int(top_k)
        if top_k_value < 0:
            raise ValueError("top_k must be non-negative.")
        peaks_value = int(peaks_per_template)
        if peaks_value < 1:
            raise ValueError("peaks_per_template must be at least 1.")

        grid_dt = self._resolve_grid_dt(float(np.min(teff_values)))
        prepared = self._prepare(time, flux, ferr, grid_dt=grid_dt)

        ranked: list[PSPLFFTCandidate] = []
        best_profile: Optional[PSPLFFTProfile] = None
        best_candidate: Optional[PSPLFFTCandidate] = None

        for u0_value in u0_values:
            for teff_value in teff_values:
                profile = self._scan_prepared(
                    prepared,
                    u0=float(u0_value),
                    teff=float(teff_value),
                )
                for index in self._peak_indices(profile, peaks_value):
                    candidate = profile.candidate(index)
                    ranked.append(candidate)
                    if best_candidate is None or candidate.delta_chi2 > best_candidate.delta_chi2:
                        best_candidate = candidate
                        best_profile = profile

        ranked.sort(key=lambda candidate: candidate.delta_chi2, reverse=True)
        if top_k_value:
            ranked = ranked[:top_k_value]
        else:
            ranked = []

        best = ranked[0] if ranked else None
        if best is None:
            best_profile = None
        elif best_candidate is not None and (
            best.u0 != best_candidate.u0
            or best.teff != best_candidate.teff
            or best.grid_index != best_candidate.grid_index
        ):
            # This is only possible if candidate ordering changes under exact
            # ties.  Recompute the corresponding profile to keep the result
            # internally consistent.
            best_profile = self._scan_prepared(prepared, u0=best.u0, teff=best.teff)

        return PSPLFFTSearchResult(
            candidates=tuple(ranked),
            best=best,
            best_profile=best_profile,
            t0_grid=prepared.t0_grid,
            grid_dt=float(prepared.grid_dt),
            null_chi2=float(prepared.null_chi2),
            n_observations=int(prepared.n_observations),
            n_grid=int(prepared.t0_grid.size),
            u0_grid=u0_values.copy(),
            teff_grid=teff_values.copy(),
        )

    def search_tE(
        self,
        time,
        flux,
        ferr,
        *,
        u0_grid: Sequence[float],
        tE_grid: Sequence[float],
        top_k: int = 8,
        peaks_per_template: int = 1,
    ) -> PSPLFFTSearchResult:
        """Search an exact PSPL bank with ``tE`` as the outer scale grid.

        For one ``tE``, every ``u0`` is a row of the radial source-plane
        magnification map

        ``A(sqrt(u0**2 + ((t - t0) / tE)**2)) - 1``.

        Those rows are transformed and correlated as a single NumPy batch.
        Thus Python only loops over ``tE``; the returned IFFT plane contains
        every ``(u0, t0)`` trial for that scale.
        """
        u0_values = self._positive_grid(u0_grid, "u0_grid")
        tE_values = self._positive_grid(tE_grid, "tE_grid")
        top_k_value, peaks_value = self._validate_search_counts(
            top_k=top_k,
            peaks_per_template=peaks_per_template,
        )

        min_teff = float(np.min(u0_values) * np.min(tE_values))
        prepared = self._prepare(
            time,
            flux,
            ferr,
            grid_dt=self._resolve_grid_dt(min_teff),
        )

        ranked: list[PSPLFFTCandidate] = []
        best_profile: Optional[PSPLFFTProfile] = None
        best_candidate: Optional[PSPLFFTCandidate] = None
        for tE_value in tE_values:
            profiles = self._scan_prepared_tE_batch(
                prepared,
                u0_values=u0_values,
                tE=float(tE_value),
            )
            for profile in profiles:
                for index in self._peak_indices(profile, peaks_value):
                    candidate = profile.candidate(index)
                    ranked.append(candidate)
                    if best_candidate is None or candidate.delta_chi2 > best_candidate.delta_chi2:
                        best_candidate = candidate
                        best_profile = profile

        ranked.sort(key=lambda candidate: candidate.delta_chi2, reverse=True)
        ranked = ranked[:top_k_value] if top_k_value else []
        best = ranked[0] if ranked else None
        if best is None:
            best_profile = None
        elif best_candidate is None or (
            best.u0 != best_candidate.u0
            or best.tE != best_candidate.tE
            or best.grid_index != best_candidate.grid_index
        ):
            best_profile = self._scan_prepared(
                prepared,
                u0=best.u0,
                teff=best.teff,
            )

        teff_values = np.multiply.outer(tE_values, u0_values).reshape(-1)
        return PSPLFFTSearchResult(
            candidates=tuple(ranked),
            best=best,
            best_profile=best_profile,
            t0_grid=prepared.t0_grid,
            grid_dt=float(prepared.grid_dt),
            null_chi2=float(prepared.null_chi2),
            n_observations=int(prepared.n_observations),
            n_grid=int(prepared.t0_grid.size),
            u0_grid=u0_values.copy(),
            teff_grid=teff_values,
            tE_grid=tE_values.copy(),
        )

    @staticmethod
    def _validate_search_counts(*, top_k: int, peaks_per_template: int) -> tuple[int, int]:
        top_k_value = int(top_k)
        if top_k_value < 0:
            raise ValueError("top_k must be non-negative.")
        peaks_value = int(peaks_per_template)
        if peaks_value < 1:
            raise ValueError("peaks_per_template must be at least 1.")
        return top_k_value, peaks_value

    def _resolve_grid_dt(self, min_teff: float) -> float:
        if self.grid_dt is not None:
            return float(self.grid_dt)
        return float(min_teff) / self.samples_per_teff

    def _prepare(self, time, flux, ferr, *, grid_dt: float) -> _PreparedFFTData:
        time_np = np.asarray(time, dtype=float)
        flux_np = np.asarray(flux, dtype=float)
        ferr_np = np.asarray(ferr, dtype=float)

        if time_np.ndim != 1 or flux_np.ndim != 1 or ferr_np.ndim != 1:
            raise ValueError("time/flux/ferr must be 1D arrays.")
        if not (time_np.size == flux_np.size == ferr_np.size):
            raise ValueError("time/flux/ferr must have the same length.")
        if time_np.size < 2:
            raise ValueError("Need at least two observations.")
        if np.any(~np.isfinite(time_np)) or np.any(~np.isfinite(flux_np)) or np.any(~np.isfinite(ferr_np)):
            raise ValueError("time/flux/ferr must be finite.")
        if np.any(ferr_np <= 0.0):
            raise ValueError("ferr must be positive.")

        grid_dt_value = self._positive_scalar(grid_dt, "grid_dt")
        t_start = float(np.min(time_np))
        span = float(np.max(time_np) - t_start)
        n_grid = int(np.ceil(span / grid_dt_value)) + 1
        n_grid = max(n_grid, 2)
        if n_grid > self.max_grid_points:
            raise ValueError(
                "FFT grid would contain "
                f"{n_grid} points, exceeding max_grid_points={self.max_grid_points}. "
                "Increase grid_dt or max_grid_points."
            )

        t0_grid = t_start + grid_dt_value * np.arange(n_grid, dtype=float)
        bin_index = np.floor((time_np - t_start) / grid_dt_value + 0.5).astype(np.int64)
        bin_index = np.clip(bin_index, 0, n_grid - 1)

        weights_obs = 1.0 / np.square(ferr_np)
        total_weight = float(np.sum(weights_obs))
        mean_flux = float(np.sum(weights_obs * flux_np) / total_weight)
        centered_flux = flux_np - mean_flux
        null_chi2 = float(np.sum(weights_obs * centered_flux * centered_flux))

        weights = np.bincount(bin_index, weights=weights_obs, minlength=n_grid).astype(float, copy=False)
        weighted_centered_flux = np.bincount(
            bin_index,
            weights=weights_obs * centered_flux,
            minlength=n_grid,
        ).astype(float, copy=False)

        convolution_size = 3 * n_grid - 2
        nfft = self._next_power_of_two(convolution_size)
        fft_weights = np.fft.rfft(weights, n=nfft)
        fft_weighted_centered_flux = np.fft.rfft(weighted_centered_flux, n=nfft)

        return _PreparedFFTData(
            t0_grid=t0_grid,
            grid_dt=grid_dt_value,
            lags=(np.arange(2 * n_grid - 1, dtype=float) - (n_grid - 1)) * grid_dt_value,
            total_weight=total_weight,
            mean_flux=mean_flux,
            null_chi2=null_chi2,
            fft_weights=fft_weights,
            fft_weighted_centered_flux=fft_weighted_centered_flux,
            nfft=nfft,
            crop=slice(n_grid - 1, 2 * n_grid - 1),
            n_observations=int(time_np.size),
        )

    def _scan_prepared(self, prepared: _PreparedFFTData, *, u0: float, teff: float) -> PSPLFFTProfile:
        n_grid = prepared.t0_grid.size
        template = pspl_excess_magnification(prepared.lags, u0, teff)

        fft_template_reversed = np.fft.rfft(template[::-1], n=prepared.nfft)
        fft_template2_reversed = np.fft.rfft(np.square(template)[::-1], n=prepared.nfft)

        qx = np.fft.irfft(
            prepared.fft_weights * fft_template_reversed,
            n=prepared.nfft,
        )[prepared.crop]
        qxx = np.fft.irfft(
            prepared.fft_weights * fft_template2_reversed,
            n=prepared.nfft,
        )[prepared.crop]
        sxy = np.fft.irfft(
            prepared.fft_weighted_centered_flux * fft_template_reversed,
            n=prepared.nfft,
        )[prepared.crop]

        sxx = qxx - np.square(qx) / prepared.total_weight
        scale = max(1.0, float(np.max(np.abs(qxx))))
        valid = np.isfinite(sxx) & np.isfinite(sxy) & (sxx > self.singular_rtol * scale)

        source_numerator = np.maximum(sxy, 0.0) if self.positive_source else sxy
        fs = np.zeros(n_grid, dtype=float)
        delta_chi2 = np.zeros(n_grid, dtype=float)
        fs[valid] = source_numerator[valid] / sxx[valid]
        delta_chi2[valid] = np.square(source_numerator[valid]) / sxx[valid]
        delta_chi2 = np.clip(delta_chi2, 0.0, max(prepared.null_chi2, 0.0))

        f0 = prepared.mean_flux - fs * qx / prepared.total_weight
        fb = f0 - fs
        chi2 = np.maximum(prepared.null_chi2 - delta_chi2, 0.0)

        return PSPLFFTProfile(
            t0=prepared.t0_grid.copy(),
            chi2=chi2,
            delta_chi2=delta_chi2,
            fs=fs,
            f0=f0,
            fb=fb,
            valid=valid,
            u0=float(u0),
            teff=float(teff),
            tE=float(teff / u0),
            null_chi2=float(prepared.null_chi2),
            grid_dt=float(prepared.grid_dt),
        )

    def _scan_prepared_tE_batch(
        self,
        prepared: _PreparedFFTData,
        *,
        u0_values: np.ndarray,
        tE: float,
    ) -> tuple[PSPLFFTProfile, ...]:
        """Return all source-plane ``u0`` rows for one ``tE`` in one FFT batch."""
        tE_value = self._positive_scalar(tE, "tE")
        u0_rows = np.asarray(u0_values, dtype=float).reshape(-1, 1)
        source_x = prepared.lags.reshape(1, -1) / tE_value
        u = np.hypot(u0_rows, source_x)
        u2 = np.square(u)
        root = np.hypot(u, 2.0)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            templates = 4.0 / (u * root * (u2 + 2.0 + u * root))
        templates = np.nan_to_num(templates, nan=0.0, posinf=0.0, neginf=0.0)

        # Thread-pool dispatch costs more than the FFT itself for small grids.
        # Keep the default ``-1`` adaptive, while honoring an explicit positive
        # worker count at every size.
        workers = (
            1
            if self.fft_workers < 0 and prepared.nfft < 32_768
            else self.fft_workers
        )
        fft_templates_reversed = scipy_fft.rfft(
            templates[:, ::-1],
            n=prepared.nfft,
            axis=-1,
            workers=workers,
        )
        fft_templates2_reversed = scipy_fft.rfft(
            np.square(templates[:, ::-1]),
            n=prepared.nfft,
            axis=-1,
            workers=workers,
        )
        crop = prepared.crop
        qx = scipy_fft.irfft(
            fft_templates_reversed * prepared.fft_weights[None, :],
            n=prepared.nfft,
            axis=-1,
            workers=workers,
        )[:, crop]
        qxx = scipy_fft.irfft(
            fft_templates2_reversed * prepared.fft_weights[None, :],
            n=prepared.nfft,
            axis=-1,
            workers=workers,
        )[:, crop]
        sxy = scipy_fft.irfft(
            fft_templates_reversed * prepared.fft_weighted_centered_flux[None, :],
            n=prepared.nfft,
            axis=-1,
            workers=workers,
        )[:, crop]

        sxx = qxx - np.square(qx) / prepared.total_weight
        scale = np.maximum(1.0, np.max(np.abs(qxx), axis=1, keepdims=True))
        valid = np.isfinite(sxx) & np.isfinite(sxy) & (sxx > self.singular_rtol * scale)
        source_numerator = np.maximum(sxy, 0.0) if self.positive_source else sxy

        fs = np.zeros_like(sxx)
        delta_chi2 = np.zeros_like(sxx)
        fs[valid] = source_numerator[valid] / sxx[valid]
        delta_chi2[valid] = np.square(source_numerator[valid]) / sxx[valid]
        delta_chi2 = np.clip(delta_chi2, 0.0, max(prepared.null_chi2, 0.0))
        f0 = prepared.mean_flux - fs * qx / prepared.total_weight
        fb = f0 - fs
        chi2 = np.maximum(prepared.null_chi2 - delta_chi2, 0.0)

        return tuple(
            PSPLFFTProfile(
                t0=prepared.t0_grid.copy(),
                chi2=chi2[row],
                delta_chi2=delta_chi2[row],
                fs=fs[row],
                f0=f0[row],
                fb=fb[row],
                valid=valid[row],
                u0=float(u0),
                teff=float(u0 * tE_value),
                tE=tE_value,
                null_chi2=float(prepared.null_chi2),
                grid_dt=float(prepared.grid_dt),
            )
            for row, u0 in enumerate(np.asarray(u0_values, dtype=float))
        )

    @staticmethod
    def _peak_indices(profile: PSPLFFTProfile, count: int) -> np.ndarray:
        score = profile.delta_chi2
        valid = profile.valid & np.isfinite(score)
        if score.size == 0 or not np.any(valid):
            return np.empty(0, dtype=int)

        left = np.empty_like(score)
        right = np.empty_like(score)
        left[0] = -np.inf
        left[1:] = score[:-1]
        right[-1] = -np.inf
        right[:-1] = score[1:]
        local = valid & (score >= left) & (score >= right)
        indices = np.flatnonzero(local & (score > 0.0))
        if indices.size == 0:
            return np.empty(0, dtype=int)
        order = np.argsort(score[indices], kind="stable")[::-1]
        return indices[order[:count]]

    @staticmethod
    def _positive_scalar(value: float, name: str) -> float:
        result = float(value)
        if not np.isfinite(result) or result <= 0.0:
            raise ValueError(f"{name} must be positive and finite.")
        return result

    @classmethod
    def _positive_grid(cls, values: Sequence[float], name: str) -> np.ndarray:
        result = np.asarray(values, dtype=float)
        if result.ndim != 1 or result.size == 0:
            raise ValueError(f"{name} must be a non-empty 1D array.")
        if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
            raise ValueError(f"{name} must contain only positive finite values.")
        return result

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        return 1 << (int(value) - 1).bit_length()
