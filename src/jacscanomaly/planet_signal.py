from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from .finder import Finder
from .anomaly_models import get_chi2_anom_masked, get_chi2_flat_masked
from .models import BestCandidate
from .plot import _adaptive_single_lens_curve
from .singlelens_fit import SingleLensFitResult
from .singlelens_model import A_pspl_func


@dataclass(frozen=True)
class PlanetSignalConfig:
    """
    Configuration for iterative single-lens refinement and signal extraction.

    This first implementation is PSPL-only. It uses the existing template scan
    as a seed finder, masks the strongest unexplained local windows, and refits
    the PSPL baseline on the remaining data.
    """

    baseline_mode: str = "mask"
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
    iterations: Tuple[PlanetSignalIteration, ...]
    candidates: Tuple[PlanetSignalCandidate, ...]
    best: Optional[PlanetSignalCandidate]

    def plot_signal(
        self,
        *,
        show: bool = True,
        peak_xlim: Optional[tuple[float, float]] = None,
        signal_xlim: Optional[tuple[float, float]] = None,
        peak_tE_width: float = 1.5,
        signal_pad: float = 0.5,
        max_signal_width: float = 6.0,
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
            t0, tE, u0 = map(float, np.asarray(self.refined_fit.params)[:3])
            effective_width = abs(tE) * max(abs(u0), 1.0)
            hw = max(float(peak_tE_width) * effective_width, 1.0)
            peak_xlim = (t0 - hw, t0 + hw)
            if self.best is not None:
                peak_xlim = (
                    min(float(peak_xlim[0]), self.best.t_start - float(signal_pad)),
                    max(float(peak_xlim[1]), self.best.t_end + float(signal_pad)),
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

        ax_peak.errorbar(
            t[normal], self.flux[normal], yerr=self.ferr[normal],
            fmt="o", markersize=2, c="C0", ecolor="C0", alpha=0.45,
            label="baseline points", zorder=0,
        )
        ax_peak.errorbar(
            t[signal], self.flux[signal], yerr=self.ferr[signal],
            fmt="o", markersize=2.5, c="C1", ecolor="C1", alpha=0.9,
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
            fmt="o", markersize=2.5, c="C1", ecolor="C1", alpha=0.9,
            zorder=2,
        )
        ax_zoom.plot(t_signal_model, f_signal_model, c="k", lw=2.0, zorder=1)
        ax_zoom.set_xlim(signal_xlim)
        ax_zoom.set_ylabel("flux")

        ax_res.axhline(0.0, c="0.5", lw=1)
        ax_res.plot(t[normal], z[normal], "o", ms=2, c="C0", alpha=0.45, zorder=0)
        ax_res.plot(t[signal], z[signal], "o", ms=2.5, c="C1", alpha=0.9, zorder=2)
        ax_res.set_xlim(signal_xlim)
        ax_res.set_ylabel("residual / error")
        ax_res.set_xlabel("time")

        for candidate in self.candidates:
            for ax in (ax_peak, ax_zoom, ax_res):
                ax.axvspan(candidate.t_start, candidate.t_end, color="C1", alpha=0.08, lw=0)

        if self.best is not None:
            ax_peak.set_title(
                f"planet-signal candidate: t={self.best.t_center:.3f}, "
                f"chi2={self.best.chi2:.1f}, n={self.best.n_points}"
            )
        ax_peak.legend(loc="best")

        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes

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
        if self.finder.config.fitter_kind != "pspl":
            raise NotImplementedError("PlanetSignalExtractor v1 supports fitter_kind='pspl' only.")

        time_j, flux_j, ferr_j, x0_j, time_np, flux_np, ferr_np = self.finder._to_arrays(
            time, flux, ferr, x0
        )
        self.finder._ensure_fitter(float(np.median(time_np)))

        if refit:
            initial_fit = self.finder.fit_single_lens(time_np, flux_np, ferr_np, x0)
        else:
            initial_fit = self._evaluate_pspl_on_full_data(
                time_j=time_j,
                flux_j=flux_j,
                ferr_j=ferr_j,
                params=x0_j,
                fs=None,
                fb=None,
            )

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

            candidate_fit = self._fit_masked_pspl_and_evaluate_full(
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

            candidate_fit = self._fit_weighted_pspl_and_evaluate_full(
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
                        candidate_fit = self._fit_masked_pspl_and_evaluate_full(
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
        return self.finder._pick_best_candidate(clusters_all, grid_metrics_all)

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

    def _fit_masked_pspl_and_evaluate_full(
        self,
        *,
        time_j: jnp.ndarray,
        flux_j: jnp.ndarray,
        ferr_j: jnp.ndarray,
        keep_mask_np: np.ndarray,
        x0_j: jnp.ndarray,
    ) -> SingleLensFitResult:
        if int(np.sum(keep_mask_np)) < 4:
            raise ValueError("Not enough unmasked points to refit PSPL.")

        keep_j = jnp.asarray(keep_mask_np)
        masked_fit = self.finder.fitter.fit(time_j[keep_j], flux_j[keep_j], ferr_j[keep_j], x0_j)
        return self._evaluate_pspl_on_full_data(
            time_j=time_j,
            flux_j=flux_j,
            ferr_j=ferr_j,
            params=jnp.asarray(masked_fit.params),
            fs=masked_fit.fs,
            fb=masked_fit.fb,
        )

    def _fit_weighted_pspl_and_evaluate_full(
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
        return self._evaluate_pspl_on_full_data(
            time_j=time_j,
            flux_j=flux_j,
            ferr_j=ferr_j,
            params=jnp.asarray(weighted_fit.params),
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
        dof = max(int(np.sum(keep)) - 3, 1)
        return float(np.sum(z * z) / dof)

    @staticmethod
    def _evaluate_pspl_on_full_data(
        *,
        time_j: jnp.ndarray,
        flux_j: jnp.ndarray,
        ferr_j: jnp.ndarray,
        params: jnp.ndarray,
        fs,
        fb,
    ) -> SingleLensFitResult:
        if params is None:
            raise ValueError("x0 is required when refit=False.")

        params = jnp.asarray(params, dtype=time_j.dtype)
        ferr_safe = jnp.maximum(ferr_j, 1e-12)
        A = A_pspl_func(params, time_j)

        if fs is None or fb is None:
            from .photometry import solve_fs_fb

            fs, fb = solve_fs_fb(A, flux_j, ferr_safe)

        model_flux = fs * A + fb
        residual = flux_j - model_flux
        z = residual / ferr_safe
        chi2 = jnp.sum(z * z)
        n = int(time_j.shape[0])

        return SingleLensFitResult(
            time=np.asarray(time_j),
            flux=np.asarray(flux_j),
            ferr=np.asarray(ferr_safe),
            params=params,
            param_names=("t0", "tE", "u0"),
            chi2=chi2,
            chi2_dof=chi2 / max(n - 3, 1),
            fs=fs,
            fb=fb,
            model_flux=model_flux,
            residual=residual,
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
