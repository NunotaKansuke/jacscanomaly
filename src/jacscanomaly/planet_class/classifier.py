from __future__ import annotations

import numpy as np

from ..planet_signal import PlanetSignalResult
from .atoms import (
    CentralDoubleCuspAtom,
    CentralPerturbationAtom,
    CanonicalCuspAtom,
    ChangRefsdalPerturbationAtom,
    CurvedFoldCausticAtom,
    CuspTailAtom,
    FiniteSourceCuspAtom,
    FoldCausticAtom,
    FullCausticCrossingAtom,
    GrazingFoldCausticAtom,
    LimbDarkenedFoldCausticAtom,
    MinorImageBoxTroughAtom,
    NegativeDipAtom,
    PositiveBumpAtom,
    PSPLPositiveBumpAtom,
    PSPLMisfitAtom,
    RimTroughCausticAtom,
    SecondPSPLAtom,
    ShearQuadrupoleAtom,
    SignedTwoFoldCausticAtom,
    SystematicsArtifactAtom,
    TwoFoldCausticAtom,
)
from .features import segment_features
from .pspl import pspl_params_from_result
from .types import (
    AtomFitResult,
    LocalPhysicalFitResult,
    PlanetAnomalyFitResult,
    PlanetClassConfig,
    SegmentData,
    SegmentModelResult,
)


class PlanetAnomalyClassifier:
    """
    Fit residual-template atoms to extracted planet-signal components.

    This class compares local residual morphologies after
    :class:`PlanetSignalExtractor` has separated candidate signal points from
    the refined single-lens baseline.  The fitted atom parameters are local
    morphology estimates and locally identifiable physical constraints.  The
    classifier does not expand those measurements into assumed 2L1S/1L2S
    parameter grids and is not a replacement for a global model comparison.
    """

    def __init__(self, config: PlanetClassConfig = PlanetClassConfig()):
        self.config = config

    def fit(self, result: PlanetSignalResult) -> PlanetAnomalyFitResult:
        """
        Fit enabled local morphology atoms to extracted signal components.

        Parameters
        ----------
        result : PlanetSignalResult
            Output of :class:`PlanetSignalExtractor`. The classifier uses its
            refined residual, component classification, and refined PSPL fit.

        Returns
        -------
        PlanetAnomalyFitResult
            Ranked atom fits for each component, local parameter estimates and
            uncertainty estimates when available, class probabilities, and
            locally identifiable physical constraints.

        Notes
        -----
        Atoms are local residual models. Their BIC ranking is useful for
        triage, but must not be treated as a global physical-model comparison.
        """
        pspl = pspl_params_from_result(result)
        classification = result.classify()
        segment_results: list[SegmentModelResult] = []
        event_warnings: list[str] = []

        for component in classification.components:
            segment = self._segment_from_component(result, component, pspl)
            features = segment_features(segment)
            warnings = self._segment_warnings(segment, features)
            atom_fits = self._fit_atoms(segment, features)
            atom_fits = tuple(sorted(atom_fits, key=lambda fit: fit.bic))
            local_physical_fits = self._fit_local_physical_atoms(
                result=result,
                segment=segment,
                features=features,
                morphology_fits=atom_fits,
            )
            if self.config.keep_top_atom_fits > 0:
                atom_fits = atom_fits[: int(self.config.keep_top_atom_fits)]
            best_fit = atom_fits[0] if atom_fits else None
            class_probabilities = self._class_probabilities(atom_fits)
            segment_results.append(
                SegmentModelResult(
                    component=component,
                    features=features,
                    atom_fits=atom_fits,
                    best_fit=best_fit,
                    class_probabilities=class_probabilities,
                    warnings=tuple(warnings),
                    local_physical_fits=local_physical_fits,
                )
            )

        class_probabilities = self._event_class_probabilities(segment_results)
        best_label = max(class_probabilities, key=class_probabilities.get) if class_probabilities else "none"
        best_atom = self._best_atom(segment_results, class_label=best_label)
        if not segment_results:
            event_warnings.append("no signal components to classify")

        return PlanetAnomalyFitResult(
            pspl=pspl,
            segment_results=tuple(segment_results),
            best_label=best_label,
            best_atom=best_atom,
            class_probabilities=class_probabilities,
            warnings=tuple(event_warnings),
            physical_max_delta_bic=float(self.config.physical_max_delta_bic),
            physical_window_max_delta_bic=float(self.config.physical_window_max_delta_bic),
        )

    def _segment_from_component(self, result, component, pspl) -> SegmentData:
        start = int(component.start_index)
        end = int(component.end_index)
        sl = slice(start, end)
        return SegmentData(
            component=component,
            time=np.asarray(result.time[sl], dtype=float),
            flux=np.asarray(result.flux[sl], dtype=float),
            ferr=np.asarray(result.ferr[sl], dtype=float),
            residual=np.asarray(result.refined_residual[sl], dtype=float),
            model_flux=np.asarray(result.refined_fit.model_flux[sl], dtype=float),
            full_indices=np.arange(start, end, dtype=int),
            pspl=pspl,
        )

    @staticmethod
    def _window_segment(result, component, pspl, *, t_start: float, t_end: float) -> SegmentData:
        time = np.asarray(result.time, dtype=float)
        indices = np.flatnonzero((time >= float(t_start)) & (time <= float(t_end)))
        return SegmentData(
            component=component,
            time=np.asarray(result.time[indices], dtype=float),
            flux=np.asarray(result.flux[indices], dtype=float),
            ferr=np.asarray(result.ferr[indices], dtype=float),
            residual=np.asarray(result.refined_residual[indices], dtype=float),
            model_flux=np.asarray(result.refined_fit.model_flux[indices], dtype=float),
            full_indices=indices.astype(int),
            pspl=pspl,
        )

    def _fit_local_physical_atoms(
        self,
        *,
        result: PlanetSignalResult,
        segment: SegmentData,
        features: dict[str, float],
        morphology_fits: tuple[AtomFitResult, ...],
    ) -> tuple[LocalPhysicalFitResult, ...]:
        windows = self._local_physical_windows(segment, features, morphology_fits)
        edge_labels = {
            "fold_caustic",
            "curved_fold_caustic",
            "full_caustic_crossing",
            "grazing_fold_caustic",
            "limb_darkened_fold_caustic",
            "two_fold_caustic",
            "signed_two_fold_caustic",
        }
        edge_routes = [
            fit for fit in morphology_fits
            if fit.success and fit.class_label in edge_labels and np.isfinite(fit.bic)
        ]
        edge_route = edge_routes[0].class_label if edge_routes else ""
        local_results: list[LocalPhysicalFitResult] = []
        for window_id, kind, center, half_width in windows:
            local_segment = self._window_segment(
                result,
                segment.component,
                segment.pspl,
                t_start=center - half_width,
                t_end=center + half_width,
            )
            if local_segment.time.size < int(self.config.local_physical_min_points):
                continue
            local_features = segment_features(local_segment)
            local_features["t_peak"] = float(center)
            local_features["locator_time"] = float(center)
            atoms = []
            is_edge = "edge" in kind
            if kind == "central_feature":
                if self.config.enable_central_perturbation:
                    atoms.append(CentralPerturbationAtom(self.config))
                if self.config.enable_central_double_cusp and self._is_mixed_signed_like(local_features, local_segment):
                    atoms.append(CentralDoubleCuspAtom(self.config))
            if is_edge:
                if self.config.enable_fold_caustic:
                    atoms.append(FoldCausticAtom(self.config))
                if self.config.enable_curved_fold_caustic and edge_route == "curved_fold_caustic":
                    atoms.append(CurvedFoldCausticAtom(self.config))
                if self.config.enable_limb_darkened_fold_caustic and edge_route in {
                    "limb_darkened_fold_caustic",
                    "full_caustic_crossing",
                }:
                    atoms.append(LimbDarkenedFoldCausticAtom(self.config))
                if self.config.enable_grazing_fold_caustic and (
                    edge_route == "grazing_fold_caustic" or kind == "grazing_edge"
                ):
                    atoms.append(GrazingFoldCausticAtom(self.config))
            compact_peak = kind in {"positive_peak", "negative_dip", "residual_peak"} and (
                float(local_features.get("snr", 0.0)) >= 15.0
                and float(local_features.get("fwhm", 0.0))
                <= float(self.config.cr_max_fwhm_tE_fraction) * max(float(local_segment.pspl.tE), 1e-12)
            )
            if self.config.enable_chang_refsdal and compact_peak:
                atoms.append(ChangRefsdalPerturbationAtom(self.config))

            fits: list[AtomFitResult] = []
            for atom in atoms:
                try:
                    fits.append(atom.fit(local_segment, local_features))
                except Exception as exc:  # pragma: no cover - defensive per-window isolation
                    fits.append(
                        AtomFitResult(
                            atom_name=atom.atom_name,
                            class_label=atom.class_label,
                            params={},
                            param_errors=None,
                            chi2=float("inf"),
                            chi2_baseline=float(local_features.get("chi2", np.inf)),
                            delta_chi2=float("-inf"),
                            bic=float("inf"),
                            aic=float("inf"),
                            score=float("-inf"),
                            n_data=int(local_segment.time.size),
                            n_params=0,
                            success=False,
                            warnings=(f"local physical fit failed: {exc}",),
                            estimation_role=atom.estimation_role,
                        )
                    )
            for fit in sorted(fits, key=lambda item: item.bic):
                local_results.append(
                    LocalPhysicalFitResult(
                        window_id=window_id,
                        locator_kind=kind,
                        locator_time=float(center),
                        t_start=float(local_segment.time[0]),
                        t_end=float(local_segment.time[-1]),
                        atom_fit=fit,
                    )
                )
        return tuple(local_results)

    def _local_physical_windows(
        self,
        segment: SegmentData,
        features: dict[str, float],
        morphology_fits: tuple[AtomFitResult, ...],
    ) -> tuple[tuple[str, str, float, float], ...]:
        t = np.asarray(segment.time, dtype=float)
        y = np.asarray(segment.residual, dtype=float)
        if t.size < 2:
            return ()
        observed_cadence = float(np.median(np.diff(t))) if t.size >= 2 else 0.0
        cadence = max(float(features.get("cadence", 0.0)), observed_cadence, 1e-8)
        duration = max(float(t[-1] - t[0]), cadence)
        baseline_half = float(self.config.local_physical_baseline_cadences) * cadence
        candidates: list[tuple[int, str, float, float]] = []
        finite_bics = [fit.bic for fit in morphology_fits if np.isfinite(fit.bic)]
        best_bic = min(finite_bics) if finite_bics else float("inf")

        central = next(
            (
                fit for fit in morphology_fits
                if fit.class_label in {"central_caustic", "central_double_cusp"}
                and fit.success
                and np.isfinite(fit.bic)
                and fit.bic - best_bic <= float(self.config.physical_max_delta_bic)
            ),
            None,
        )
        if central is not None:
            center = float(central.params.get("t_center", np.nan))
            scale = max(0.5 * float(central.params.get("duration", cadence)), cadence)
            if np.isfinite(center):
                half = min(max(baseline_half, 3.0 * scale), max(0.4 * duration, baseline_half))
                candidates.append((0, "central_feature", center, half))

        full = next(
            (
                fit for fit in morphology_fits
                if fit.class_label == "full_caustic_crossing" and fit.success
            ),
            None,
        )
        if full is not None:
            entry = float(full.params.get("t_entry", np.nan))
            exit_ = float(full.params.get("t_exit", np.nan))
            if np.isfinite(entry) and np.isfinite(exit_) and exit_ - entry >= 4.0 * cadence:
                separation = exit_ - entry
                for kind, center, scale_key in (
                    ("entry_edge", entry, "entry_edge_scale"),
                    ("exit_edge", exit_, "exit_edge_scale"),
                ):
                    scale = max(float(full.params.get(scale_key, cadence)), cadence)
                    half = max(baseline_half, 4.0 * scale, 0.12 * separation)
                    half = max(baseline_half, min(half, 0.4 * separation))
                    candidates.append((0, kind, center, half))

        # Whole-segment fold models are locators only.  Refit each indicated
        # edge in its own data window before exposing a physical constraint.
        locator_specs = {
            "fold_caustic": (("tc", "morphology_edge", "tstar"),),
            "limb_darkened_fold_caustic": (("tc", "morphology_edge", "tstar"),),
            "curved_fold_caustic": (("tc", "morphology_edge", "tstar"),),
            "two_fold_caustic": (
                ("tc1", "entry_edge", "tstar_1"),
                ("tc2", "exit_edge", "tstar_2"),
            ),
            "signed_two_fold_caustic": (
                ("tc1", "entry_edge", "tstar_1"),
                ("tc2", "exit_edge", "tstar_2"),
            ),
            "grazing_fold_caustic": (
                ("t_contact_1", "entry_edge", "tstar_contact_1"),
                ("t_contact_2", "exit_edge", "tstar_contact_2"),
                ("t_closest", "grazing_edge", "width"),
                ("ta", "grazing_edge", "width"),
            ),
        }
        for fit in morphology_fits:
            specs = locator_specs.get(fit.class_label)
            if specs is None or not fit.success or not np.isfinite(fit.bic):
                continue
            if fit.bic - best_bic > float(self.config.physical_max_delta_bic):
                continue
            for time_key, kind, scale_key in specs:
                center = float(fit.params.get(time_key, np.nan))
                if not np.isfinite(center):
                    continue
                scale = max(float(fit.params.get(scale_key, cadence)), cadence)
                half = max(baseline_half, 4.0 * scale)
                half = min(half, max(0.3 * duration, baseline_half))
                candidates.append((1, kind, center, half))

        for kind, extrema in (("positive_peak", segment.component.peaks), ("negative_dip", segment.component.dips)):
            for extremum in extrema:
                scale = float(extremum.fitted_teff) if np.isfinite(extremum.fitted_teff) else float(extremum.timescale)
                scale = max(scale, cadence)
                half = max(baseline_half, 2.5 * scale)
                half = min(half, max(0.3 * duration, baseline_half))
                candidates.append((2, kind, float(extremum.time), half))

        if segment.component.signal_type in {"whole_event_anomaly", "caustic_crossing", "complex"} and t.size >= 5:
            kernel_size = min(7, t.size if t.size % 2 else t.size - 1)
            kernel_size = max(kernel_size, 3)
            smooth = np.convolve(y, np.ones(kernel_size) / kernel_size, mode="same")
            derivative = np.abs(np.gradient(smooth, t))
            local_max = np.flatnonzero(
                (derivative[1:-1] >= derivative[:-2]) & (derivative[1:-1] >= derivative[2:])
            ) + 1
            ranked = sorted(local_max, key=lambda index: derivative[index], reverse=True)
            derivative_half = max(baseline_half, 0.08 * duration)
            derivative_half = min(derivative_half, max(0.25 * duration, baseline_half))
            for index in ranked[:4]:
                candidates.append((3, "derivative_edge", float(t[index]), derivative_half))

        if not candidates:
            index = int(np.argmax(np.abs(y)))
            candidates.append((4, "residual_peak", float(t[index]), max(baseline_half, 0.15 * duration)))

        selected: list[tuple[int, str, float, float]] = []
        for candidate in sorted(candidates, key=lambda item: (item[0], item[2])):
            _, _kind, center, half = candidate
            duplicate = any(
                abs(center - existing[2]) <= max(2.0 * cadence, 0.25 * min(half, existing[3]))
                for existing in selected
            )
            if not duplicate:
                selected.append(candidate)
            if len(selected) >= int(self.config.local_physical_max_windows):
                break
        return tuple(
            (f"w{index}_{kind}", kind, center, half)
            for index, (_priority, kind, center, half) in enumerate(selected)
        )

    def _segment_warnings(self, segment: SegmentData, features: dict[str, float]) -> list[str]:
        warnings: list[str] = []
        if int(features.get("n_points", 0)) < int(self.config.min_points_per_segment):
            warnings.append("segment has too few points")
        if float(features.get("fwhm", 0.0)) <= 1.5 * float(features.get("cadence", 0.0)):
            warnings.append("segment width is close to cadence")
        return warnings

    def _fit_atoms(self, segment: SegmentData, features: dict[str, float]) -> tuple[AtomFitResult, ...]:
        if int(features.get("n_points", 0)) < 2:
            return ()
        atoms = []
        sign = float(features.get("sign", 0.0))
        if self.config.enable_positive_bump and sign >= 0.0:
            atoms.append(PositiveBumpAtom(self.config))
        if self.config.enable_pspl_positive_bump and sign >= 0.0:
            atoms.append(PSPLPositiveBumpAtom(self.config))
        if self.config.enable_negative_dip and sign <= 0.0:
            atoms.append(NegativeDipAtom(self.config))
        if self.config.enable_minor_image_box_trough and sign <= 0.0:
            atoms.append(MinorImageBoxTroughAtom(self.config))
        if self.config.enable_central_perturbation and self._is_central(features, segment):
            atoms.append(CentralPerturbationAtom(self.config))
        if (
            self.config.enable_central_double_cusp
            and self._is_central(features, segment)
            and self._is_mixed_signed_like(features, segment)
        ):
            atoms.append(CentralDoubleCuspAtom(self.config))
        if self.config.enable_fold_caustic and self._is_fold_like(features, segment):
            atoms.append(FoldCausticAtom(self.config))
        if self.config.enable_curved_fold_caustic and self._is_fold_like(features, segment):
            atoms.append(CurvedFoldCausticAtom(self.config))
        if self.config.enable_full_caustic_crossing and self._is_full_caustic_crossing_like(features, segment):
            atoms.append(FullCausticCrossingAtom(self.config))
        if self.config.enable_grazing_fold_caustic and self._is_fold_like(features, segment):
            atoms.append(GrazingFoldCausticAtom(self.config))
        if self.config.enable_limb_darkened_fold_caustic and self._is_strong_caustic_like(features, segment):
            atoms.append(LimbDarkenedFoldCausticAtom(self.config))
        if self.config.enable_rim_trough_caustic and self._is_rim_trough_like(features, segment):
            atoms.append(RimTroughCausticAtom(self.config))
        if self.config.enable_two_fold_caustic and self._is_strong_caustic_like(features, segment):
            atoms.append(TwoFoldCausticAtom(self.config))
        if self.config.enable_signed_two_fold_caustic and self._is_strong_caustic_like(features, segment):
            atoms.append(SignedTwoFoldCausticAtom(self.config))
        if self.config.enable_cusp_tail and self._is_cusp_like(features, segment):
            atoms.append(CuspTailAtom(self.config))
        if self.config.enable_canonical_cusp and self._is_strong_caustic_like(features, segment):
            atoms.append(CanonicalCuspAtom(self.config))
        if self.config.enable_finite_source_cusp and self._is_strong_caustic_like(features, segment):
            atoms.append(FiniteSourceCuspAtom(self.config))
        if self.config.enable_chang_refsdal and self._is_image_perturbation_like(features, segment):
            atoms.append(ChangRefsdalPerturbationAtom(self.config))
        if self.config.enable_second_pspl and sign >= 0.0:
            atoms.append(SecondPSPLAtom(self.config))
        if self.config.enable_shear_quadrupole and self._is_smooth_distortion_like(features, segment):
            atoms.append(ShearQuadrupoleAtom(self.config))
        if self.config.enable_systematics_diagnostic and self._is_systematics_like(features, segment):
            atoms.append(SystematicsArtifactAtom(self.config))
        if self.config.enable_pspl_misfit:
            atoms.append(PSPLMisfitAtom(self.config))

        fits: list[AtomFitResult] = []
        for atom in atoms:
            try:
                fit = atom.fit(segment, features)
            except Exception as exc:  # pragma: no cover - defensive per-atom isolation
                fits.append(
                    AtomFitResult(
                        atom_name=atom.atom_name,
                        class_label=atom.class_label,
                        params={},
                        param_errors=None,
                        chi2=float("inf"),
                        chi2_baseline=float(features.get("chi2", np.inf)),
                        delta_chi2=float("-inf"),
                        bic=float("inf"),
                        aic=float("inf"),
                        score=float("-inf"),
                        n_data=int(features.get("n_points", 0)),
                        n_params=0,
                        success=False,
                        warnings=(f"fit failed: {exc}",),
                    )
                )
                continue
            fits.append(fit)
        return tuple(fits)

    def _is_central(self, features: dict[str, float], segment: SegmentData) -> bool:
        window = float(self.config.central_window_factor) * max(
            abs(float(segment.pspl.u0)) * float(segment.pspl.tE),
            float(features.get("fwhm", 0.0)),
            1e-12,
        )
        return abs(float(features.get("t_peak", segment.pspl.t0)) - float(segment.pspl.t0)) <= window

    @staticmethod
    def _is_fold_like(features: dict[str, float], segment: SegmentData) -> bool:
        if segment.component.signal_type == "caustic_crossing":
            return True
        if float(features.get("edge_sharpness", 0.0)) >= 0.15:
            return True
        return int(features.get("n_points", 0)) >= 6 and float(features.get("snr", 0.0)) >= 20.0

    @staticmethod
    def _is_cusp_like(features: dict[str, float], segment: SegmentData) -> bool:
        if segment.component.signal_type in {"caustic_crossing", "complex"}:
            return True
        return (
            int(features.get("n_points", 0)) >= 8
            and float(features.get("snr", 0.0)) >= 20.0
            and float(features.get("duration", 0.0)) > 2.0 * float(features.get("cadence", 0.0))
        )

    @staticmethod
    def _is_strong_caustic_like(features: dict[str, float], segment: SegmentData) -> bool:
        if segment.component.signal_type in {"caustic_crossing", "complex"}:
            return True
        return float(features.get("edge_sharpness", 0.0)) >= 0.15 and float(features.get("snr", 0.0)) >= 20.0

    @staticmethod
    def _is_full_caustic_crossing_like(features: dict[str, float], segment: SegmentData) -> bool:
        if int(features.get("n_points", 0)) < 20:
            return False
        if segment.component.signal_type in {"whole_event_anomaly", "caustic_crossing"}:
            return True
        if segment.component.signal_type == "complex" and float(features.get("snr", 0.0)) >= 30.0:
            return True
        return (
            float(features.get("duration", 0.0)) >= 0.05 * max(float(segment.pspl.tE), 1e-12)
            and float(features.get("edge_sharpness", 0.0)) >= 0.08
            and float(features.get("snr", 0.0)) >= 30.0
        )

    @staticmethod
    def _is_rim_trough_like(features: dict[str, float], segment: SegmentData) -> bool:
        if int(features.get("n_points", 0)) < 8:
            return False
        positive = float(features.get("positive_chi2", 0.0))
        negative = float(features.get("negative_chi2", 0.0))
        if min(positive, negative) <= 0.05 * max(positive, negative, 1e-12):
            return False
        if segment.component.signal_type in {"dip", "complex", "caustic_crossing"}:
            return True
        return float(features.get("edge_sharpness", 0.0)) >= 0.12 and float(features.get("snr", 0.0)) >= 20.0

    @staticmethod
    def _is_mixed_signed_like(features: dict[str, float], segment: SegmentData) -> bool:
        if int(features.get("n_points", 0)) < 8:
            return False
        positive = float(features.get("positive_chi2", 0.0))
        negative = float(features.get("negative_chi2", 0.0))
        if min(positive, negative) > 0.03 * max(positive, negative, 1e-12):
            return True
        return segment.component.signal_type in {"complex", "caustic_crossing"}

    def _is_image_perturbation_like(self, features: dict[str, float], segment: SegmentData) -> bool:
        signal_type = segment.component.signal_type
        if signal_type not in {"single_peak", "dip", "weakpeak", "weakdip", "complex", "caustic_crossing"}:
            return False
        if float(features.get("snr", 0.0)) < 15.0:
            return False
        fwhm_fraction = float(features.get("fwhm", 0.0)) / max(float(segment.pspl.tE), 1e-12)
        if fwhm_fraction > float(self.config.cr_max_fwhm_tE_fraction):
            return False
        if signal_type in {"complex", "caustic_crossing"}:
            n_extrema = len(segment.component.peaks) + len(segment.component.dips)
            return n_extrema <= 3
        return True

    @staticmethod
    def _is_smooth_distortion_like(features: dict[str, float], segment: SegmentData) -> bool:
        if segment.component.signal_type in {"complex", "low_significance"}:
            return True
        duration = float(features.get("duration", 0.0))
        return duration >= 0.05 * max(float(segment.pspl.tE), 1e-12) and float(features.get("snr", 0.0)) >= 15.0

    @staticmethod
    def _is_systematics_like(features: dict[str, float], segment: SegmentData) -> bool:
        n_points = int(features.get("n_points", 0))
        if n_points <= 3:
            return True
        cadence = float(features.get("cadence", 0.0))
        if cadence > 0.0 and float(features.get("fwhm", 0.0)) <= 1.5 * cadence:
            return True
        if segment.component.signal_type in {"low_significance", "weakpeak", "weakdip"}:
            return True
        return float(features.get("kurtosis", 0.0)) >= 8.0 and float(features.get("edge_sharpness", 0.0)) >= 0.5

    @staticmethod
    def _class_probabilities(atom_fits: tuple[AtomFitResult, ...]) -> dict[str, float]:
        finite = [fit for fit in atom_fits if np.isfinite(fit.bic)]
        if not finite:
            return {}
        bic_min = min(float(fit.bic) for fit in finite)
        weights: dict[str, float] = {}
        for fit in finite:
            weight = float(np.exp(-0.5 * (float(fit.bic) - bic_min)))
            weights[fit.class_label] = weights.get(fit.class_label, 0.0) + weight
        total = sum(weights.values())
        if total <= 0.0:
            return {}
        return {key: float(value / total) for key, value in sorted(weights.items())}

    @staticmethod
    def _event_class_probabilities(segment_results: list[SegmentModelResult]) -> dict[str, float]:
        weights: dict[str, float] = {}
        for segment in segment_results:
            strength = max(float(segment.features.get("chi2", 0.0)), 1.0)
            for label, prob in segment.class_probabilities.items():
                weights[label] = weights.get(label, 0.0) + strength * float(prob)
        total = sum(weights.values())
        if total <= 0.0:
            return {}
        return {key: float(value / total) for key, value in sorted(weights.items())}

    @staticmethod
    def _best_atom(segment_results: list[SegmentModelResult], *, class_label: str | None = None) -> AtomFitResult | None:
        if class_label is None or class_label == "none":
            candidates = [
                segment.best_fit
                for segment in segment_results
                if segment.best_fit is not None and np.isfinite(segment.best_fit.bic)
            ]
            if not candidates:
                return None
            return min(candidates, key=lambda fit: fit.bic)

        candidates: list[tuple[float, AtomFitResult]] = []
        for segment in segment_results:
            segment_candidates = [
                fit
                for fit in segment.atom_fits
                if fit.class_label == class_label and np.isfinite(fit.bic)
            ]
            if not segment_candidates:
                continue
            strength = max(float(segment.features.get("chi2", 0.0)), 1.0)
            contribution = strength * float(segment.class_probabilities.get(class_label, 0.0))
            candidates.append((contribution, min(segment_candidates, key=lambda fit: fit.bic)))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]
