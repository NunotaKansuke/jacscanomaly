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
from .seeds import seeds_from_atom
from .types import (
    AtomFitResult,
    PlanetAnomalyFitResult,
    PlanetClassConfig,
    SegmentData,
    SegmentModelResult,
    SeedCandidate,
)


class PlanetAnomalyClassifier:
    """
    Fit residual-template atoms to extracted planet-signal components.

    This class compares local residual morphologies after
    :class:`PlanetSignalExtractor` has separated candidate signal points from
    the refined single-lens baseline.  The fitted atom parameters are local
    morphology estimates and seed generators for downstream physical models;
    they are not a replacement for global 2L1S/1L2S model comparison.
    """

    def __init__(self, config: PlanetClassConfig = PlanetClassConfig()):
        self.config = config

    def fit(self, result: PlanetSignalResult) -> PlanetAnomalyFitResult:
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
            if self.config.keep_top_atom_fits > 0:
                atom_fits = atom_fits[: int(self.config.keep_top_atom_fits)]
            best_fit = atom_fits[0] if atom_fits else None
            class_probabilities = self._class_probabilities(atom_fits)
            seeds = self._seeds_from_fits(atom_fits, pspl)
            segment_results.append(
                SegmentModelResult(
                    component=component,
                    features=features,
                    atom_fits=atom_fits,
                    best_fit=best_fit,
                    class_probabilities=class_probabilities,
                    seeds=seeds,
                    warnings=tuple(warnings),
                )
            )

        event_seeds = self._deduplicate_seeds(
            tuple(seed for segment in segment_results for seed in segment.seeds)
        )
        class_probabilities = self._event_class_probabilities(segment_results)
        best_label = max(class_probabilities, key=class_probabilities.get) if class_probabilities else "none"
        best_atom = self._best_atom(segment_results, class_label=best_label)
        if not segment_results:
            event_warnings.append("no signal components to classify")

        return PlanetAnomalyFitResult(
            pspl=pspl,
            segment_results=tuple(segment_results),
            event_seeds=event_seeds,
            best_label=best_label,
            best_atom=best_atom,
            class_probabilities=class_probabilities,
            warnings=tuple(event_warnings),
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

    @staticmethod
    def _is_image_perturbation_like(features: dict[str, float], segment: SegmentData) -> bool:
        return segment.component.signal_type in {"single_peak", "dip", "weakpeak", "weakdip", "complex"} or float(
            features.get("snr", 0.0)
        ) >= 15.0

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

    def _seeds_from_fits(self, atom_fits: tuple[AtomFitResult, ...], pspl) -> tuple[SeedCandidate, ...]:
        seeds: list[SeedCandidate] = []
        for fit in atom_fits:
            if not fit.success:
                continue
            seeds.extend(seeds_from_atom(fit, pspl, self.config))
        ranked = tuple(sorted(seeds, key=lambda seed: seed.score, reverse=True))
        ranked = self._deduplicate_seeds(ranked)
        if self.config.keep_top_seeds_per_segment > 0:
            ranked = ranked[: int(self.config.keep_top_seeds_per_segment)]
        return ranked

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

    @staticmethod
    def _deduplicate_seeds(seeds: tuple[SeedCandidate, ...]) -> tuple[SeedCandidate, ...]:
        unique: dict[tuple, SeedCandidate] = {}
        for seed in seeds:
            key = (
                seed.model_type,
                seed.class_label,
                seed.degeneracy_tag,
                tuple(sorted((k, round(float(v), 10)) for k, v in seed.params.items() if np.isfinite(v))),
            )
            current = unique.get(key)
            if current is None or seed.score > current.score:
                unique[key] = seed
        return tuple(sorted(unique.values(), key=lambda seed: seed.score, reverse=True))
