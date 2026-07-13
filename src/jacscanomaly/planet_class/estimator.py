from __future__ import annotations

from typing import Optional

import numpy as np

from ..planet_signal import PlanetSignalResult
from .features import segment_features
from .geometry import (
    BRANCH_MAJOR,
    BRANCH_MINOR,
    BRANCH_UNKNOWN,
    REGIME_CENTRAL_RESONANT,
    AnomalyGeometry,
    anomaly_geometry,
    q_from_bump,
    q_from_dip,
)
from .pspl import pspl_params_from_result
from .templates import fit_bump, fit_caustic_crossing, fit_dip, fit_fold, fit_null
from .types import (
    AnomalyScales,
    AnomalyShapeFit,
    ComponentAnomalyResult,
    GridSeed,
    PlanetAnomalyFitResult,
    PlanetClassConfig,
    PSPLParams,
    SegmentData,
)


SHAPE_NO_COHERENT = "no_coherent_shape"
SHAPE_LOW_SIGNIFICANCE = "low_significance"
SHAPE_INSUFFICIENT = "insufficient_data"

_BRANCH_BY_SHAPE = {
    "bump": BRANCH_MAJOR,
    "dip": BRANCH_MINOR,
    "fold": BRANCH_UNKNOWN,
    "caustic_crossing": BRANCH_UNKNOWN,
}


class PlanetAnomalyClassifier:
    """
    Heuristic planetary-anomaly estimator on a refined PSPL baseline.

    For each extracted anomaly component the estimator measures the local
    shape with a small template set (bump, dip, fold, caustic crossing,
    null), then derives the deterministic geometry
    ``(tau_anom, u_anom, alpha, s_dagger)`` and assumption-tagged scale
    estimates (``dt/tE``, ``q``, ``tstar/tE``).  See
    :mod:`jacscanomaly.planet_class.geometry` for the formalism and
    references.
    """

    def __init__(self, config: PlanetClassConfig = PlanetClassConfig()):
        self.config = config

    def fit(self, result: PlanetSignalResult) -> PlanetAnomalyFitResult:
        pspl = pspl_params_from_result(result)
        classification = result.classify()
        components = tuple(
            self._analyze_component(self._segment_from_component(result, component, pspl))
            for component in classification.components
        )

        best_index: Optional[int] = None
        best_delta = -np.inf
        for i, component in enumerate(components):
            if not component.significant:
                continue
            if component.shape in {SHAPE_LOW_SIGNIFICANCE, SHAPE_NO_COHERENT, SHAPE_INSUFFICIENT}:
                continue
            delta = float(component.best_fit.delta_chi2)
            if delta > best_delta:
                best_delta = delta
                best_index = i

        event_warnings: list[str] = []
        if not components:
            event_warnings.append("no signal components to classify")
        best_shape = components[best_index].shape if best_index is not None else "none"
        return PlanetAnomalyFitResult(
            pspl=pspl,
            components=components,
            best_component_index=best_index,
            best_shape=best_shape,
            warnings=tuple(event_warnings),
        )

    @staticmethod
    def _segment_from_component(result, component, pspl: PSPLParams) -> SegmentData:
        sl = slice(int(component.start_index), int(component.end_index))
        return SegmentData(
            component=component,
            time=np.asarray(result.time[sl], dtype=float),
            flux=np.asarray(result.flux[sl], dtype=float),
            ferr=np.asarray(result.ferr[sl], dtype=float),
            residual=np.asarray(result.refined_residual[sl], dtype=float),
            model_flux=np.asarray(result.refined_fit.model_flux[sl], dtype=float),
            full_indices=np.arange(sl.start, sl.stop, dtype=int),
            pspl=pspl,
        )

    def _analyze_component(self, segment: SegmentData) -> ComponentAnomalyResult:
        config = self.config
        features = segment_features(segment)
        n_points = int(features.get("n_points", 0))
        warnings: list[str] = []
        min_points = max(int(config.min_points_per_segment), int(config.polynomial_order) + 3)
        if n_points < min_points:
            return ComponentAnomalyResult(
                component=segment.component,
                features=features,
                shape_fits=(),
                best_fit=None,
                shape=SHAPE_INSUFFICIENT,
                geometry=None,
                scales=None,
                warnings=("segment has too few points",),
            )
        if float(features.get("fwhm", 0.0)) <= 1.5 * float(features.get("cadence", 0.0)):
            warnings.append("segment width is close to cadence")

        fits = self._fit_templates(segment, features)
        fits = tuple(sorted(fits, key=lambda fit: fit.bic))
        # A template can be disqualified (success=False, e.g. an amplitude
        # with the wrong sign) yet still have the lowest BIC; prefer the best
        # qualified fit and keep the disqualified one in the ranking.
        best = next((fit for fit in fits if fit.success), fits[0])
        shape = self._shape_label(best)
        geometry = None
        scales = None
        grid_seed = None
        if shape not in {SHAPE_NO_COHERENT, SHAPE_LOW_SIGNIFICANCE}:
            geometry = self._geometry(best, segment.pspl)
            scales = self._scales(best, geometry, segment.pspl)
            grid_seed = self._grid_seed(geometry, scales)
        return ComponentAnomalyResult(
            component=segment.component,
            features=features,
            shape_fits=fits,
            best_fit=best,
            shape=shape,
            geometry=geometry,
            scales=scales,
            grid_seed=grid_seed,
            warnings=tuple(warnings),
        )

    def _fit_templates(
        self, segment: SegmentData, features: dict[str, float]
    ) -> list[AnomalyShapeFit]:
        config = self.config
        null = fit_null(segment, config, center=float(features["t_peak"]))
        chi2_null = null.chi2
        fits = [null]
        n_points = int(features["n_points"])
        total_power = max(
            float(features["positive_chi2"]) + float(features["negative_chi2"]), 1e-12
        )
        frac = float(config.mixed_sign_power_fraction)
        if float(features["positive_chi2"]) >= frac * total_power:
            fits.append(fit_bump(segment, config, features, chi2_null=chi2_null))
        if float(features["negative_chi2"]) >= frac * total_power:
            fits.append(fit_dip(segment, config, features, chi2_null=chi2_null))
        if n_points >= int(config.min_points_fold):
            fits.append(fit_fold(segment, config, features, chi2_null=chi2_null))
        if n_points >= int(config.min_points_crossing):
            fits.append(fit_caustic_crossing(segment, config, features, chi2_null=chi2_null))
        return fits

    def _shape_label(self, best: AnomalyShapeFit) -> str:
        if best.name == "null":
            return SHAPE_NO_COHERENT
        if not best.success or best.delta_chi2 < float(self.config.min_delta_chi2):
            return SHAPE_LOW_SIGNIFICANCE
        return best.name

    def _geometry(self, best: AnomalyShapeFit, pspl: PSPLParams) -> Optional[AnomalyGeometry]:
        t_anom = float(best.params.get("t_anom", np.nan))
        if not np.isfinite(t_anom):
            return None
        t_anom_err = float((best.param_errors or {}).get("t_anom", np.nan))
        return anomaly_geometry(
            t_anom,
            t0=pspl.t0,
            tE=pspl.tE,
            u0=pspl.u0,
            t_anom_err=t_anom_err,
            preferred_branch=_BRANCH_BY_SHAPE.get(best.name, BRANCH_UNKNOWN),
            central_u_anom_max=float(self.config.central_u_anom_max),
        )

    @staticmethod
    def _scales(
        best: AnomalyShapeFit,
        geometry: Optional[AnomalyGeometry],
        pspl: PSPLParams,
    ) -> Optional[AnomalyScales]:
        tE = float(pspl.tE)
        errors = best.param_errors or {}
        notes: list[str] = []
        central = geometry is not None and geometry.regime == REGIME_CENTRAL_RESONANT
        if central:
            notes.append(
                "central_or_resonant regime (u_anom small): s_dagger -> 1 and "
                "no q estimate is reported"
            )

        if best.name == "bump":
            dt = float(best.params.get("fwhm", np.nan))
            dt_err = float(errors.get("fwhm", np.nan))
            if central:
                q = q_err = float("nan")
            else:
                q, q_err = q_from_bump(dt, tE=tE, fwhm_err=dt_err)
                notes.append(
                    "q assumes the bump FWHM is the planet Einstein-ring "
                    "crossing diameter (Gould & Loeb 1992; sim-calibrated, "
                    "order-of-magnitude only)"
                )
            return AnomalyScales(
                dt=dt,
                dt_err=dt_err,
                dt_over_tE=dt / tE,
                q=q,
                q_err=q_err,
                q_method="" if central else "bump_planet_einstein_crossing",
                notes=tuple(notes),
            )

        if best.name == "dip":
            dt = float(best.params.get("dt_dip", np.nan))
            dt_err = float(errors.get("dt_dip", np.nan))
            if central or geometry is None:
                q = q_err = float("nan")
            else:
                q, q_err = q_from_dip(dt, tE=tE, geometry=geometry, dt_dip_err=dt_err)
                notes.append(
                    "q from the minor-image dip relation "
                    "q = (dt_dip/4tE)^2 (s_dagger_minus/u_anom) sin^2(alpha) "
                    "(Han 2006; Hwang et al. 2022)"
                )
            return AnomalyScales(
                dt=dt,
                dt_err=dt_err,
                dt_over_tE=dt / tE,
                q=q,
                q_err=q_err,
                q_method="" if central or not np.isfinite(q) else "dip_han2006",
                notes=tuple(notes),
            )

        if best.name == "fold":
            tstar = float(best.params.get("tstar", np.nan))
            notes.append(
                "single fold crossing constrains tstar/tE = rho/|sin(psi)| only; "
                "q is not determined by local data"
            )
            return AnomalyScales(
                tstar_over_tE=tstar / tE,
                notes=tuple(notes),
            )

        if best.name == "caustic_crossing":
            dt = float(best.params.get("dt_cc", np.nan))
            dt_err = float(errors.get("dt_cc", np.nan))
            notes.append(
                "caustic crossing constrains dt_cc/tE and per-edge "
                "tstar/tE = rho/|sin(psi)|; q is not determined by local data"
            )
            return AnomalyScales(
                dt=dt,
                dt_err=dt_err,
                dt_over_tE=dt / tE,
                tstar_entry_over_tE=float(best.params.get("tstar_entry", np.nan)) / tE,
                tstar_exit_over_tE=float(best.params.get("tstar_exit", np.nan)) / tE,
                notes=tuple(notes),
            )

        return None

    def _grid_seed(
        self,
        geometry: Optional[AnomalyGeometry],
        scales: Optional[AnomalyScales],
    ) -> Optional[GridSeed]:
        if geometry is None:
            return None
        config = self.config
        if geometry.preferred_branch == BRANCH_MINOR:
            s_candidates = (geometry.s_dagger_minus, geometry.s_dagger_plus)
        else:
            # Major image first also when the branch is unknown: bumps and
            # crossings are more often major-image perturbations.
            s_candidates = (geometry.s_dagger_plus, geometry.s_dagger_minus)
        alpha = float(geometry.alpha)
        two_pi = 2.0 * np.pi
        alpha_candidates = tuple(
            sorted({(a % two_pi) for a in (alpha, -alpha, np.pi - alpha, np.pi + alpha)})
        )
        q_center = float("nan")
        q_width = float("nan")
        q_quality = "none"
        if scales is not None and np.isfinite(scales.q):
            q_center = float(scales.q)
            if scales.q_method == "dip_han2006":
                q_width = float(config.seed_q_width_factor_dip)
                q_quality = "calibrated"
            else:
                q_width = float(config.seed_q_width_factor_bump)
                q_quality = "order_of_magnitude"
        return GridSeed(
            s_candidates=s_candidates,
            s_width_factor=float(config.seed_s_width_factor),
            alpha_candidates=alpha_candidates,
            alpha_tolerance=float(np.deg2rad(config.seed_alpha_tolerance_deg)),
            q_center=q_center,
            q_width_factor=q_width,
            q_quality=q_quality,
        )

