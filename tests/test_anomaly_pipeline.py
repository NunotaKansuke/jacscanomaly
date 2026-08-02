from types import SimpleNamespace

import numpy as np

from jacscanomaly import (
    AnomalyPipelineConfig,
    Finder,
    PlanetFeature,
    PlanetFeatureResult,
    PlanetSignalConfig,
    TemplateFreeCandidate,
    TemplateFreeSearchResult,
    build_anomaly_candidates,
)
from jacscanomaly.planet_signal import PlanetSignalExtractor


def _fit(*, chi2_dof=1.0, marker=0.0):
    return SimpleNamespace(
        params=np.array([10.0 + marker, 2.0, 0.1]),
        chi2_dof=float(chi2_dof),
        model_kind="pspl",
    )


def _install_pipeline_stubs(
    monkeypatch,
    finder,
    *,
    parent_fit,
    post_fit,
    post_iterations=(),
    post_mask=(False, False, False),
    post_weights=(1.0, 1.0, 1.0),
    fallback_accepted=True,
):
    refinement = SimpleNamespace(
        refined_fit=post_fit,
        iterations=tuple(post_iterations),
        signal_mask=np.asarray(post_mask, dtype=bool),
        point_weight=np.asarray(post_weights, dtype=float),
    )
    effect_result = SimpleNamespace(
        selected_fit=parent_fit,
        fallback_result=(
            SimpleNamespace(success=True) if fallback_accepted else None
        ),
        planet_before=None if fallback_accepted else refinement,
        planet_after=refinement if fallback_accepted else None,
        reason_codes=(
            ("fallback_accepted",)
            if fallback_accepted
            else ("fallback_skipped",)
        ),
        diagnostics={"fallback_accepted": fallback_accepted},
    )
    monkeypatch.setattr(
        finder,
        "run_effect_aware",
        lambda *args, **kwargs: effect_result,
    )

    features = SimpleNamespace(n_peaks=1, n_dips=0)
    measurement = SimpleNamespace(
        refined_fit=parent_fit,
        signal_mask=np.array([False, True, False]),
        measure_features=lambda config: features,
    )
    seen = {}

    def fake_measurement_run(extractor, *args, **kwargs):
        seen["config"] = extractor.config
        seen["freeze_baseline"] = kwargs.get("freeze_baseline")
        seen["initial_fit"] = kwargs.get("initial_fit")
        measurement.refined_fit = kwargs["initial_fit"]
        return measurement

    monkeypatch.setattr(PlanetSignalExtractor, "run", fake_measurement_run)
    template_free = object()
    monkeypatch.setattr(
        finder,
        "run_template_free",
        lambda *args, **kwargs: template_free,
    )
    return features, template_free, seen


def test_complete_pipeline_measures_features_when_post_refit_is_empty(monkeypatch):
    finder = Finder()
    parent = _fit()
    features, template_free, seen = _install_pipeline_stubs(
        monkeypatch,
        finder,
        parent_fit=parent,
        post_fit=parent,
    )
    custom_measurement = PlanetSignalConfig(
        max_signal_span_over_tE=0.01,
        frozen_measurement_windows=False,
    )

    result = finder.run_anomaly_pipeline(
        np.array([9.0, 10.0, 11.0]),
        np.ones(3),
        np.full(3, 0.1),
        config=AnomalyPipelineConfig(final_measurement=custom_measurement),
    )

    assert result.adopted_fit is parent
    assert not result.fit_exclusion_mask.any()
    np.testing.assert_array_equal(
        result.measurement_mask,
        np.array([False, True, False]),
    )
    assert result.features is features
    assert result.template_free is template_free
    assert result.has_anomaly_candidate is False
    assert result.anomaly_candidates == []
    assert result.best_anomaly_candidate is None
    assert seen["freeze_baseline"] is True
    assert seen["initial_fit"] is parent
    assert np.isinf(seen["config"].max_signal_span_over_tE)
    assert seen["config"].frozen_measurement_windows is True
    assert "final_residual_measurement_completed" in result.reason_codes


def test_complete_pipeline_rolls_back_degraded_post_refinement(monkeypatch):
    finder = Finder()
    parent = _fit(chi2_dof=1.0)
    degraded = _fit(chi2_dof=2.0, marker=1.0)
    _install_pipeline_stubs(
        monkeypatch,
        finder,
        parent_fit=parent,
        post_fit=degraded,
        post_iterations=(object(),),
        post_mask=(False, True, False),
        post_weights=(1.0, 0.0, 1.0),
    )

    result = finder.run_anomaly_pipeline(
        np.array([9.0, 10.0, 11.0]),
        np.ones(3),
        np.full(3, 0.1),
    )

    assert result.adopted_fit is parent
    assert not result.fit_exclusion_mask.any()
    assert result.diagnostics["post_physical_refinement_reset"] is True
    assert result.diagnostics["post_physical_refits_completed"] == 0


def test_complete_pipeline_keeps_actual_exclusion_mask_for_accepted_refit(monkeypatch):
    finder = Finder()
    parent = _fit(chi2_dof=1.0)
    accepted = _fit(chi2_dof=1.1, marker=1.0)
    _install_pipeline_stubs(
        monkeypatch,
        finder,
        parent_fit=parent,
        post_fit=accepted,
        post_iterations=(object(),),
        post_mask=(False, True, True),
        post_weights=(1.0, 0.0, 1.0),
    )

    result = finder.run_anomaly_pipeline(
        np.array([9.0, 10.0, 11.0]),
        np.ones(3),
        np.full(3, 0.1),
    )

    assert result.adopted_fit is accepted
    np.testing.assert_array_equal(
        result.fit_exclusion_mask,
        np.array([False, True, False]),
    )
    assert result.diagnostics["post_physical_refinement_reset"] is False
    assert result.diagnostics["post_physical_refits_completed"] == 1


def test_complete_pipeline_uses_prephysical_refit_when_fallback_is_skipped(monkeypatch):
    finder = Finder()
    initial = _fit(chi2_dof=2.0)
    refined = _fit(chi2_dof=1.0, marker=1.0)
    _install_pipeline_stubs(
        monkeypatch,
        finder,
        parent_fit=initial,
        post_fit=refined,
        post_iterations=(object(),),
        post_mask=(False, True, False),
        post_weights=(1.0, 0.0, 1.0),
        fallback_accepted=False,
    )

    result = finder.run_anomaly_pipeline(
        np.array([9.0, 10.0, 11.0]),
        np.ones(3),
        np.full(3, 0.1),
    )

    assert result.adopted_fit is refined
    np.testing.assert_array_equal(
        result.fit_exclusion_mask,
        np.array([False, True, False]),
    )
    assert result.diagnostics["planet_before_refits_completed"] == 1
    assert result.diagnostics["post_physical_refits_completed"] == 0
    assert result.diagnostics["post_physical_mask_points"] == 0
    assert result.diagnostics["fit_exclusion_mask_points"] == 1


def test_unified_candidates_merge_feature_with_template_free_statistics():
    feature = PlanetFeature(
        kind="peak",
        index=2,
        time=10.0,
        t_start=9.8,
        t_end=10.3,
        timescale=0.5,
        strength=12.0,
        signed_z=12.0,
        residual=0.2,
        fractional_deviation=0.18,
        magnification_ratio=1.18,
    )
    free_candidate = TemplateFreeCandidate(
        kind="zero_crossing",
        season_idx=0,
        start_index=1,
        end_index=4,
        t_start=9.7,
        t_end=10.5,
        t_center=10.1,
        n_points=4,
        chi2=220.0,
        reduced_chi2=55.0,
        max_abs_z=11.0,
    )
    template_free = TemplateFreeSearchResult(
        time=np.array([9.5, 9.7, 10.0, 10.3, 10.5]),
        residual=np.zeros(5),
        ferr=np.ones(5),
        z=np.array([0.0, 3.0, 11.0, 4.0, 0.0]),
        candidates=(free_candidate,),
        fixed_window_candidates=(),
        hybrid_candidates=(),
        blind_reduced_candidates=(),
        best=free_candidate,
    )

    candidates = build_anomaly_candidates(
        PlanetFeatureResult(peaks=(feature,), dips=()),
        template_free,
        time=template_free.time,
        fit_exclusion_mask=np.array([False, False, True, False, False]),
        adopted_model="fspl",
    )

    assert len(candidates) == 1
    row = candidates[0].to_dict()
    assert row == {
        "rank": 1,
        "kind": "peak",
        "t_center": 10.0,
        "t_start": 9.8,
        "t_end": 10.3,
        "timescale": 0.5,
        "half_width": 0.25,
        "max_abs_z": 12.0,
        "signed_z": 12.0,
        "fractional_deviation": 0.18,
        "chi2": 220.0,
        "reduced_chi2": 55.0,
        "n_points": 4,
        "sources": ["final_residual_feature", "template_free"],
        "fit_excluded": True,
        "adopted_model": "fspl",
    }


def test_unified_candidates_keep_separated_detections_and_rank_by_z():
    features = PlanetFeatureResult(
        peaks=(
            PlanetFeature(
                kind="peak",
                index=0,
                time=1.0,
                t_start=0.9,
                t_end=1.1,
                timescale=0.2,
                strength=8.0,
                signed_z=8.0,
                residual=0.1,
                fractional_deviation=0.1,
                magnification_ratio=1.1,
            ),
        ),
        dips=(),
    )
    free = TemplateFreeCandidate(
        kind="zero_crossing",
        season_idx=0,
        start_index=3,
        end_index=4,
        t_start=5.0,
        t_end=5.2,
        t_center=5.1,
        n_points=2,
        chi2=300.0,
        reduced_chi2=150.0,
        max_abs_z=15.0,
    )
    search = TemplateFreeSearchResult(
        time=np.array([0.9, 1.0, 1.1, 5.0, 5.2]),
        residual=np.zeros(5),
        ferr=np.ones(5),
        z=np.array([0.0, 8.0, 0.0, -15.0, 0.0]),
        candidates=(free,),
        fixed_window_candidates=(),
        hybrid_candidates=(),
        blind_reduced_candidates=(),
        best=free,
    )

    candidates = build_anomaly_candidates(
        features,
        search,
        time=search.time,
        fit_exclusion_mask=np.zeros(5, dtype=bool),
        adopted_model="pspl",
    )

    assert [candidate.rank for candidate in candidates] == [1, 2]
    assert [candidate.t_center for candidate in candidates] == [5.1, 1.0]
    assert candidates[0].signed_z == -15.0
