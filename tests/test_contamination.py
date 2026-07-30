import numpy as np
import pytest
from types import SimpleNamespace

from jacscanomaly.contamination import (
    ContaminationConfig,
    ContaminationSegmentation,
    RobustFitResult,
    _blocks_from_state,
    contamination_objective,
    protected_support_mask,
    robust_refine_with_fitter,
    segment_anomaly_dp,
    segmentation_weights,
    scaled_parameter_distance,
)
from jacscanomaly.config import FinderConfig
from jacscanomaly.finder import Finder
from jacscanomaly.effect_detection import EffectCandidate
from jacscanomaly.singlelens_fallback import (
    EffectFitterSpec,
    FallbackConfig,
    FallbackResult,
    detector_seed_parameters,
    make_effect_fitter,
    run_robust_fallback,
    run_staged_joint_fallback,
)


def test_dp_prefers_one_contiguous_anomaly_block():
    time = np.arange(100.0)
    z = np.zeros(100)
    z[45:52] = 12.0

    result = segment_anomaly_dp(time, z)

    assert result.blocks
    assert result.anomaly_fraction < 0.2
    assert any(start <= 45 and end >= 51 for start, end in result.blocks)
    assert np.all(segmentation_weights(result)[45:52] < 1.0)


def test_broad_residual_is_not_discarded_as_planet_mask():
    time = np.arange(100.0)
    z = np.full(100, 6.0)
    result = segment_anomaly_dp(
        time,
        z,
        config=ContaminationConfig(max_anomaly_fraction=0.25, max_anomaly_span_fraction=0.25),
    )

    assert result.anomaly_fraction <= 0.25
    assert "broad_residual_protected" in result.diagnostics


def test_season_gap_resets_anomaly_transition():
    time = np.r_[np.arange(5.0), np.arange(100.0, 105.0)]
    z = np.zeros_like(time)
    z[2] = 12.0
    z[7] = 12.0
    result = segment_anomaly_dp(time, z)

    assert len(result.blocks) == 2


def test_continuous_anomaly_state_starts_a_new_block_after_season_gap():
    time = np.asarray([0.0, 1.0, 100.0, 101.0])
    state = np.ones(time.size, dtype=bool)

    assert _blocks_from_state(time, state, season_gap=10.0) == ((0, 1), (2, 3))


def test_fspl_support_protects_both_sides_of_peak():
    time = np.linspace(-10.0, 10.0, 101)
    seed = np.asarray([0.0, 10.0, 0.1, np.log(0.02)])
    protected = protected_support_mask(time, "fspl", seed)

    assert protected[50]
    assert protected[49]
    assert protected[51]
    assert not protected[0]


def test_detector_seeded_fallback_reuses_existing_fitter_interface():
    class FakeFitter:
        def fit(self, time, flux, ferr, x0):
            model = np.ones_like(flux)
            residual = flux - model
            return SimpleNamespace(
                params=np.asarray(x0, dtype=float),
                raw_params=None,
                residual=residual,
                chi2=float(np.sum((residual / ferr) ** 2)),
            )

    time = np.arange(40.0)
    flux = np.ones(40)
    flux[18:22] += 0.4
    ferr = np.full(40, 0.05)
    result = run_robust_fallback(
        FakeFitter(),
        time,
        flux,
        ferr,
        [0.0, 10.0, 0.1],
        effect="mixed",
        config=FallbackConfig(max_seeds=3),
    )

    assert result.attempts
    assert result.selected_seed.shape == (3,)
    assert not result.success
    assert "effect_score_not_checked" in result.reason_codes


def test_bound_stuck_fallback_is_never_success():
    class FakeFitter:
        def fit(self, time, flux, ferr, x0):
            return SimpleNamespace(
                params=np.asarray(x0, dtype=float),
                raw_params=None,
                residual=np.zeros_like(flux),
                chi2=0.0,
            )

    baseline = SimpleNamespace(chi2=2.0)
    result = run_robust_fallback(
        FakeFitter(),
        np.arange(20.0),
        np.ones(20),
        np.ones(20),
        [0.0, 10.0, 0.1, 0.99999, 0.0],
        effect="space_parallax",
        config=FallbackConfig(
            parameter_dimension=5,
            max_seeds=1,
            max_piE=1.0,
            contamination=ContaminationConfig(max_iter=3),
        ),
        baseline_fit=baseline,
        effect_score_fn=lambda fit: float(fit.chi2),
    )

    assert not result.success
    assert "parameter_at_bound" in result.reason_codes


def test_fallback_rejects_fit_that_only_improves_the_planet_interval(monkeypatch):
    import jacscanomaly.singlelens_fallback as fallback_module

    time = np.arange(20.0)
    ferr = np.ones(20)
    known_planet = time < 10.0
    baseline = SimpleNamespace(
        params=np.asarray([0.0, 10.0, 0.1]),
        residual=np.where(known_planet, 3.0, 0.0),
        chi2=90.0,
    )
    selected = SimpleNamespace(
        params=np.asarray([0.0, 10.0, 0.1, 0.05]),
        residual=np.where(known_planet, 0.0, 1.0),
        chi2=10.0,
        optimizer_success=True,
    )
    segmentation = ContaminationSegmentation(
        state=known_planet.astype(np.int8),
        anomaly_probability=known_planet.astype(float),
        blocks=((0, 9),),
        objective=1.0,
        anomaly_fraction=0.5,
        anomaly_span_fraction=0.5,
        protected_fraction=0.0,
    )

    monkeypatch.setattr(
        fallback_module,
        "robust_refine_with_fitter",
        lambda *args, **kwargs: RobustFitResult(
            fit=selected,
            initial_fit=baseline,
            final_weights=np.ones(20),
            segmentation=segmentation,
            iterations=(),
            converged=True,
            segmentation_stable=True,
        ),
    )
    candidate = EffectCandidate(
        effect="fspl",
        score=100.0,
        score_without_compact_blocks=100.0,
        effective_rank=1,
        condition_number=1.0,
        coverage=1.0,
        max_point_influence=0.0,
        max_block_influence=0.0,
        subset_stability=1.0,
        morphology="mixed_or_planet",
    )
    result = run_robust_fallback(
        object(),
        time,
        np.ones(20),
        ferr,
        [0.0, 10.0, 0.1, 0.05],
        candidates=(candidate,),
        extra_seeds=([0.1, 10.0, 0.1, 0.05],),
        effect="fspl",
        config=FallbackConfig(parameter_dimension=4, max_seeds=2),
        known_anomaly_mask=known_planet,
        baseline_fit=baseline,
        effect_score_fn=lambda fit, *_: 100.0 if fit is baseline else 1.0,
    )

    assert not result.success
    assert "non_planet_region_bic_not_improved" in result.reason_codes


def test_partial_fspl_signed_topology_cannot_override_clean_region_bic(monkeypatch):
    import jacscanomaly.singlelens_fallback as fallback_module

    time = np.arange(20.0)
    ferr = np.ones(20)
    known_planet = time < 10.0
    baseline = SimpleNamespace(
        params=np.asarray([0.0, 10.0, 0.1]),
        residual=np.where(known_planet, 3.0, 0.0),
        chi2=90.0,
    )
    selected = SimpleNamespace(
        params=np.asarray([0.0, 10.0, 0.1, 0.05]),
        residual=np.where(known_planet, 0.0, 1.0),
        chi2=10.0,
        optimizer_success=True,
    )
    segmentation = ContaminationSegmentation(
        state=known_planet.astype(np.int8),
        anomaly_probability=known_planet.astype(float),
        blocks=((0, 9),),
        objective=1.0,
        anomaly_fraction=0.5,
        anomaly_span_fraction=0.5,
        protected_fraction=0.0,
    )
    monkeypatch.setattr(
        fallback_module,
        "robust_refine_with_fitter",
        lambda *args, **kwargs: RobustFitResult(
            fit=selected,
            initial_fit=baseline,
            final_weights=np.ones(20),
            segmentation=segmentation,
            iterations=(),
            converged=True,
            segmentation_stable=True,
        ),
    )
    candidate = EffectCandidate(
        effect="fspl",
        score=100.0,
        score_without_compact_blocks=100.0,
        effective_rank=1,
        condition_number=1.0,
        coverage=1.0,
        max_point_influence=0.0,
        max_block_influence=0.0,
        subset_stability=1.0,
        morphology="fspl_partial_peak",
        subset_diagnostics=({
            "name": "fspl_morphology",
            "partial": True,
            "central_symmetry": 0.98,
            "template_explained_fraction": 0.9,
            "core_mean_z": -30.0,
            "left_shoulder_mean_z": 8.0,
            "right_shoulder_mean_z": float("nan"),
        },),
    )

    result = run_robust_fallback(
        object(),
        time,
        np.ones(20),
        ferr,
        [0.0, 10.0, 0.1, 0.05],
        candidates=(candidate,),
        extra_seeds=([0.1, 10.0, 0.1, 0.05],),
        effect="fspl",
        config=FallbackConfig(parameter_dimension=4, max_seeds=2),
        known_anomaly_mask=known_planet,
        baseline_fit=baseline,
        effect_score_fn=lambda fit, *_: 100.0 if fit is baseline else 1.0,
    )

    assert not result.success
    assert "non_planet_region_bic_not_improved" in result.reason_codes
    assert "fallback_acceptance_failed" in result.reason_codes


def test_clear_fspl_topology_can_override_segmenter_oscillation(monkeypatch):
    import jacscanomaly.singlelens_fallback as fallback_module

    time = np.arange(200.0)
    baseline = SimpleNamespace(
        params=np.asarray([100.0, 40.0, 0.03]),
        residual=np.zeros(200),
        chi2=10_000.0,
    )
    selected = SimpleNamespace(
        params=np.asarray([100.0, 44.0, 0.03, 0.055]),
        residual=np.zeros(200),
        chi2=100.0,
        optimizer_success=True,
    )
    segmentation = ContaminationSegmentation(
        state=np.zeros(200, dtype=np.int8),
        anomaly_probability=np.zeros(200),
        blocks=(),
        objective=1.0,
        anomaly_fraction=0.0,
        anomaly_span_fraction=0.0,
        protected_fraction=0.0,
    )
    monkeypatch.setattr(
        fallback_module,
        "robust_refine_with_fitter",
        lambda *args, **kwargs: RobustFitResult(
            fit=selected,
            initial_fit=baseline,
            final_weights=np.ones(200),
            segmentation=segmentation,
            iterations=(),
            converged=False,
            segmentation_stable=False,
        ),
    )
    candidate = EffectCandidate(
        effect="fspl",
        score=10_000.0,
        score_without_compact_blocks=10_000.0,
        effective_rank=1,
        condition_number=1.0,
        coverage=1.0,
        max_point_influence=0.0,
        max_block_influence=0.0,
        subset_stability=1.0,
        morphology="fspl_flattened_peak",
    )

    result = run_robust_fallback(
        object(),
        time,
        np.ones(200),
        np.ones(200),
        selected.params,
        candidates=(candidate,),
        extra_seeds=([100.1, 44.0, 0.03, 0.055],),
        effect="fspl",
        config=FallbackConfig(parameter_dimension=4, max_seeds=2),
        baseline_fit=baseline,
        effect_score_fn=lambda fit, *_: 10_000.0 if fit is baseline else 10.0,
    )

    assert result.success
    assert "clear_fspl_topology_overrides_contamination_stability" in result.reason_codes


def test_overwhelming_reproduced_parallax_can_override_support_guard(monkeypatch):
    import jacscanomaly.singlelens_fallback as fallback_module

    time = np.arange(200.0)
    baseline = SimpleNamespace(
        params=np.asarray([100.0, 200.0, 0.5]),
        residual=np.zeros(200),
        chi2=10_000.0,
    )
    selected = SimpleNamespace(
        params=np.asarray([100.0, 200.0, 0.5, -0.1, 0.03]),
        residual=np.zeros(200),
        chi2=100.0,
        optimizer_success=True,
    )
    segmentation = ContaminationSegmentation(
        state=np.zeros(200, dtype=np.int8),
        anomaly_probability=np.zeros(200),
        blocks=(),
        objective=1.0,
        anomaly_fraction=0.0,
        anomaly_span_fraction=0.0,
        protected_fraction=1.0,
        protected_component_retained_fractions=(0.0,),
    )
    monkeypatch.setattr(
        fallback_module,
        "robust_refine_with_fitter",
        lambda *args, **kwargs: RobustFitResult(
            fit=selected,
            initial_fit=baseline,
            final_weights=np.ones(200),
            segmentation=segmentation,
            iterations=(),
            converged=True,
            segmentation_stable=True,
        ),
    )

    result = run_robust_fallback(
        object(),
        time,
        np.ones(200),
        np.ones(200),
        selected.params,
        extra_seeds=([100.1, 200.0, 0.5, -0.1, 0.03],),
        effect="space_parallax",
        config=FallbackConfig(parameter_dimension=5, max_seeds=2),
        baseline_fit=baseline,
        effect_score_fn=lambda fit, *_: 10_000.0 if fit is baseline else 1.0,
    )

    assert result.success
    assert "insufficient_identifiability" in result.reason_codes
    assert "overwhelming_parallax_evidence_overrides_support_guard" in result.reason_codes


def test_fallback_accepts_bic_gain_even_when_global_reduced_chi2_is_bad(monkeypatch):
    import jacscanomaly.singlelens_fallback as fallback_module

    time = np.arange(200.0)
    baseline = SimpleNamespace(
        params=np.asarray([100.0, 20.0, 0.1]),
        residual=np.zeros(200),
        chi2=1_000_000.0,
    )
    selected = SimpleNamespace(
        params=np.asarray([100.0, 20.0, 0.1, 0.01]),
        residual=np.zeros(200),
        chi2=900_000.0,
        optimizer_success=True,
    )
    segmentation = ContaminationSegmentation(
        state=np.zeros(200, dtype=np.int8),
        anomaly_probability=np.zeros(200),
        blocks=(),
        objective=1.0,
        anomaly_fraction=0.0,
        anomaly_span_fraction=0.0,
        protected_fraction=0.0,
    )
    monkeypatch.setattr(
        fallback_module,
        "robust_refine_with_fitter",
        lambda *args, **kwargs: RobustFitResult(
            fit=selected,
            initial_fit=baseline,
            final_weights=np.ones(200),
            segmentation=segmentation,
            iterations=(),
            converged=True,
            segmentation_stable=True,
        ),
    )

    result = run_robust_fallback(
        object(),
        time,
        np.ones(200),
        np.ones(200),
        selected.params,
        extra_seeds=([100.1, 20.0, 0.1, 0.01],),
        effect="fspl",
        config=FallbackConfig(parameter_dimension=4, max_seeds=2),
        baseline_fit=baseline,
        effect_score_fn=lambda fit, *_: (
            1_000_000.0 if fit is baseline else 900_000.0
        ),
    )

    assert result.success
    assert "global_fit_improvement_insufficient" in result.reason_codes
    assert "accepted_by_postfit_validity_and_bic" in result.reason_codes


def test_protected_support_is_soft_and_keeps_regularization_finite():
    time = np.arange(60.0)
    z = np.zeros(60)
    z[28:32] = 15.0
    protected = np.zeros(60, dtype=bool)
    protected[28:32] = True

    result = segment_anomaly_dp(
        time,
        z,
        protected_mask=protected,
        config=ContaminationConfig(protected_anomaly_penalty=3.0),
    )

    assert np.all(np.isfinite(result.state))
    assert np.isfinite(result.objective)
    assert result.protected_fraction > 0.0
    assert result.contamination_penalty >= 0.0
    assert "inf" not in repr(result)


def test_fspl_protected_peak_is_soft_and_wing_anomalies_remain_available():
    time = np.linspace(-10.0, 10.0, 201)
    z = np.zeros_like(time)
    z[(time > -7.0) & (time < -6.0)] = 12.0
    z[(time > 6.0) & (time < 7.0)] = 12.0
    protected = protected_support_mask(
        time, "fspl", np.asarray([0.0, 10.0, 0.1, np.log(0.02)])
    )
    result = segment_anomaly_dp(
        time,
        z,
        protected_mask=protected,
        config=ContaminationConfig(protected_anomaly_penalty=1.0),
    )
    weights = segmentation_weights(result)

    assert protected[time.size // 2]
    assert np.any(weights[~protected] < 1.0)
    assert np.all(np.isfinite(result.objective))


def test_parallax_protected_wings_are_soft():
    time = np.linspace(-20.0, 20.0, 201)
    protected = protected_support_mask(
        time, "space_parallax", np.asarray([0.0, 10.0, 0.1])
    )
    z = np.zeros_like(time)
    z[protected] = 10.0
    result = segment_anomaly_dp(
        time,
        z,
        protected_mask=protected,
        config=ContaminationConfig(protected_anomaly_penalty=1.0),
    )

    assert protected[0] and protected[-1]
    assert result.protected_anomaly_fraction < 0.35
    assert np.isfinite(result.objective)


def test_known_planet_block_is_forced_into_anomaly_state():
    time = np.linspace(-5.0, 5.0, 101)
    forced = np.abs(time - 1.0) < 0.3
    result = segment_anomaly_dp(
        time,
        np.zeros_like(time),
        forced_anomaly_mask=forced,
    )

    assert np.all(result.state[forced] == 1)
    assert np.all(result.anomaly_probability[forced] == 1.0)
    assert np.isfinite(result.objective)


def test_mixed_supports_remain_independent_in_common_objective():
    time = np.arange(20.0)
    first = np.zeros(20, dtype=bool)
    second = np.zeros(20, dtype=bool)
    first[4:8] = True
    second[12:16] = True
    result = segment_anomaly_dp(
        time,
        np.zeros_like(time),
        protected_masks=(first, second),
    )

    assert result.protected_components == ("support_0", "support_1")
    assert len(result.protected_component_anomaly_fractions) == 2
    assert len(result.protected_component_retained_fractions) == 2
    assert result.protected_fraction == 0.4
    assert np.isfinite(result.objective)


def test_common_contamination_objective_has_explicit_span_cost():
    time = np.arange(8.0)
    z = np.zeros(8)
    state = np.zeros(8, dtype=bool)
    state[2:6] = True
    config = ContaminationConfig(span_penalty=0.5)

    value = contamination_objective(time, z, state, config=config)
    no_span = contamination_objective(
        time,
        z,
        state,
        config=ContaminationConfig(span_penalty=0.0),
    )

    assert value > no_span


def test_adaptive_constraint_search_reports_the_fixed_canonical_objective():
    time = np.arange(40.0)
    z = np.full(40, 6.0)
    protected = np.zeros(40, dtype=bool)
    protected[10:20] = True
    config = ContaminationConfig(
        max_anomaly_fraction=0.1,
        max_protected_anomaly_fraction=0.1,
    )

    result = segment_anomaly_dp(
        time,
        z,
        protected_mask=protected,
        config=config,
    )
    expected = contamination_objective(
        time,
        z,
        result.state,
        config=config,
        protected_mask=protected,
    )

    assert result.objective == pytest.approx(expected)


def test_span_penalty_changes_long_block_selection():
    time = np.arange(100.0)
    z = np.zeros(100)
    z[20:80] = 3.0

    without_span = segment_anomaly_dp(
        time,
        z,
        config=ContaminationConfig(
            span_penalty=0.0,
            max_anomaly_fraction=0.9,
            max_anomaly_span_fraction=0.9,
        ),
    )
    with_span = segment_anomaly_dp(
        time,
        z,
        config=ContaminationConfig(
            span_penalty=1.0,
            max_anomaly_fraction=0.9,
            max_anomaly_span_fraction=0.9,
        ),
    )

    assert without_span.anomaly_span_fraction > with_span.anomaly_span_fraction


def test_scaled_parameter_distance_uses_dimensionless_raw_contract():
    first = np.asarray([1000.0, 10.0, 0.1, -3.0, 0.2, -0.1])
    second = np.asarray([1001.0, 20.0, 0.2, -2.0, 0.2, -0.1])

    distance = scaled_parameter_distance(first, second)

    assert distance > 0.0
    assert scaled_parameter_distance(first, first) == 0.0
    assert scaled_parameter_distance(
        np.asarray([1000.0, 10.0, 0.1]),
        np.asarray([1001.0, 20.0, 0.1]),
    ) < 0.5


def test_effect_factory_exposes_effect_specific_dimensions():
    config = FinderConfig(fitter_kind="pspl", single_fit_backend="jax")
    fspl = make_effect_fitter(config, "fspl", 100.0)

    assert fspl.parameter_dimension == 4
    assert fspl.raw_parameter_names == ("t0", "tE", "u0", "logrho")
    assert fspl.fitter.profile_peak_only is False


def test_parallax_seed_atlas_covers_multiple_directions():
    candidate = EffectCandidate(
        effect="space_parallax",
        score=100.0,
        score_without_compact_blocks=100.0,
        effective_rank=2,
        condition_number=10.0,
        coverage=0.8,
        max_point_influence=0.1,
        max_block_influence=0.1,
        subset_stability=1.0,
        best_template_or_direction=np.asarray([1.0, 0.0]),
        seed_parameters=np.asarray([0.0, 30.0, 0.1]),
    )
    seeds = detector_seed_parameters(
        [0.0, 30.0, 0.1],
        [candidate],
        config=FallbackConfig(parameter_dimension=5, max_seeds=16),
    )

    pi = np.asarray([seed[3:] for seed in seeds])
    assert np.count_nonzero(np.linalg.norm(pi, axis=1) > 0.1) >= 4


def test_fspl_detector_local_variants_precede_broad_baseline_atlas():
    detector = np.asarray([100.0, 12.0, -0.2, -1.5])
    candidate = EffectCandidate(
        effect="fspl",
        score=100.0,
        score_without_compact_blocks=80.0,
        effective_rank=1,
        condition_number=1.0,
        coverage=1.0,
        max_point_influence=0.0,
        max_block_influence=0.0,
        subset_stability=1.0,
        seed_parameters=detector,
    )

    seeds = detector_seed_parameters(
        [100.0, 10.0, -0.1],
        (candidate,),
        config=FallbackConfig(parameter_dimension=4, max_seeds=6),
    )

    np.testing.assert_allclose(seeds[1], detector)
    assert all(
        np.linalg.norm(np.asarray(seed) - detector) < 4.0
        for seed in seeds[2:6]
    )


def test_fspl_small_seed_budget_covers_factor_e_rho_bias():
    detector = np.asarray([100.0, 12.0, -0.2, -2.0])
    candidate = EffectCandidate(
        effect="fspl",
        score=100.0,
        score_without_compact_blocks=80.0,
        effective_rank=1,
        condition_number=1.0,
        coverage=1.0,
        max_point_influence=0.0,
        max_block_influence=0.0,
        subset_stability=1.0,
        seed_parameters=detector,
    )

    seeds = detector_seed_parameters(
        [100.0, 10.0, -0.1],
        (candidate,),
        config=FallbackConfig(parameter_dimension=4, max_seeds=8),
    )

    assert any(np.isclose(seed[3], detector[3] + 1.0) for seed in seeds)
    assert any(
        np.isclose(seed[3], detector[3] + 1.0)
        and np.signbit(seed[2]) != np.signbit(detector[2])
        for seed in seeds
    )


def test_fspl_seed_families_change_rho_and_keep_four_dimensions():
    candidate = EffectCandidate(
        effect="fspl",
        score=100.0,
        score_without_compact_blocks=100.0,
        effective_rank=1,
        condition_number=1.0,
        coverage=0.8,
        max_point_influence=0.1,
        max_block_influence=0.1,
        subset_stability=1.0,
        seed_parameters=np.asarray([0.0, 10.0, 0.1, -3.0]),
    )
    seeds = detector_seed_parameters(
        [0.0, 10.0, 0.1, -3.0],
        [candidate],
        config=FallbackConfig(parameter_dimension=4, max_seeds=48),
    )

    assert seeds
    assert all(seed.shape == (4,) for seed in seeds)
    assert len({round(float(seed[3]), 6) for seed in seeds}) > 1


def test_mixed_candidates_produce_valid_six_dimension_seeds():
    candidates = [
        EffectCandidate(
            effect="fspl",
            score=100.0,
            score_without_compact_blocks=100.0,
            effective_rank=1,
            condition_number=1.0,
            coverage=0.8,
            max_point_influence=0.1,
            max_block_influence=0.1,
            subset_stability=1.0,
            seed_parameters=np.asarray([0.0, 10.0, 0.1, -1.2]),
        ),
        EffectCandidate(
            effect="space_parallax",
            score=100.0,
            score_without_compact_blocks=100.0,
            effective_rank=2,
            condition_number=1.0,
            coverage=0.8,
            max_point_influence=0.1,
            max_block_influence=0.1,
            subset_stability=1.0,
            best_template_or_direction=np.asarray([1.0, 0.0]),
            seed_parameters=np.asarray([0.0, 10.0, 0.1]),
        ),
    ]
    seeds = detector_seed_parameters(
        [0.0, 10.0, 0.1],
        candidates,
        config=FallbackConfig(parameter_dimension=6, max_seeds=64),
    )

    assert seeds
    assert all(seed.shape == (6,) for seed in seeds)
    assert any(
        abs(float(seed[3]) + 3.0) > 1.0e-6
        and np.linalg.norm(seed[4:6]) > 0.0
        for seed in seeds
    )


def test_joint_fallback_bridges_stage_seeds_into_six_dimensions():
    class FakeFitter:
        def fit(self, time, flux, ferr, x0):
            return SimpleNamespace(
                params=np.asarray(x0, dtype=float),
                raw_params=None,
                residual=np.zeros_like(flux),
                chi2=0.0,
                optimizer_success=True,
            )

    result = run_robust_fallback(
        FakeFitter(),
        np.arange(20.0),
        np.ones(20),
        np.ones(20),
        [0.0, 10.0, 0.1],
        effect="fspl_space_parallax",
        extra_seeds=(
            np.asarray([0.0, 10.0, 0.1, -1.2, 0.2, -0.1]),
            np.asarray([0.0, 10.0, 0.1, 0.2, -0.1]),
        ),
        config=FallbackConfig(
            parameter_dimension=6,
            max_seeds=3,
            contamination=ContaminationConfig(max_iter=3),
        ),
    )

    assert result.attempts
    assert all(attempt.seed.shape == (6,) for attempt in result.attempts)


def test_staged_fallback_keeps_viable_stage_when_joint_ephemeris_is_unavailable(monkeypatch):
    import jacscanomaly.singlelens_fallback as fallback_module

    fit = SimpleNamespace(
        params=np.asarray([0.0, 10.0, 0.1, 0.05]),
        raw_params=np.asarray([0.0, 10.0, 0.1, np.log(0.05)]),
        chi2=1.0,
    )

    def fake_factory(_config, effect, _tref):
        if "space_parallax" in effect:
            raise ValueError("observer ephemeris does not cover the requested data time range")
        return EffectFitterSpec(
            effect=effect,
            fitter=object(),
            parameter_dimension=4,
            parameter_names=("t0", "tE", "u0", "rho"),
            raw_parameter_names=("t0", "tE", "u0", "logrho"),
            backend="synthetic",
            convention="synthetic",
        )

    def fake_run(*args, **kwargs):
        effect = kwargs["effect"]
        return FallbackResult(
            fit=fit,
            initial_fit=fit,
            effect=effect,
            attempts=(),
            selected_seed=np.asarray([0.0, 10.0, 0.1, np.log(0.05)]),
            success=effect == "fspl",
            reason_codes=("synthetic",),
            selected_original_chi2=1.0,
            selected_robust_objective=1.0,
        )

    monkeypatch.setattr(fallback_module, "make_effect_fitter", fake_factory)
    monkeypatch.setattr(fallback_module, "run_robust_fallback", fake_run)
    candidates = (
        EffectCandidate(
            effect="fspl",
            score=10.0,
            score_without_compact_blocks=10.0,
            effective_rank=1,
            condition_number=1.0,
            coverage=1.0,
            max_point_influence=0.0,
            max_block_influence=0.0,
            subset_stability=1.0,
        ),
        EffectCandidate(
            effect="space_parallax",
            score=10.0,
            score_without_compact_blocks=10.0,
            effective_rank=2,
            condition_number=1.0,
            coverage=1.0,
            max_point_influence=0.0,
            max_block_influence=0.0,
            subset_stability=1.0,
        ),
    )

    result = run_staged_joint_fallback(
        FinderConfig(),
        np.arange(20.0),
        np.ones(20),
        np.ones(20),
        [0.0, 10.0, 0.1],
        candidates=candidates,
        effect="fspl_space_parallax",
    )

    assert result.success
    assert result.effect == "fspl"
    assert "accepted_single_effect_stage" in result.reason_codes


def test_staged_fallback_selects_lower_order_model_when_joint_bic_is_worse(
    monkeypatch,
):
    import jacscanomaly.singlelens_fallback as fallback_module

    dimensions = {
        "fspl": 4,
        "annual_parallax": 5,
        "space_parallax": 5,
        "fspl_space_parallax": 6,
    }
    selected_bics = {
        "fspl": 71_550.0,
        "annual_parallax": 40_357.0,
        "space_parallax": 40_341.0,
        "fspl_space_parallax": 40_350.0,
    }

    def fake_factory(config, effect, tref):
        dimension = dimensions[effect]
        return EffectFitterSpec(
            effect=effect,
            fitter=object(),
            parameter_dimension=dimension,
            parameter_names=tuple(f"p{i}" for i in range(dimension)),
            raw_parameter_names=tuple(f"p{i}" for i in range(dimension)),
            backend="synthetic",
            convention="synthetic",
        )

    def fake_run(*args, **kwargs):
        effect = kwargs["effect"]
        dimension = dimensions[effect]
        fit = SimpleNamespace(
            params=np.zeros(dimension),
            chi2=selected_bics[effect],
        )
        return FallbackResult(
            fit=fit,
            initial_fit=fit,
            effect=effect,
            attempts=(),
            selected_seed=np.zeros(dimension),
            success=True,
            reason_codes=("accepted_by_postfit_validity_and_bic",),
            selected_original_chi2=selected_bics[effect],
            selected_bic=selected_bics[effect],
            numerically_valid=True,
        )

    monkeypatch.setattr(fallback_module, "make_effect_fitter", fake_factory)
    monkeypatch.setattr(fallback_module, "run_robust_fallback", fake_run)
    candidates = tuple(
        EffectCandidate(
            effect=effect,
            score=100.0,
            score_without_compact_blocks=100.0,
            effective_rank=1,
            condition_number=1.0,
            coverage=1.0,
            max_point_influence=0.0,
            max_block_influence=0.0,
            subset_stability=1.0,
        )
        for effect in ("fspl", "annual_parallax", "space_parallax")
    )

    result = run_staged_joint_fallback(
        FinderConfig(),
        np.arange(20.0),
        np.ones(20),
        np.ones(20),
        [0.0, 10.0, 0.1],
        candidates=candidates,
        effect="fspl_space_parallax",
    )

    assert result.effect == "space_parallax"
    assert "selected_by_hierarchical_bic" in result.reason_codes
    assert "accepted_lower_order_model" in result.reason_codes


def test_staged_joint_synthetic_regression_preserves_each_stage_seed(monkeypatch):
    import jacscanomaly.singlelens_fallback as fallback_module

    joint_calls = []

    def fake_factory(config, effect, tref):
        dimension = {
            "fspl": 4,
            "space_parallax": 5,
            "fspl_space_parallax": 6,
        }[effect]
        return EffectFitterSpec(
            effect=effect,
            fitter=object(),
            parameter_dimension=dimension,
            parameter_names=tuple(f"p{i}" for i in range(dimension)),
            raw_parameter_names=tuple(f"p{i}" for i in range(dimension)),
            backend="synthetic",
            convention="synthetic",
        )

    def fake_run(*args, **kwargs):
        effect = kwargs["effect"]
        dimension = kwargs["config"].parameter_dimension
        if effect == "fspl_space_parallax":
            joint_calls.append(tuple(kwargs.get("extra_seeds", ())))
        fit = SimpleNamespace(
            params=np.arange(float(dimension)),
            raw_params=None,
            chi2=1.0,
        )
        return FallbackResult(
            fit=fit,
            initial_fit=fit,
            effect=effect,
            attempts=(),
            selected_seed=np.zeros(dimension),
            success=False,
            reason_codes=("synthetic",),
        )

    monkeypatch.setattr(fallback_module, "make_effect_fitter", fake_factory)
    monkeypatch.setattr(fallback_module, "run_robust_fallback", fake_run)
    candidates = (
        EffectCandidate(
            effect="fspl",
            score=10.0,
            score_without_compact_blocks=10.0,
            effective_rank=1,
            condition_number=1.0,
            coverage=1.0,
            max_point_influence=0.0,
            max_block_influence=0.0,
            subset_stability=1.0,
            seed_parameters=np.asarray([0.0, 10.0, 0.1, -3.0]),
        ),
        EffectCandidate(
            effect="space_parallax",
            score=10.0,
            score_without_compact_blocks=10.0,
            effective_rank=2,
            condition_number=1.0,
            coverage=1.0,
            max_point_influence=0.0,
            max_block_influence=0.0,
            subset_stability=1.0,
            best_template_or_direction=np.asarray([1.0, 0.0]),
            seed_parameters=np.asarray([0.0, 10.0, 0.1]),
        ),
    )

    result = run_staged_joint_fallback(
        FinderConfig(),
        np.arange(10.0),
        np.ones(10),
        np.ones(10),
        [0.0, 10.0, 0.1],
        candidates=candidates,
        fallback_config=FallbackConfig(max_seeds=4),
    )

    assert len(result.stage_results) == 2
    assert len(joint_calls) == 1
    assert joint_calls[0]
    assert np.asarray(joint_calls[0][0]).size == 6
    assert any(
        np.asarray(seed)[3] == 3.0
        and np.allclose(np.asarray(seed)[4:6], [3.0, 4.0])
        for seed in joint_calls[0]
    )


def test_robust_refine_returns_segmentation_for_the_final_fit():
    class MovingResidualFitter:
        def __init__(self):
            self.calls = 0

        def fit(self, time, flux, ferr, x0):
            residual = np.zeros_like(flux)
            if self.calls == 0:
                residual[4:7] = 12.0
            else:
                residual[13:16] = 12.0
            self.calls += 1
            return SimpleNamespace(
                params=np.asarray(x0, dtype=float),
                raw_params=None,
                residual=residual,
                chi2=float(np.sum((residual / ferr) ** 2)),
                optimizer_success=True,
            )

    result = robust_refine_with_fitter(
        MovingResidualFitter(),
        np.arange(20.0),
        np.ones(20),
        np.ones(20),
        [0.0, 10.0, 0.1],
        config=ContaminationConfig(max_iter=1),
    )

    assert any(start <= 13 and end >= 15 for start, end in result.segmentation.blocks)
    assert not any(start <= 4 and end >= 6 for start, end in result.segmentation.blocks)


def test_baseline_residual_downweights_contamination_before_first_physical_fit():
    class RecordingFitter:
        def __init__(self):
            self.first_ferr = None

        def fit(self, time, flux, ferr, x0):
            if self.first_ferr is None:
                self.first_ferr = np.asarray(ferr, dtype=float).copy()
            return SimpleNamespace(
                params=np.asarray(x0, dtype=float),
                raw_params=None,
                residual=np.zeros_like(flux),
                chi2=0.0,
                optimizer_success=True,
            )

    fitter = RecordingFitter()
    initial_z = np.zeros(30)
    initial_z[12:16] = 12.0
    robust_refine_with_fitter(
        fitter,
        np.arange(30.0),
        np.ones(30),
        np.ones(30),
        [0.0, 10.0, 0.1],
        initial_standardized_residual=initial_z,
        config=ContaminationConfig(max_iter=1),
    )

    assert np.all(fitter.first_ferr[12:16] > 1.0)
    assert np.min(fitter.first_ferr[12:16]) > np.max(fitter.first_ferr[:10])


def test_local_segmentation_boundary_jitter_does_not_block_convergence():
    class OnePointJitterFitter:
        def __init__(self):
            self.calls = 0

        def fit(self, time, flux, ferr, x0):
            residual = np.zeros_like(flux)
            start = 400 + min(self.calls, 1)
            residual[start : start + 20] = 12.0
            self.calls += 1
            return SimpleNamespace(
                params=np.asarray(x0, dtype=float),
                raw_params=None,
                residual=residual,
                chi2=float(np.sum((residual / ferr) ** 2)),
                optimizer_success=True,
            )

    result = robust_refine_with_fitter(
        OnePointJitterFitter(),
        np.arange(1000.0),
        np.ones(1000),
        np.ones(1000),
        [500.0, 100.0, 0.1],
        config=ContaminationConfig(
            max_iter=4,
            min_weight_change=1.0e-2,
            weight_damping=1.0,
        ),
    )

    assert result.converged
    assert result.segmentation_stable
    assert result.iterations[-1].weight_change < 1.0e-2


def test_wrong_dimension_seed_is_reported_before_fit_attempt():
    with pytest.raises(ValueError, match="Cannot coerce seed"):
        detector_seed_parameters(
            [0.0, 10.0, 0.1, -3.0, 0.2, -0.1, 0.4],
            config=FallbackConfig(parameter_dimension=4),
        )


def test_finder_fspl_acceptance_score_keeps_the_clear_compact_peak(monkeypatch):
    import jacscanomaly.finder as finder_module

    baseline = SimpleNamespace(
        params=np.asarray([0.0, 10.0, 0.1]),
        raw_params=None,
        chi2=10.0,
        marker="baseline",
    )
    selected = SimpleNamespace(
        params=np.asarray([0.0, 10.0, 0.1, 0.05]),
        raw_params=np.asarray([0.0, 10.0, 0.1, np.log(0.05)]),
        chi2=10.0,
        marker="selected",
    )
    observed_scores = []
    observed_max_pi_e = []

    monkeypatch.setattr(
        finder_module,
        "make_effect_fitter",
        lambda config, effect, tref: EffectFitterSpec(
            effect="fspl",
            fitter=object(),
            parameter_dimension=4,
            parameter_names=("t0", "tE", "u0", "rho"),
            raw_parameter_names=("t0", "tE", "u0", "logrho"),
            backend="synthetic",
            convention="synthetic",
        ),
    )
    monkeypatch.setattr(
        finder_module,
        "detect_fspl_from_pspl_fit",
        lambda fit: SimpleNamespace(
            score=20.0 if fit.marker == "baseline" else 2.0,
            score_without_compact_blocks=100.0
            if fit.marker == "baseline"
            else 5.0
        ),
    )

    def fake_fallback(*args, **kwargs):
        score_fn = kwargs["effect_score_fn"]
        observed_scores.extend([score_fn(baseline), score_fn(selected)])
        observed_max_pi_e.append(kwargs["config"].max_piE)
        return FallbackResult(
            fit=selected,
            initial_fit=baseline,
            effect="fspl",
            attempts=(),
            selected_seed=np.asarray([0.0, 10.0, 0.1, np.log(0.05)]),
            success=False,
            reason_codes=("synthetic",),
        )

    monkeypatch.setattr(finder_module, "run_robust_fallback", fake_fallback)
    finder = Finder(FinderConfig())
    monkeypatch.setattr(finder, "_ensure_fitter", lambda tref: None)
    candidate = EffectCandidate(
        effect="fspl",
        score=400.0,
        score_without_compact_blocks=100.0,
        effective_rank=1,
        condition_number=1.0,
        coverage=1.0,
        max_point_influence=0.0,
        max_block_influence=0.0,
        subset_stability=1.0,
        seed_parameters=np.asarray([0.0, 10.0, 0.1, np.log(0.05)]),
        morphology="fspl_even_peak",
    )

    finder.robust_fallback(
        np.arange(20.0),
        np.ones(20),
        np.ones(20),
        fit=baseline,
        candidates=(candidate,),
        effect="fspl",
    )

    assert observed_scores == [400.0, 2.0]
    assert observed_max_pi_e == [FinderConfig().max_piE]
