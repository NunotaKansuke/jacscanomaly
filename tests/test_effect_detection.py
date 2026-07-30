import numpy as np
import pytest
import jax
import jax.numpy as jnp
import json

from jacscanomaly.effect_detection import (
    EffectCandidate,
    _fspl_sparse_high_snr_topology,
    _fspl_signed_topology,
    _pspl_nuisance_and_parallax_jacobians,
    _projected_score,
    _projected_template_scores,
    _pspl_magnification_and_jacobian,
    build_fspl_template_bank,
    detect_fspl_from_pspl_fit,
    find_compact_blocks,
    parallax_score_test,
    project_out_nuisance,
)
from jacscanomaly.singlelens_model import (
    A_fspl_from_u,
    A_fspl_logrho_func,
    A_pspl_func,
    A_pspl_parallax_func,
    A_pspl_space_parallax_func,
    u_rectilinear,
)
from jacscanomaly.effect_routing import route_candidate, routing_pareto_curve
from jacscanomaly.exact_probe import run_fspl_exact_probe
from jacscanomaly import parallax
from jacscanomaly.trajectory import make_parallax_projector


def test_projection_removes_nuisance_tangent_without_dense_projector():
    time = np.linspace(-2.0, 2.0, 41)
    nuisance = np.column_stack([np.ones_like(time), time])
    vector = 3.0 * nuisance[:, 0] - 2.0 * nuisance[:, 1]

    projected, diagnostics = project_out_nuisance(vector, nuisance)

    np.testing.assert_allclose(projected, 0.0, atol=1.0e-10)
    assert diagnostics.effective_rank == 2
    assert diagnostics.condition_number < 3.0


def test_batched_template_scores_match_individual_projection():
    rng = np.random.default_rng(42)
    n_points = 101
    z = rng.normal(size=n_points)
    nuisance = np.column_stack(
        [np.ones(n_points), np.linspace(-1.0, 1.0, n_points)]
    )
    templates = rng.normal(size=(7, n_points))
    mask = np.ones(n_points, dtype=bool)
    mask[[3, 17]] = False

    batched = _projected_template_scores(
        z, templates, nuisance, mask, rtol=1.0e-10
    )
    individual = np.asarray(
        [
            _projected_score(
                z, template, nuisance, mask, rtol=1.0e-10
            ).score
            for template in templates
        ]
    )

    np.testing.assert_allclose(batched, individual, rtol=1.0e-12, atol=1.0e-12)


def test_degenerate_projected_template_has_boolean_support():
    z = np.ones(10)
    nuisance = np.ones((10, 1))
    template = np.ones(10)
    score = _projected_score(
        z,
        template,
        nuisance,
        np.ones(10, dtype=bool),
        rtol=1.0e-10,
    )

    assert score.support.dtype == np.bool_
    assert not np.any(score.support)


def test_analytic_pspl_jacobian_matches_jax():
    time = np.linspace(-4.0, 7.0, 83)
    params = np.asarray([0.3, 2.7, -0.18])

    magnification, jacobian = _pspl_magnification_and_jacobian(time, params)
    expected_magnification = np.asarray(
        A_pspl_func(jnp.asarray(params), jnp.asarray(time))
    )
    expected_jacobian = np.asarray(
        __import__("jax").jacfwd(
            lambda q: A_pspl_func(q, jnp.asarray(time))
        )(jnp.asarray(params))
    )

    np.testing.assert_allclose(
        magnification, expected_magnification, rtol=2.0e-6, atol=2.0e-6
    )
    np.testing.assert_allclose(
        jacobian, expected_jacobian, rtol=3.0e-5, atol=3.0e-5
    )


def test_analytic_annual_parallax_tangent_matches_jax():
    from types import SimpleNamespace

    time = np.linspace(8950.0, 9050.0, 121)
    params = np.asarray([9001.0, 27.0, -0.13])
    projector = make_parallax_projector(
        267.6, -29.1, params[0], use_HJD=True
    )
    fit = SimpleNamespace(
        params=params,
        fs=1.7,
        ferr=np.linspace(0.01, 0.02, time.size),
        residual=np.zeros(time.size),
    )

    _, _, tangent = _pspl_nuisance_and_parallax_jacobians(
        fit, time, projector, space=False
    )
    p0 = jnp.asarray([*params, 0.0, 0.0])
    expected = jax.jacfwd(
        lambda q: A_pspl_parallax_func(q, jnp.asarray(time), projector)
    )(p0)[:, 3:]
    expected = fit.fs * np.asarray(expected) / fit.ferr[:, None]

    np.testing.assert_allclose(tangent, expected, rtol=5.0e-5, atol=5.0e-5)


def test_analytic_gulls_parallax_tangent_matches_jax():
    from types import SimpleNamespace

    sample_time = np.linspace(2450000.0, 2450040.0, 9)
    phase = np.linspace(0.0, 0.3, sample_time.size)
    positions = np.column_stack(
        [np.cos(phase), np.sin(phase), 0.1 * np.sin(2.0 * phase)]
    )
    velocities = np.gradient(positions, sample_time, axis=0)
    observer = parallax.SatelliteEphemeris(
        jnp.asarray(sample_time), jnp.asarray(positions), jnp.asarray(velocities)
    )
    projector = parallax.GullsSpaceParallaxProjector(
        observer, 267.6, -29.1, 2450020.0
    )
    time = np.linspace(5.0, 35.0, 101)
    params = np.asarray([20.0, 12.0, 0.2])
    fit = SimpleNamespace(
        params=params,
        fs=0.8,
        ferr=np.full(time.size, 0.015),
        residual=np.zeros(time.size),
    )

    _, _, tangent = _pspl_nuisance_and_parallax_jacobians(
        fit, time, projector, space=True
    )
    p0 = jnp.asarray([*params, 0.0, 0.0])
    expected = jax.jacfwd(
        lambda q: A_pspl_space_parallax_func(
            q, jnp.asarray(time), projector
        )
    )(p0)[:, 3:]
    expected = fit.fs * np.asarray(expected) / fit.ferr[:, None]

    np.testing.assert_allclose(tangent, expected, rtol=5.0e-5, atol=5.0e-5)


def test_parallax_score_matches_linear_projected_delta_chi2():
    time = np.linspace(-5.0, 5.0, 101)
    nuisance = np.column_stack([np.ones_like(time), time])
    H = np.column_stack([time * time, np.sin(time)])
    beta = np.asarray([1.7, -0.8])
    z = nuisance @ np.asarray([2.0, -0.25]) + H @ beta

    candidate = parallax_score_test(
        time,
        z,
        nuisance,
        H,
        compact_sigma=100.0,
    )

    projected_H = H - nuisance @ np.linalg.lstsq(nuisance, H, rcond=None)[0]
    expected = float((projected_H @ beta) @ (projected_H @ beta))
    np.testing.assert_allclose(candidate.score, expected, rtol=1.0e-8, atol=1.0e-8)
    assert candidate.effective_rank == 2
    assert candidate.score_without_compact_blocks == candidate.score


def test_rank_deficient_parallax_geometry_is_reported():
    time = np.linspace(-1.0, 1.0, 31)
    nuisance = np.column_stack([np.ones_like(time), time])
    H = np.column_stack([time * time, 2.0 * time * time])
    z = H[:, 0]

    candidate = parallax_score_test(time, z, nuisance, H, compact_sigma=100.0)

    assert candidate.effective_rank == 1
    assert "rank_deficient" in candidate.reason_codes


def test_parallax_subset_stability_uses_direction_not_raw_subset_size():
    time = np.linspace(-5.0, 5.0, 200)
    nuisance = np.ones((time.size, 1))
    H = np.column_stack([np.sin(time), np.cos(time)])
    z = H @ np.asarray([1.5, -0.7])

    candidate = parallax_score_test(
        time,
        z,
        nuisance,
        H,
        compact_sigma=100.0,
    )

    assert candidate.subset_stability > 0.25
    assert all(
        row.get("reason") != "insufficient_subset_information"
        for row in candidate.subset_diagnostics
        if row.get("name") in {"pre", "post"}
    )


def test_parallax_direction_reversal_is_unstable():
    time = np.linspace(-5.0, 5.0, 200)
    nuisance = np.ones((time.size, 1))
    H = np.column_stack([np.sin(time), np.cos(time)])
    z = H @ np.asarray([1.5, -0.7])
    z[time > 0.0] = -(H @ np.asarray([1.5, -0.7]))[time > 0.0]

    candidate = parallax_score_test(
        time,
        z,
        nuisance,
        H,
        compact_sigma=100.0,
    )

    assert candidate.subset_stability < 0.25
    assert "subset_unstable" in candidate.reason_codes


def test_compact_block_helper_does_not_remove_broad_residual():
    time = np.linspace(0.0, 20.0, 101)
    z = np.full_like(time, 6.0)

    mask = find_compact_blocks(time, z, sigma=5.0, max_blocks=1, max_span=2.0)

    assert not np.any(mask)


def test_fspl_joint_template_bank_recovers_injected_template():
    pytest.importorskip("microjax")
    from types import SimpleNamespace

    time = np.linspace(-20.0, 20.0, 161)
    pspl = np.asarray([0.0, 10.0, 0.1])
    fspl = np.asarray([0.0, 10.0, 0.1, np.log(0.08)])
    pspl_model = np.asarray(A_pspl_func(jnp.asarray(pspl), jnp.asarray(time)))
    fspl_model = np.asarray(A_fspl_logrho_func(jnp.asarray(fspl), jnp.asarray(time)))
    fit = SimpleNamespace(
        time=time,
        params=jnp.asarray(pspl),
        fs=jnp.asarray(1.0),
        ferr=np.full(time.size, 0.01),
        residual=fspl_model - pspl_model,
    )

    candidate = detect_fspl_from_pspl_fit(
        fit,
        rho_over_u0=(0.5, 1.0),
        tE_factors=(1.0,),
    )

    assert candidate.effect == "fspl"
    assert candidate.score > 20.0
    assert candidate.seed_parameters is not None
    assert candidate.seed_parameters.shape == (4,)


def test_fspl_template_bank_profiles_pspl_nuisance_and_forwards_fft_size():
    pytest.importorskip("microjax")

    time = np.linspace(-5.0, 5.0, 41)
    pspl = np.asarray([0.0, 3.0, 0.2])
    bank, metadata = build_fspl_template_bank(
        time,
        pspl,
        rho_over_u0=(1.0,),
        tE_factors=(1.0,),
        u0_signs=(1.0,),
        N_fft=128,
        backend="microjax",
    )
    rho = float(metadata[0][3])
    tE = float(metadata[0][4])
    u0 = float(metadata[0][5])
    u = u_rectilinear(0.0, tE, u0, jnp.asarray(time))
    A_fspl = np.asarray(A_fspl_from_u(u, rho, N_fft=128))
    A_pspl = np.asarray(A_pspl_func(jnp.asarray([0.0, tE, u0]), jnp.asarray(time)))
    design = np.column_stack([A_pspl, np.ones_like(A_pspl)])
    coefficients, *_ = np.linalg.lstsq(design, A_fspl, rcond=None)
    expected = A_fspl - design @ coefficients

    assert bank.shape == (1, time.size)
    np.testing.assert_allclose(bank[0], expected, rtol=1.0e-5, atol=1.0e-5)


def test_fspl_sign_degeneracy_reuses_one_physical_curve(monkeypatch):
    calls = []

    def fake_magnification(u, rho):
        calls.append((np.asarray(u).copy(), float(rho)))
        return 1.0 + 0.1 / (np.asarray(u) + float(rho))

    monkeypatch.setattr(
        "jacscanomaly.effect_detection._native_fspl_magnification",
        fake_magnification,
    )
    bank, metadata = build_fspl_template_bank(
        np.linspace(-3.0, 3.0, 31),
        (0.0, 2.0, -0.2),
        rho_over_u0=(0.5, 1.0),
        tE_factors=(0.5, 1.0),
        u0_signs=(-1.0, 1.0),
    )

    assert len(calls) == 4
    assert bank.shape == (8, 31)
    np.testing.assert_allclose(bank[0], bank[1])
    assert metadata[0][2] == -1.0
    assert metadata[1][2] == 1.0


def test_native_fspl_bank_caps_rho_at_validated_vbm_limit(monkeypatch):
    evaluated_rho = []

    def fake_magnification(u, rho):
        evaluated_rho.append(float(rho))
        return np.ones_like(u, dtype=float) + 0.01

    monkeypatch.setattr(
        "jacscanomaly.effect_detection._native_fspl_magnification",
        fake_magnification,
    )
    _, metadata = build_fspl_template_bank(
        np.linspace(-2.0, 2.0, 21),
        (0.0, 1.0, 20.0),
        rho_over_u0=(4.0,),
        tE_factors=(1.0,),
        u0_signs=(1.0,),
    )

    assert evaluated_rho == [10.0]
    assert metadata[0][3] == 10.0


def test_routing_has_three_stages_and_pareto_metrics():
    candidate = EffectCandidate(
        effect="annual_parallax",
        score=40.0,
        score_without_compact_blocks=38.0,
        effective_rank=2,
        condition_number=10.0,
        coverage=0.8,
        max_point_influence=0.1,
        max_block_influence=0.2,
        subset_stability=0.9,
    )
    routed = route_candidate(candidate)
    assert routed.decision == "fallback"

    null_candidate = EffectCandidate(
        effect="annual_parallax",
        score=8.0,
        score_without_compact_blocks=8.0,
        effective_rank=2,
        condition_number=10.0,
        coverage=0.8,
        max_point_influence=0.1,
        max_block_influence=0.2,
        subset_stability=0.9,
    )
    curve = routing_pareto_curve([candidate, null_candidate], [True, False], [9.0, 50.0])
    assert curve[0]["recall"] == 1.0
    assert curve[0]["fallback_rate"] == 0.0


def test_high_score_with_unstable_subsets_requires_model_comparison():
    candidate = EffectCandidate(
        effect="space_parallax",
        score=100.0,
        score_without_compact_blocks=90.0,
        effective_rank=2,
        condition_number=10.0,
        coverage=0.8,
        max_point_influence=0.1,
        max_block_influence=0.2,
        subset_stability=0.0,
    )

    routed = route_candidate(candidate)

    assert routed.decision == "exact_probe"
    assert "diagnostic_uncertainty" in routed.reason_codes


def test_short_event_parallax_needs_exact_comparison_even_with_coherent_wings():
    candidate = EffectCandidate(
        effect="annual_parallax",
        score=100.0,
        score_without_compact_blocks=90.0,
        effective_rank=2,
        condition_number=10.0,
        coverage=0.8,
        max_point_influence=0.1,
        max_block_influence=0.2,
        subset_stability=0.9,
        seed_parameters=np.asarray([0.0, 7.0, 0.1]),
        morphology="parallax_coherent_wings",
    )

    routed = route_candidate(candidate)

    assert routed.decision == "exact_probe"
    assert "short_event_parallax_requires_model_comparison" in routed.reason_codes


def test_coherent_parallax_wings_cannot_bypass_planet_mask_conflict():
    candidate = EffectCandidate(
        effect="annual_parallax",
        score=1000.0,
        score_without_compact_blocks=950.0,
        effective_rank=2,
        condition_number=10.0,
        coverage=0.8,
        max_point_influence=0.1,
        max_block_influence=0.2,
        subset_stability=0.9,
        seed_parameters=np.asarray([0.0, 80.0, 0.1]),
        score_without_planet=5.0,
        planet_overlap=0.6,
        morphology="parallax_coherent_wings",
        reason_codes=("planet_morphology_dominated",),
    )

    routed = route_candidate(candidate)

    assert routed.decision == "exact_probe"
    assert (
        "parallax_planet_conflict_requires_model_comparison"
        in routed.reason_codes
    )


def test_planet_dominated_physical_score_is_routed_as_planet():
    candidate = EffectCandidate(
        effect="fspl",
        score=1000.0,
        score_without_compact_blocks=900.0,
        effective_rank=1,
        condition_number=2.0,
        coverage=0.9,
        max_point_influence=0.01,
        max_block_influence=0.05,
        subset_stability=0.9,
        score_without_planet=20.0,
        planet_overlap=0.8,
        morphology="planet_like",
        reason_codes=("planet_morphology_dominated",),
    )

    routed = route_candidate(candidate)

    assert routed.decision == "planet"
    assert "routed_as_planet_anomaly" in routed.reason_codes


def test_symmetric_flattened_fspl_peak_routes_to_fallback():
    candidate = EffectCandidate(
        effect="fspl",
        score=2.0e6,
        score_without_compact_blocks=1.8e6,
        effective_rank=1,
        condition_number=2.0,
        coverage=0.9,
        max_point_influence=0.01,
        max_block_influence=0.05,
        subset_stability=0.99,
        score_without_planet=1.0e4,
        planet_overlap=0.9,
        morphology="fspl_flattened_peak",
        reason_codes=("planet_morphology_dominated",),
    )

    routed = route_candidate(candidate)

    assert routed.decision == "fallback"
    assert "strong_physical_score" in routed.reason_codes


def test_fspl_topology_requires_central_dip_and_two_positive_shoulders():
    time = np.linspace(-2.0, 2.0, 81)
    radius = np.abs(time)
    residual = np.zeros_like(time)
    residual[radius <= 0.5] = -8.0
    residual[(radius > 0.5) & (radius <= 1.5)] = 4.0

    topology = _fspl_signed_topology(
        time,
        residual,
        t0=0.0,
        t_star=1.0,
        valid_mask=np.ones(time.size, dtype=bool),
    )

    assert topology["valid"]
    assert not topology["partial"]


def test_fspl_topology_does_not_accept_a_compact_planet_dip():
    time = np.linspace(-2.0, 2.0, 81)
    residual = np.zeros_like(time)
    residual[np.abs(time) <= 0.15] = -12.0

    topology = _fspl_signed_topology(
        time,
        residual,
        t0=0.0,
        t_star=1.0,
        valid_mask=np.ones(time.size, dtype=bool),
    )

    assert not topology["valid"]
    assert not topology["partial"]


def test_sparse_high_snr_fspl_topology_accepts_two_symmetric_samples_per_region():
    topology = {
        "core_mean_z": -1032.0,
        "left_shoulder_mean_z": 48.0,
        "right_shoulder_mean_z": 36.0,
        "core_points": 2,
        "left_shoulder_points": 2,
        "right_shoulder_points": 2,
    }

    assert _fspl_sparse_high_snr_topology(
        topology,
        symmetry=0.9503,
        template_explained_fraction=0.327,
    )


def test_sparse_high_snr_fspl_topology_rejects_a_one_sign_compact_dip():
    topology = {
        "core_mean_z": -100.0,
        "left_shoulder_mean_z": 8.0,
        "right_shoulder_mean_z": -4.0,
        "core_points": 2,
        "left_shoulder_points": 2,
        "right_shoulder_points": 2,
    }

    assert not _fspl_sparse_high_snr_topology(
        topology,
        symmetry=0.99,
        template_explained_fraction=0.8,
    )


def test_fspl_topology_marks_one_sided_observation_for_exact_comparison():
    time = np.linspace(-2.0, 0.0, 41)
    radius = np.abs(time)
    residual = np.zeros_like(time)
    residual[radius <= 0.5] = -8.0
    residual[(radius > 0.5) & (radius <= 1.5)] = 4.0

    topology = _fspl_signed_topology(
        time,
        residual,
        t0=0.0,
        t_star=1.0,
        valid_mask=np.ones(time.size, dtype=bool),
    )

    assert not topology["valid"]
    assert topology["partial"]


def test_candidate_summary_is_recursively_json_serializable():
    candidate = EffectCandidate(
        effect="fspl",
        score=np.float64(12.0),
        score_without_compact_blocks=11.0,
        effective_rank=1,
        condition_number=2.0,
        coverage=0.8,
        max_point_influence=0.1,
        max_block_influence=0.2,
        subset_stability=0.9,
        subset_diagnostics=(
            {
                "direction": np.asarray([-0.5]),
                "nested": {"score": np.float64(3.0)},
            },
        ),
    )

    encoded = json.dumps(candidate.summary_dict())

    assert '"direction": [-0.5]' in encoded
    assert '"score": 3.0' in encoded


def test_fspl_exact_probe_promotes_an_exact_template():
    pytest.importorskip("microjax")
    from types import SimpleNamespace

    time = np.linspace(-2.0, 2.0, 81)
    pspl = np.asarray([0.0, 1.0, 0.1])
    fspl = np.asarray([0.0, 1.0, 0.1, np.log(0.08)])
    pspl_model = np.asarray(A_pspl_func(jnp.asarray(pspl), jnp.asarray(time)))
    flux = np.asarray(A_fspl_logrho_func(jnp.asarray(fspl), jnp.asarray(time)))
    fit = SimpleNamespace(
        time=time,
        flux=flux,
        params=jnp.asarray(pspl),
        fs=jnp.asarray(1.0),
        fb=jnp.asarray(0.0),
        ferr=np.full(time.size, 0.01),
        residual=flux - pspl_model,
    )
    candidate = EffectCandidate(
        effect="fspl",
        score=100.0,
        score_without_compact_blocks=100.0,
        effective_rank=1,
        condition_number=1.0,
        coverage=1.0,
        max_point_influence=0.1,
        max_block_influence=0.1,
        subset_stability=1.0,
        seed_parameters=fspl,
    )

    result = run_fspl_exact_probe(
        fit,
        candidate,
        rho_over_u0=(0.75,),
        tE_factors=(1.0,),
        t0_offsets=(0.0,),
    )

    assert result.decision == "fallback"
    assert result.best is not None
    assert result.best.delta_chi2 > 9.0
