import numpy as np
import pytest
import jax.numpy as jnp

from jacscanomaly.effect_detection import (
    EffectCandidate,
    build_fspl_template_bank,
    detect_fspl_from_pspl_fit,
    find_compact_blocks,
    parallax_score_test,
    project_out_nuisance,
)
from jacscanomaly.singlelens_model import A_fspl_from_u, A_fspl_logrho_func, A_pspl_func, u_rectilinear
from jacscanomaly.effect_routing import route_candidate, routing_pareto_curve
from jacscanomaly.exact_probe import run_fspl_exact_probe


def test_projection_removes_nuisance_tangent_without_dense_projector():
    time = np.linspace(-2.0, 2.0, 41)
    nuisance = np.column_stack([np.ones_like(time), time])
    vector = 3.0 * nuisance[:, 0] - 2.0 * nuisance[:, 1]

    projected, diagnostics = project_out_nuisance(vector, nuisance)

    np.testing.assert_allclose(projected, 0.0, atol=1.0e-10)
    assert diagnostics.effective_rank == 2
    assert diagnostics.condition_number < 3.0


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


def test_high_score_with_unstable_subsets_fails_open_to_fallback():
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

    assert routed.decision == "fallback"
    assert "fallback_after_probe_unavailable" in routed.reason_codes


def test_fspl_exact_probe_promotes_an_exact_template():
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
