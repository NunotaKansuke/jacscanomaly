import numpy as np

from jacscanomaly import (
    FlatBaselineDiagnostic,
    PlanetAnomalyClassifier,
    PlanetClassConfig,
    PlanetSignalCandidate,
    PlanetSignalResult,
)
from jacscanomaly.planet_class import (
    PSPLParams,
    fold_g0,
    fold_g0_integral,
    fold_limb_darkened,
    q_grid_from_width,
    r_major,
    r_minor,
)
from jacscanomaly.singlelens_fit import SingleLensFitResult
from jacscanomaly.singlelens_model import A_pspl_func


def _flat_diagnostic():
    return FlatBaselineDiagnostic(
        use_flat_baseline=False,
        peak_in_mask=False,
        u0=0.2,
        n_peak_support=0,
        n_unmasked_peak_support=0,
        masked_peak_fraction=0.0,
        dchi2_flat_minus_pspl=0.0,
        improvement_n_eff=0.0,
        improvement_peak_frac=0.0,
    )


def _result_from_residual(time, residual, mask, *, ferr_value=0.02):
    ferr = np.full_like(time, ferr_value)
    params = np.array([10.0, 3.0, 0.2])
    fs = 2.0
    fb = 1.0
    model = fs * np.asarray(A_pspl_func(params, time)) + fb
    flux = model + residual
    fit = SingleLensFitResult(
        time=time,
        flux=flux,
        ferr=ferr,
        params=params,
        param_names=("t0", "tE", "u0"),
        chi2=np.array(float(np.sum((residual / ferr) ** 2))),
        chi2_dof=np.array(1.0),
        fs=np.array(fs),
        fb=np.array(fb),
        model_flux=model,
        residual=residual,
    )
    idx = np.flatnonzero(mask)
    candidate = PlanetSignalCandidate(
        start_index=int(idx[0]),
        end_index=int(idx[-1]) + 1,
        t_start=float(time[idx[0]]),
        t_end=float(time[idx[-1]]),
        t_center=float(0.5 * (time[idx[0]] + time[idx[-1]])),
        n_points=int(idx.size),
        chi2=float(np.sum((residual[mask] / ferr[mask]) ** 2)),
        reduced_chi2=1.0,
        max_abs_z=float(np.max(np.abs(residual[mask] / ferr[mask]))),
        peak_time=float(time[np.argmax(np.abs(residual / ferr))]),
        peak_z=float((residual / ferr)[np.argmax(np.abs(residual / ferr))]),
        signed_sum_z=float(np.sum(residual[mask] / ferr[mask])),
    )
    return PlanetSignalResult(
        time=time,
        flux=flux,
        ferr=ferr,
        initial_fit=fit,
        refined_fit=fit,
        initial_residual=residual,
        refined_residual=residual,
        signal_mask=mask,
        point_weight=np.where(mask, 0.0, 1.0),
        flat_baseline_diagnostic=_flat_diagnostic(),
        iterations=(),
        candidates=(candidate,),
        best=candidate,
    )


def test_pspl_image_radii_are_reciprocal():
    u = np.logspace(-3, 2, 100)
    assert np.allclose(r_major(u) * r_minor(u), 1.0)


def test_q_grid_width_scaling_is_quadratic():
    pspl = PSPLParams(t0=0.0, tE=10.0, u0=0.1, Fs=1.0, Fb=0.0)
    q1 = q_grid_from_width(0.5, pspl.tE, factors=(1.0,), q_floor=1e-9, q_ceil=1.0)[0]
    q2 = q_grid_from_width(1.0, pspl.tE, factors=(1.0,), q_floor=1e-9, q_ceil=1.0)[0]
    assert np.isclose(q2 / q1, 4.0)


def test_planet_anomaly_classifier_fits_positive_bump_and_generates_counterpart_seeds():
    time = np.linspace(8.0, 12.0, 161)
    residual = 0.25 / (1.0 + ((time - 9.3) / 0.12) ** 2)
    mask = np.abs(time - 9.3) < 0.5
    result = _result_from_residual(time, residual, mask)

    fit = PlanetAnomalyClassifier(
        PlanetClassConfig(min_delta_chi2_for_seed=5.0)
    ).fit(result)

    assert fit.best_label == "major_image_bump"
    assert fit.best_atom is not None
    assert abs(fit.best_atom.params["t_peak"] - 9.3) < 0.05
    assert fit.best_atom.param_errors is not None
    assert fit.best_atom.param_errors["t_peak"] > 0.0
    assert fit.best_atom.param_errors["width"] > 0.0
    assert any(seed.degeneracy_tag == "wide_major" and seed.params["s"] > 1.0 for seed in fit.event_seeds)
    assert any(seed.degeneracy_tag == "close_counterpart" and seed.params["s"] < 1.0 for seed in fit.event_seeds)


def test_planet_anomaly_classifier_fits_negative_dip_and_generates_minor_image_seeds():
    time = np.linspace(8.0, 12.0, 161)
    residual = -0.22 / (1.0 + ((time - 10.6) / 0.15) ** 2)
    mask = np.abs(time - 10.6) < 0.55
    result = _result_from_residual(time, residual, mask)

    fit = PlanetAnomalyClassifier(
        PlanetClassConfig(min_delta_chi2_for_seed=5.0)
    ).fit(result)

    assert fit.best_label == "minor_image_dip"
    assert fit.best_atom is not None
    assert abs(fit.best_atom.params["t_peak"] - 10.6) < 0.05
    assert any(seed.degeneracy_tag == "close_minor" and seed.params["s"] < 1.0 for seed in fit.event_seeds)
    assert any(seed.degeneracy_tag == "wide_counterpart" and seed.params["s"] > 1.0 for seed in fit.event_seeds)


def test_planet_anomaly_classifier_can_identify_second_pspl_like_bump():
    time = np.linspace(0.0, 30.0, 400)
    params2 = np.array([20.0, 4.0, 0.35])
    residual = 0.07 * (np.asarray(A_pspl_func(params2, time)) - 1.0)
    mask = np.abs(time - 20.0) < 5.0
    result = _result_from_residual(time, residual, mask, ferr_value=0.01)

    fit = PlanetAnomalyClassifier(
        PlanetClassConfig(min_delta_chi2_for_seed=5.0)
    ).fit(result)

    second = [
        atom
        for segment in fit.segment_results
        for atom in segment.atom_fits
        if atom.class_label == "second_source"
    ]
    assert second
    assert second[0].success
    assert abs(second[0].params["t0_2"] - 20.0) < 0.2
    assert abs(second[0].params["tE_2"] - 4.0) < 0.2
    assert any(seed.model_type == "1L2S" for seed in fit.event_seeds)


def test_planet_anomaly_classifier_generates_central_caustic_seed_family():
    time = np.linspace(8.0, 12.0, 241)
    residual = (
        0.18 / (1.0 + ((time - 9.82) / 0.10) ** 2)
        + 0.15 / (1.0 + ((time - 10.18) / 0.10) ** 2)
        - 0.06 / (1.0 + ((time - 10.0) / 0.22) ** 2)
    )
    mask = np.abs(time - 10.0) < 0.7
    result = _result_from_residual(time, residual, mask)

    fit = PlanetAnomalyClassifier(
        PlanetClassConfig(min_delta_chi2_for_seed=5.0)
    ).fit(result)

    central = [
        atom
        for segment in fit.segment_results
        for atom in segment.atom_fits
        if atom.class_label == "central_caustic"
    ]
    assert central
    assert central[0].success
    central_seeds = [seed for seed in fit.event_seeds if seed.class_label == "central_caustic"]
    assert central_seeds
    assert any(seed.params["s"] < 1.0 for seed in central_seeds)
    assert any(seed.params["s"] > 1.0 for seed in central_seeds)
    assert all(1e-7 <= seed.params["q"] <= 1.0 for seed in central_seeds)


def test_fold_kernel_support_and_positive_tail():
    assert np.allclose(fold_g0(np.array([-3.0, -1.5, -1.0])), 0.0)
    z = np.array([10.0, 100.0, 1000.0])
    values = fold_g0(z)
    assert np.all(values > 0.0)
    scaled = values * np.sqrt(z)
    assert np.max(scaled) / np.min(scaled) < 1.2
    direct = fold_g0_integral(np.array([-0.5, 0.0, 2.0]))
    lookup = fold_g0(np.array([-0.5, 0.0, 2.0]))
    assert np.allclose(lookup, direct, rtol=2e-3, atol=2e-3)


def test_planet_anomaly_classifier_fits_fold_caustic_atom_and_seed():
    time = np.linspace(8.8, 10.8, 220)
    tc = 9.75
    tstar = 0.08
    sign = 1.0
    residual = 0.18 * fold_g0(sign * (time - tc) / tstar)
    mask = (time > 9.55) & (time < 10.25)
    result = _result_from_residual(time, residual, mask, ferr_value=0.01)

    fit = PlanetAnomalyClassifier(
        PlanetClassConfig(min_delta_chi2_for_seed=5.0)
    ).fit(result)

    fold = [
        atom
        for segment in fit.segment_results
        for atom in segment.atom_fits
        if atom.class_label == "fold_caustic"
    ]
    assert fold
    assert fold[0].success
    assert abs(fold[0].params["tc"] - tc) < 0.04
    assert abs(fold[0].params["tstar"] - tstar) < 0.04
    assert any(
        seed.class_label == "fold_caustic"
        and seed.degeneracy_tag == "local_caustic_only"
        for seed in fit.event_seeds
    )


def test_planet_anomaly_classifier_fits_curved_fold_caustic_atom():
    time = np.linspace(8.8, 10.8, 220)
    tc = 9.75
    tstar = 0.08
    q_curv = 0.45
    sign = 1.0
    tau = (time - tc) / tstar
    residual = 0.18 * fold_g0(sign * (tau + q_curv * tau * tau))
    mask = (time > 9.55) & (time < 10.25)
    result = _result_from_residual(time, residual, mask, ferr_value=0.01)

    fit = PlanetAnomalyClassifier(
        PlanetClassConfig(min_delta_chi2_for_seed=5.0, keep_top_atom_fits=10)
    ).fit(result)

    curved = [
        atom
        for segment in fit.segment_results
        for atom in segment.atom_fits
        if atom.class_label == "curved_fold_caustic"
    ]
    assert curved
    assert curved[0].success
    assert np.isfinite(curved[0].params["tc"])
    assert abs(curved[0].params["q_curv"]) > 0.05
    assert any(seed.class_label == "curved_fold_caustic" for seed in fit.event_seeds)


def test_planet_anomaly_classifier_fits_cusp_tail_atom_and_seed():
    time = np.linspace(8.0, 12.0, 220)
    ta = 10.15
    width = 0.35
    b = 0.25
    p = 1.0
    residual = 0.10 * (b * b + ((time - ta) / width) ** 2) ** (-0.5 * p)
    mask = np.abs(time - ta) < 1.0
    result = _result_from_residual(time, residual, mask, ferr_value=0.015)

    fit = PlanetAnomalyClassifier(
        PlanetClassConfig(min_delta_chi2_for_seed=5.0, keep_top_atom_fits=10)
    ).fit(result)

    cusp = [
        atom
        for segment in fit.segment_results
        for atom in segment.atom_fits
        if atom.class_label == "cusp_caustic"
    ]
    assert cusp
    assert cusp[0].success
    assert abs(cusp[0].params["ta"] - ta) < 0.08
    assert any(
        seed.class_label == "cusp_caustic"
        and seed.degeneracy_tag == "local_cusp_only"
        for seed in fit.event_seeds
    )


def test_limb_darkened_fold_atom_fits_gamma_and_seed():
    time = np.linspace(8.8, 10.8, 140)
    tc = 9.75
    tstar = 0.08
    residual = 0.16 * fold_limb_darkened((time - tc) / tstar, gamma=0.55)
    mask = (time > 9.55) & (time < 10.25)
    result = _result_from_residual(time, residual, mask, ferr_value=0.012)

    fit = PlanetAnomalyClassifier(
        PlanetClassConfig(
            min_delta_chi2_for_seed=5.0,
            enable_fold_caustic=False,
            enable_curved_fold_caustic=False,
            enable_grazing_fold_caustic=False,
            enable_two_fold_caustic=False,
            enable_cusp_tail=False,
            enable_canonical_cusp=False,
            enable_chang_refsdal=False,
        )
    ).fit(result)

    atoms = [atom for seg in fit.segment_results for atom in seg.atom_fits if atom.class_label == "limb_darkened_fold_caustic"]
    assert atoms
    assert atoms[0].success
    assert 0.0 <= atoms[0].params["Gamma"] <= 1.0
    assert any(seed.class_label == "limb_darkened_fold_caustic" for seed in fit.event_seeds)


def test_grazing_fold_atom_fits_limb_contact_seed():
    time = np.linspace(8.8, 10.8, 140)
    ta = 9.9
    width = 0.16
    residual = 0.15 * fold_g0(-0.75 + (time - ta) / width + 0.4 * ((time - ta) / width) ** 2)
    mask = (time > 9.45) & (time < 10.35)
    result = _result_from_residual(time, residual, mask, ferr_value=0.012)

    fit = PlanetAnomalyClassifier(
        PlanetClassConfig(
            min_delta_chi2_for_seed=5.0,
            enable_fold_caustic=False,
            enable_curved_fold_caustic=False,
            enable_limb_darkened_fold_caustic=False,
            enable_two_fold_caustic=False,
            enable_cusp_tail=False,
            enable_canonical_cusp=False,
            enable_chang_refsdal=False,
        )
    ).fit(result)

    atoms = [atom for seg in fit.segment_results for atom in seg.atom_fits if atom.class_label == "grazing_fold_caustic"]
    assert atoms
    assert atoms[0].success
    assert -1.5 <= atoms[0].params["z0"] <= 3.0
    assert any(seed.class_label == "grazing_fold_caustic" for seed in fit.event_seeds)


def test_two_fold_atom_fits_unresolved_pair_seed():
    time = np.linspace(8.8, 10.8, 150)
    tc1, tc2, tstar = 9.55, 10.15, 0.08
    residual = 0.12 * fold_g0((time - tc1) / tstar) + 0.10 * fold_g0(-(time - tc2) / tstar)
    mask = (time > 9.35) & (time < 10.35)
    result = _result_from_residual(time, residual, mask, ferr_value=0.012)

    fit = PlanetAnomalyClassifier(
        PlanetClassConfig(
            min_delta_chi2_for_seed=5.0,
            enable_fold_caustic=False,
            enable_curved_fold_caustic=False,
            enable_grazing_fold_caustic=False,
            enable_limb_darkened_fold_caustic=False,
            enable_cusp_tail=False,
            enable_canonical_cusp=False,
            enable_chang_refsdal=False,
        )
    ).fit(result)

    atoms = [atom for seg in fit.segment_results for atom in seg.atom_fits if atom.class_label == "two_fold_caustic"]
    assert atoms
    assert atoms[0].success
    assert atoms[0].params["tc2"] > atoms[0].params["tc1"]
    assert any(seed.class_label == "two_fold_caustic" for seed in fit.event_seeds)


def test_canonical_and_finite_source_cusp_atoms_fit_seed():
    time = np.linspace(9.0, 11.0, 80)
    residual = (
        0.16 / (1.0 + ((time - 9.7) / 0.08) ** 2)
        + 0.14 / (1.0 + ((time - 10.3) / 0.08) ** 2)
        + 0.03 / (1.0 + ((time - 10.0) / 0.35) ** 2)
    )
    mask = (time > 9.35) & (time < 10.65)
    result = _result_from_residual(time, residual, mask, ferr_value=0.012)

    fit = PlanetAnomalyClassifier(
        PlanetClassConfig(
            min_delta_chi2_for_seed=5.0,
            keep_top_atom_fits=20,
            enable_positive_bump=False,
            enable_negative_dip=False,
            enable_central_perturbation=False,
            enable_fold_caustic=False,
            enable_curved_fold_caustic=False,
            enable_grazing_fold_caustic=False,
            enable_limb_darkened_fold_caustic=False,
            enable_two_fold_caustic=False,
            enable_cusp_tail=False,
            enable_finite_source_cusp=True,
            enable_chang_refsdal=False,
            enable_second_pspl=False,
            enable_pspl_misfit=False,
        )
    ).fit(result)

    labels = {atom.class_label for seg in fit.segment_results for atom in seg.atom_fits if atom.success}
    assert "canonical_cusp" in labels
    assert "finite_source_cusp" in labels
    assert any(seed.source_atom == "canonical_cusp_map" for seed in fit.event_seeds)
    assert any(seed.source_atom == "finite_source_cusp_lookup" for seed in fit.event_seeds)


def test_chang_refsdal_atom_fits_local_image_perturbation_seed():
    time = np.linspace(8.0, 12.0, 120)
    residual = 0.22 / (1.0 + ((time - 9.35) / 0.13) ** 2)
    mask = np.abs(time - 9.35) < 0.55
    result = _result_from_residual(time, residual, mask, ferr_value=0.015)

    fit = PlanetAnomalyClassifier(
        PlanetClassConfig(
            min_delta_chi2_for_seed=5.0,
            enable_positive_bump=False,
            enable_negative_dip=False,
            enable_central_perturbation=False,
            enable_fold_caustic=False,
            enable_curved_fold_caustic=False,
            enable_grazing_fold_caustic=False,
            enable_limb_darkened_fold_caustic=False,
            enable_two_fold_caustic=False,
            enable_cusp_tail=False,
            enable_canonical_cusp=False,
            enable_chang_refsdal=True,
            enable_second_pspl=False,
            enable_pspl_misfit=False,
        )
    ).fit(result)

    atoms = [atom for seg in fit.segment_results for atom in seg.atom_fits if atom.class_label == "chang_refsdal"]
    assert atoms
    assert atoms[0].success
    assert any(seed.class_label == "chang_refsdal" for seed in fit.event_seeds)


def test_class_probabilities_use_pure_bic_weights():
    time = np.linspace(8.0, 12.0, 161)
    residual = 0.25 / (1.0 + ((time - 9.3) / 0.12) ** 2)
    mask = np.abs(time - 9.3) < 0.5
    result = _result_from_residual(time, residual, mask)

    fit = PlanetAnomalyClassifier(
        PlanetClassConfig(min_delta_chi2_for_seed=5.0)
    ).fit(result)
    segment = fit.segment_results[0]
    bic_min = min(atom.bic for atom in segment.atom_fits if np.isfinite(atom.bic))
    expected = {}
    for atom in segment.atom_fits:
        if np.isfinite(atom.bic):
            expected[atom.class_label] = expected.get(atom.class_label, 0.0) + np.exp(-0.5 * (atom.bic - bic_min))
    norm = sum(expected.values())
    expected = {key: value / norm for key, value in expected.items()}

    assert segment.class_probabilities == expected


def test_validity_penalty_marks_cadence_limited_width_without_changing_bic_probability():
    time = np.linspace(8.0, 12.0, 81)
    residual = np.zeros_like(time)
    residual[np.argmin(np.abs(time - 9.3))] = 0.4
    mask = np.abs(time - 9.3) < 0.2
    result = _result_from_residual(time, residual, mask, ferr_value=0.02)

    fit = PlanetAnomalyClassifier(
        PlanetClassConfig(
            min_delta_chi2_for_seed=1.0,
            cadence_width_penalty=50.0,
            enable_central_perturbation=False,
            enable_fold_caustic=False,
            enable_curved_fold_caustic=False,
            enable_grazing_fold_caustic=False,
            enable_limb_darkened_fold_caustic=False,
            enable_two_fold_caustic=False,
            enable_cusp_tail=False,
            enable_canonical_cusp=False,
            enable_chang_refsdal=False,
            enable_second_pspl=False,
            enable_pspl_misfit=False,
        )
    ).fit(result)

    atom = fit.segment_results[0].atom_fits[0]
    assert atom.validity_penalty > 0.0
    assert any("width is close to cadence" in warning for warning in atom.warnings)
    assert np.isclose(atom.score, atom.delta_chi2 - atom.n_params * np.log(atom.n_data) - atom.validity_penalty)
