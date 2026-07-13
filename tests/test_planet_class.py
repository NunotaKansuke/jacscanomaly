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
    chang_refsdal_flux_residual,
    cr_relative_magnification,
    fold_g0,
    fold_g0_integral,
    fold_limb_darkened,
    r_major,
    r_minor,
    warm_chang_refsdal_lookup_cache,
)
from jacscanomaly.planet_class.atoms.chang_refsdal import pspl_image_position
from jacscanomaly.planet_class.atoms.fold import CurvedFoldCausticAtom
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


def test_chang_refsdal_lookup_has_caustic_structure_and_far_field_normalization():
    center = cr_relative_magnification(np.asarray([0.0]), np.asarray([0.0]), 0.2)[0]
    far = cr_relative_magnification(np.asarray([10.0]), np.asarray([0.0]), 0.2)[0]
    flank = cr_relative_magnification(np.asarray([1.2]), np.asarray([0.0]), 0.2)[0]

    assert center > 1.0
    assert np.isclose(far, 1.0)
    assert np.isfinite(flank)


def test_chang_refsdal_finite_source_lookup_and_cache_warmup():
    x = np.linspace(-1.5, 1.5, 21)
    y = np.zeros_like(x)
    point = cr_relative_magnification(x, y, 0.2, source_radius_hat=0.0, grid_size=96)
    finite = cr_relative_magnification(x, y, 0.2, source_radius_hat=0.4, grid_size=96)

    assert point.shape == finite.shape
    assert np.all(np.isfinite(finite))
    assert np.std(finite) < np.std(point)
    warm_chang_refsdal_lookup_cache(
        gamma_values=(0.0, 0.2),
        source_radius_hat_values=(0.0, 0.4),
        grid_size=64,
    )


def test_planet_anomaly_classifier_fits_positive_bump_without_assumed_planet_parameters():
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
    assert set(fit.best_atom.params) >= {"t_peak", "width"}
    assert "q" not in fit.best_atom.params
    assert "s" not in fit.best_atom.params


def test_planet_anomaly_classifier_fits_pspl_positive_bump_atom():
    time = np.linspace(8.0, 16.0, 240)
    u0 = 0.35
    tE = 1.2
    u = np.sqrt(u0 * u0 + ((time - 11.0) / tE) ** 2)
    residual = 0.04 * ((u * u + 2.0) / (u * np.sqrt(u * u + 4.0)) - 1.0)
    mask = np.abs(time - 11.0) < 2.8
    result = _result_from_residual(time, residual, mask, ferr_value=0.01)

    fit = PlanetAnomalyClassifier(
        PlanetClassConfig(
            min_delta_chi2_for_seed=5.0,
            enable_positive_bump=False,
            enable_second_pspl=False,
        )
    ).fit(result)

    atoms = [
        atom
        for segment in fit.segment_results
        for atom in segment.atom_fits
        if atom.class_label == "major_image_pspl_bump"
    ]
    assert atoms
    assert atoms[0].success
    assert abs(atoms[0].params["t_peak"] - 11.0) < 0.15


def test_planet_anomaly_classifier_fits_negative_dip_without_assumed_planet_parameters():
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
    assert "q" not in fit.best_atom.params
    assert "s" not in fit.best_atom.params


def test_planet_anomaly_classifier_fits_minor_image_box_trough_atom():
    time = np.linspace(8.0, 12.0, 220)
    edge = 0.06
    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
    residual = -0.18 * (sigmoid((time - 9.3) / edge) - sigmoid((time - 10.7) / edge))
    mask = (time > 9.0) & (time < 11.0)
    result = _result_from_residual(time, residual, mask, ferr_value=0.01)

    fit = PlanetAnomalyClassifier(
        PlanetClassConfig(
            min_delta_chi2_for_seed=5.0,
            enable_negative_dip=False,
        )
    ).fit(result)

    atoms = [
        atom
        for segment in fit.segment_results
        for atom in segment.atom_fits
        if atom.class_label == "minor_image_box_trough"
    ]
    assert atoms
    assert atoms[0].success
    assert abs(atoms[0].params["t_start"] - 9.3) < 0.1
    assert abs(atoms[0].params["t_end"] - 10.7) < 0.1


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
        if atom.class_label == "second_pspl_like"
    ]
    assert second
    assert second[0].success
    assert abs(second[0].params["t0_2"] - 20.0) < 0.2
    assert abs(second[0].params["tE_2"] - 4.0) < 0.2
    assert np.isfinite(second[0].params["q_flux"])
    assert np.isclose(second[0].params["tE_ratio"], second[0].params["tE_2"] / fit.pspl.tE)
    assert not {"q_wide_repeating", "s_plus", "s_minus", "alpha_plus", "alpha_minus"} & set(second[0].params)


def test_shear_quadrupole_atom_reports_only_fitted_diagnostics():
    time = np.linspace(0.0, 30.0, 300)
    tau = (time - 16.0) / 5.0
    residual = 0.08 * (tau * tau - np.mean(tau * tau)) / (1.0 + tau * tau) + 0.03 * tau / (1.0 + tau * tau)
    mask = np.abs(time - 16.0) < 8.0
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
            enable_rim_trough_caustic=False,
            enable_two_fold_caustic=False,
            enable_cusp_tail=False,
            enable_canonical_cusp=False,
            enable_chang_refsdal=False,
            enable_second_pspl=False,
            enable_pspl_misfit=False,
        )
    ).fit(result)

    atoms = [atom for seg in fit.segment_results for atom in seg.atom_fits if atom.class_label == "shear_quadrupole"]
    assert atoms
    assert atoms[0].success
    assert atoms[0].params["gamma"] > 0.0


def test_systematics_candidate_atom_fits_sparse_outliers_without_seeds():
    time = np.linspace(8.0, 12.0, 80)
    residual = np.zeros_like(time)
    residual[30] = 0.22
    residual[32] = -0.18
    mask = np.zeros_like(time, dtype=bool)
    mask[29:34] = True
    result = _result_from_residual(time, residual, mask, ferr_value=0.01)

    fit = PlanetAnomalyClassifier(
        PlanetClassConfig(
            min_delta_chi2_for_seed=5.0,
            enable_positive_bump=False,
            enable_pspl_positive_bump=False,
            enable_negative_dip=False,
            enable_minor_image_box_trough=False,
            enable_central_perturbation=False,
            enable_central_double_cusp=False,
            enable_fold_caustic=False,
            enable_curved_fold_caustic=False,
            enable_grazing_fold_caustic=False,
            enable_limb_darkened_fold_caustic=False,
            enable_rim_trough_caustic=False,
            enable_two_fold_caustic=False,
            enable_cusp_tail=False,
            enable_canonical_cusp=False,
            enable_chang_refsdal=False,
            enable_second_pspl=False,
            enable_shear_quadrupole=False,
            enable_pspl_misfit=False,
        )
    ).fit(result)

    atoms = [
        atom
        for segment in fit.segment_results
        for atom in segment.atom_fits
        if atom.class_label == "systematics_candidate"
    ]
    assert atoms
    assert atoms[0].success
    assert atoms[0].params["n_spikes"] >= 2.0


def test_planet_anomaly_classifier_reports_projected_central_constraint():
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
    local = [
        item.atom_fit
        for segment in fit.segment_results
        for item in segment.local_physical_fits
        if item.atom_fit.class_label == "central_caustic"
    ]
    assert local
    assert "chord_factor_times_q_over_s_minus_inv_s_sq" in local[0].physical_params
    assert "q" not in local[0].physical_params
    assert "s" not in local[0].physical_params
    assert any("unknown chord factor" in relation for relation in local[0].constraint_relations)


def test_planet_anomaly_classifier_fits_central_double_cusp_atom():
    time = np.linspace(8.0, 12.0, 260)
    residual = (
        0.20 / (1.0 + ((time - 9.75) / 0.08) ** 2)
        + 0.16 / (1.0 + ((time - 10.25) / 0.08) ** 2)
        - 0.12 / (1.0 + ((time - 10.0) / 0.20) ** 2)
    )
    mask = np.abs(time - 10.0) < 0.75
    result = _result_from_residual(time, residual, mask, ferr_value=0.01)

    fit = PlanetAnomalyClassifier(
        PlanetClassConfig(
            min_delta_chi2_for_seed=5.0,
            enable_central_perturbation=False,
            enable_rim_trough_caustic=False,
        )
    ).fit(result)

    atoms = [
        atom
        for segment in fit.segment_results
        for atom in segment.atom_fits
        if atom.class_label == "central_double_cusp"
    ]
    assert atoms
    assert atoms[0].success
    assert abs(atoms[0].params["t_center"] - 10.0) < 0.1
    assert atoms[0].params["cusp_separation"] > 0.2


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


def test_planet_anomaly_classifier_fits_fold_caustic_atom_and_local_constraint():
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
    assert np.isclose(fold[0].params["t_limb"], fold[0].params["tc"] - fold[0].params["entry_exit_sign"] * fold[0].params["tstar"])
    local_fold = [
        local
        for segment in fit.segment_results
        for local in segment.local_physical_fits
        if local.atom_fit.class_label == "fold_caustic" and local.atom_fit.physical_valid
    ]
    assert local_fold
    assert local_fold[0].locator_kind == "morphology_edge"
    assert abs(local_fold[0].atom_fit.params["tstar"] - tstar) < 0.04


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
    assert any(
        local.atom_fit.physical_valid
        for segment in fit.segment_results
        for local in segment.local_physical_fits
        if "edge" in local.locator_kind
    )


def test_curved_fold_caustic_reports_entry_exit_times_from_quadratic_roots():
    tc = 9.75
    tstar = 0.08
    q_curv = 0.2
    sign = 1.0

    limb = CurvedFoldCausticAtom._solve_time_roots(tc, tstar, q_curv, sign, z_value=-1.0)
    center = CurvedFoldCausticAtom._solve_time_roots(tc, tstar, q_curv, sign, z_value=0.0)

    assert len(limb) == 2
    assert len(center) == 2
    assert limb[0] < limb[1]
    assert center[0] < center[1]
    assert np.isclose(sign * (((limb[0] - tc) / tstar) + q_curv * ((limb[0] - tc) / tstar) ** 2), -1.0)
    assert np.isclose(sign * (((limb[1] - tc) / tstar) + q_curv * ((limb[1] - tc) / tstar) ** 2), -1.0)
    assert np.isclose(sign * (((center[0] - tc) / tstar) + q_curv * ((center[0] - tc) / tstar) ** 2), 0.0)
    assert np.isclose(sign * (((center[1] - tc) / tstar) + q_curv * ((center[1] - tc) / tstar) ** 2), 0.0)


def test_planet_anomaly_classifier_fits_cusp_tail_atom():
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


def test_limb_darkened_fold_atom_fits_gamma():
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


def test_grazing_fold_atom_fits_limb_contact_constraint():
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
    assert np.isclose(atoms[0].params["a1"], 1.0 / atoms[0].params["width"])
    if "t_contact_1" in atoms[0].params:
        x = (atoms[0].params["t_contact_1"] - atoms[0].params["ta"]) / atoms[0].params["width"]
        z = atoms[0].params["z0"] + x + atoms[0].params["q_curv"] * x * x
        assert np.isclose(z, -1.0)
        assert atoms[0].params["rho_over_sinalpha_contact_1"] > 0.0
    assert any(
        local.atom_fit.class_label == "grazing_fold_caustic"
        for segment in fit.segment_results
        for local in segment.local_physical_fits
    )


def test_two_fold_atom_fits_unresolved_pair():
    time = np.linspace(8.8, 10.8, 150)
    tc1, tc2, tstar = 9.55, 10.15, 0.08
    residual = 0.12 * fold_g0((time - tc1) / tstar) + 0.10 * fold_g0(-(time - tc2) / tstar)
    mask = (time > 9.35) & (time < 10.35)
    result = _result_from_residual(time, residual, mask, ferr_value=0.012)

    fit = PlanetAnomalyClassifier(
        PlanetClassConfig(
            min_delta_chi2_for_seed=5.0,
            enable_fold_caustic=True,
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
    assert atoms[0].atom_name == "two_fold_caustic"
    assert atoms[0].params["tc2"] > atoms[0].params["tc1"]
    assert atoms[0].params["fold_ratio"] > 0.0
    assert np.isclose(atoms[0].params["contact_separation_over_2tstar"], (atoms[0].params["tc2"] - atoms[0].params["tc1"]) / (2.0 * atoms[0].params["tstar"]))
    assert "amplitude_2" not in atoms[0].params
    constraints = fit.physical_constraint_dicts()
    assert {row["locator_kind"] for row in constraints} >= {"entry_edge", "exit_edge"}
    assert fit.physical_relation_dicts()


def test_full_caustic_crossing_atom_fits_entry_exit_transit():
    time = np.linspace(8.0, 13.0, 260)
    t_entry, t_exit = 9.2, 11.7
    tstar_entry, tstar_exit = 0.07, 0.10
    center = 0.5 * (t_entry + t_exit)
    half = 0.5 * (t_exit - t_entry)
    tau = (time - center) / half
    window = 1.0 / (1.0 + np.exp(-(time - t_entry) / 0.08)) / (1.0 + np.exp(-(t_exit - time) / 0.08))
    residual = (
        0.18 * fold_g0((time - t_entry) / tstar_entry)
        + 0.14 * fold_g0(-(time - t_exit) / tstar_exit)
        + window * (0.05 - 0.03 * tau + 0.02 * (tau * tau - 1.0 / 3.0))
    )
    mask = (time > 8.8) & (time < 12.1)
    result = _result_from_residual(time, residual, mask, ferr_value=0.01)

    fit = PlanetAnomalyClassifier(
        PlanetClassConfig(
            min_delta_chi2_for_seed=5.0,
            enable_positive_bump=False,
            enable_pspl_positive_bump=False,
            enable_negative_dip=False,
            enable_minor_image_box_trough=False,
            enable_central_perturbation=False,
            enable_central_double_cusp=False,
            enable_fold_caustic=True,
            enable_curved_fold_caustic=False,
            enable_grazing_fold_caustic=False,
            enable_limb_darkened_fold_caustic=False,
            enable_rim_trough_caustic=False,
            enable_two_fold_caustic=False,
            enable_cusp_tail=False,
            enable_canonical_cusp=False,
            enable_chang_refsdal=False,
            enable_second_pspl=False,
            enable_shear_quadrupole=False,
            enable_systematics_diagnostic=False,
            enable_pspl_misfit=False,
            local_physical_max_windows=2,
        )
    ).fit(result)

    atoms = [atom for seg in fit.segment_results for atom in seg.atom_fits if atom.class_label == "full_caustic_crossing"]
    assert atoms
    assert atoms[0].success
    assert atoms[0].estimation_role == "morphology"
    assert not atoms[0].physical_params
    assert "rho_over_sinalpha" not in atoms[0].params
    assert atoms[0].params["t_entry"] < atoms[0].params["t_exit"]
    assert atoms[0].params["caustic_inside_duration"] > 0.5
    local = [
        item
        for segment in fit.segment_results
        for item in segment.local_physical_fits
        if item.atom_fit.class_label == "fold_caustic" and item.atom_fit.physical_valid
    ]
    assert {item.locator_kind for item in local} == {"entry_edge", "exit_edge"}
    relations = fit.physical_relation_dicts()
    assert len(relations) == 1
    assert abs(relations[0]["t_entry"] - t_entry) < 0.15
    assert abs(relations[0]["t_exit"] - t_exit) < 0.15


def test_rim_trough_atom_fits_bump_dip_bump():
    time = np.linspace(8.8, 10.8, 180)
    left, trough, right = 9.65, 10.0, 10.32
    residual = (
        0.11 / (1.0 + ((time - left) / 0.08) ** 2)
        - 0.22 / (1.0 + ((time - trough) / 0.16) ** 2)
        + 0.15 / (1.0 + ((time - right) / 0.08) ** 2)
    )
    mask = (time > 9.35) & (time < 10.55)
    result = _result_from_residual(time, residual, mask, ferr_value=0.012)

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
            enable_chang_refsdal=False,
            enable_second_pspl=False,
            enable_pspl_misfit=False,
        )
    ).fit(result)

    atoms = [atom for seg in fit.segment_results for atom in seg.atom_fits if atom.class_label == "rim_trough_caustic"]
    assert atoms
    assert atoms[0].success
    assert atoms[0].params["tc1"] < atoms[0].params["t_trough"] < atoms[0].params["tc2"]
    assert atoms[0].params["rim_ratio"] > 0.0
    assert atoms[0].params["trough_ratio"] > 0.0
    assert atoms[0].params["polarity"] == 1.0


def test_canonical_and_finite_source_cusp_atoms_fit():
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
    cusp_atoms = [atom for seg in fit.segment_results for atom in seg.atom_fits if atom.class_label in {"canonical_cusp", "finite_source_cusp"}]
    assert all("t_cusp_closest" in atom.params and "cusp_impact" in atom.params for atom in cusp_atoms)
    finite = next(atom for atom in cusp_atoms if atom.class_label == "finite_source_cusp")
    assert np.isclose(finite.params["rho_over_sinalpha_cusp_local"], finite.params["width"] / fit.pspl.tE)


def test_chang_refsdal_atom_does_not_turn_an_arbitrary_bump_into_a_physical_constraint():
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
    assert not atoms[0].physical_valid
    assert "n_event_seeds" not in fit.summary_dict()
    assert not fit.physical_constraint_dicts()


def test_physical_chang_refsdal_fit_recovers_local_planet_scale_without_free_amplitude():
    time = np.linspace(8.8, 10.2, 180)
    pspl = PSPLParams(t0=10.0, tE=3.0, u0=0.2, Fs=2.0, Fb=1.0)
    ta = 9.45
    xp, yp = pspl_image_position(np.asarray([ta]), pspl, branch="major")
    q_true = 8e-4
    residual = chang_refsdal_flux_residual(
        time,
        pspl,
        x_planet=float(xp[0]),
        y_planet=float(yp[0]),
        q=q_true,
        rho_over_sqrt_q=0.1,
        branch="major",
        grid_size=96,
    )
    mask = np.abs(time - ta) < 0.45
    result = _result_from_residual(time, residual, mask, ferr_value=0.01)
    config = PlanetClassConfig(
        min_delta_chi2_for_seed=1.0,
        cr_lookup_grid_size=96,
        cr_lookup_source_radius_grid=(0.1,),
        cr_lookup_sqrt_q_factors=(0.1, 0.3, 0.6, 1.0, 2.0, 4.0),
        enable_positive_bump=False,
        enable_pspl_positive_bump=False,
        enable_negative_dip=False,
        enable_minor_image_box_trough=False,
        enable_central_perturbation=False,
        enable_central_double_cusp=False,
        enable_fold_caustic=False,
        enable_curved_fold_caustic=False,
        enable_full_caustic_crossing=False,
        enable_cusp_tail=False,
        enable_grazing_fold_caustic=False,
        enable_two_fold_caustic=False,
        enable_rim_trough_caustic=False,
        enable_limb_darkened_fold_caustic=False,
        enable_canonical_cusp=False,
        enable_second_pspl=False,
        enable_shear_quadrupole=False,
        enable_systematics_diagnostic=False,
        enable_pspl_misfit=False,
    )
    fit = PlanetAnomalyClassifier(config).fit(result)
    local = next(
        item
        for segment in fit.segment_results
        for item in segment.local_physical_fits
        if item.atom_fit.class_label == "chang_refsdal"
    )
    atom = local.atom_fit

    assert atom.estimation_role == "physical_local"
    assert local.locator_kind in {"positive_peak", "negative_dip"}
    assert atom.physical_valid
    assert "amplitude" not in atom.params
    assert np.isclose(atom.params["q"], atom.params["sqrt_q"] ** 2)
    assert np.isclose(atom.params["s"], np.hypot(atom.params["x_planet"], atom.params["y_planet"]))
    assert 0.2 < atom.params["q"] / q_true < 5.0
    assert abs(atom.params["s"] - np.hypot(xp[0], yp[0])) < 0.15
    assert np.isclose(atom.params["rho"], atom.params["rho_over_sqrt_q"] * np.sqrt(atom.params["q"]))
    constraints = fit.physical_constraint_dicts()
    assert len(constraints) == 1
    assert constraints[0]["class_label"] == "chang_refsdal"


def test_central_duration_retains_unknown_chord_factor_in_identifiable_combination():
    time = np.linspace(8.5, 11.5, 121)
    residual = 0.2 / (1.0 + ((time - 10.0) / 0.12) ** 2)
    result = _result_from_residual(time, residual, np.abs(time - 10.0) < 0.5)
    fit = PlanetAnomalyClassifier(
        PlanetClassConfig(
            enable_chang_refsdal=False,
            keep_top_atom_fits=30,
        )
    ).fit(result)
    central = next(
        atom
        for segment in fit.segment_results
        for atom in segment.atom_fits
        if atom.class_label == "central_caustic"
    )

    assert central.estimation_role == "physical_constraint"
    combo = central.physical_params["chord_factor_times_q_over_s_minus_inv_s_sq"]
    assert np.isclose(combo, central.params["central_projected_width_over_tE"] / 4.0)
    assert "q" not in central.physical_params
    assert "s" not in central.physical_params
    assert any("C_chord" in relation for relation in central.constraint_relations)


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


def test_planet_anomaly_fit_result_summary_helpers_and_signal_convenience_api():
    time = np.linspace(8.0, 12.0, 161)
    residual = 0.25 / (1.0 + ((time - 9.3) / 0.12) ** 2)
    mask = np.abs(time - 9.3) < 0.5
    result = _result_from_residual(time, residual, mask)

    fit = result.classify_anomaly(
        PlanetClassConfig(
            min_delta_chi2_for_seed=5.0,
            enable_fold_caustic=False,
            enable_curved_fold_caustic=False,
            enable_grazing_fold_caustic=False,
            enable_limb_darkened_fold_caustic=False,
            enable_two_fold_caustic=False,
            enable_cusp_tail=False,
            enable_canonical_cusp=False,
            enable_chang_refsdal=False,
        )
    )

    summary = fit.summary_dict()
    assert summary["best_label"] == fit.best_label
    assert summary["n_segments"] == 1
    assert "best_bic" in summary
    assert "best_delta_chi2" in summary

    segment_rows = fit.segment_summary_dicts()
    atom_rows = fit.atom_summary_dicts()
    assert len(segment_rows) == 1
    assert atom_rows
    assert "n_event_seeds" not in summary
    assert "n_seeds" not in segment_rows[0]
    assert "best_atom:" in fit.summary_text()

    segment_table = fit.summary_table()
    atom_table = fit.atom_table()
    assert len(segment_table) == 1
    assert len(atom_table) == len(atom_rows)

    fig, ax = fit.plot_summary(signal_result=result, show=False)
    assert fig is not None
    assert ax.get_title().startswith("Planet anomaly classification")
