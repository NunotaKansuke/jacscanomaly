import numpy as np

from jacscanomaly import (
    FlatBaselineDiagnostic,
    PlanetAnomalyClassifier,
    PlanetClassConfig,
    PlanetSignalCandidate,
    PlanetSignalResult,
)
from jacscanomaly.planet_class import (
    anomaly_geometry,
    fold_g0,
    fold_g0_integral,
    fold_limb_darkened,
    q_from_bump,
    q_from_dip,
    r_major,
    r_minor,
)
from jacscanomaly.planet_class.geometry import (
    BRANCH_MAJOR,
    BRANCH_MINOR,
    REGIME_CENTRAL_RESONANT,
    REGIME_PLANETARY,
)
from jacscanomaly.planet_class.templates import pspl_bump_fwhm
from jacscanomaly.singlelens_fit import SingleLensFitResult
from jacscanomaly.singlelens_model import A_pspl_func


PSPL_T0 = 10.0
PSPL_TE = 3.0
PSPL_U0 = 0.2


def _flat_diagnostic():
    return FlatBaselineDiagnostic(
        use_flat_baseline=False,
        peak_in_mask=False,
        u0=PSPL_U0,
        n_peak_support=0,
        n_unmasked_peak_support=0,
        masked_peak_fraction=0.0,
        dchi2_flat_minus_pspl=0.0,
        improvement_n_eff=0.0,
        improvement_peak_frac=0.0,
    )


def _result_from_residual(time, residual, mask, *, ferr_value=0.02):
    ferr = np.full_like(time, ferr_value)
    params = np.array([PSPL_T0, PSPL_TE, PSPL_U0])
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


def _noise(time, ferr_value, scale=0.3, seed=42):
    rng = np.random.default_rng(seed)
    return scale * ferr_value * rng.standard_normal(time.size)


# ---------------------------------------------------------------------------
# Deterministic geometry


def test_pspl_image_radii_are_reciprocal():
    u = np.logspace(-3, 2, 100)
    assert np.allclose(r_major(u) * r_minor(u), 1.0)


def test_anomaly_geometry_matches_hand_computation():
    g = anomaly_geometry(11.5, t0=PSPL_T0, tE=PSPL_TE, u0=PSPL_U0)
    tau = 1.5 / 3.0
    u_anom = np.hypot(tau, PSPL_U0)
    assert np.isclose(g.tau_anom, tau)
    assert np.isclose(g.u_anom, u_anom)
    assert np.isclose(g.alpha, np.arctan2(PSPL_U0, tau))
    assert np.isclose(g.sin_alpha, PSPL_U0 / u_anom)
    assert np.isclose(g.s_dagger_plus, 0.5 * (np.sqrt(u_anom**2 + 4.0) + u_anom))
    assert g.regime == REGIME_PLANETARY


def test_anomaly_geometry_s_dagger_identities():
    for t_anom in (8.0, 10.7, 13.0, 25.0):
        g = anomaly_geometry(t_anom, t0=PSPL_T0, tE=PSPL_TE, u0=PSPL_U0)
        assert np.isclose(g.s_dagger_plus * g.s_dagger_minus, 1.0)
        assert np.isclose(g.s_dagger_plus - g.s_dagger_minus, g.u_anom)


def test_anomaly_geometry_round_trip_recovers_planet_separation():
    # Place the planet at the major-image position: u_anom = s - 1/s.
    s = 1.4
    u_anom = s - 1.0 / s
    tau = np.sqrt(u_anom**2 - PSPL_U0**2)
    t_anom = PSPL_T0 + PSPL_TE * tau
    g = anomaly_geometry(t_anom, t0=PSPL_T0, tE=PSPL_TE, u0=PSPL_U0)
    assert np.isclose(g.s_dagger_plus, s)
    # The minor-image branch of the same geometry is the reciprocal solution.
    assert np.isclose(g.s_dagger_minus, 1.0 / s)


def test_anomaly_geometry_flags_central_regime_near_peak():
    g = anomaly_geometry(PSPL_T0 + 0.01, t0=PSPL_T0, tE=PSPL_TE, u0=0.05)
    assert g.regime == REGIME_CENTRAL_RESONANT
    assert np.isclose(g.s_dagger_plus, 1.0, atol=0.05)


def test_anomaly_geometry_propagates_t_anom_error():
    g = anomaly_geometry(11.5, t0=PSPL_T0, tE=PSPL_TE, u0=PSPL_U0, t_anom_err=0.06)
    assert np.isclose(g.tau_anom_err, 0.02)
    step = 1e-6
    gp = anomaly_geometry(11.5 + step, t0=PSPL_T0, tE=PSPL_TE, u0=PSPL_U0)
    numeric = abs(gp.u_anom - g.u_anom) / step * 0.06
    assert np.isclose(g.u_anom_err, numeric, rtol=1e-3)


def test_q_from_dip_matches_reference_forms():
    g = anomaly_geometry(11.5, t0=PSPL_T0, tE=PSPL_TE, u0=PSPL_U0)
    dt_dip = 0.3
    q, _ = q_from_dip(dt_dip, tE=PSPL_TE, geometry=g)
    expected = (
        (dt_dip / (4.0 * PSPL_TE)) ** 2 * (g.s_dagger_minus / g.u_anom) * g.sin_alpha**2
    )
    assert np.isclose(q, expected)
    # Equivalent published form: (dt/4tE)^2 (s_dagger/|u0|) |sin^3 alpha|.
    published = (
        (dt_dip / (4.0 * PSPL_TE)) ** 2
        * (g.s_dagger_minus / PSPL_U0)
        * abs(np.sin(g.alpha)) ** 3
    )
    assert np.isclose(q, published)


def test_q_from_bump_is_planet_einstein_timescale_squared():
    q, q_err = q_from_bump(0.3, tE=PSPL_TE, t_p_err=0.03)
    assert np.isclose(q, 0.01)
    assert np.isclose(q_err, 2.0 * 0.01 * 0.1)


# ---------------------------------------------------------------------------
# Fold kernel


def test_fold_kernel_support_and_positive_tail():
    z = np.array([-2.0, -1.0, 0.0, 1.0, 10.0, 1e5])
    g = fold_g0(z)
    assert g[0] == 0.0
    assert g[1] == 0.0
    assert np.all(g[2:] > 0.0)
    # z^-1/2 asymptote in the far tail.
    assert np.isclose(g[5] / g[4], np.sqrt(10.0 / 1e5), rtol=0.05)
    direct = fold_g0_integral(z)
    assert np.allclose(g, direct, rtol=1e-3, atol=1e-9)
    ld = fold_limb_darkened(z, gamma=0.5)
    assert np.all(np.isfinite(ld))


# ---------------------------------------------------------------------------
# Template shape measurement and the full estimator


def _bump_profile(time, t_c, t_p, u_p):
    u = np.sqrt(u_p**2 + ((time - t_c) / t_p) ** 2)
    return (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0)) - 1.0


def test_classifier_measures_injected_bump_and_derives_major_image_geometry():
    time = np.linspace(8.0, 14.0, 241)
    t_c, t_p, u_p, amp = 11.5, 0.25, 0.3, 0.25
    residual = amp * _bump_profile(time, t_c, t_p, u_p) + _noise(time, 0.02)
    mask = np.abs(time - t_c) < 1.0
    result = _result_from_residual(time, residual, mask)

    fit = PlanetAnomalyClassifier(PlanetClassConfig()).fit(result)
    assert fit.best_shape == "bump"
    best = fit.best_component
    assert best is not None and best.best_fit is not None
    assert abs(best.best_fit.params["t_anom"] - t_c) < 0.05
    assert np.isclose(best.best_fit.params["t_p"], t_p, rtol=0.3)

    g = best.geometry
    assert g is not None
    assert g.preferred_branch == BRANCH_MAJOR
    expected = anomaly_geometry(
        best.best_fit.params["t_anom"], t0=PSPL_T0, tE=PSPL_TE, u0=PSPL_U0
    )
    assert np.isclose(g.s_dagger_plus, expected.s_dagger_plus)

    scales = best.scales
    assert scales is not None
    assert scales.q_method == "bump_planet_einstein_crossing"
    assert np.isclose(scales.q, (t_p / PSPL_TE) ** 2, rtol=0.6)
    assert np.isclose(scales.dt_over_tE, pspl_bump_fwhm(t_p, u_p) / PSPL_TE, rtol=0.3)


def test_classifier_measures_injected_dip_and_estimates_q():
    time = np.linspace(8.0, 14.0, 241)
    t_c, dt_dip, edge, depth = 11.5, 0.5, 0.06, 0.25

    def sig(x):
        return 1.0 / (1.0 + np.exp(-x))

    box = sig((time - (t_c - 0.5 * dt_dip)) / edge) * sig(((t_c + 0.5 * dt_dip) - time) / edge)
    residual = -depth * box + _noise(time, 0.02)
    mask = np.abs(time - t_c) < 1.0
    result = _result_from_residual(time, residual, mask)

    fit = PlanetAnomalyClassifier(PlanetClassConfig()).fit(result)
    assert fit.best_shape == "dip"
    best = fit.best_component
    assert abs(best.best_fit.params["t_anom"] - t_c) < 0.05
    assert np.isclose(best.best_fit.params["dt_dip"], dt_dip, rtol=0.3)

    g = best.geometry
    assert g.preferred_branch == BRANCH_MINOR
    assert g.regime == REGIME_PLANETARY

    scales = best.scales
    assert scales.q_method == "dip_han2006"
    q_expected, _ = q_from_dip(dt_dip, tE=PSPL_TE, geometry=g)
    assert np.isclose(scales.q, q_expected, rtol=0.7)


def test_classifier_measures_injected_caustic_crossing_edges():
    time = np.linspace(8.0, 14.0, 481)
    t_entry, t_exit, tstar = 11.0, 12.2, 0.08
    residual = (
        0.30 * fold_g0((time - t_entry) / tstar)
        + 0.30 * fold_g0(-(time - t_exit) / tstar)
        + 0.10 * ((time > t_entry) & (time < t_exit))
        + _noise(time, 0.02)
    )
    mask = (time > t_entry - 0.6) & (time < t_exit + 0.6)
    result = _result_from_residual(time, residual, mask)

    fit = PlanetAnomalyClassifier(PlanetClassConfig()).fit(result)
    assert fit.best_shape == "caustic_crossing"
    best = fit.best_component
    params = best.best_fit.params
    assert abs(params["t_entry"] - t_entry) < 0.1
    assert abs(params["t_exit"] - t_exit) < 0.1
    assert np.isclose(params["dt_cc"], t_exit - t_entry, rtol=0.15)

    scales = best.scales
    assert np.isclose(scales.dt_over_tE, (t_exit - t_entry) / PSPL_TE, rtol=0.15)
    assert np.isclose(scales.tstar_entry_over_tE, tstar / PSPL_TE, rtol=0.5)
    assert np.isclose(scales.tstar_exit_over_tE, tstar / PSPL_TE, rtol=0.5)
    # q must not be reported for a crossing: local data do not determine it.
    assert not np.isfinite(scales.q)


def test_classifier_reports_no_coherent_shape_for_pure_noise():
    time = np.linspace(8.0, 14.0, 241)
    residual = _noise(time, 0.02, scale=1.0)
    mask = np.abs(time - 11.5) < 1.0
    result = _result_from_residual(time, residual, mask)

    fit = PlanetAnomalyClassifier(PlanetClassConfig()).fit(result)
    assert fit.best_shape == "none"
    assert fit.best_component is None
    for component in fit.components:
        assert component.shape in {"no_coherent_shape", "low_significance", "insufficient_data"}
        assert component.geometry is None
        assert component.scales is None


def test_summary_helpers_and_plot_run():
    time = np.linspace(8.0, 14.0, 241)
    t_c, t_p, u_p = 11.5, 0.25, 0.3
    residual = 0.25 * _bump_profile(time, t_c, t_p, u_p) + _noise(time, 0.02)
    mask = np.abs(time - t_c) < 1.0
    result = _result_from_residual(time, residual, mask)

    fit = result.classify_anomaly()
    row = fit.summary_dict()
    assert row["best_shape"] == "bump"
    assert "u_anom" in row and "s_dagger_plus" in row
    assert len(fit.component_summary_dicts()) == len(fit.components)
    shape_rows = fit.shape_fit_dicts()
    assert any(r["shape"] == "null" for r in shape_rows)
    assert fit.summary_text()
    fig, _ax = fit.plot_summary(signal_result=result, show=False)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_insufficient_data_component_is_flagged():
    time = np.linspace(8.0, 14.0, 61)
    residual = np.zeros_like(time)
    residual[30] = 0.4
    mask = np.zeros(time.size, dtype=bool)
    mask[29:32] = True
    result = _result_from_residual(time, residual, mask)

    fit = PlanetAnomalyClassifier(PlanetClassConfig()).fit(result)
    assert fit.best_shape == "none"
    assert all(
        component.shape in {"insufficient_data", "no_coherent_shape", "low_significance"}
        for component in fit.components
    )
