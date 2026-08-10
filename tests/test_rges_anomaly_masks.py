from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_rges_anomaly_html import _canonical_scripts, _roman_payload
from tools.rges_anomaly_pipeline import (
    _ecliptic_to_icrf,
    _earth_position_at,
    _fit_exclusion_mask,
    _write_rges_space_ephemeris,
)
from jacscanomaly.parallax import load_vbm_satellite_file


def test_fit_exclusion_mask_is_empty_without_accepted_refit():
    result = SimpleNamespace(
        signal_mask=np.array([True, True, False]),
        point_weight=np.array([0.0, 0.0, 1.0]),
        iterations=(),
    )
    assert not _fit_exclusion_mask(result).any()


def test_fit_exclusion_mask_keeps_only_zero_weight_signal_points():
    result = SimpleNamespace(
        signal_mask=np.array([True, True, True, False]),
        point_weight=np.array([0.0, 1.0, 0.0, 1.0]),
        iterations=(object(),),
    )
    np.testing.assert_array_equal(
        _fit_exclusion_mask(result),
        np.array([True, False, True, False]),
    )


def test_rges_observer_positions_become_geocentric_space_ephemeris(tmp_path):
    times = np.array([2459000.0, 2459001.0, 2459002.0])
    earth = _earth_position_at(times)
    geocentric_icrf = np.array(
        [[0.012, -0.001, 0.0005], [0.0121, -0.0009, 0.0005], [0.0122, -0.0008, 0.0005]]
    )
    # RGES stores the complete observer orbit in ecliptic Cartesian AU.
    observer_ecliptic = _ecliptic_to_icrf(earth + geocentric_icrf, -1.0)
    path = _write_rges_space_ephemeris(
        tmp_path, "beginner", times, observer_ecliptic
    )
    table = load_vbm_satellite_file(str(path))
    assert table.shape == (3, 4)
    np.testing.assert_allclose(table[:, 3], np.linalg.norm(geocentric_icrf, axis=1), atol=2e-6)


def _row(series):
    return {
        "event": "RMDC26_TEST",
        "tier": "beginner",
        "score": 1.0,
        "stage": "scan",
        "payload": {
            "metadata": {"tier": "beginner", "event": "RMDC26_TEST"},
            "fit": {
                "refined": {
                    "model_kind": "pspl",
                    "params": {"t0": 10.0, "tE": 1.0, "u0": 1.0},
                }
            },
            "series": series,
            "features": {"items": []},
            "template_free": {"candidates": []},
            "physical_fallback": {"accepted": False, "result": None},
        },
    }


def test_html_never_falls_back_to_analysis_signal_mask_for_orange_points():
    series = {
        "n_total": 5,
        "time": [8.0, 9.0, 10.0, 11.0, 12.0],
        "flux": [1.0] * 5,
        "ferr": [0.1] * 5,
        "model_flux": [1.0] * 5,
        "residual": [0.0] * 5,
        # This is deliberately a broad analysis mask, with no display mask.
        "signal_mask": [1, 1, 1, 1, 1],
    }
    output = _roman_payload(_row(series))
    assert sum(output["series"]["display_signal_mask"]) == 0


def test_html_preserves_explicit_partial_display_mask():
    series = {
        "n_total": 5,
        "time": [8.0, 9.0, 10.0, 11.0, 12.0],
        "flux": [1.0] * 5,
        "ferr": [0.1] * 5,
        "model_flux": [1.0] * 5,
        "residual": [0.0] * 5,
        "signal_mask": [1, 1, 1, 1, 1],
        "display_signal_mask": [0, 1, 0, 0, 1],
        "fit_exclusion_mask": [0, 1, 0, 0, 1],
    }
    output = _roman_payload(_row(series))
    assert sum(output["series"]["display_signal_mask"]) == 2


def test_html_plot_scripts_use_display_mask_and_final_detection_marker():
    scripts = _canonical_scripts()
    assert "s.display_signal_mask" in scripts["EVENT_JS"]
    assert "d.series?.display_signal_mask" in scripts["EVENT_JS"]
    assert "s.display_signal_mask" in scripts["FEATURE_EVENT_JS"]
    assert "s.signal_mask || []" not in scripts["EVENT_JS"]
    assert "detectionT0" in scripts["EVENT_JS"]


def test_html_exposes_final_detection_with_legacy_candidate_fallback():
    series = {
        "n_total": 2,
        "time": [9.0, 10.0],
        "flux": [1.0, 1.0],
        "ferr": [0.1, 0.1],
        "model_flux": [1.0, 1.0],
        "residual": [0.0, 0.0],
        "display_signal_mask": [0, 1],
    }
    row = _row(series)
    row["payload"]["anomaly_candidates"] = [{"t_center": 9.5, "kind": "peak"}]
    output = _roman_payload(row)
    assert output["final_detection"] == {"detected": True, "t0": 9.5}


def test_html_uses_adaptive_model_curve_instead_of_observation_model_samples():
    series = {
        "n_total": 3,
        "time": [9.0, 10.0, 11.0],
        "flux": [1.0] * 3,
        "ferr": [0.1] * 3,
        "model_flux": [9.0, 9.0, 9.0],
        "residual": [0.0] * 3,
        "signal_mask": [0, 0, 0],
        "display_signal_mask": [0, 0, 0],
    }
    row = _row(series)
    row["payload"]["plot"] = {
        "model_curve": {"time": [9.5, 10.0, 10.5], "flux": [1.1, 1.2, 1.1]}
    }
    output = _roman_payload(row)
    assert output["plot"]["model_curve"] == {
        "time": [9.5, 10.0, 10.5],
        "flux": [1.1, 1.2, 1.1],
    }


def test_html_uses_unified_anomaly_candidates_as_canonical_list():
    series = {
        "n_total": 3,
        "time": [9.0, 10.0, 11.0],
        "flux": [1.0] * 3,
        "ferr": [0.1] * 3,
        "model_flux": [1.0] * 3,
        "residual": [0.0] * 3,
        "display_signal_mask": [0, 0, 0],
    }
    row = _row(series)
    unified = {
        "rank": 1,
        "kind": "peak",
        "t_center": 10.0,
        "t_start": 9.8,
        "t_end": 10.2,
        "timescale": 0.4,
        "max_abs_z": 12.0,
        "sources": ["final_residual_feature", "template_free"],
    }
    row["payload"]["anomaly_candidates"] = [unified]
    row["payload"]["template_free"]["candidates"] = [
        {"t_start": 1.0, "t_end": 2.0}
    ]

    output = _roman_payload(row)

    assert output["has_anomaly_candidate"] is True
    assert output["best_anomaly_candidate"] == unified
    assert output["anomaly_candidates"] == [unified]
    assert output["candidates"] == [unified]
