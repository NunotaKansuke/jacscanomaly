#!/usr/bin/env python3
"""Run the two curated FSPL/parallax fallback regressions.

The benchmark intentionally obtains its seeds from the PSPL fit and detector
only.  The truth values in the instruction note are not used during fitting;
the JSON artifact is a reproducible acceptance record for the implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from jacscanomaly import (
    ContaminationConfig,
    FallbackConfig,
    Finder,
    FinderConfig,
    RoutingThresholds,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = PROJECT_ROOT.parent / "sample_rtmodel_v2.4"
SATELLITE = SAMPLE_ROOT / "satellitedir" / "satellite1.txt"
EVENTS = {
    "2_755_3280": {
        "data": SAMPLE_ROOT / "event_2_755_3280" / "Data" / "RomanW146sat1.dat",
        "ra_deg": 267.765654,
        "dec_deg": -28.807404,
        "effect": "fspl",
    },
    "0_599_2302": {
        "data": SAMPLE_ROOT / "event_0_599_2302" / "Data" / "RomanW146sat1.dat",
        "ra_deg": 268.17211,
        "dec_deg": -29.827315,
        "effect": "space_parallax",
    },
}
SCHEMA_VERSION = 2
REQUIRED_EVENT_FIELDS = (
    "event_id",
    "metadata",
    "baseline",
    "candidates",
    "selected_candidate_decision",
    "fallback",
    "runtime_seconds",
)


def _load_curve(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.genfromtxt(path)
    time = np.asarray(raw[:, 2], dtype=float)
    flux = 10.0 ** (-np.asarray(raw[:, 0], dtype=float) / 2.5)
    ferr = flux * np.log(10.0) / 2.5 * np.asarray(raw[:, 1], dtype=float)
    return time, flux, np.maximum(ferr, 1.0e-12)


def _json_safe(value):
    """Convert numpy values nested in detector diagnostics to JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _fit_summary(fit) -> dict[str, object]:
    return {
        "params": np.asarray(fit.params, dtype=float).tolist(),
        "raw_params": None if getattr(fit, "raw_params", None) is None else np.asarray(fit.raw_params, dtype=float).tolist(),
        "chi2_original_ferr": float(np.asarray(fit.chi2)),
        "optimizer_success": None
        if getattr(fit, "optimizer_success", None) is None
        else bool(getattr(fit, "optimizer_success")),
    }


def _candidate_summary(candidate) -> dict[str, object]:
    return _json_safe(candidate.summary_dict())


def _attempt_summary(attempt) -> dict[str, object]:
    last_iteration = attempt.result.iterations[-1] if attempt.result.iterations else None
    return {
        "seed": np.asarray(attempt.seed, dtype=float).tolist(),
        "original_chi2": float(attempt.original_chi2),
        "robust_objective": float(attempt.robust_objective),
        "contamination_penalty": float(attempt.contamination_penalty),
        "parameter_distance": float(attempt.parameter_distance),
        "optimizer_success": bool(attempt.optimizer_success),
        "stable": bool(attempt.stable),
        "alternating_converged": bool(attempt.result.converged),
        "segmentation_stable": bool(attempt.result.segmentation_stable),
        "parameter_at_bound": bool(attempt.parameter_at_bound),
        "n_alternating_iterations": len(attempt.result.iterations),
        "last_parameter_change": None
        if last_iteration is None
        else float(last_iteration.parameter_change),
        "last_weight_change": None
        if last_iteration is None
        else float(last_iteration.weight_change),
        "segmentation": {
            "anomaly_fraction": float(attempt.result.segmentation.anomaly_fraction),
            "anomaly_span_fraction": float(attempt.result.segmentation.anomaly_span_fraction),
            "protected_fraction": float(attempt.result.segmentation.protected_fraction),
            "protected_anomaly_fraction": float(attempt.result.segmentation.protected_anomaly_fraction),
            "protected_component_anomaly_fractions": list(
                attempt.result.segmentation.protected_component_anomaly_fractions
            ),
            "protected_component_retained_fractions": list(
                attempt.result.segmentation.protected_component_retained_fractions
            ),
            "diagnostics": list(attempt.result.segmentation.diagnostics),
        },
    }


def run_event(event_id: str, max_seeds: int, max_iter: int) -> dict[str, object]:
    spec = EVENTS[event_id]
    started = perf_counter()
    time, flux, ferr = _load_curve(spec["data"])
    baseline_finder = Finder(
        FinderConfig(
            fitter_kind="pspl",
            single_fit_backend="cpp",
            auto_init_max_clusters=1,
            auto_init_tE_grid_n=3,
        )
    )
    baseline = baseline_finder.fit_single_lens(time, flux, ferr, None)
    detector_finder = baseline_finder
    detector_kwargs: dict[str, object] = {"include_fspl": spec["effect"] == "fspl"}
    if spec["effect"] == "space_parallax":
        detector_finder = Finder(
            FinderConfig(
                fitter_kind="pspl",
                single_fit_backend="cpp",
                ra_deg=spec["ra_deg"],
                dec_deg=spec["dec_deg"],
                tref=float(np.median(time)),
                satellite_ephemeris_path=str(SATELLITE),
                parallax_observer_convention="gulls",
                parallax_time_scale="hjd",
                parallax_time_offset=2450000.0,
            )
        )
    candidates = detector_finder.detect_effects(
        baseline,
        routing_thresholds=RoutingThresholds(exact_probe_available=True),
        execute_exact_probe=True,
        **detector_kwargs,
    )
    selected = [candidate for candidate in candidates if candidate.effect == spec["effect"]]
    fallback = None
    if selected and selected[0].decision == "fallback":
        fallback = detector_finder.robust_fallback(
            time,
            flux,
            ferr,
            fit=baseline,
            candidates=selected,
            effect=spec["effect"],
            config=FallbackConfig(
                max_seeds=max_seeds,
                contamination=ContaminationConfig(max_iter=max_iter),
            ),
        )
    result: dict[str, object] = {
        "event_id": event_id,
        "data_path": str(spec["data"]),
        "effect": spec["effect"],
        "metadata": {
            "ra_deg": float(spec["ra_deg"]),
            "dec_deg": float(spec["dec_deg"]),
            "observer_ephemeris": None
            if spec["effect"] == "fspl"
            else str(SATELLITE),
            "seed_policy": "baseline_pspl_fit_plus_detector_candidates",
        },
        "n_points": int(time.size),
        "runtime_seconds": float(perf_counter() - started),
        "baseline": _fit_summary(baseline),
        "candidates": [_candidate_summary(candidate) for candidate in candidates],
        "selected_candidate_decision": None if not selected else selected[0].decision,
        "fallback": None,
    }
    if fallback is not None:
        selected_attempt = next(
            (
                attempt
                for attempt in fallback.attempts
                if attempt.result.fit is fallback.fit
            ),
            None,
        )
        result["fallback"] = {
            "effect": fallback.effect,
            "success": bool(fallback.success),
            "reason_codes": list(fallback.reason_codes),
            "baseline_original_chi2": float(fallback.baseline_original_chi2),
            "selected_original_chi2": float(fallback.selected_original_chi2),
            "selected_robust_objective": float(fallback.selected_robust_objective),
            "baseline_effect_score": fallback.baseline_effect_score,
            "selected_effect_score": fallback.selected_effect_score,
            "convergence": {
                "alternating": bool(
                    selected_attempt is not None
                    and selected_attempt.result.converged
                ),
                "segmentation_stable": bool(
                    selected_attempt is not None
                    and selected_attempt.result.segmentation_stable
                ),
            },
            "selected_seed": np.asarray(fallback.selected_seed, dtype=float).tolist(),
            "fit": _fit_summary(fallback.fit),
            "model_spec": fallback.model_spec,
            "attempts": [_attempt_summary(attempt) for attempt in fallback.attempts],
        }
    return result


def _validate_event_schema(result: dict[str, object]) -> None:
    """Fail early if a benchmark artifact loses a required diagnostic field."""
    missing = [field for field in REQUIRED_EVENT_FIELDS if field not in result]
    if missing:
        raise ValueError(f"Benchmark event is missing schema fields: {missing}")
    if result["fallback"] is not None:
        fallback = result["fallback"]
        for field in (
            "reason_codes",
            "attempts",
            "selected_original_chi2",
            "selected_robust_objective",
            "baseline_effect_score",
            "selected_effect_score",
            "fit",
        ):
            if field not in fallback:
                raise ValueError(f"Benchmark fallback is missing schema field: {field}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", nargs="+", choices=tuple(EVENTS), default=list(EVENTS))
    parser.add_argument("--max-seeds", type=int, default=24)
    parser.add_argument("--max-iter", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/robust_fspl_parallax_regression.json"),
    )
    args = parser.parse_args()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "seed_policy": "pspl_fit_and_detector_only",
        "events": [run_event(event_id, args.max_seeds, args.max_iter) for event_id in args.events],
    }
    for event in payload["events"]:
        _validate_event_schema(event)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
