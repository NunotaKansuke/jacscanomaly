#!/usr/bin/env python3
"""Run jacscanomaly on the RGES Beginner/Experienced Parquet light curves.

The RGES files are row-oriented Parquet tables, not one-file-per-event light
curves.  This runner deliberately processes one ``name`` at a time and only
reads F146 rows for that name.  It writes an atomic JSON artifact after every
successful event so a long run can be resumed safely.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATA_ROOT = Path("/moao39_13/nunota/rges-data")
DEFAULT_OUTPUT = DATA_ROOT / "anomaly_finder_result"
DEFAULT_HTML_OUTPUT = Path(__file__).resolve().parents[2] / "html_portal" / "rges_anomaly_finder"
DEFAULT_SYNC_SCRIPT = Path(__file__).resolve().parents[2] / "html_portal" / "tool" / "request_sync.sh"
DEFAULT_PROGRESS_FILE = DATA_ROOT / "anomaly_finder_progress.txt"
TIERS = {
    "beginner": {
        "path": DATA_ROOT / "beginner" / "RMDC26_Beginner_Tier_test.parquet",
        "value": "mag",
        "error": "mag_err",
        "data_kind": "mag",
    },
    "experienced": {
        "path": DATA_ROOT / "experienced" / "RMDC26_Experienced_Tier_test.parquet",
        "value": "flux_uJy",
        "error": "flux_err_uJy",
        "data_kind": "flux",
    },
}
FILT = "F146"
SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
OBLIQUITY_DEG = 23.4392911


def _json_safe(value: Any) -> Any:
    """Convert NumPy/dataclass-like values into strict JSON values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: _json_safe(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _safe_name(name: str) -> str:
    value = SAFE_NAME.sub("_", str(name)).strip("._")
    return value or hashlib.sha1(str(name).encode()).hexdigest()[:12]


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _require_pyarrow():
    try:
        import pyarrow.compute as pc
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "RGES processing requires pyarrow; install it in the runtime "
            "environment before running this script."
        ) from exc
    return ds, pc, pq


def event_names(tier: str, output_dir: Path, *, refresh: bool = False) -> list[str]:
    """Enumerate names without materialising the 184 GB ML table."""
    _, _, pq = _require_pyarrow()
    cache = output_dir / "metadata" / f"event_names_{tier}.json"
    if cache.exists() and not refresh:
        return [str(item) for item in json.loads(cache.read_text(encoding="utf-8"))]

    path = Path(TIERS[tier]["path"])
    if not path.is_file():
        raise FileNotFoundError(f"missing RGES tier file: {path}")
    names: set[str] = set()
    parquet = pq.ParquetFile(path)
    for row_group in range(parquet.metadata.num_row_groups):
        table = parquet.read_row_group(row_group, columns=["name"])
        names.update(str(value) for value in table.column("name").unique().to_pylist())
        del table
        if row_group % 10 == 9:
            print(f"[{tier}] indexed row groups {row_group + 1}/{parquet.metadata.num_row_groups}", flush=True)
    ordered = sorted(names)
    _atomic_json(cache, ordered)
    return ordered


def _load_f146_event_with_observer(
    tier: str, name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
    """Read one F146 curve and retain the RGES observer positions internally."""
    ds, pc, _ = _require_pyarrow()
    spec = TIERS[tier]
    dataset = ds.dataset(spec["path"], format="parquet")
    expression = pc.equal(ds.field("name"), name) & pc.equal(ds.field("filt"), FILT)
    table = dataset.to_table(
        columns=[
            "bjd",
            spec["value"],
            spec["error"],
            "saturation_flag",
            "ra_deg",
            "dec_deg",
            "obs_x",
            "obs_y",
            "obs_z",
        ],
        filter=expression,
    )
    time_values = np.asarray(table["bjd"].to_numpy(zero_copy_only=False), dtype=float)
    values = np.asarray(table[spec["value"]].to_numpy(zero_copy_only=False), dtype=float)
    errors = np.asarray(table[spec["error"]].to_numpy(zero_copy_only=False), dtype=float)
    saturation = np.asarray(table["saturation_flag"].to_numpy(zero_copy_only=False))
    ra_values = np.asarray(table["ra_deg"].to_numpy(zero_copy_only=False), dtype=float)
    dec_values = np.asarray(table["dec_deg"].to_numpy(zero_copy_only=False), dtype=float)
    observer_values = np.column_stack(
        [
            np.asarray(table[column].to_numpy(zero_copy_only=False), dtype=float)
            for column in ("obs_x", "obs_y", "obs_z")
        ]
    )

    valid = (
        np.isfinite(time_values)
        & np.isfinite(values)
        & np.isfinite(errors)
        & (errors > 0.0)
        & (saturation == 0)
        & np.isfinite(ra_values)
        & np.isfinite(dec_values)
        & np.all(np.isfinite(observer_values), axis=1)
    )
    time_values = time_values[valid]
    values = values[valid]
    errors = errors[valid]
    ra_values = ra_values[valid]
    dec_values = dec_values[valid]
    observer_values = observer_values[valid]
    order = np.argsort(time_values, kind="mergesort")
    time_values = time_values[order]
    values = values[order]
    errors = errors[order]
    ra_values = ra_values[order]
    dec_values = dec_values[order]
    observer_values = observer_values[order]
    if time_values.size == 0:
        raise ValueError("no valid F146 points after quality filtering")

    # Keep both tiers in a numerically stable relative-flux scale.  The
    # magnitudes use the same stable conversion as Finder(data_kind="mag").
    if spec["data_kind"] == "mag":
        zero_point = float(np.median(values))
        exponent = -0.4 * np.log(10.0) * (values - zero_point)
        exponent = np.clip(exponent, np.log(np.finfo(np.float32).tiny), np.log(np.finfo(np.float32).max))
        flux = np.exp(exponent)
        ferr = np.maximum((np.log(10.0) / 2.5) * flux * errors, np.finfo(np.float32).tiny)
        source_scale = "relative_flux_from_mag"
    else:
        scale = float(np.median(np.abs(values)))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("F146 flux has no positive finite scale")
        flux = values / scale
        ferr = errors / scale
        source_scale = "relative_flux_from_uJy"

    metadata = {
        "tier": tier,
        "event": name,
        "filter": FILT,
        "source_value_column": spec["value"],
        "source_error_column": spec["error"],
        "source_data_kind": spec["data_kind"],
        "source_scale": source_scale,
        "n_raw_f146": int(table.num_rows),
        "n_valid": int(time_values.size),
        "n_rejected_quality": int(table.num_rows - time_values.size),
        "ra_deg": float(np.median(ra_values)),
        "dec_deg": float(np.median(dec_values)),
        "space_parallax_ephemeris": "rges_obs_xyz",
    }
    return time_values, flux, ferr, metadata, observer_values


def load_f146_event(
    tier: str, name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Read, quality-filter, sort, and normalise one F146 light curve."""
    time_values, flux, ferr, metadata, _ = _load_f146_event_with_observer(tier, name)
    return time_values, flux, ferr, metadata


def _earth_position_at(time_values: np.ndarray) -> np.ndarray:
    """Interpolate the package Earth ephemeris at full-JD RGES timestamps."""
    from importlib import resources
    from jacscanomaly.parallax import load_horizons_vectors_file

    earth_table = load_horizons_vectors_file(
        str(resources.files("jacscanomaly.data").joinpath("earth_orbital_parallax_table.txt"))
    )
    times = np.asarray(time_values, dtype=float)
    if times.size == 0 or times.min() < earth_table[0, 0] or times.max() > earth_table[-1, 0]:
        raise ValueError("RGES observer timestamps fall outside the package Earth ephemeris support")
    return np.column_stack(
        [np.interp(times, earth_table[:, 0], earth_table[:, axis]) for axis in range(1, 4)]
    )


def _ecliptic_to_icrf(position: np.ndarray, sign: float = 1.0) -> np.ndarray:
    """Rotate ecliptic Cartesian coordinates into the ICRF equatorial frame."""
    angle = np.deg2rad(float(sign) * OBLIQUITY_DEG)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    return np.asarray(position, dtype=float) @ rotation.T


def _write_rges_space_ephemeris(
    output_dir: Path,
    tier: str,
    time_values: np.ndarray,
    observer_values: np.ndarray,
) -> Path:
    """Create the common RGES geocentric satellite table used by GULLS mode.

    RGES ``obs_x/y/z`` are complete observer positions in ecliptic Cartesian
    AU.  The native/GULLS backend consumes geocentric ICRF positions, so the
    frame is selected by agreement with the package Earth ephemeris, converted
    to ICRF, and the Earth position is removed before writing JD/RA/Dec/AU.
    """
    time_values = np.asarray(time_values, dtype=float).reshape(-1)
    observer_values = np.asarray(observer_values, dtype=float).reshape(-1, 3)
    if time_values.size != observer_values.shape[0] or time_values.size < 2:
        raise ValueError("RGES observer ephemeris requires at least two time-position rows")
    order = np.argsort(time_values, kind="mergesort")
    times = time_values[order]
    positions = observer_values[order]
    unique, unique_index = np.unique(times, return_index=True)
    times = unique
    positions = positions[unique_index]

    ephemeris_dir = Path(output_dir) / "ephemeris"
    ephemeris_dir.mkdir(parents=True, exist_ok=True)
    path = ephemeris_dir / "rges_space_parallax.txt"
    # The cadence/observer orbit is shared by the RGES tiers.  Reuse the
    # validated table for subsequent event subprocesses instead of rewriting
    # tens of thousands of rows for every light curve.
    if path.is_file() and path.stat().st_size > 0:
        return path

    earth = _earth_position_at(times)

    candidates = {
        "identity": positions,
        "ecliptic_to_icrf": _ecliptic_to_icrf(positions, 1.0),
        "ecliptic_to_icrf_reverse": _ecliptic_to_icrf(positions, -1.0),
    }
    # The RGES columns are complete heliocentric positions (radius ~1 AU).
    # Choose the frame that agrees with the package ICRF Earth ephemeris.  The
    # radius check prevents a genuinely geocentric input from being mistaken
    # for a complete observer orbit.
    scores = {
        key: float(np.median(np.linalg.norm(value - earth, axis=1)))
        for key, value in candidates.items()
    }
    frame_name = min(scores, key=scores.get)
    complete_observer = candidates[frame_name]
    if float(np.median(np.linalg.norm(complete_observer, axis=1))) > 0.2:
        geocentric = complete_observer - earth
        observer_kind = "complete_heliocentric"
    else:
        geocentric = complete_observer
        observer_kind = "geocentric"
    distances = np.linalg.norm(geocentric, axis=1)
    if not np.all(np.isfinite(distances)) or np.any(distances <= 1.0e-8):
        raise ValueError("RGES observer positions contain invalid geocentric distances")
    ra = np.degrees(np.arctan2(geocentric[:, 1], geocentric[:, 0])) % 360.0
    dec = np.degrees(np.arcsin(np.clip(geocentric[:, 2] / distances, -1.0, 1.0)))
    table = np.column_stack((times, ra, dec, distances))

    temporary = path.with_suffix(path.suffix + ".tmp")
    header = (
        f"# RGES obs_xyz -> geocentric ICRF GULLS table; frame={frame_name}; "
        f"observer={observer_kind}; score={scores[frame_name]:.8g}\n"
    )
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(header)
        np.savetxt(handle, table, fmt="%.12f")
    temporary.replace(path)
    return path


def _fit_summary(fit: Any) -> dict[str, Any]:
    params = np.asarray(getattr(fit, "params", ()), dtype=float).reshape(-1)
    names = list(getattr(fit, "param_names", ()))
    named = {
        str(name): float(params[index])
        for index, name in enumerate(names)
        if index < params.size
    }
    kind = str(getattr(fit, "model_kind", "pspl"))
    return {
        "model_kind": kind,
        "param_names": names,
        "params": named,
        "params_vector": params,
        "chi2": getattr(fit, "chi2", None),
        "chi2_dof": getattr(fit, "chi2_dof", None),
        "bic": getattr(fit, "bic", None),
        "fs": getattr(fit, "fs", None),
        "fb": getattr(fit, "fb", None),
        "optimizer_status": getattr(fit, "optimizer_status", None),
    }


def _adaptive_model_curve(fit: Any, xlim: tuple[float, float]) -> dict[str, Any]:
    """Evaluate the adopted model on an adaptive, non-observation grid."""
    from jacscanomaly.plot import _adaptive_single_lens_curve

    curve_time, curve_flux = _adaptive_single_lens_curve(
        fit,
        (float(xlim[0]), float(xlim[1])),
        base_points=192,
        max_points=4000,
    )
    return {
        "time": np.asarray(curve_time, dtype=float),
        "flux": np.asarray(curve_flux, dtype=float),
    }
def _candidate_summary(candidate: Any) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return _json_safe(candidate)


def _plot_indices(time_values: np.ndarray, signal_mask: np.ndarray, max_points: int) -> np.ndarray:
    n_points = time_values.size
    if max_points <= 0 or n_points <= max_points:
        return np.arange(n_points, dtype=int)
    uniform = np.linspace(0, n_points - 1, max_points, dtype=int)
    signal = np.flatnonzero(signal_mask)
    return np.unique(np.concatenate((uniform, signal))).astype(int)


def _fit_exclusion_mask(result: Any) -> np.ndarray:
    """Return only points actually excluded by an accepted continuation fit.

    ``PlanetSignalResult.signal_mask`` is the extractor's working mask.  It
    is suitable for analysis provenance, but it must not be assumed to be an
    HTML display mask: a frozen/measurement pass can use the same field for a
    broad residual-measurement window.  In the RGES beam path, an actual mask
    is represented by an accepted iteration and zero point weights.
    """
    signal = np.asarray(getattr(result, "signal_mask", ()), dtype=bool).reshape(-1)
    if signal.size == 0 or not tuple(getattr(result, "iterations", ())):
        return np.zeros(signal.shape, dtype=bool)
    weights = np.asarray(getattr(result, "point_weight", ()), dtype=float).reshape(-1)
    if weights.size == signal.size:
        return signal & (weights <= 0.0)
    return signal.copy()


def _write_figures(
    result: Any,
    template_free: Any,
    features: Any,
    figure_dir: Path,
    tier: str,
    safe_event: str,
    display_mask: np.ndarray | None = None,
) -> dict[str, str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    time_values = np.asarray(result.time, dtype=float)
    flux = np.asarray(result.flux, dtype=float)
    ferr = np.asarray(result.ferr, dtype=float)
    residual = np.asarray(result.refined_residual, dtype=float)
    mask = (
        _fit_exclusion_mask(result)
        if display_mask is None
        else np.asarray(display_mask, dtype=bool)
    )
    indices = _plot_indices(time_values, mask, 12000)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax, rx = axes
    ax.errorbar(time_values[indices], flux[indices], yerr=ferr[indices], fmt=".", ms=2, alpha=0.45, label="F146")
    model_kind = str(getattr(result.refined_fit, "model_kind", "pspl"))
    model_curve = _adaptive_model_curve(
        result.refined_fit,
        (float(np.min(time_values)), float(np.max(time_values))),
    )
    model_time = model_curve["time"]
    model_values = model_curve["flux"]
    ax.plot(model_time, model_values, color="black", lw=1.4, label=f"refined {model_kind}")
    if np.any(mask):
        ax.scatter(time_values[mask], flux[mask], s=8, color="tab:orange", label="refined signal", zorder=3)
    ax.set_ylabel("relative F146 flux")
    ax.legend(loc="best")

    z = residual / np.maximum(ferr, 1e-12)
    rx.axhline(0.0, color="0.5", lw=1)
    rx.plot(time_values[indices], z[indices], ".", ms=2, alpha=0.55, color="tab:blue")
    if np.any(mask):
        rx.scatter(time_values[mask], z[mask], s=8, color="tab:orange", zorder=3)
    for feature in features.features:
        rx.axvspan(feature.t_start, feature.t_end, color="tab:red" if feature.kind == "dip" else "tab:green", alpha=0.18)
        rx.axvline(feature.time, color="tab:red" if feature.kind == "dip" else "tab:green", lw=1)
    rx.set_ylabel("residual / error")
    rx.set_xlabel("BJD")
    fig.tight_layout()
    template_path = figure_dir / f"{safe_event}_template_pspl.png"
    fig.savefig(template_path, dpi=130)
    plt.close(fig)

    tf_fig, _ = template_free.plot(show=False, use_normalized_residual=True)
    tf_fig.tight_layout()
    tf_path = figure_dir / f"{safe_event}_template_free.png"
    tf_fig.savefig(tf_path, dpi=130)
    plt.close(tf_fig)
    tf_zoom_fig, _ = template_free.plot(
        show=False,
        use_normalized_residual=True,
        zoom=True,
        zoom_pad=5.0,
    )
    tf_zoom_fig.tight_layout()
    tf_zoom_path = figure_dir / f"{safe_event}_template_free_zoom.png"
    tf_zoom_fig.savefig(tf_zoom_path, dpi=130)
    plt.close(tf_zoom_fig)
    return {
        "template_pspl": str(template_path.relative_to(figure_dir.parent.parent)),
        "template_free": str(tf_path.relative_to(figure_dir.parent.parent)),
        "template_free_zoom": str(tf_zoom_path.relative_to(figure_dir.parent.parent)),
    }


def _run_one(
    tier: str,
    name: str,
    output_dir: Path,
    *,
    min_points: int,
    plot_points: int,
    fallback_max_seeds: int,
    fallback_max_iter: int,
    fallback_angle_steps: int,
) -> dict[str, Any]:
    from jacscanomaly import (
        AnomalyPipelineConfig,
        ContaminationConfig,
        FallbackConfig,
        Finder,
        FinderConfig,
        PlanetSignalConfig,
    )

    time_values, flux, ferr, metadata, observer_values = _load_f146_event_with_observer(
        tier, name
    )
    if time_values.size < min_points:
        raise ValueError(f"only {time_values.size} valid F146 points; need {min_points}")
    space_ephemeris_path = _write_rges_space_ephemeris(
        output_dir, tier, time_values, observer_values
    )

    finder = Finder(
        FinderConfig(
            fitter_kind="pspl",
            single_fit_backend="cpp",
            grid_backend="cpp",
            ra_deg=float(metadata["ra_deg"]),
            dec_deg=float(metadata["dec_deg"]),
            satellite_ephemeris_path=str(space_ephemeris_path),
            parallax_geometry="space",
            parallax_observer_convention="gulls",
            # RGES bjd values are full Julian Dates, unlike the Roman/GULLS
            # helper tables that use an offset HJD convention.
            parallax_time_scale="jd",
            parallax_time_offset=0.0,
        )
    )
    planet_config = PlanetSignalConfig(
        baseline_mode="beam_interval",
        beam_max_iter=3,
        beam_width=1,
        beam_candidates_per_iter=1,
    )
    # Keep the complete event in every fitting stage.  In particular, do not
    # replace the fallback input with a representative/subsampled curve: that
    # changes the model comparison and is not equivalent to the Roman flow.
    pipeline_result = finder.run_anomaly_pipeline(
        time_values,
        flux,
        ferr,
        config=AnomalyPipelineConfig(
            planet=planet_config,
            planet_fast_mode=False,
            post_physical_max_refits=3,
            fallback=FallbackConfig(
                max_seeds=max(1, int(fallback_max_seeds)),
                parallax_angle_steps=max(1, int(fallback_angle_steps)),
                contamination=ContaminationConfig(
                    max_iter=max(1, int(fallback_max_iter))
                ),
            ),
        ),
        verbose=False,
    )
    physical_run = pipeline_result.effect_aware
    base_result = physical_run.planet_before
    if base_result is None:
        raise RuntimeError("effect-aware run did not produce the pre-fallback result")
    full_effects = tuple(physical_run.effect_candidates)
    fallback_candidates = tuple(
        candidate
        for candidate in full_effects
        if getattr(candidate, "decision", "skip") == "fallback"
    )
    fallback_result = physical_run.fallback_result
    selected_physical_fit = physical_run.selected_fit
    result = pipeline_result.final_measurement
    post_physical_refits = int(
        pipeline_result.diagnostics["post_physical_refits_completed"]
    )
    post_physical_reset = bool(
        pipeline_result.diagnostics["post_physical_refinement_reset"]
    )
    display_mask = np.asarray(
        pipeline_result.fit_exclusion_mask, dtype=bool
    )
    physical_reason_codes: list[str] = list(pipeline_result.reason_codes)
    physical_reason_codes.append("full_f146_fallback_input")

    features = pipeline_result.features
    template_free = pipeline_result.template_free

    # Use the same event/saved-window rules as the HTML builder, then sample
    # the adopted model independently of the observation times.  This keeps
    # the public line smooth across cadence gaps and prevents it from being
    # mistaken for another data trace.
    try:
        from build_rges_anomaly_html import _data_xlim, _event_xlim
    except ModuleNotFoundError:
        from tools.build_rges_anomaly_html import _data_xlim, _event_xlim
    window_seed = {
        "fit": {"selected_physical": _fit_summary(result.refined_fit)},
        "series": {
            "time": time_values.tolist(),
            "display_signal_mask": display_mask.astype(int).tolist(),
        },
        "anomaly_candidates": pipeline_result.anomaly_candidates,
    }
    display_xlim = _event_xlim(window_seed, time_values.tolist())
    saved_xlim = _data_xlim(display_xlim, time_values.tolist())
    model_curve = _adaptive_model_curve(result.refined_fit, tuple(saved_xlim))

    safe_event = _safe_name(name)
    figure_dir = output_dir / "figures" / tier
    figure_paths = _write_figures(
        result,
        template_free,
        features,
        figure_dir,
        tier,
        safe_event,
        display_mask=display_mask,
    )
    model_comparison = []
    if fallback_result is not None:
        for stage in tuple(getattr(fallback_result, "stage_results", ())):
            model_comparison.append(
                {
                    "effect": getattr(stage, "effect", None),
                    "success": getattr(stage, "success", None),
                    "bic_improvement": getattr(stage, "bic_improvement", None),
                    "baseline_bic": getattr(stage, "baseline_bic", None),
                    "selected_bic": getattr(stage, "selected_bic", None),
                    "selected_original_chi2": getattr(
                        stage, "selected_original_chi2", None
                    ),
                    "reason_codes": getattr(stage, "reason_codes", ()),
                }
            )
    plot_indices = _plot_indices(time_values, display_mask, plot_points)
    payload = {
        "schema_version": 2,
        "pipeline": {
            "name": "rges_f146_template_pspl_refine_features_template_free",
            "stage": "event_complete",
            "jacscanomaly": "source_checkout",
        },
        "metadata": metadata,
        "fit": {
            "initial": _fit_summary(base_result.initial_fit),
            "selected_physical": _fit_summary(selected_physical_fit),
            "refined": _fit_summary(pipeline_result.adopted_fit),
        },
        "physical_fallback": {
            "reason_codes": physical_reason_codes,
            "diagnostics": {
                "n_effect_candidates": len(full_effects),
                "n_fallback_candidates": len(fallback_candidates),
                "full_f146_points": int(time_values.size),
                "fallback_search_points": int(time_values.size),
                "post_physical_refits_completed": post_physical_refits,
                "post_physical_mask_points": int(np.sum(display_mask)),
                "post_physical_refinement_reset": post_physical_reset,
            },
            "effect_candidates": full_effects,
            "accepted": bool(
                fallback_result is not None and fallback_result.success
            ),
            "result": (
                {
                    "effect": fallback_result.effect,
                    "success": fallback_result.success,
                    "reason_codes": fallback_result.reason_codes,
                    "bic_improvement": fallback_result.bic_improvement,
                    "baseline_bic": fallback_result.baseline_bic,
                    "selected_bic": fallback_result.selected_bic,
                    "n_attempts": len(fallback_result.attempts),
                    "model_spec": fallback_result.model_spec,
                    "model_comparison": model_comparison,
                }
                if fallback_result is not None
                else None
            ),
            "space_parallax_skipped": False,
            "space_parallax_ephemeris": str(space_ephemeris_path),
            "space_parallax_observer_convention": "gulls",
        },
        "template_scan": {
            "seed": _candidate_summary(result.initial_seed),
            "n_refinement_candidates": len(result.candidates),
            "best_refinement_candidate": _candidate_summary(result.best),
            "timing": _json_safe(result.timing),
        },
        "features": {
            **features.summary_dict(),
            "items": features.feature_dicts(),
        },
        "template_free": {
            "n_candidates": len(template_free.candidates),
            "best": _candidate_summary(template_free.best),
            "candidates": [_candidate_summary(item) for item in template_free.candidates],
        },
        "has_anomaly_candidate": pipeline_result.has_anomaly_candidate,
        "best_anomaly_candidate": pipeline_result.best_anomaly_candidate,
        "anomaly_candidates": pipeline_result.anomaly_candidates,
        "plot": {
            "peak_xlim": display_xlim,
            "saved_xlim": saved_xlim,
            "model_curve": model_curve,
        },
        "plots": figure_paths,
        "series": {
            "n_total": int(time_values.size),
            "n_saved": int(plot_indices.size),
            "time": time_values[plot_indices],
            "flux": np.asarray(result.flux)[plot_indices],
            "ferr": np.asarray(result.ferr)[plot_indices],
            "model_flux": np.asarray(result.refined_fit.model_flux)[plot_indices],
            "residual": np.asarray(result.refined_residual)[plot_indices],
            "signal_mask": np.asarray(result.signal_mask, dtype=bool)[plot_indices].astype(int),
            "fit_exclusion_mask": display_mask[plot_indices].astype(int),
            "display_signal_mask": display_mask[plot_indices].astype(int),
        },
    }
    event_json = output_dir / "events" / tier / f"{safe_event}.json"
    _atomic_json(event_json, payload)
    payload["_json_path"] = str(event_json)
    return payload


def _write_error(output_dir: Path, tier: str, name: str, exc: BaseException) -> None:
    error_path = output_dir / "errors.jsonl"
    with error_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "tier": tier,
                    "event": name,
                    "error": type(exc).__name__,
                    "message": str(exc),
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def _publish_event(output_dir: Path, event_json: Path) -> None:
    """Build one event page and request portal synchronization."""
    try:
        from build_rges_anomaly_html import build_html
    except ModuleNotFoundError:
        from tools.build_rges_anomaly_html import build_html

    build_html(output_dir, DEFAULT_HTML_OUTPUT, event_json=event_json)
    subprocess.run(
        [str(DEFAULT_SYNC_SCRIPT)],
        cwd=str(DEFAULT_SYNC_SCRIPT.parent.parent),
        check=True,
    )


def _write_progress(
    progress_path: Path,
    processed: int,
    total: int,
    tier: str,
    name: str,
    status: str,
) -> None:
    percent = 100.0 * processed / max(total, 1)
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{processed}/{total} ({percent:.2f}%) tier={tier} "
            f"event={name} status={status}\n"
        )


def _refresh_manifest(output_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted((output_dir / "events").glob("*/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metadata = payload.get("metadata", {}) or {}
        refined = (payload.get("fit", {}) or {}).get("refined", {}) or {}
        features = payload.get("features", {}) or {}
        tf = payload.get("template_free", {}) or {}
        anomaly_candidates = list(payload.get("anomaly_candidates", []) or [])
        entries.append(
            {
                "tier": metadata.get("tier", path.parent.name),
                "event": metadata.get("event", path.stem),
                "json": str(path.relative_to(output_dir)),
                "plots": payload.get("plots", {}),
                "n_points": metadata.get("n_valid", 0),
                "chi2_dof": refined.get("chi2_dof"),
                "n_peaks": features.get("n_peaks", 0),
                "n_dips": features.get("n_dips", 0),
                "template_free_candidates": tf.get("n_candidates", 0),
                "anomaly_candidates": len(anomaly_candidates),
                "has_anomaly_candidate": bool(anomaly_candidates),
                "best_anomaly_candidate": (
                    anomaly_candidates[0] if anomaly_candidates else None
                ),
                "score": (payload.get("template_scan", {}).get("seed") or {}).get("score"),
            }
        )
    entries.sort(key=lambda row: (row["tier"], row["event"]))
    _atomic_json(output_dir / "manifest.json", entries)
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("beginner", "experienced", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None, help="Maximum events per selected tier.")
    parser.add_argument("--names", default=None, help="Comma-separated event names; overrides --limit.")
    parser.add_argument("--refresh-event-list", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute existing successful event JSON files.")
    parser.add_argument("--min-points", type=int, default=32)
    parser.add_argument("--plot-points", type=int, default=12000)
    parser.add_argument("--fallback-max-seeds", type=int, default=8)
    parser.add_argument("--fallback-max-iter", type=int, default=8)
    parser.add_argument("--fallback-angle-steps", type=int, default=8)
    parser.add_argument("--progress-file", type=Path, default=DEFAULT_PROGRESS_FILE)
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Process events in this Python process. By default each event gets a fresh child process.",
    )
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--build-html", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.progress_file.resolve()
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    selected_tiers = list(TIERS) if args.tier == "both" else [args.tier]
    explicit_names = None
    if args.names:
        explicit_names = {item.strip() for item in args.names.split(",") if item.strip()}

    batches: list[tuple[str, list[str]]] = []
    for tier in selected_tiers:
        names = event_names(tier, output_dir, refresh=args.refresh_event_list)
        if explicit_names is not None:
            names = [name for name in names if name in explicit_names]
        elif args.limit is not None:
            names = names[: max(0, args.limit)]
        batches.append((tier, names))

    total_events = sum(len(names) for _, names in batches)
    processed_events = 0
    completed = _refresh_manifest(output_dir)
    portal_manifest_keys: set[tuple[str, str]] = set()
    portal_template_ready = False
    if args.build_html:
        try:
            portal_info = json.loads(
                (DEFAULT_HTML_OUTPUT / "build_info.json").read_text(encoding="utf-8")
            )
            portal_template_ready = portal_info.get("ui_template") in {
                "roman_simu/planet_signal_result",
                "roman_simu/anomaly_finder_result",
            }
            if portal_template_ready:
                portal_manifest_keys = {
                    (str(row.get("tier", "")), str(row.get("event", "")))
                    for row in json.loads(
                        (DEFAULT_HTML_OUTPUT / "manifest.json").read_text(encoding="utf-8")
                    )
                }
        except (OSError, json.JSONDecodeError, TypeError):
            portal_template_ready = False
    print(f"[overall] events selected: {total_events}", flush=True)
    for tier, names in batches:
        print(f"[{tier}] events selected: {len(names)}", flush=True)
        for position, name in enumerate(names, start=1):
            safe_event = _safe_name(name)
            event_json = output_dir / "events" / tier / f"{safe_event}.json"
            if (
                event_json.exists()
                and not args.force
                and (
                    not args.build_html
                    or (
                        portal_template_ready
                        and (tier, name) in portal_manifest_keys
                    )
                )
            ):
                processed_events += 1
                percent = 100.0 * processed_events / max(total_events, 1)
                print(
                    f"[{tier}] {position}/{len(names)} {name}: skip existing "
                    f"| overall {processed_events}/{total_events} ({percent:.2f}%)",
                    flush=True,
                )
                if not args._child:
                    _write_progress(
                        progress_path, processed_events, total_events, tier, name, "skip"
                    )
                continue
            started = time.time()
            print(f"[{tier}] {position}/{len(names)} {name}: start F146", flush=True)
            previous_mtime = (
                event_json.stat().st_mtime_ns if event_json.exists() else None
            )
            try:
                if args.in_process or args._child:
                    _run_one(
                        tier,
                        name,
                        output_dir,
                        min_points=max(1, args.min_points),
                        plot_points=args.plot_points,
                        fallback_max_seeds=args.fallback_max_seeds,
                        fallback_max_iter=args.fallback_max_iter,
                        fallback_angle_steps=args.fallback_angle_steps,
                    )
                else:
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--tier",
                        tier,
                        "--names",
                        name,
                        "--output-dir",
                        str(output_dir),
                        "--min-points",
                        str(args.min_points),
                        "--plot-points",
                        str(args.plot_points),
                        "--fallback-max-seeds",
                        str(args.fallback_max_seeds),
                        "--fallback-max-iter",
                        str(args.fallback_max_iter),
                        "--fallback-angle-steps",
                        str(args.fallback_angle_steps),
                        "--_child",
                    ]
                    if args.force:
                        command.append("--force")
                    completed_process = subprocess.run(
                        command,
                        cwd=str(Path(__file__).resolve().parents[1]),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT,
                    )
                    if completed_process.returncode != 0:
                        raise RuntimeError(f"isolated event process exited with {completed_process.returncode}")
                    if not event_json.exists() or (
                        previous_mtime is not None
                        and event_json.stat().st_mtime_ns <= previous_mtime
                    ):
                        raise RuntimeError(
                            "isolated event process exited without writing a fresh event artifact"
                        )
                if args.build_html and not args._child:
                    _publish_event(output_dir, event_json)
            except Exception as exc:  # continue to the next event by design
                _write_error(output_dir, tier, name, exc)
                print(f"[{tier}] {position}/{len(names)} {name}: ERROR {type(exc).__name__}: {exc}", flush=True)
            else:
                print(f"[{tier}] {position}/{len(names)} {name}: done {time.time() - started:.1f}s", flush=True)
            processed_events += 1
            percent = 100.0 * processed_events / max(total_events, 1)
            print(
                f"[overall] progress {processed_events}/{total_events} ({percent:.2f}%)",
                flush=True,
            )
            if not args._child:
                _write_progress(
                    progress_path,
                    processed_events,
                    total_events,
                    tier,
                    name,
                    "done" if event_json.exists() else "error",
                )
            completed = _refresh_manifest(output_dir)
            try:
                import jax

                jax.clear_caches()
            except Exception:
                pass
            gc.collect()

    print(f"completed artifacts: {len(completed)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
