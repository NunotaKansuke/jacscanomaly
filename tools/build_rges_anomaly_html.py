#!/usr/bin/env python3
"""Build RGES pages using the canonical roman_simu anomaly-finder HTML."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import shutil
from typing import Any


FILT = "F146"
DATA_ROOT = Path("/moao39_13/nunota/rges-data")
DEFAULT_RESULT_DIR = DATA_ROOT / "anomaly_finder_result"
DEFAULT_PORTAL_OUTPUT = Path(__file__).resolve().parents[2] / "html_portal" / "rges_anomaly_finder"
ROMAN_MAKE_HTML = Path(__file__).resolve().parents[2] / "roman_simu" / "tool" / "make_html.py"
PLOTLY_SOURCE = Path("/moao39_13/nunota/autolens/html/assets/plotly-1.58.5.min.js")


def _canonical_scripts() -> dict[str, str]:
    """Read the literal CSS/JS blocks from the real Roman HTML generator."""
    tree = ast.parse(ROMAN_MAKE_HTML.read_text(encoding="utf-8"))
    wanted = {"CSS", "EVENT_JS", "FEATURE_EVENT_JS", "JS"}
    found: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in wanted:
                value = ast.literal_eval(node.value)
                if isinstance(value, str):
                    found[target.id] = value
    missing = wanted - found.keys()
    if missing:
        raise RuntimeError(f"canonical Roman HTML blocks missing: {sorted(missing)}")
    return found


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _fmt(value: Any) -> str:
    number = _finite(value)
    if number is None:
        return ""
    if abs(number) >= 1.0e6 or (0 < abs(number) < 0.01):
        return f"{number:.3e}"
    return f"{number:.4f}"


def _num_cell(value: Any) -> str:
    shown = _fmt(value)
    raw = ""
    number = _finite(value)
    if number is not None:
        raw = f"{number:.8g}"
    return f'<td class="r" data-v="{_esc(raw)}">{_esc(shown)}</td>'


def _xlim(values: list[Any], mask: list[Any] | None = None) -> list[float]:
    selected = values
    if mask is not None and len(mask) == len(values):
        masked = [value for value, flag in zip(values, mask) if bool(flag)]
        if masked:
            selected = masked
    numbers = [value for value in (_finite(item) for item in selected) if value is not None]
    if not numbers:
        numbers = [value for value in (_finite(item) for item in values) if value is not None]
    if not numbers:
        return [0.0, 1.0]
    lo, hi = min(numbers), max(numbers)
    if lo == hi:
        return [lo - 1.0, hi + 1.0]
    pad = 0.03 * (hi - lo)
    return [lo - pad, hi + pad]


def _event_xlim(payload: dict[str, Any], time_values: list[Any]) -> list[float]:
    """Return the initial main-panel range used by the canonical plot tool.

    This is deliberately a display range, not the range saved in the public
    JSON.  The canonical Roman pipeline uses a several-tE event view and then
    saves an additional padded window behind it.
    """
    fit = payload.get("fit", {}) or {}
    selected = fit.get("selected_physical") or fit.get("refined") or fit.get("initial") or {}
    params = selected.get("params", {}) or {}
    t0, t_e, u0 = (_finite(params.get(key)) for key in ("t0", "tE", "u0"))
    finite_time = [value for value in (_finite(item) for item in time_values) if value is not None]
    if t0 is None or t_e is None or u0 is None:
        return _xlim(time_values)

    # Same scale as the canonical Roman event plot: 3 tE effective widths,
    # with a five-day minimum half-width.  This keeps the whole event visible
    # without opening the plot to an entire observing season.
    half_width = max(3.0 * abs(t_e) * max(abs(u0), 1.0), 5.0)
    lo, hi = t0 - half_width, t0 + half_width

    # Keep a separated anomaly and every adopted planet-signal point in the
    # initial view as well.  The latter is important: a planet signal can be
    # farther from the lens peak than the nominal tE window.
    raw_series = payload.get("series", {}) or {}
    # The extractor's signal_mask is provenance for the analysis and may be
    # a broad residual-measurement window.  Only the explicit display mask is
    # allowed to expand the initial HTML view.
    signal_mask = list(raw_series.get("display_signal_mask", []) or [])
    if len(signal_mask) == len(time_values):
        signal_times = [
            _finite(value)
            for value, flag in zip(time_values, signal_mask)
            if bool(flag)
        ]
        signal_times = [value for value in signal_times if value is not None]
        if signal_times:
            lo = min(lo, min(signal_times) - 0.5)
            hi = max(hi, max(signal_times) + 0.5)

    candidates = _anomaly_candidate_rows(payload)
    for candidate in candidates:
        start = _finite(candidate.get("t_start"))
        end = _finite(candidate.get("t_end"))
        if start is None or end is None:
            continue
        lo = min(lo, start - 0.5)
        hi = max(hi, end + 0.5)

    if finite_time:
        lo = max(lo, min(finite_time))
        hi = min(hi, max(finite_time))
    return [lo, hi] if lo < hi else _xlim(time_values)


def _anomaly_candidate_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return canonical candidates, with a compatibility path for old JSON."""

    if "anomaly_candidates" in payload:
        return list(payload.get("anomaly_candidates", []) or [])
    candidates = list(
        (payload.get("template_free", {}) or {}).get("candidates", []) or []
    )
    candidates += list((payload.get("features", {}) or {}).get("items", []) or [])
    return candidates


def _data_xlim(display_xlim: list[float], time_values: list[Any]) -> list[float]:
    """Make the saved data window wider than the initial display window.

    Two days of padding follows the canonical Roman event-window writer.  It
    allows users to zoom out slightly and retains the baseline around the
    displayed event, while still avoiding a full-season payload.
    """
    numbers = [value for value in (_finite(item) for item in time_values) if value is not None]
    if not numbers:
        return list(display_xlim)
    pad = 2.0
    lo = max(min(numbers), float(display_xlim[0]) - pad)
    hi = min(max(numbers), float(display_xlim[1]) + pad)
    return [lo, hi] if lo < hi else _xlim(time_values)


def _public_series(
    series: dict[str, Any],
    xlim: list[float],
    max_points: int = 12000,
    preserve_mask: list[Any] | None = None,
) -> dict[str, Any]:
    """Trim public arrays to the wider saved data window, preserving signal."""
    raw_time = list(series.get("time", []) or [])
    n_raw = len(raw_time)
    if not n_raw:
        return dict(series)
    lo, hi = xlim
    indices = [index for index, value in enumerate(raw_time) if _finite(value) is not None and lo <= float(value) <= hi]
    if not indices:
        nearest = min(range(n_raw), key=lambda index: abs(float(raw_time[index]) - (lo + hi) / 2.0))
        indices = [nearest]
    if len(indices) > max_points:
        signal_mask = list(preserve_mask or [])
        selected = set()
        denominator = max_points - 1
        for position in range(max_points):
            selected.add(indices[round(position * (len(indices) - 1) / denominator)])
        selected.update(index for index in indices if index < len(signal_mask) and bool(signal_mask[index]))
        indices = sorted(selected)

    result: dict[str, Any] = {}
    for key, value in series.items():
        if isinstance(value, list) and len(value) == n_raw:
            result[key] = [value[index] for index in indices]
        else:
            result[key] = value
    result["n_saved"] = len(indices)
    result["n_total"] = series.get("n_total", n_raw)
    return result


def _read_events(result_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((result_dir / "events").glob("*/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metadata = payload.get("metadata", {}) or {}
        fit = payload.get("fit", {}) or {}
        refined = fit.get("refined", {}) or {}
        selected = fit.get("selected_physical") or refined
        scan = payload.get("template_scan", {}) or {}
        seed = scan.get("seed", {}) or {}
        features = payload.get("features", {}) or {}
        anomaly_candidates = _anomaly_candidate_rows(payload)
        best_candidate = anomaly_candidates[0] if anomaly_candidates else {}
        physical = payload.get("physical_fallback", {}) or {}
        physical_result = physical.get("result", {}) or {}
        timescale = None
        if best_candidate:
            timescale = best_candidate.get("timescale")
        if timescale is None:
            start = _finite(best_candidate.get("t_start"))
            end = _finite(best_candidate.get("t_end"))
            if start is not None and end is not None:
                timescale = end - start
        score = _finite(seed.get("score"))
        if score is None:
            score = _finite(seed.get("dchi2"))
        event = str(metadata.get("event", path.stem))
        tier = str(metadata.get("tier", path.parent.name))
        rows.append(
            {
                "payload": payload,
                "source": path,
                "tier": tier,
                "event": event,
                "score": score,
                "timescale": timescale,
                "n_peaks": int(features.get("n_peaks", 0) or 0),
                "n_dips": int(features.get("n_dips", 0) or 0),
                "n_candidates": len(anomaly_candidates),
                "has_anomaly_candidate": bool(anomaly_candidates),
                "best_candidate_time": best_candidate.get("t_center"),
                "best_candidate_max_abs_z": best_candidate.get("max_abs_z"),
                "chi2_dof": _finite(refined.get("chi2_dof")),
                "stage": "physical" if physical.get("accepted") else "scan",
                "model_kind": str(selected.get("model_kind", "pspl")),
                "physical_effect": physical_result.get("effect"),
                "physical_accepted": bool(physical.get("accepted")),
                "json": f"planet_signal_data/{event}.json",
            }
        )
    rows.sort(
        key=lambda row: (
            -(row["score"] if row["score"] is not None else float("-inf")),
            row["event"],
        )
    )
    return rows


def _roman_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = row["payload"]
    raw_series = payload.get("series", {}) or {}
    raw_time_values = list(raw_series.get("time", []))
    display_xlim = _event_xlim(payload, raw_time_values)
    saved_xlim = _data_xlim(display_xlim, raw_time_values)
    raw_display_mask = list(raw_series.get("display_signal_mask", []) or [])
    if len(raw_display_mask) != len(raw_time_values):
        raw_display_mask = [0] * len(raw_time_values)
    series = _public_series(
        raw_series,
        saved_xlim,
        preserve_mask=raw_display_mask,
    )
    time_values = list(series.get("time", []))
    signal_mask = list(series.get("display_signal_mask", []) or [])
    if len(signal_mask) != len(time_values):
        signal_mask = [0] * len(time_values)
        series["display_signal_mask"] = signal_mask
    fit_mask = list(series.get("fit_exclusion_mask", []) or [])
    if len(fit_mask) != len(time_values):
        series["fit_exclusion_mask"] = list(signal_mask)
    physical = payload.get("physical_fallback", {}) or {}
    fit = payload.get("fit", {}) or {}
    selected_fit = fit.get("selected_physical") or fit.get("refined") or fit.get("initial") or {}
    features = payload.get("features", {}) or {}
    template_free = payload.get("template_free", {}) or {}
    anomaly_candidates = _anomaly_candidate_rows(payload)
    metadata = payload.get("metadata", {}) or {}
    result = physical.get("result") or {}
    source_plot = payload.get("plot", {}) or {}
    model_curve = source_plot.get("model_curve", {}) or {}
    if not (
        isinstance(model_curve.get("time"), list)
        and isinstance(model_curve.get("flux"), list)
        and len(model_curve["time"]) == len(model_curve["flux"])
    ):
        # Old artifacts are rebuilt without silently turning observation
        # samples into a model line.  Newly processed events always carry the
        # adaptive curve generated from the adopted fit.
        model_curve = {"time": [], "flux": []}
    return {
        "event": row["event"],
        "tier": row["tier"],
        "filter": FILT,
        "score": row["score"],
        "pipeline": {
            "stage": row["stage"],
            "template": "roman_simu/anomaly_finder_result",
            "template_scan_visible": False,
        },
        "fit": selected_fit,
        "features": features,
        "has_anomaly_candidate": bool(anomaly_candidates),
        "best_anomaly_candidate": (
            anomaly_candidates[0] if anomaly_candidates else None
        ),
        "anomaly_candidates": anomaly_candidates,
        "candidates": anomaly_candidates,
        "flat_baseline": {"use_flat_baseline": False},
        "physical": {
            "accepted": bool(physical.get("accepted")),
            "effect": result.get("effect"),
            "reason_codes": physical.get("reason_codes", []),
        },
        "template_free": template_free,
        "metadata": metadata,
        "window": {
            "n_total": int(raw_series.get("n_total", len(raw_time_values)) or len(raw_time_values)),
            "n_saved": len(time_values),
            "t_min": saved_xlim[0],
            "t_max": saved_xlim[1],
        },
        "series": series,
        "plot": {
            # The main panel starts at the canonical event scale.  Its JSON
            # data is intentionally saved on the wider saved_xlim so that
            # Plotly zooming can reveal the surrounding baseline and any
            # signal points just outside the first view.
            "peak_xlim": display_xlim,
            "saved_xlim": saved_xlim,
            "signal_xlim": _xlim(time_values, signal_mask)
            if any(bool(value) for value in signal_mask)
            else display_xlim,
            "model_curve": {
                "time": list(model_curve.get("time", [])),
                "flux": list(model_curve.get("flux", [])),
            },
            "signal_model_curve": {
                "time": list(model_curve.get("time", [])),
                "flux": list(model_curve.get("flux", [])),
            },
        },
    }


def _existing_manifest(out_dir: Path) -> dict[str, dict[str, Any]]:
    path = out_dir / "manifest.json"
    if not path.exists():
        return {}
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        str(row.get("event", "")): row
        for row in values
        if isinstance(row, dict) and row.get("event")
    }


def _page(title: str, body: str, scripts: dict[str, str], event_page: bool = False) -> str:
    if event_page:
        js = (
            '<script src="../assets/plotly-1.58.5.min.js"></script>'
            f'<script>{scripts["EVENT_JS"]}</script>'
            f'<script>{scripts["FEATURE_EVENT_JS"]}</script>'
            f'<script>{scripts["JS"]}</script>'
        )
    else:
        js = f'<script>{scripts["JS"]}</script>'
    return (
        '<!DOCTYPE html>\n<html lang="en">\n'
        f'<head><meta charset="UTF-8"><title>{_esc(title)}</title>'
        f'<style>{scripts["CSS"]}</style></head>\n'
        f'<body><div class="page"><div class="build-version">build RGES {datetime.now(timezone.utc).strftime("v%Y.%m.%d.%H%M")}</div>{body}</div>'
        f'{js}</body>\n</html>\n'
    )


def _stats_block(title: str, rows: list[tuple[str, Any]], open_block: bool = False) -> str:
    def shown(value: Any) -> Any:
        if isinstance(value, bool):
            return str(value)
        number = _finite(value)
        return _fmt(number) if number is not None else value

    cells = "".join(
        f'<tr><th>{_esc(key)}</th><td>{_esc(shown(value))}</td></tr>'
        for key, value in rows
        if value is not None and value != ""
    )
    opened = " open" if open_block else ""
    return f'<details class="stats-block"{opened}><summary>{_esc(title)}</summary><table class="stats">{cells}</table></details>'


def _event_stats(row: dict[str, Any]) -> str:
    payload = row["payload"]
    free = payload.get("template_free", {}) or {}
    best_free = free.get("best", {}) or {}
    anomaly_candidates = _anomaly_candidate_rows(payload)
    best_anomaly = anomaly_candidates[0] if anomaly_candidates else {}
    fit = payload.get("fit", {}) or {}
    selected = fit.get("selected_physical") or fit.get("refined") or fit.get("initial") or {}
    params = selected.get("params", {}) or {}
    physical = payload.get("physical_fallback", {}) or {}
    physical_result = physical.get("result", {}) or {}
    features = payload.get("features", {}) or {}
    rows: list[str] = []

    rows.append(_stats_block("Anomaly candidates", [
        ("has candidate", bool(anomaly_candidates)),
        ("n_candidates", len(anomaly_candidates)),
        ("kind", best_anomaly.get("kind")),
        ("t_center", best_anomaly.get("t_center")),
        ("t_start", best_anomaly.get("t_start")),
        ("t_end", best_anomaly.get("t_end")),
        ("timescale", best_anomaly.get("timescale")),
        ("max|z|", best_anomaly.get("max_abs_z")),
        ("signed z", best_anomaly.get("signed_z")),
        ("chi2", best_anomaly.get("chi2")),
        ("sources", ", ".join(best_anomaly.get("sources", []) or [])),
        ("fit excluded", best_anomaly.get("fit_excluded")),
    ], open_block=True))
    rows.append(_stats_block("Template-free", [
        ("t0", selected.get("params", {}).get("t0")),
        ("tE", selected.get("params", {}).get("tE")),
        ("u0", selected.get("params", {}).get("u0")),
        ("chi2/dof", best_free.get("reduced_chi2")),
        ("n_candidates", free.get("n_candidates")),
        ("kind", best_free.get("kind")),
        ("t_start", best_free.get("t_start")),
        ("t_end", best_free.get("t_end")),
        ("n_points", best_free.get("n_points")),
        ("chi2", best_free.get("chi2")),
        ("chi2/n", best_free.get("reduced_chi2")),
        ("max|z|", best_free.get("max_abs_z")),
    ]))
    pi_en = _finite(params.get("piEN", 0.0)) or 0.0
    pi_ee = _finite(params.get("piEE", 0.0)) or 0.0
    rows.append(_stats_block("Planet signal", [
        ("model", selected.get("model_kind")),
        ("BIC", selected.get("bic")),
        ("score", row.get("score")),
        *[(str(key), value) for key, value in params.items()],
        ("chi2/dof", selected.get("chi2_dof")),
        ("piE", math.hypot(pi_en, pi_ee)),
        ("fs", selected.get("fs")),
        ("fb", selected.get("fb")),
        ("n_candidates", len(anomaly_candidates)),
        ("n_peaks", features.get("n_peaks")),
        ("n_dips", features.get("n_dips")),
        ("signal_points", sum(bool(v) for v in payload.get("series", {}).get("display_signal_mask", []))),
    ]))
    rows.append(_stats_block("Fallback / adopted model", [
        ("accepted", physical.get("accepted")),
        ("effect", physical_result.get("effect")),
        ("bic improvement", physical_result.get("bic_improvement")),
        ("reason codes", ", ".join(physical.get("reason_codes", []) or [])),
    ], open_block=bool(physical.get("accepted"))))
    items = features.get("items", []) or []
    strongest = items[0] if items else {}
    rows.append(_stats_block("Peak/dip measurements", [
        ("n_peaks", features.get("n_peaks", 0)),
        ("n_dips", features.get("n_dips", 0)),
        ("strongest kind", strongest.get("kind")),
        ("strongest time", strongest.get("time")),
        ("strongest timescale", strongest.get("timescale")),
        ("strongest strength", strongest.get("strength")),
    ]))
    return '<div class="stats-group">' + "".join(rows) + "</div>"


def _make_template_free_zoom(payload: dict[str, Any], target: Path) -> bool:
    """Make the extra candidate-window figure for older JSON artifacts."""
    series = payload.get("series", {}) or {}
    time_values = [_finite(value) for value in series.get("time", [])]
    residual = [_finite(value) for value in series.get("residual", [])]
    ferr = [_finite(value) for value in series.get("ferr", [])]
    time_values = [value for value in time_values]
    if not time_values or len(time_values) != len(residual) or len(residual) != len(ferr):
        return False
    best = (payload.get("template_free", {}) or {}).get("best") or {}
    start = _finite(best.get("t_start"))
    end = _finite(best.get("t_end"))
    if start is None or end is None:
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    t = np.asarray(time_values, dtype=float)
    z = np.asarray(
        [r / max(float(e), 1e-30) for r, e in zip(residual, ferr)], dtype=float
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, z, ".", ms=3, alpha=0.6, zorder=1)
    ax.axhline(0.0, lw=1, c="0.5", zorder=0)
    ax.axvspan(start, end, color="C3", alpha=0.25, label="anomaly candidate", zorder=2)
    center = _finite(best.get("t_center"))
    if center is not None:
        ax.axvline(center, color="C3", lw=1, zorder=3)
    pad = 5.0
    ax.set_xlim(start - pad, end + pad)
    ax.set_xlabel("time")
    ax.set_ylabel("residual / error")
    ax.set_title(
        f"anomaly candidate: chi2={_fmt(best.get('chi2'))}, "
        f"chi2/n={_fmt(best.get('reduced_chi2'))}, n={best.get('n_points', '')}"
    )
    ax.minorticks_on()
    ax.legend()
    fig.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return True


def _event_page(row: dict[str, Any], manifest: list[dict[str, Any]], scripts: dict[str, str], out_dir: Path) -> None:
    event = row["event"]
    ordered = [item["event"] for item in manifest]
    position = ordered.index(event)
    previous = ordered[position - 1]
    following = ordered[(position + 1) % len(ordered)]
    payload_path = out_dir / "planet_signal_data" / f"{event}.json"
    metadata = row["payload"].get("metadata", {}) or {}
    plots = row["payload"].get("plots", {}) or {}
    figure_names: list[tuple[str, Path, str]] = []
    for key, label, subdir in (
        ("template_free", "Template-free Result", "template_free_figures"),
        ("template_free_zoom", "Template-free Result (zoomed)", "template_free_figures"),
    ):
        source_name = plots.get(key)
        source = row["source"].parents[2] / str(source_name) if source_name else Path("/nonexistent")
        if source_name and not source.exists():
            source = row["source"].parents[2] / "figures" / row["tier"] / Path(str(source_name)).name
        if not source.exists() and key == "template_free_zoom":
            source = out_dir / "template_free_figures" / f"{event}_template_free_zoom.png"
            _make_template_free_zoom(row["payload"], source)
        if source.exists():
            target = out_dir / subdir / f"{event}_{key}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
            figure_names.append((label, target, subdir))

    free_figs = "".join(
        f'<figure><figcaption>{_esc(label)}</figcaption><img src="../{subdir}/{_esc(path.name)}" alt="{_esc(label)}"></figure>'
        for label, path, subdir in figure_names if subdir == "template_free_figures"
    )
    nav = (
        '<div class="event-nav">'
        '<a href="../index.html">&#x2302;&nbsp;Index</a>'
        f'<a href="{_esc(previous)}.html">&#x2190;&nbsp;{_esc(previous)}</a>'
        f'<span>{_esc(event)}</span>'
        f'<a href="{_esc(following)}.html">{_esc(following)}&nbsp;&#x2192;</a>'
        '</div>'
    )
    series = row["payload"].get("series", {}) or {}
    body = (
        nav
        + f'<h1 class="ename">{_esc(event)}</h1>'
        + f'<p class="emeta">Tier&nbsp;{_esc(metadata.get("tier", row["tier"]))}&nbsp;&nbsp;|&nbsp;&nbsp;'
        + f'Filter&nbsp;{_esc(metadata.get("filter", FILT))}&nbsp;&nbsp;|&nbsp;&nbsp;'
        + f'Points&nbsp;{_esc(metadata.get("n_valid", len(series.get("time", []))))}</p>'
        + '<section class="interactive-panel" id="planet-signal-root" '
        + f'data-json="../{_esc(payload_path.relative_to(out_dir).as_posix())}" data-feature-overlay="true">'
        + '<h2>Planet Signal</h2><div id="planet-plot-status" class="plot-status">loading planet signal JSON</div>'
        + '<div id="planet-plot" class="planet-plot"></div></section>'
        + '<section class="interactive-panel" id="planet-feature-root" '
        + f'data-signal-json="../{_esc(payload_path.relative_to(out_dir).as_posix())}">'
        + '<h2>Residual and Finder Peaks/Dips</h2>'
        + '<div id="feature-plot-status" class="plot-status">loading adopted-model residual and Finder peak/dip measurements</div>'
        + '<div id="feature-plot" class="anomaly-plot"></div>'
        + '<div class="hint">Residual is always shown. Red marks positive Finder peaks and green marks Finder dips; shaded widths are their direct threshold-crossing timescales.</div>'
        + '<div id="feature-table" class="mini-table"></div></section>'
        + _event_stats(row)
        + f'<p class="planet-signal-links"><a href="../{_esc(payload_path.relative_to(out_dir).as_posix())}">planet signal JSON</a></p>'
        + '<h2 class="section-title">Template-free</h2>'
        + f'<div class="images">{free_figs or "<p>No template-free figure.</p>"}</div>'
    )
    (out_dir / "events").mkdir(parents=True, exist_ok=True)
    (out_dir / "events" / f"{event}.html").write_text(
        _page(event, body, scripts, event_page=True), encoding="utf-8"
    )


def _index_page(manifest: list[dict[str, Any]], scripts: dict[str, str], out_dir: Path) -> None:
    rows = []
    for row in manifest:
        rows.append(
            f'<tr><td class="mono"><a href="events/{_esc(row["event"])}.html">{_esc(row["event"])}</a></td>'
            + _num_cell(row.get("score"))
            + _num_cell(row.get("timescale"))
            + _num_cell(row.get("n_candidates"))
            + _num_cell(row.get("best_candidate_time"))
            + _num_cell(row.get("best_candidate_max_abs_z"))
            + f'<td>{_esc(row.get("model_kind", ""))}</td>'
            + f'<td>{_esc(row.get("stage", ""))}</td>'
            + f'<td>{_esc(row.get("tier", ""))}</td></tr>'
        )
    body = (
        '<div class="topbar"><p class="crumbs"><a href="/ou-moa/index.html">main</a></p>'
        '<h1>Roman Simulation &mdash; Anomaly Finder Results</h1>'
        f'<p class="sub">RGES {FILT} &mdash; {len(manifest)} events &mdash; click column headers to sort</p></div>'
        '<div class="search-bar"><input id="q" type="text" placeholder="Filter by event name&hellip;"></div>'
        '<table id="etable"><thead><tr><th>Event</th><th>score</th><th>timescale</th><th>candidates</th><th>best time</th><th>max|z|</th><th>model</th><th>stage</th><th>tier</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )
    (out_dir / "index.html").write_text(
        _page("Roman Simu — Anomaly Finder Results", body, scripts), encoding="utf-8"
    )


def _write_legacy_redirects(out_dir: Path, manifest: list[dict[str, Any]]) -> None:
    """Prevent old nested RGES pages from serving the former broken UI."""
    for row in manifest:
        legacy = out_dir / "events" / row.get("tier", "") / f'{row["event"]}.html'
        if not legacy.exists():
            continue
        target = f"../{row['event']}.html"
        legacy.write_text(
            f'<!doctype html><meta http-equiv="refresh" content="0; url={_esc(target)}">'
            f'<a href="{_esc(target)}">Open { _esc(row["event"]) }</a>',
            encoding="utf-8",
        )


def build_html(
    result_dir: Path,
    out_dir: Path,
    version: str | None = None,
    event_json: Path | None = None,
) -> int:
    result_dir = Path(result_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scripts = _canonical_scripts()
    # Do not leave the copied template-scan plot accessible from the public
    # portal.  The scan remains an internal pipeline stage, not a user-facing
    # result.
    for old_figure in (out_dir / "figures").rglob("*_template_pspl.png") if (out_dir / "figures").exists() else []:
        old_figure.unlink(missing_ok=True)
    all_rows = _read_events(result_dir)
    target = Path(event_json).resolve() if event_json is not None else None
    rows_to_write = [row for row in all_rows if target is None or row["source"].resolve() == target]
    manifest_by_event = _existing_manifest(out_dir) if target is not None else {}
    (out_dir / "planet_signal_data").mkdir(parents=True, exist_ok=True)
    for row in rows_to_write:
        (out_dir / "planet_signal_data" / f'{row["event"]}.json').write_text(
            json.dumps(_roman_payload(row), separators=(",", ":")) + "\n", encoding="utf-8"
        )
        manifest_by_event[row["event"]] = {
            key: value for key, value in row.items() if key not in {"payload", "source"}
        }
        # Keep the payload available under the previous data path as well.
        data_dir = out_dir / "data" / row["tier"]
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / f'{row["event"]}.json').write_text(
            json.dumps(_roman_payload(row), separators=(",", ":")) + "\n", encoding="utf-8"
        )

    manifest = sorted(
        manifest_by_event.values(),
        key=lambda row: (-(row.get("score") if row.get("score") is not None else float("-inf")), row.get("event", "")),
    )
    if target is None:
        manifest = [
            {key: value for key, value in row.items() if key not in {"payload", "source"}}
            for row in all_rows
        ]
        manifest.sort(key=lambda row: (-(row.get("score") if row.get("score") is not None else float("-inf")), row.get("event", "")))

    (out_dir / "assets").mkdir(parents=True, exist_ok=True)
    if PLOTLY_SOURCE.exists():
        shutil.copy2(PLOTLY_SOURCE, out_dir / "assets" / "plotly-1.58.5.min.js")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    _index_page(manifest, scripts, out_dir)

    by_event = {row["event"]: row for row in rows_to_write}
    for manifest_row in manifest:
        row = by_event.get(manifest_row["event"])
        if row is not None:
            _event_page(row, manifest, scripts, out_dir)
    _write_legacy_redirects(out_dir, manifest)

    build_version = version or f"v{datetime.now(timezone.utc).strftime('%Y.%m.%d.%H%M')}"
    (out_dir / "build_info.json").write_text(
        json.dumps(
            {
                "version": build_version,
                "event_count": len(manifest),
                "filter": FILT,
                "ui_template": "roman_simu/anomaly_finder_result",
                "template_scan_visible": False,
                "legacy_template_scan_imported": False,
                "built_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(manifest)} RGES events to {out_dir}")
    return len(manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PORTAL_OUTPUT)
    parser.add_argument("--event-json", type=Path, default=None)
    parser.add_argument("--version", default=None)
    args = parser.parse_args()
    build_html(args.result_dir, args.out_dir, args.version, args.event_json)
