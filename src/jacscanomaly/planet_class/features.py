from __future__ import annotations

import numpy as np

from .types import SegmentData
from .pspl import u_abs


def segment_features(segment: SegmentData) -> dict[str, float]:
    t = np.asarray(segment.time, dtype=float)
    residual = np.asarray(segment.residual, dtype=float)
    ferr = np.maximum(np.asarray(segment.ferr, dtype=float), 1e-12)
    z = residual / ferr
    abs_z = np.abs(z)
    if t.size == 0:
        return {}

    peak_index = int(np.argmax(abs_z))
    t_peak = float(t[peak_index])
    peak_z = float(z[peak_index])
    pos_index = int(np.argmax(z))
    neg_index = int(np.argmin(z))
    t_positive_peak = float(t[pos_index])
    t_negative_peak = float(t[neg_index])
    positive_peak_z = float(z[pos_index])
    negative_peak_z = float(z[neg_index])
    duration = float(t[-1] - t[0]) if t.size > 1 else 0.0
    chi2 = float(np.sum(z * z))
    positive_chi2 = float(np.sum(np.where(z > 0.0, z * z, 0.0)))
    negative_chi2 = float(np.sum(np.where(z < 0.0, z * z, 0.0)))
    sign = 1.0 if positive_chi2 >= negative_chi2 else -1.0
    cadence = float(np.median(np.diff(t))) if t.size > 1 else 0.0

    half = 0.5 * float(np.max(abs_z))
    above = abs_z >= half
    if np.any(above):
        fwhm = float(t[np.flatnonzero(above)[-1]] - t[np.flatnonzero(above)[0]])
    else:
        fwhm = max(duration, cadence)

    centered = residual - float(np.mean(residual))
    rms = float(np.sqrt(np.mean(centered * centered))) if centered.size else 0.0
    skewness = float(np.mean((centered / rms) ** 3)) if rms > 0.0 else 0.0
    kurtosis = float(np.mean((centered / rms) ** 4)) if rms > 0.0 else 0.0
    edge_sharpness = (
        float(np.max(np.abs(np.diff(residual)))) / max(float(np.max(np.abs(residual))), 1e-12)
        if residual.size > 1
        else 0.0
    )

    return {
        "t_peak": t_peak,
        "peak_z": peak_z,
        "t_positive_peak": t_positive_peak,
        "positive_peak_z": positive_peak_z,
        "t_negative_peak": t_negative_peak,
        "negative_peak_z": negative_peak_z,
        "sign": sign,
        "duration": duration,
        "fwhm": max(fwhm, cadence, 0.0),
        "cadence": cadence,
        "n_points": float(t.size),
        "chi2": chi2,
        "positive_chi2": positive_chi2,
        "negative_chi2": negative_chi2,
        "snr": float(np.sqrt(max(chi2, 0.0))),
        "skewness": skewness,
        "kurtosis": kurtosis,
        "edge_sharpness": edge_sharpness,
        "distance_from_pspl_peak": abs(t_peak - segment.pspl.t0) / max(segment.pspl.tE, 1e-12),
        "u_at_peak": float(u_abs(t_peak, segment.pspl)),
    }
