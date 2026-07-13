from __future__ import annotations

import numpy as np

from .types import SegmentData


def segment_features(segment: SegmentData) -> dict[str, float]:
    """
    Minimal per-component measurements used for template routing and initial
    guesses: extremum times, signed residual power, duration, FWHM-like
    width, and cadence.
    """
    t = np.asarray(segment.time, dtype=float)
    residual = np.asarray(segment.residual, dtype=float)
    ferr = np.maximum(np.asarray(segment.ferr, dtype=float), 1e-12)
    if t.size == 0:
        return {}
    z = residual / ferr
    abs_z = np.abs(z)

    duration = float(t[-1] - t[0]) if t.size > 1 else 0.0
    cadence = float(np.median(np.diff(t))) if t.size > 1 else 0.0
    chi2 = float(np.sum(z * z))
    positive_chi2 = float(np.sum(np.where(z > 0.0, z * z, 0.0)))
    negative_chi2 = float(np.sum(np.where(z < 0.0, z * z, 0.0)))

    half = 0.5 * float(np.max(abs_z))
    above = np.flatnonzero(abs_z >= half)
    fwhm = float(t[above[-1]] - t[above[0]]) if above.size else max(duration, cadence)

    return {
        "t_peak": float(t[int(np.argmax(abs_z))]),
        "t_positive_peak": float(t[int(np.argmax(z))]),
        "t_negative_peak": float(t[int(np.argmin(z))]),
        "duration": duration,
        "fwhm": max(fwhm, cadence),
        "cadence": cadence,
        "n_points": float(t.size),
        "chi2": chi2,
        "positive_chi2": positive_chi2,
        "negative_chi2": negative_chi2,
        "snr": float(np.sqrt(max(chi2, 0.0))),
    }
