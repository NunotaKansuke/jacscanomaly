from __future__ import annotations

import numpy as np

from .base import ResidualAtom
from ..types import AtomFitResult, SegmentData


class PositiveBumpAtom(ResidualAtom):
    atom_name = "lorentzian_positive_bump"
    class_label = "major_image_bump"

    def fit(self, segment: SegmentData, features: dict[str, float]) -> AtomFitResult:
        t = np.asarray(segment.time, dtype=float)
        if t.size == 0:
            raise ValueError("Cannot fit an empty segment.")
        t_peak = float(features.get("t_peak", t[int(np.argmax(segment.residual))]))
        width = max(float(features.get("fwhm", 0.0)), float(features.get("cadence", 0.0)), 1e-6)
        lo_t, hi_t = float(t[0]), float(t[-1])
        min_width = max(float(features.get("cadence", 0.0)), 1e-6)
        max_width = max(float(features.get("duration", width)), min_width) * 3.0

        guesses = [
            np.asarray([t_peak, np.log(width)], dtype=float),
            np.asarray([t_peak, np.log(max(0.5 * width, min_width))], dtype=float),
            np.asarray([t_peak, np.log(min(2.0 * width, max_width))], dtype=float),
        ]

        return self._fit_profiled(
            segment=segment,
            features=features,
            theta0_list=guesses,
            bounds=[(lo_t, hi_t), (np.log(min_width), np.log(max_width))],
            shape_from_theta=lambda theta, time: 1.0 / (1.0 + ((time - theta[0]) / np.exp(theta[1])) ** 2),
            params_from_theta=lambda theta: {
                "t_peak": float(theta[0]),
                "width": float(np.exp(theta[1])),
                "nu": 1.0,
            },
            expected_amplitude_sign=1.0,
        )


class NegativeDipAtom(ResidualAtom):
    atom_name = "lorentzian_negative_dip"
    class_label = "minor_image_dip"

    def fit(self, segment: SegmentData, features: dict[str, float]) -> AtomFitResult:
        t = np.asarray(segment.time, dtype=float)
        if t.size == 0:
            raise ValueError("Cannot fit an empty segment.")
        t_peak = float(features.get("t_peak", t[int(np.argmin(segment.residual))]))
        width = max(float(features.get("fwhm", 0.0)), float(features.get("cadence", 0.0)), 1e-6)
        lo_t, hi_t = float(t[0]), float(t[-1])
        min_width = max(float(features.get("cadence", 0.0)), 1e-6)
        max_width = max(float(features.get("duration", width)), min_width) * 3.0

        guesses = [
            np.asarray([t_peak, np.log(width)], dtype=float),
            np.asarray([t_peak, np.log(max(0.5 * width, min_width))], dtype=float),
            np.asarray([t_peak, np.log(min(2.0 * width, max_width))], dtype=float),
        ]

        return self._fit_profiled(
            segment=segment,
            features=features,
            theta0_list=guesses,
            bounds=[(lo_t, hi_t), (np.log(min_width), np.log(max_width))],
            shape_from_theta=lambda theta, time: -1.0 / (1.0 + ((time - theta[0]) / np.exp(theta[1])) ** 2),
            params_from_theta=lambda theta: {
                "t_peak": float(theta[0]),
                "width": float(np.exp(theta[1])),
                "nu": 1.0,
            },
            expected_amplitude_sign=1.0,
        )
