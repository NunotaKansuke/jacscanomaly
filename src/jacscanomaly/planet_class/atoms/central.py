from __future__ import annotations

import numpy as np

from .base import ResidualAtom
from ..types import AtomFitResult, SegmentData


class CentralPerturbationAtom(ResidualAtom):
    atom_name = "central_symmetric_perturbation"
    class_label = "central_caustic"

    def fit(self, segment: SegmentData, features: dict[str, float]) -> AtomFitResult:
        t = np.asarray(segment.time, dtype=float)
        if t.size == 0:
            raise ValueError("Cannot fit an empty segment.")

        width = max(float(features.get("fwhm", 0.0)), float(features.get("cadence", 0.0)), 1e-6)
        min_width = max(float(features.get("cadence", 0.0)), 1e-6)
        max_width = max(float(features.get("duration", width)), min_width) * 3.0
        t0 = float(segment.pspl.t0)
        lo_t, hi_t = float(t[0]), float(t[-1])
        center0 = float(np.clip(t0, lo_t, hi_t))

        guesses = [
            np.asarray([center0, np.log(width)], dtype=float),
            np.asarray([center0, np.log(max(0.5 * width, min_width))], dtype=float),
            np.asarray([center0, np.log(min(2.0 * width, max_width))], dtype=float),
            np.asarray([float(features.get("t_peak", center0)), np.log(width)], dtype=float),
        ]

        return self._fit_profiled(
            segment=segment,
            features=features,
            theta0_list=guesses,
            bounds=[(lo_t, hi_t), (np.log(min_width), np.log(max_width))],
            shape_from_theta=lambda theta, time: 1.0 / (1.0 + ((time - theta[0]) / np.exp(theta[1])) ** 2),
            params_from_theta=lambda theta: {
                "t_center": float(theta[0]),
                "width": float(np.exp(theta[1])),
                "duration": float(max(2.0 * np.exp(theta[1]), features.get("duration", 0.0))),
                "offset_from_t0": float(theta[0] - t0),
            },
            expected_amplitude_sign=None,
            extra_warnings=self._warnings(segment, features),
        )

    def _warnings(self, segment: SegmentData, features: dict[str, float]) -> tuple[str, ...]:
        window = float(self.config.central_window_factor) * max(
            abs(float(segment.pspl.u0)) * float(segment.pspl.tE),
            float(features.get("fwhm", 0.0)),
            1e-12,
        )
        distance = abs(float(features.get("t_peak", segment.pspl.t0)) - float(segment.pspl.t0))
        if distance > window:
            return ("feature is outside the configured central window",)
        return ()


class CentralDoubleCuspAtom(ResidualAtom):
    atom_name = "central_double_cusp"
    class_label = "central_double_cusp"

    def fit(self, segment: SegmentData, features: dict[str, float]) -> AtomFitResult:
        t = np.asarray(segment.time, dtype=float)
        if t.size == 0:
            raise ValueError("Cannot fit an empty segment.")

        t0 = float(segment.pspl.t0)
        lo_t, hi_t = float(t[0]), float(t[-1])
        center0 = float(np.clip(t0, lo_t, hi_t))
        cadence = max(float(features.get("cadence", 0.0)), 1e-6)
        width = max(float(features.get("fwhm", 0.0)), cadence, 1e-6)
        duration = max(float(features.get("duration", width)), width)
        min_width = max(0.5 * cadence, 1e-6)
        max_width = max(duration, min_width) * 2.0
        min_sep = max(cadence, 1e-6)
        max_sep = max(duration, min_sep) * 1.5

        guesses = [
            np.asarray([center0, np.log(max(0.35 * duration, min_sep)), np.log(max(0.20 * width, min_width)), np.log(max(0.35 * width, min_width)), np.log(1.0), np.log(0.5)]),
            np.asarray([center0, np.log(max(0.50 * duration, min_sep)), np.log(max(0.15 * width, min_width)), np.log(max(0.50 * width, min_width)), np.log(0.7), np.log(0.7)]),
            np.asarray([float(features.get("t_negative_peak", center0)), np.log(max(0.30 * duration, min_sep)), np.log(max(0.25 * width, min_width)), np.log(max(0.60 * width, min_width)), np.log(1.3), np.log(0.4)]),
        ]

        def lorentz(time: np.ndarray, center: float, scale: float) -> np.ndarray:
            return 1.0 / (1.0 + ((time - center) / max(scale, 1e-12)) ** 2)

        def shape(theta: np.ndarray, time: np.ndarray) -> np.ndarray:
            center = float(theta[0])
            sep = float(np.exp(theta[1]))
            rim_width = float(np.exp(theta[2]))
            mid_width = float(np.exp(theta[3]))
            right_ratio = float(np.exp(theta[4]))
            trough_ratio = float(np.exp(theta[5]))
            left = lorentz(time, center - 0.5 * sep, rim_width)
            right = lorentz(time, center + 0.5 * sep, rim_width)
            middle = lorentz(time, center, mid_width)
            return left + right_ratio * right - trough_ratio * middle

        return self._fit_profiled(
            segment=segment,
            features=features,
            theta0_list=guesses,
            bounds=[
                (lo_t, hi_t),
                (np.log(min_sep), np.log(max_sep)),
                (np.log(min_width), np.log(max_width)),
                (np.log(min_width), np.log(max_width)),
                (np.log(0.05), np.log(20.0)),
                (np.log(0.05), np.log(20.0)),
            ],
            shape_from_theta=shape,
            params_from_theta=lambda theta: {
                "t_center": float(theta[0]),
                "t_cusp_1": float(theta[0] - 0.5 * np.exp(theta[1])),
                "t_cusp_2": float(theta[0] + 0.5 * np.exp(theta[1])),
                "cusp_separation": float(np.exp(theta[1])),
                "width": float(np.exp(theta[2])),
                "rim_width": float(np.exp(theta[2])),
                "mid_width": float(np.exp(theta[3])),
                "right_cusp_ratio": float(np.exp(theta[4])),
                "trough_ratio": float(np.exp(theta[5])),
                "duration": float(max(np.exp(theta[1]), 2.0 * np.exp(theta[3]))),
                "offset_from_t0": float(theta[0] - t0),
            },
            expected_amplitude_sign=1.0,
            extra_warnings=self._warnings(segment, features),
        )

    def _warnings(self, segment: SegmentData, features: dict[str, float]) -> tuple[str, ...]:
        window = float(self.config.central_window_factor) * max(
            abs(float(segment.pspl.u0)) * float(segment.pspl.tE),
            float(features.get("fwhm", 0.0)),
            1e-12,
        )
        distance = abs(float(features.get("t_peak", segment.pspl.t0)) - float(segment.pspl.t0))
        if distance > window:
            return ("feature is outside the configured central window",)
        return ()
