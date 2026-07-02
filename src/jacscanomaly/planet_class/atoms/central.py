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
