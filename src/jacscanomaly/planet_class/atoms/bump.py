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
        t_peak = float(features.get("t_positive_peak", t[int(np.argmax(segment.residual))]))
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


class PSPLPositiveBumpAtom(ResidualAtom):
    atom_name = "pspl_positive_bump"
    class_label = "major_image_pspl_bump"

    def fit(self, segment: SegmentData, features: dict[str, float]) -> AtomFitResult:
        t = np.asarray(segment.time, dtype=float)
        if t.size == 0:
            raise ValueError("Cannot fit an empty segment.")
        t_peak = float(features.get("t_positive_peak", t[int(np.argmax(segment.residual))]))
        width = max(float(features.get("fwhm", 0.0)), float(features.get("cadence", 0.0)), 1e-6)
        lo_t, hi_t = float(t[0]), float(t[-1])
        min_width = max(float(features.get("cadence", 0.0)), 1e-6)
        max_width = max(float(features.get("duration", width)), min_width) * 5.0
        min_u0 = 1e-3
        max_u0 = 10.0

        guesses = [
            np.asarray([t_peak, np.log(width), np.log(0.5)], dtype=float),
            np.asarray([t_peak, np.log(max(0.5 * width, min_width)), np.log(0.2)], dtype=float),
            np.asarray([t_peak, np.log(min(2.0 * width, max_width)), np.log(1.0)], dtype=float),
        ]

        def shape(theta: np.ndarray, time: np.ndarray) -> np.ndarray:
            t0 = float(theta[0])
            tE = float(np.exp(theta[1]))
            u0 = float(np.exp(theta[2]))
            u = np.sqrt(u0 * u0 + ((time - t0) / max(tE, 1e-12)) ** 2)
            return (u * u + 2.0) / (u * np.sqrt(u * u + 4.0)) - 1.0

        return self._fit_profiled(
            segment=segment,
            features=features,
            theta0_list=guesses,
            bounds=[
                (lo_t, hi_t),
                (np.log(min_width), np.log(max_width)),
                (np.log(min_u0), np.log(max_u0)),
            ],
            shape_from_theta=shape,
            params_from_theta=lambda theta: {
                "t_peak": float(theta[0]),
                "width": float(np.exp(theta[1])),
                "tE_pert": float(np.exp(theta[1])),
                "u0_pert": float(np.exp(theta[2])),
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
        t_peak = float(features.get("t_negative_peak", t[int(np.argmin(segment.residual))]))
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


class MinorImageBoxTroughAtom(ResidualAtom):
    atom_name = "softened_box_negative_trough"
    class_label = "minor_image_box_trough"

    def fit(self, segment: SegmentData, features: dict[str, float]) -> AtomFitResult:
        t = np.asarray(segment.time, dtype=float)
        if t.size == 0:
            raise ValueError("Cannot fit an empty segment.")
        t_peak = float(features.get("t_negative_peak", t[int(np.argmin(segment.residual))]))
        width = max(float(features.get("fwhm", 0.0)), float(features.get("cadence", 0.0)), 1e-6)
        lo_t, hi_t = float(t[0]), float(t[-1])
        cadence = max(float(features.get("cadence", 0.0)), 1e-6)
        min_half_width = max(0.5 * cadence, 1e-6)
        max_half_width = max(float(features.get("duration", width)), min_half_width) * 1.5
        min_edge = max(0.25 * cadence, 1e-6)
        max_edge = max(max_half_width, min_edge)

        guesses = [
            np.asarray([t_peak, np.log(max(0.5 * width, min_half_width)), np.log(max(0.25 * width, min_edge))]),
            np.asarray([t_peak, np.log(max(width, min_half_width)), np.log(min(max(0.15 * width, min_edge), max_edge))]),
            np.asarray([t_peak, np.log(max(0.3 * width, min_half_width)), np.log(min(max(0.1 * width, min_edge), max_edge))]),
        ]

        def sigmoid(x: np.ndarray) -> np.ndarray:
            x = np.clip(x, -60.0, 60.0)
            return 1.0 / (1.0 + np.exp(-x))

        def shape(theta: np.ndarray, time: np.ndarray) -> np.ndarray:
            center = float(theta[0])
            half_width = float(np.exp(theta[1]))
            edge = float(np.exp(theta[2]))
            t1 = center - half_width
            t2 = center + half_width
            return -(sigmoid((time - t1) / edge) - sigmoid((time - t2) / edge))

        return self._fit_profiled(
            segment=segment,
            features=features,
            theta0_list=guesses,
            bounds=[
                (lo_t, hi_t),
                (np.log(min_half_width), np.log(max_half_width)),
                (np.log(min_edge), np.log(max_edge)),
            ],
            shape_from_theta=shape,
            params_from_theta=lambda theta: {
                "t_peak": float(theta[0]),
                "t_start": float(theta[0] - np.exp(theta[1])),
                "t_end": float(theta[0] + np.exp(theta[1])),
                "width": float(2.0 * np.exp(theta[1])),
                "edge_width": float(np.exp(theta[2])),
            },
            expected_amplitude_sign=1.0,
        )
