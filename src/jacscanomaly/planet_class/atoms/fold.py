from __future__ import annotations

import numpy as np

from .base import ResidualAtom
from ..fold_kernel import fold_g0, fold_limb_darkened
from ..types import AtomFitResult, SegmentData


class FoldCausticAtom(ResidualAtom):
    atom_name = "straight_fold_caustic"
    class_label = "fold_caustic"

    def fit(self, segment: SegmentData, features: dict[str, float]) -> AtomFitResult:
        t = np.asarray(segment.time, dtype=float)
        if t.size == 0:
            raise ValueError("Cannot fit an empty segment.")

        t_peak = float(features.get("t_peak", t[int(np.argmax(np.abs(segment.residual)))]))
        width = max(float(features.get("fwhm", 0.0)), float(features.get("cadence", 0.0)), 1e-6)
        lo_t, hi_t = float(t[0]), float(t[-1])
        min_tstar = max(0.5 * float(features.get("cadence", 0.0)), 1e-6)
        max_tstar = max(float(features.get("duration", width)), min_tstar) * 2.0

        best: AtomFitResult | None = None
        for entry_exit_sign in (1.0, -1.0):
            guesses = [
                np.asarray([t_peak, np.log(width)], dtype=float),
                np.asarray([t_peak, np.log(max(0.5 * width, min_tstar))], dtype=float),
                np.asarray([t_peak, np.log(min(2.0 * width, max_tstar))], dtype=float),
            ]
            fit = self._fit_profiled(
                segment=segment,
                features=features,
                theta0_list=guesses,
                bounds=[(lo_t, hi_t), (np.log(min_tstar), np.log(max_tstar))],
                shape_from_theta=lambda theta, time, sign=entry_exit_sign: fold_g0(
                    sign * (time - theta[0]) / np.exp(theta[1])
                ),
                params_from_theta=lambda theta, sign=entry_exit_sign: {
                    "tc": float(theta[0]),
                    "tstar": float(np.exp(theta[1])),
                    "entry_exit_sign": float(sign),
                    "rho_over_sinalpha": float(np.exp(theta[1]) / max(segment.pspl.tE, 1e-12)),
                },
                expected_amplitude_sign=None,
                extra_warnings=self._warnings(features, min_tstar=min_tstar, max_tstar=max_tstar),
            )
            if best is None or fit.bic < best.bic:
                best = fit
        if best is None:
            raise RuntimeError("Fold fit did not produce any candidate.")
        return best

    @staticmethod
    def _warnings(features: dict[str, float], *, min_tstar: float, max_tstar: float) -> tuple[str, ...]:
        warnings = []
        if float(features.get("edge_sharpness", 0.0)) < 0.05:
            warnings.append("fold atom fitted to a smooth feature")
        if min_tstar <= 0.0 or max_tstar <= min_tstar:
            warnings.append("invalid fold tstar bounds")
        return tuple(warnings)


class CurvedFoldCausticAtom(ResidualAtom):
    atom_name = "curved_fold_caustic"
    class_label = "curved_fold_caustic"

    def fit(self, segment: SegmentData, features: dict[str, float]) -> AtomFitResult:
        t = np.asarray(segment.time, dtype=float)
        if t.size == 0:
            raise ValueError("Cannot fit an empty segment.")

        t_peak = float(features.get("t_peak", t[int(np.argmax(np.abs(segment.residual)))]))
        width = max(float(features.get("fwhm", 0.0)), float(features.get("cadence", 0.0)), 1e-6)
        lo_t, hi_t = float(t[0]), float(t[-1])
        min_tstar = max(0.5 * float(features.get("cadence", 0.0)), 1e-6)
        max_tstar = max(float(features.get("duration", width)), min_tstar) * 2.0

        best: AtomFitResult | None = None
        for entry_exit_sign in (1.0, -1.0):
            shifted_left = float(np.clip(t_peak - width, lo_t, hi_t))
            shifted_right = float(np.clip(t_peak + width, lo_t, hi_t))
            guesses = [
                np.asarray([t_peak, np.log(width), 0.0], dtype=float),
                np.asarray([t_peak, np.log(max(0.5 * width, min_tstar)), 0.2], dtype=float),
                np.asarray([t_peak, np.log(min(2.0 * width, max_tstar)), -0.2], dtype=float),
                np.asarray([shifted_left, np.log(width), 0.5], dtype=float),
                np.asarray([shifted_left, np.log(width), -0.5], dtype=float),
                np.asarray([shifted_right, np.log(width), 0.5], dtype=float),
                np.asarray([shifted_right, np.log(width), -0.5], dtype=float),
            ]
            fit = self._fit_profiled(
                segment=segment,
                features=features,
                theta0_list=guesses,
                bounds=[(lo_t, hi_t), (np.log(min_tstar), np.log(max_tstar)), (-2.0, 2.0)],
                shape_from_theta=lambda theta, time, sign=entry_exit_sign: fold_g0(
                    sign
                    * (
                        (time - theta[0]) / np.exp(theta[1])
                        + theta[2] * ((time - theta[0]) / np.exp(theta[1])) ** 2
                    )
                ),
                params_from_theta=lambda theta, sign=entry_exit_sign: {
                    "tc": float(theta[0]),
                    "tstar": float(np.exp(theta[1])),
                    "q_curv": float(theta[2]),
                    "entry_exit_sign": float(sign),
                    "rho_over_sinalpha": float(np.exp(theta[1]) / max(segment.pspl.tE, 1e-12)),
                },
                expected_amplitude_sign=None,
                extra_warnings=FoldCausticAtom._warnings(features, min_tstar=min_tstar, max_tstar=max_tstar),
            )
            if best is None or fit.bic < best.bic:
                best = fit
        if best is None:
            raise RuntimeError("Curved fold fit did not produce any candidate.")
        return best


class GrazingFoldCausticAtom(ResidualAtom):
    atom_name = "grazing_fold_caustic"
    class_label = "grazing_fold_caustic"

    def fit(self, segment: SegmentData, features: dict[str, float]) -> AtomFitResult:
        t = np.asarray(segment.time, dtype=float)
        if t.size == 0:
            raise ValueError("Cannot fit an empty segment.")
        t_peak = float(features.get("t_peak", t[int(np.argmax(np.abs(segment.residual)))]))
        width = max(float(features.get("fwhm", 0.0)), float(features.get("cadence", 0.0)), 1e-6)
        lo_t, hi_t = float(t[0]), float(t[-1])
        min_width = max(float(features.get("cadence", 0.0)), 1e-6)
        max_width = max(float(features.get("duration", width)), min_width) * 3.0
        guesses = [
            np.asarray([t_peak, -0.8, np.log(width), 0.0], dtype=float),
            np.asarray([t_peak, -0.3, np.log(width), 0.3], dtype=float),
            np.asarray([t_peak, 0.2, np.log(width), -0.3], dtype=float),
        ]
        return self._fit_profiled(
            segment=segment,
            features=features,
            theta0_list=guesses,
            bounds=[(lo_t, hi_t), (-1.5, 3.0), (np.log(min_width), np.log(max_width)), (-3.0, 3.0)],
            shape_from_theta=lambda theta, time: fold_g0(
                theta[1] + (time - theta[0]) / np.exp(theta[2]) + theta[3] * ((time - theta[0]) / np.exp(theta[2])) ** 2
            ),
            params_from_theta=lambda theta: {
                "ta": float(theta[0]),
                "z0": float(theta[1]),
                "width": float(np.exp(theta[2])),
                "q_curv": float(theta[3]),
            },
            expected_amplitude_sign=None,
            extra_warnings=(),
        )


class LimbDarkenedFoldCausticAtom(ResidualAtom):
    atom_name = "limb_darkened_fold_caustic"
    class_label = "limb_darkened_fold_caustic"

    def fit(self, segment: SegmentData, features: dict[str, float]) -> AtomFitResult:
        t = np.asarray(segment.time, dtype=float)
        if t.size == 0:
            raise ValueError("Cannot fit an empty segment.")
        t_peak = float(features.get("t_peak", t[int(np.argmax(np.abs(segment.residual)))]))
        width = max(float(features.get("fwhm", 0.0)), float(features.get("cadence", 0.0)), 1e-6)
        lo_t, hi_t = float(t[0]), float(t[-1])
        min_tstar = max(0.5 * float(features.get("cadence", 0.0)), 1e-6)
        max_tstar = max(float(features.get("duration", width)), min_tstar) * 2.0
        best: AtomFitResult | None = None
        for entry_exit_sign in (1.0, -1.0):
            guesses = [
                np.asarray([t_peak, np.log(width), 0.0], dtype=float),
                np.asarray([t_peak, np.log(width), 0.4], dtype=float),
                np.asarray([t_peak, np.log(max(0.5 * width, min_tstar)), 0.7], dtype=float),
            ]
            fit = self._fit_profiled(
                segment=segment,
                features=features,
                theta0_list=guesses,
                bounds=[(lo_t, hi_t), (np.log(min_tstar), np.log(max_tstar)), (0.0, 1.0)],
                shape_from_theta=lambda theta, time, sign=entry_exit_sign: fold_limb_darkened(
                    sign * (time - theta[0]) / np.exp(theta[1]),
                    gamma=float(theta[2]),
                ),
                params_from_theta=lambda theta, sign=entry_exit_sign: {
                    "tc": float(theta[0]),
                    "tstar": float(np.exp(theta[1])),
                    "Gamma": float(theta[2]),
                    "entry_exit_sign": float(sign),
                    "rho_over_sinalpha": float(np.exp(theta[1]) / max(segment.pspl.tE, 1e-12)),
                },
                expected_amplitude_sign=None,
                extra_warnings=FoldCausticAtom._warnings(features, min_tstar=min_tstar, max_tstar=max_tstar),
            )
            if best is None or fit.bic < best.bic:
                best = fit
        if best is None:
            raise RuntimeError("Limb-darkened fold fit did not produce any candidate.")
        return best


class TwoFoldCausticAtom(ResidualAtom):
    atom_name = "two_fold_caustic"
    class_label = "two_fold_caustic"

    def fit(self, segment: SegmentData, features: dict[str, float]) -> AtomFitResult:
        t = np.asarray(segment.time, dtype=float)
        if t.size == 0:
            raise ValueError("Cannot fit an empty segment.")
        width = max(float(features.get("fwhm", 0.0)), float(features.get("cadence", 0.0)), 1e-6)
        t_peak = float(features.get("t_peak", t[int(np.argmax(np.abs(segment.residual)))]))
        lo_t, hi_t = float(t[0]), float(t[-1])
        min_tstar = max(0.5 * float(features.get("cadence", 0.0)), 1e-6)
        max_tstar = max(float(features.get("duration", width)), min_tstar) * 2.0
        sep0 = max(0.5 * width, min_tstar)
        best: AtomFitResult | None = None
        for signs in ((1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (-1.0, -1.0)):
            guesses = [
                np.asarray([t_peak - sep0, t_peak + sep0, np.log(width)], dtype=float),
                np.asarray([t_peak - width, t_peak + width, np.log(max(0.5 * width, min_tstar))], dtype=float),
            ]
            fit = self._fit_profiled(
                segment=segment,
                features=features,
                theta0_list=guesses,
                bounds=[(lo_t, hi_t), (lo_t, hi_t), (np.log(min_tstar), np.log(max_tstar))],
                shape_from_theta=lambda theta, time, ss=signs: np.column_stack(
                    (
                        fold_g0(ss[0] * (time - min(theta[0], theta[1])) / np.exp(theta[2])),
                        fold_g0(ss[1] * (time - max(theta[0], theta[1])) / np.exp(theta[2])),
                    )
                ),
                params_from_theta=lambda theta, ss=signs: {
                    "tc1": float(min(theta[0], theta[1])),
                    "tc2": float(max(theta[0], theta[1])),
                    "tstar": float(np.exp(theta[2])),
                    "entry_exit_sign_1": float(ss[0]),
                    "entry_exit_sign_2": float(ss[1]),
                    "rho_over_sinalpha": float(np.exp(theta[2]) / max(segment.pspl.tE, 1e-12)),
                },
                expected_amplitude_sign=None,
                extra_warnings=(),
            )
            if best is None or fit.bic < best.bic:
                best = fit
        if best is None:
            raise RuntimeError("Two-fold fit did not produce any candidate.")
        return best
