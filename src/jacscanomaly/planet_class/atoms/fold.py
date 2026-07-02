from __future__ import annotations

import numpy as np

from .base import ResidualAtom
from ..fold_kernel import fold_g0
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
