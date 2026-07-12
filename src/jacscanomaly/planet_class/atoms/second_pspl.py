from __future__ import annotations

from dataclasses import replace

import numpy as np

from .base import ResidualAtom
from ..pspl import pspl_magnification_from_u
from ..types import AtomFitResult, SegmentData


class SecondPSPLAtom(ResidualAtom):
    atom_name = "second_pspl_like_residual"
    class_label = "second_pspl_like"

    def fit(self, segment: SegmentData, features: dict[str, float]) -> AtomFitResult:
        t = np.asarray(segment.time, dtype=float)
        if t.size == 0:
            raise ValueError("Cannot fit an empty segment.")
        t_peak = float(features.get("t_positive_peak", t[int(np.argmax(segment.residual))]))
        width = max(float(features.get("fwhm", 0.0)), float(features.get("cadence", 0.0)), 1e-6)
        lo_t, hi_t = float(t[0]), float(t[-1])
        min_tE = max(float(features.get("cadence", 0.0)), 1e-5)
        max_tE = max(3.0 * float(segment.pspl.tE), 10.0 * width, min_tE)
        min_u0 = 1e-3
        max_u0 = 10.0

        guesses = []
        for tE2 in (width, max(2.0 * width, min_tE), max(0.5 * segment.pspl.tE, min_tE), segment.pspl.tE):
            for u02 in (0.1, 0.3, 1.0):
                guesses.append(np.asarray([t_peak, np.log(np.clip(tE2, min_tE, max_tE)), np.log(u02)], dtype=float))

        def shape(theta: np.ndarray, time: np.ndarray) -> np.ndarray:
            t0_2 = float(theta[0])
            tE_2 = float(np.exp(theta[1]))
            u0_2 = float(np.exp(theta[2]))
            u = np.sqrt(u0_2 * u0_2 + ((time - t0_2) / tE_2) ** 2)
            return pspl_magnification_from_u(u) - 1.0

        fit = self._fit_profiled(
            segment=segment,
            features=features,
            theta0_list=guesses,
            bounds=[(lo_t, hi_t), (np.log(min_tE), np.log(max_tE)), (np.log(min_u0), np.log(max_u0))],
            shape_from_theta=shape,
            params_from_theta=lambda theta: {
                "t0_2": float(theta[0]),
                "tE_2": float(np.exp(theta[1])),
                "u0_2": float(np.exp(theta[2])),
                "Fs_2_over_Fs_1": float("nan"),
            },
            expected_amplitude_sign=1.0,
        )
        params = dict(fit.params)
        t0_2 = float(params["t0_2"])
        tE_2 = float(params["tE_2"])
        u0_2 = float(params["u0_2"])
        ratio = tE_2 / max(float(segment.pspl.tE), 1e-12)
        q_wide = ratio * ratio
        dx = (t0_2 - float(segment.pspl.t0)) / max(float(segment.pspl.tE), 1e-12)
        dy_offset = u0_2 * np.sqrt(max(q_wide, 0.0))
        params.update(
            {
                "Fs_2_over_Fs_1": float(params.get("amplitude", np.nan)) / max(abs(float(segment.pspl.Fs)), 1e-12),
                "q_flux": float(params.get("amplitude", np.nan)) / max(abs(float(segment.pspl.Fs)), 1e-12),
                "tE_ratio": ratio,
                "q_wide_repeating": q_wide,
                "separation_x": dx,
            }
        )
        for suffix, sign in (("plus", 1.0), ("minus", -1.0)):
            dy = float(segment.pspl.u0) + sign * dy_offset
            params[f"separation_y_{suffix}"] = dy
            params[f"s_{suffix}"] = float(np.hypot(dx, dy))
            params[f"alpha_{suffix}"] = float(np.arctan2(dy, dx))
        errors = dict(fit.param_errors or {})
        if "amplitude" in errors:
            errors["Fs_2_over_Fs_1"] = errors["amplitude"] / max(abs(float(segment.pspl.Fs)), 1e-12)
            errors["q_flux"] = errors["Fs_2_over_Fs_1"]
        return replace(fit, params=params, param_errors=errors or None)
