from __future__ import annotations

import numpy as np

from .base import ResidualAtom
from ..types import AtomFitResult, SegmentData


class CuspTailAtom(ResidualAtom):
    atom_name = "phenomenological_cusp_tail"
    class_label = "cusp_caustic"

    def fit(self, segment: SegmentData, features: dict[str, float]) -> AtomFitResult:
        t = np.asarray(segment.time, dtype=float)
        if t.size == 0:
            raise ValueError("Cannot fit an empty segment.")

        t_peak = float(features.get("t_peak", t[int(np.argmax(np.abs(segment.residual)))]))
        width = max(float(features.get("fwhm", 0.0)), float(features.get("cadence", 0.0)), 1e-6)
        lo_t, hi_t = float(t[0]), float(t[-1])
        min_width = max(float(features.get("cadence", 0.0)), 1e-6)
        max_width = max(float(features.get("duration", width)), min_width) * 4.0

        best: AtomFitResult | None = None
        for p in tuple(self.config.cusp_tail_powers):
            p = float(p)
            guesses = [
                np.asarray([t_peak, np.log(width), np.log(0.2)], dtype=float),
                np.asarray([t_peak, np.log(max(0.5 * width, min_width)), np.log(0.5)], dtype=float),
                np.asarray([t_peak, np.log(min(2.0 * width, max_width)), np.log(1.0)], dtype=float),
            ]
            fit = self._fit_profiled(
                segment=segment,
                features=features,
                theta0_list=guesses,
                bounds=[
                    (lo_t, hi_t),
                    (np.log(min_width), np.log(max_width)),
                    (np.log(0.03), np.log(10.0)),
                ],
                shape_from_theta=lambda theta, time, power=p: (
                    np.exp(2.0 * theta[2]) + ((time - theta[0]) / np.exp(theta[1])) ** 2
                )
                ** (-0.5 * power),
                params_from_theta=lambda theta, power=p: {
                    "ta": float(theta[0]),
                    "width": float(np.exp(theta[1])),
                    "b": float(np.exp(theta[2])),
                    "p": float(power),
                },
                expected_amplitude_sign=None,
                extra_warnings=self._warnings(features),
            )
            if best is None or fit.bic < best.bic:
                best = fit
        if best is None:
            raise RuntimeError("Cusp-tail fit did not produce any candidate.")
        return best

    @staticmethod
    def _warnings(features: dict[str, float]) -> tuple[str, ...]:
        if float(features.get("edge_sharpness", 0.0)) > 0.5:
            return ("cusp-tail atom fitted to a very sharp feature",)
        return ()
