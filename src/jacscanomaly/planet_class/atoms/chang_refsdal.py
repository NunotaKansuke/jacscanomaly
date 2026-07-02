from __future__ import annotations

import numpy as np

from .base import ResidualAtom
from ..pspl import u_abs, u_vec
from ..types import AtomFitResult, SegmentData


def pspl_image_position(time: np.ndarray, pspl, *, branch: str) -> tuple[np.ndarray, np.ndarray]:
    uv = u_vec(time, pspl)
    u = np.maximum(u_abs(time, pspl), 1e-12)
    ux = uv[0] / u
    uy = uv[1] / u
    r_plus = 0.5 * (np.sqrt(u * u + 4.0) + u)
    r_minus = 1.0 / r_plus
    if branch == "major":
        return r_plus * ux, r_plus * uy
    return -r_minus * ux, -r_minus * uy


class ChangRefsdalPerturbationAtom(ResidualAtom):
    atom_name = "chang_refsdal_local_perturbation"
    class_label = "chang_refsdal"

    def fit(self, segment: SegmentData, features: dict[str, float]) -> AtomFitResult:
        t = np.asarray(segment.time, dtype=float)
        if t.size == 0:
            raise ValueError("Cannot fit an empty segment.")
        t_peak = float(features.get("t_peak", t[int(np.argmax(np.abs(segment.residual)))]))
        width_t = max(float(features.get("fwhm", 0.0)), float(features.get("cadence", 0.0)), 1e-6)
        image_width0 = max(width_t / max(segment.pspl.tE, 1e-12), 1e-3)
        best: AtomFitResult | None = None
        for branch in ("major", "minor"):
            x, y = pspl_image_position(t, segment.pspl, branch=branch)
            x_peak, y_peak = pspl_image_position(np.asarray([t_peak]), segment.pspl, branch=branch)
            guesses = [
                np.asarray([float(x_peak[0]), float(y_peak[0]), np.log(image_width0)], dtype=float),
                np.asarray([float(x_peak[0]), float(y_peak[0]), np.log(2.0 * image_width0)], dtype=float),
            ]
            span = max(float(np.max(np.hypot(x - x_peak[0], y - y_peak[0]))), image_width0, 0.1)
            fit = self._fit_profiled(
                segment=segment,
                features=features,
                theta0_list=guesses,
                bounds=[
                    (float(x_peak[0]) - 2.0 * span, float(x_peak[0]) + 2.0 * span),
                    (float(y_peak[0]) - 2.0 * span, float(y_peak[0]) + 2.0 * span),
                    (np.log(1e-4), np.log(max(5.0 * span, 1e-3))),
                ],
                shape_from_theta=lambda theta, time, br=branch: self._shape(theta, time, segment, branch=br),
                params_from_theta=lambda theta, br=branch: {
                    "image_branch": 1.0 if br == "major" else -1.0,
                    "x_planet": float(theta[0]),
                    "y_planet": float(theta[1]),
                    "image_width": float(np.exp(theta[2])),
                    "gamma_local": float(self._local_shear(theta[0], theta[1])),
                },
                expected_amplitude_sign=None,
                extra_warnings=(),
            )
            if best is None or fit.bic < best.bic:
                best = fit
        if best is None:
            raise RuntimeError("Chang-Refsdal fit did not produce any candidate.")
        return best

    @staticmethod
    def _shape(theta: np.ndarray, time: np.ndarray, segment: SegmentData, *, branch: str) -> np.ndarray:
        x, y = pspl_image_position(time, segment.pspl, branch=branch)
        width = np.exp(theta[2])
        r2 = (x - theta[0]) ** 2 + (y - theta[1]) ** 2
        return 1.0 / (1.0 + r2 / max(width * width, 1e-12))

    @staticmethod
    def _local_shear(x: float, y: float) -> float:
        r = max(float(np.hypot(x, y)), 1e-6)
        return 1.0 / (r * r)
