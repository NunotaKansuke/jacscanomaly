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


def canonical_cusp_magnification(u1, u2) -> np.ndarray:
    p, u2b = np.broadcast_arrays(np.asarray(u1, dtype=float), np.asarray(u2, dtype=float))
    q = -u2b
    discr = (q / 2.0) ** 2 + (p / 3.0) ** 3
    out = np.zeros_like(p, dtype=float)

    one = discr >= 0.0
    if np.any(one):
        sqrt_d = np.sqrt(np.maximum(discr[one], 0.0))
        y = np.cbrt(-q[one] / 2.0 + sqrt_d) + np.cbrt(-q[one] / 2.0 - sqrt_d)
        out[one] = 1.0 / np.maximum(np.abs(p[one] + 3.0 * y * y), 1e-8)

    three = ~one
    if np.any(three):
        pp = p[three]
        qq = q[three]
        radius = 2.0 * np.sqrt(np.maximum(-pp / 3.0, 0.0))
        arg = (3.0 * qq / (2.0 * pp)) * np.sqrt(np.maximum(-3.0 / pp, 0.0))
        phi = np.arccos(np.clip(arg, -1.0, 1.0)) / 3.0
        total = np.zeros_like(pp)
        for k in (0, 1, 2):
            y = radius * np.cos(phi - 2.0 * np.pi * k / 3.0)
            total += 1.0 / np.maximum(np.abs(pp + 3.0 * y * y), 1e-8)
        out[three] = total
    return out


class FiniteSourceCuspLookup:
    def __init__(self, *, n_radius: int = 5, n_angle: int = 16):
        radii = (np.arange(int(n_radius), dtype=float) + 0.5) / max(int(n_radius), 1)
        angles = np.linspace(0.0, 2.0 * np.pi, int(n_angle), endpoint=False)
        rr, aa = np.meshgrid(radii, angles, indexing="ij")
        self.dx = (rr * np.cos(aa)).ravel()
        self.dy = (rr * np.sin(aa)).ravel()

    def __call__(self, eta1, eta2) -> np.ndarray:
        eta1_arr = np.asarray(eta1, dtype=float)
        eta2_arr = np.asarray(eta2, dtype=float)
        acc = np.zeros(np.broadcast(eta1_arr, eta2_arr).shape, dtype=float)
        for dx, dy in zip(self.dx, self.dy):
            acc += canonical_cusp_magnification(eta1_arr + dx, eta2_arr + dy)
        return acc / max(self.dx.size, 1)


class CanonicalCuspAtom(ResidualAtom):
    atom_name = "canonical_cusp_map"
    class_label = "canonical_cusp"

    def fit(self, segment: SegmentData, features: dict[str, float]) -> AtomFitResult:
        return self._fit_cusp(segment, features, finite_source=False)

    def _fit_cusp(self, segment: SegmentData, features: dict[str, float], *, finite_source: bool) -> AtomFitResult:
        t = np.asarray(segment.time, dtype=float)
        if t.size == 0:
            raise ValueError("Cannot fit an empty segment.")
        t_peak = float(features.get("t_peak", t[int(np.argmax(np.abs(segment.residual)))]))
        width = max(float(features.get("fwhm", 0.0)), float(features.get("cadence", 0.0)), 1e-6)
        lo_t, hi_t = float(t[0]), float(t[-1])
        min_width = max(float(features.get("cadence", 0.0)), 1e-6)
        max_width = max(float(features.get("duration", width)), min_width) * 4.0
        lookup = FiniteSourceCuspLookup() if finite_source else None
        guesses = [
            np.asarray([t_peak, np.log(width), 0.0, 0.2], dtype=float),
            np.asarray([t_peak, np.log(max(0.5 * width, min_width)), -0.3, 0.3], dtype=float),
            np.asarray([t_peak, np.log(min(2.0 * width, max_width)), 0.3, -0.3], dtype=float),
        ]

        def shape(theta: np.ndarray, time: np.ndarray) -> np.ndarray:
            tau = (time - theta[0]) / np.exp(theta[1])
            u1 = theta[2] + tau
            u2 = theta[3]
            if lookup is not None:
                return lookup(u1, u2)
            return canonical_cusp_magnification(u1, u2)

        fit = self._fit_profiled(
            segment=segment,
            features=features,
            theta0_list=guesses,
            bounds=[(lo_t, hi_t), (np.log(min_width), np.log(max_width)), (-3.0, 3.0), (-3.0, 3.0)],
            shape_from_theta=shape,
            params_from_theta=lambda theta: {
                "ta": float(theta[0]),
                "width": float(np.exp(theta[1])),
                "eta1_0": float(theta[2]),
                "eta2_0": float(theta[3]),
            },
            expected_amplitude_sign=None,
            extra_warnings=(),
        )
        return fit


class FiniteSourceCuspAtom(CanonicalCuspAtom):
    atom_name = "finite_source_cusp_lookup"
    class_label = "finite_source_cusp"

    def fit(self, segment: SegmentData, features: dict[str, float]) -> AtomFitResult:
        return self._fit_cusp(segment, features, finite_source=True)
