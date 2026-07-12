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
                    "t_limb": float(theta[0] - sign * np.exp(theta[1])),
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
                params_from_theta=lambda theta, sign=entry_exit_sign: self._params(theta, segment, sign),
                expected_amplitude_sign=None,
                extra_warnings=FoldCausticAtom._warnings(features, min_tstar=min_tstar, max_tstar=max_tstar),
            )
            if best is None or fit.bic < best.bic:
                best = fit
        if best is None:
            raise RuntimeError("Curved fold fit did not produce any candidate.")
        return best

    @classmethod
    def _params(cls, theta: np.ndarray, segment: SegmentData, sign: float) -> dict[str, float]:
        tc = float(theta[0])
        tstar = float(np.exp(theta[1]))
        q_curv = float(theta[2])
        params = {
            "tc": tc,
            "tstar": tstar,
            "q_curv": q_curv,
            "entry_exit_sign": float(sign),
            "rho_over_sinalpha": tstar / max(segment.pspl.tE, 1e-12),
        }

        limb_contacts = cls._solve_time_roots(tc, tstar, q_curv, sign, z_value=-1.0)
        if len(limb_contacts) >= 2:
            t_entry, t_exit = limb_contacts[0], limb_contacts[-1]
            local_entry = cls._local_crossing_scale(t_entry, tc, tstar, q_curv, sign)
            local_exit = cls._local_crossing_scale(t_exit, tc, tstar, q_curv, sign)
            params.update(
                {
                    "t_entry": t_entry,
                    "t_exit": t_exit,
                    "caustic_inside_duration": t_exit - t_entry,
                    "tstar_entry": local_entry,
                    "tstar_exit": local_exit,
                    "rho_over_sinalpha_entry": local_entry / max(segment.pspl.tE, 1e-12),
                    "rho_over_sinalpha_exit": local_exit / max(segment.pspl.tE, 1e-12),
                    "entry_exit_asymmetry": (local_exit - local_entry) / max(local_exit + local_entry, 1e-12),
                }
            )

        center_crossings = cls._solve_time_roots(tc, tstar, q_curv, sign, z_value=0.0)
        if len(center_crossings) >= 2:
            params["tc1"] = center_crossings[0]
            params["tc2"] = center_crossings[-1]
            params["t_center"] = 0.5 * (center_crossings[0] + center_crossings[-1])
        elif len(center_crossings) == 1:
            params["tc1"] = center_crossings[0]
        return params

    @staticmethod
    def _local_crossing_scale(t: float, tc: float, tstar: float, q_curv: float, sign: float) -> float:
        x = (float(t) - float(tc)) / max(float(tstar), 1e-12)
        dz_dt = float(sign) * (1.0 + 2.0 * float(q_curv) * x) / max(float(tstar), 1e-12)
        return 1.0 / max(abs(dz_dt), 1e-12)

    @staticmethod
    def _solve_time_roots(tc: float, tstar: float, q_curv: float, sign: float, *, z_value: float) -> list[float]:
        """Solve sign * (x + q_curv*x^2) = z_value for t = tc + x*tstar."""

        if not all(np.isfinite(value) for value in (tc, tstar, q_curv, sign, z_value)):
            return []
        if tstar <= 0.0 or sign == 0.0:
            return []
        c = -float(z_value) / float(sign)
        if abs(q_curv) < 1e-10:
            x_roots = [-c]
        else:
            disc = 1.0 - 4.0 * float(q_curv) * c
            if disc < -1e-12:
                return []
            sqrt_disc = float(np.sqrt(max(disc, 0.0)))
            x_roots = [(-1.0 - sqrt_disc) / (2.0 * q_curv), (-1.0 + sqrt_disc) / (2.0 * q_curv)]
        roots = sorted({round(float(tc + x * tstar), 12) for x in x_roots if np.isfinite(x)})
        return [float(value) for value in roots]


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
            params_from_theta=lambda theta: self._params(theta, segment),
            expected_amplitude_sign=None,
            extra_warnings=(),
        )

    @staticmethod
    def _roots(ta: float, width: float, z0: float, q_curv: float, z_value: float) -> list[float]:
        # z = z0 + x + q*x^2, x=(t-ta)/width.
        c = z0 - z_value
        if abs(q_curv) < 1e-10:
            xs = [-c]
        else:
            disc = 1.0 - 4.0 * q_curv * c
            if disc < -1e-12:
                return []
            root = float(np.sqrt(max(disc, 0.0)))
            xs = [(-1.0 - root) / (2.0 * q_curv), (-1.0 + root) / (2.0 * q_curv)]
        return sorted({float(ta + width * x) for x in xs if np.isfinite(x)})

    @classmethod
    def _params(cls, theta: np.ndarray, segment: SegmentData) -> dict[str, float]:
        ta, z0 = float(theta[0]), float(theta[1])
        width, q_curv = float(np.exp(theta[2])), float(theta[3])
        tE = max(float(segment.pspl.tE), 1e-12)
        params = {
            "ta": ta,
            "z0": z0,
            "width": width,
            "q_curv": q_curv,
            "a1": 1.0 / width,
            "a2": q_curv / (width * width),
            "local_scale_over_tE_at_ta": width / tE,
        }
        if abs(q_curv) > 1e-10:
            x_vertex = -1.0 / (2.0 * q_curv)
            t_stationary = ta + width * x_vertex
            z_stationary = z0 + x_vertex + q_curv * x_vertex * x_vertex
            params["t_stationary"] = t_stationary
            params["z_stationary"] = z_stationary
            params["stationary_curvature"] = 2.0 * q_curv / (width * width)
            if q_curv > 0.0:
                params["t_closest"] = t_stationary
                params["z_closest"] = z_stationary
        for prefix, z_value in (("contact", -1.0), ("center_crossing", 0.0)):
            roots = cls._roots(ta, width, z0, q_curv, z_value)
            for index, root in enumerate(roots, start=1):
                params[f"t_{prefix}_{index}"] = root
                slope = abs((1.0 + 2.0 * q_curv * ((root - ta) / width)) / width)
                local_tstar = 1.0 / max(slope, 1e-12)
                params[f"tstar_{prefix}_{index}"] = local_tstar
                params[f"rho_over_sinalpha_{prefix}_{index}"] = local_tstar / tE
        contacts = cls._roots(ta, width, z0, q_curv, -1.0)
        if len(contacts) == 2:
            params["contact_duration"] = contacts[1] - contacts[0]
        return params


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
                    "t_limb": float(theta[0] - sign * np.exp(theta[1])),
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


class RimTroughCausticAtom(ResidualAtom):
    atom_name = "rim_trough_caustic"
    class_label = "rim_trough_caustic"

    def fit(self, segment: SegmentData, features: dict[str, float]) -> AtomFitResult:
        t = np.asarray(segment.time, dtype=float)
        y = np.asarray(segment.residual, dtype=float)
        if t.size == 0:
            raise ValueError("Cannot fit an empty segment.")
        lo_t, hi_t = float(t[0]), float(t[-1])
        duration = max(float(features.get("duration", hi_t - lo_t)), 1e-6)
        cadence = max(float(features.get("cadence", 0.0)), 1e-6)
        width = max(float(features.get("fwhm", 0.0)), cadence, 1e-6)
        min_gap = max(0.5 * cadence, 1e-6)
        max_gap = max(duration, min_gap) * 1.5
        min_width = max(0.5 * cadence, 1e-6)
        max_width = max(duration, min_width) * 1.5
        best: AtomFitResult | None = None

        for polarity in (1.0, -1.0):
            center, left, right = self._initial_times(t, y, polarity=polarity, features=features)
            left_gap = max(center - left, min_gap)
            right_gap = max(right - center, min_gap)
            rim_width0 = max(0.35 * min(left_gap, right_gap), min_width)
            trough_width0 = max(0.5 * (left_gap + right_gap), min_width)
            guesses = []
            for rim_scale in (0.7, 1.0, 1.4):
                for trough_scale in (0.7, 1.0, 1.4):
                    guesses.append(
                        np.asarray(
                            [
                                center,
                                np.log(left_gap),
                                np.log(right_gap),
                                np.log(min(max(rim_width0 * rim_scale, min_width), max_width)),
                                np.log(min(max(trough_width0 * trough_scale, min_width), max_width)),
                                0.0,
                                np.log(2.0),
                            ],
                            dtype=float,
                        )
                    )
            fit = self._fit_profiled(
                segment=segment,
                features=features,
                theta0_list=guesses,
                bounds=[
                    (lo_t, hi_t),
                    (np.log(min_gap), np.log(max_gap)),
                    (np.log(min_gap), np.log(max_gap)),
                    (np.log(min_width), np.log(max_width)),
                    (np.log(min_width), np.log(max_width)),
                    (np.log(0.1), np.log(10.0)),
                    (np.log(0.1), np.log(30.0)),
                ],
                shape_from_theta=lambda theta, time, sign=polarity: self._shape(theta, time, sign),
                params_from_theta=lambda theta, sign=polarity: self._params(theta, segment, sign),
                expected_amplitude_sign=1.0,
                extra_warnings=(),
            )
            if best is None or fit.bic < best.bic:
                best = fit
        if best is None:
            raise RuntimeError("Rim-trough fit did not produce any candidate.")
        return best

    @staticmethod
    def _lorentz(time: np.ndarray, center: float, width: float) -> np.ndarray:
        u = (time - center) / max(width, 1e-12)
        return 1.0 / (1.0 + u * u)

    @classmethod
    def _shape(cls, theta: np.ndarray, time: np.ndarray, polarity: float) -> np.ndarray:
        t0 = float(theta[0])
        left_gap = float(np.exp(theta[1]))
        right_gap = float(np.exp(theta[2]))
        rim_width = float(np.exp(theta[3]))
        trough_width = float(np.exp(theta[4]))
        rim_ratio = float(np.exp(theta[5]))
        trough_ratio = float(np.exp(theta[6]))
        t1 = t0 - left_gap
        t2 = t0 + right_gap
        rims = cls._lorentz(time, t1, rim_width) + rim_ratio * cls._lorentz(time, t2, rim_width)
        trough = trough_ratio * cls._lorentz(time, t0, trough_width)
        return float(polarity) * (rims - trough)

    @staticmethod
    def _params(theta: np.ndarray, segment: SegmentData, polarity: float) -> dict[str, float]:
        t0 = float(theta[0])
        left_gap = float(np.exp(theta[1]))
        right_gap = float(np.exp(theta[2]))
        rim_width = float(np.exp(theta[3]))
        trough_width = float(np.exp(theta[4]))
        return {
            "tc1": t0 - left_gap,
            "t_trough": t0,
            "tc2": t0 + right_gap,
            "rim_width": rim_width,
            "trough_width": trough_width,
            "rim_ratio": float(np.exp(theta[5])),
            "trough_ratio": float(np.exp(theta[6])),
            "polarity": float(polarity),
            "rim_separation": left_gap + right_gap,
            "rim_time_asymmetry": (right_gap - left_gap) / max(right_gap + left_gap, 1e-12),
            "characteristic_scale_over_tE": float(max(rim_width, trough_width) / max(segment.pspl.tE, 1e-12)),
        }

    @staticmethod
    def _initial_times(
        t: np.ndarray,
        residual: np.ndarray,
        *,
        polarity: float,
        features: dict[str, float],
    ) -> tuple[float, float, float]:
        target = float(polarity) * np.asarray(residual, dtype=float)
        center_idx = int(np.argmin(target))
        center = float(t[center_idx])
        left_mask = t < center
        right_mask = t > center
        if np.any(left_mask):
            left = float(t[np.flatnonzero(left_mask)[int(np.argmax(target[left_mask]))]])
        else:
            left = center - 0.25 * max(float(features.get("duration", 0.0)), 1e-6)
        if np.any(right_mask):
            right = float(t[np.flatnonzero(right_mask)[int(np.argmax(target[right_mask]))]])
        else:
            right = center + 0.25 * max(float(features.get("duration", 0.0)), 1e-6)
        if not left < center:
            left = center - 0.25 * max(float(features.get("duration", 0.0)), 1e-6)
        if not right > center:
            right = center + 0.25 * max(float(features.get("duration", 0.0)), 1e-6)
        return center, left, right


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
        min_gap = max(2.0 * float(features.get("cadence", 0.0)), 0.25 * width, min_tstar, 1e-6)
        max_gap = max(hi_t - lo_t, min_gap)
        sep0 = min(max(0.5 * width, min_gap), max_gap)
        best: AtomFitResult | None = None
        ratio_guesses = (0.5, 1.0, 2.0)
        for signs in ((1.0, -1.0), (-1.0, 1.0)):
            left0 = min(max(t_peak - 0.5 * sep0, lo_t), hi_t - min_gap)
            guesses = [
                np.asarray([left0, np.log(sep0), np.log(width), np.log(ratio)], dtype=float)
                for ratio in ratio_guesses
            ]
            left1 = min(max(t_peak - width, lo_t), hi_t - min_gap)
            gap1 = min(max(2.0 * width, min_gap), max_gap)
            guesses.extend(
                np.asarray(
                    [left1, np.log(gap1), np.log(max(0.5 * width, min_tstar)), np.log(ratio)],
                    dtype=float,
                )
                for ratio in ratio_guesses
            )
            fit = self._fit_profiled(
                segment=segment,
                features=features,
                theta0_list=guesses,
                bounds=[
                    (lo_t, hi_t - min_gap),
                    (np.log(min_gap), np.log(max_gap)),
                    (np.log(min_tstar), np.log(max_tstar)),
                    (np.log(0.05), np.log(20.0)),
                ],
                shape_from_theta=lambda theta, time, ss=signs: (
                    fold_g0(ss[0] * (time - theta[0]) / np.exp(theta[2]))
                    + np.exp(theta[3]) * fold_g0(ss[1] * (time - (theta[0] + np.exp(theta[1]))) / np.exp(theta[2]))
                ),
                params_from_theta=lambda theta, ss=signs: {
                    "tc1": float(theta[0]),
                    "tc2": float(theta[0] + np.exp(theta[1])),
                    "caustic_inside_duration": float(np.exp(theta[1])),
                    "tstar": float(np.exp(theta[2])),
                    "tstar_1": float(np.exp(theta[2])),
                    "tstar_2": float(np.exp(theta[2])),
                    "fold_ratio": float(np.exp(theta[3])),
                    "entry_exit_sign_1": float(ss[0]),
                    "entry_exit_sign_2": float(ss[1]),
                    "rho_over_sinalpha": float(np.exp(theta[2]) / max(segment.pspl.tE, 1e-12)),
                    "rho_over_sinalpha_1": float(np.exp(theta[2]) / max(segment.pspl.tE, 1e-12)),
                    "rho_over_sinalpha_2": float(np.exp(theta[2]) / max(segment.pspl.tE, 1e-12)),
                    "contact_separation_over_2tstar": float(np.exp(theta[1]) / (2.0 * np.exp(theta[2]))),
                },
                expected_amplitude_sign=None,
                extra_warnings=(),
            )
            if best is None or fit.bic < best.bic:
                best = fit
        if best is None:
            raise RuntimeError("Two-fold fit did not produce any candidate.")
        return best


class FullCausticCrossingAtom(ResidualAtom):
    atom_name = "full_caustic_crossing"
    class_label = "full_caustic_crossing"

    def fit(self, segment: SegmentData, features: dict[str, float]) -> AtomFitResult:
        t = np.asarray(segment.time, dtype=float)
        if t.size == 0:
            raise ValueError("Cannot fit an empty segment.")
        lo_t, hi_t = float(t[0]), float(t[-1])
        duration = max(float(features.get("duration", hi_t - lo_t)), 1e-6)
        width = max(float(features.get("fwhm", 0.0)), float(features.get("cadence", 0.0)), 1e-6)
        cadence = max(float(features.get("cadence", 0.0)), 1e-6)
        min_tstar = max(0.5 * cadence, 1e-6)
        max_tstar = max(width, duration, min_tstar) * 0.5
        peak_times = [float(getattr(p, "time", np.nan)) for p in segment.component.peaks + segment.component.dips]
        peak_times = [value for value in peak_times if np.isfinite(value)]
        peak_span = max(peak_times) - min(peak_times) if len(peak_times) >= 2 else 0.0
        full_event_gap = 0.8 * peak_span if segment.component.signal_type == "whole_event_anomaly" else 0.0
        min_gap = max(3.0 * cadence, min_tstar, full_event_gap, 1e-6)
        max_gap = max(duration, min_gap) * 1.25
        guesses = self._initial_guesses(segment, features, min_gap=min_gap, max_gap=max_gap, min_tstar=min_tstar, max_tstar=max_tstar)
        bounds = [
            (lo_t, hi_t),
            (np.log(min_gap), np.log(max_gap)),
            (np.log(min_tstar), np.log(max_tstar)),
            (np.log(min_tstar), np.log(max_tstar)),
        ]
        fit_specs = [(guesses, bounds)]
        if peak_span > 0.0:
            first_peak = min(peak_times)
            peak_pad = max(0.15 * peak_span, 2.0 * cadence)
            anchored_gap_min = max(0.7 * peak_span, min_gap)
            anchored_gap_max = min(max(1.4 * peak_span, anchored_gap_min), max_gap)
            if anchored_gap_max > anchored_gap_min:
                anchored_bounds = [
                    (max(lo_t, first_peak - peak_pad), min(hi_t, first_peak + peak_pad)),
                    (np.log(anchored_gap_min), np.log(anchored_gap_max)),
                    (np.log(min_tstar), np.log(max_tstar)),
                    (np.log(min_tstar), np.log(max_tstar)),
                ]
                anchored_guesses = [
                    np.asarray(
                        [
                            first_peak,
                            np.log(min(max(peak_span, anchored_gap_min), anchored_gap_max)),
                            np.log(min(max(scale * width, min_tstar), max_tstar)),
                            np.log(min(max(scale * width, min_tstar), max_tstar)),
                        ],
                        dtype=float,
                    )
                    for scale in (0.5, 1.0, 1.5)
                ]
                if segment.component.signal_type == "whole_event_anomaly":
                    fit_specs = [(anchored_guesses, anchored_bounds)]
                else:
                    fit_specs.append((anchored_guesses, anchored_bounds))
        best: AtomFitResult | None = None
        for signs in ((1.0, -1.0), (-1.0, 1.0)):
            for theta0_list, fit_bounds in fit_specs:
                fit = self._fit_profiled(
                    segment=segment,
                    features=features,
                    theta0_list=theta0_list,
                    bounds=fit_bounds,
                    shape_from_theta=lambda theta, time, ss=signs: self._shape(theta, time, ss),
                    params_from_theta=lambda theta, ss=signs: self._params(theta, segment, ss),
                    expected_amplitude_sign=None,
                    extra_warnings=self._warnings(segment, features),
                )
                if best is None or fit.bic < best.bic:
                    best = fit
        if best is None:
            raise RuntimeError("Full caustic-crossing fit did not produce any candidate.")
        return best

    @staticmethod
    def _smooth_window(time: np.ndarray, t_entry: float, t_exit: float, edge: float) -> np.ndarray:
        edge = max(float(edge), 1e-12)
        left = np.clip((time - t_entry) / edge, -60.0, 60.0)
        right = np.clip((t_exit - time) / edge, -60.0, 60.0)
        return (1.0 / (1.0 + np.exp(-left))) * (1.0 / (1.0 + np.exp(-right)))

    @classmethod
    def _shape(cls, theta: np.ndarray, time: np.ndarray, signs: tuple[float, float]) -> np.ndarray:
        t_entry = float(theta[0])
        gap = float(np.exp(theta[1]))
        t_exit = t_entry + gap
        center = 0.5 * (t_entry + t_exit)
        tstar_entry = float(np.exp(theta[2]))
        tstar_exit = float(np.exp(theta[3]))
        entry = fold_g0(float(signs[0]) * (time - t_entry) / max(tstar_entry, 1e-12))
        exit_ = fold_g0(float(signs[1]) * (time - t_exit) / max(tstar_exit, 1e-12))
        edge = max(0.5 * (tstar_entry + tstar_exit), 1e-12)
        window = cls._smooth_window(time, t_entry, t_exit, edge)
        tau = (time - center) / max(0.5 * gap, 1e-12)
        tail_softening = 0.35
        d_entry = np.maximum((time - t_entry) / max(tstar_entry, 1e-12), 0.0)
        d_exit = np.maximum((t_exit - time) / max(tstar_exit, 1e-12), 0.0)
        entry_tail = window / np.sqrt(d_entry + tail_softening)
        exit_tail = window / np.sqrt(d_exit + tail_softening)
        tail_sum = 0.5 * (entry_tail + exit_tail)
        tail_diff = 0.5 * (entry_tail - exit_tail)
        # Keep the bridge low-rank; otherwise it can chase small residual wiggles
        # and produce a jagged visual model despite improving BIC.
        return np.column_stack(
            (
                entry,
                exit_,
                tail_sum,
                tail_diff,
                window,
                window * tau,
            )
        )

    @staticmethod
    def _params(theta: np.ndarray, segment: SegmentData, signs: tuple[float, float]) -> dict[str, float]:
        t_entry = float(theta[0])
        gap = float(np.exp(theta[1]))
        t_exit = t_entry + gap
        center = 0.5 * (t_entry + t_exit)
        tstar_entry = float(np.exp(theta[2]))
        tstar_exit = float(np.exp(theta[3]))
        return {
            "t_entry": t_entry,
            "t_exit": t_exit,
            "tc1": t_entry,
            "tc2": t_exit,
            "t_center": center,
            "caustic_inside_duration": gap,
            "tstar_entry": tstar_entry,
            "tstar_exit": tstar_exit,
            "tstar": 0.5 * (tstar_entry + tstar_exit),
            "rho_over_sinalpha_entry": tstar_entry / max(segment.pspl.tE, 1e-12),
            "rho_over_sinalpha_exit": tstar_exit / max(segment.pspl.tE, 1e-12),
            "rho_over_sinalpha": 0.5 * (tstar_entry + tstar_exit) / max(segment.pspl.tE, 1e-12),
            "entry_exit_sign_1": float(signs[0]),
            "entry_exit_sign_2": float(signs[1]),
            "entry_exit_asymmetry": float((tstar_exit - tstar_entry) / max(tstar_exit + tstar_entry, 1e-12)),
            "tail_softening": 0.35,
        }

    @staticmethod
    def _initial_guesses(
        segment: SegmentData,
        features: dict[str, float],
        *,
        min_gap: float,
        max_gap: float,
        min_tstar: float,
        max_tstar: float,
    ) -> list[np.ndarray]:
        t = np.asarray(segment.time, dtype=float)
        z = np.abs(np.asarray(segment.residual, dtype=float) / np.maximum(np.asarray(segment.ferr, dtype=float), 1e-12))
        lo_t, hi_t = float(t[0]), float(t[-1])
        duration = max(hi_t - lo_t, min_gap)
        width = max(float(features.get("fwhm", 0.0)), min_tstar)
        pairs: list[tuple[float, float]] = []
        extrema = sorted(
            [(abs(float(getattr(p, "z", 0.0))), float(getattr(p, "time", np.nan))) for p in segment.component.peaks + segment.component.dips],
            reverse=True,
        )
        times = [time for _strength, time in extrema if np.isfinite(time)]
        if len(times) >= 2:
            a, b = sorted(times[:2])
            if b - a >= min_gap:
                pairs.append((a, b))
        peak_time = float(features.get("t_peak", t[int(np.argmax(z))]))
        pairs.extend(
            [
                (lo_t + 0.1 * duration, hi_t - 0.1 * duration),
                (lo_t + 0.2 * duration, hi_t - 0.2 * duration),
                (max(lo_t, peak_time - max(width, 0.15 * duration)), min(hi_t, peak_time + max(width, 0.15 * duration))),
            ]
        )
        if z.size >= 2:
            order = np.argsort(z)[::-1]
            first = int(order[0])
            for second in order[1: min(order.size, 32)]:
                a, b = sorted((float(t[first]), float(t[int(second)])))
                if b - a >= min_gap:
                    pairs.append((a, b))
                    break
        guesses = []
        for a, b in pairs:
            gap = min(max(float(b - a), min_gap), max_gap)
            center = float(np.clip(0.5 * (a + b), lo_t, hi_t))
            for scale in (0.7, 1.0, 1.5):
                tstar = min(max(width * scale, min_tstar), max_tstar)
                guesses.append(np.asarray([center - 0.5 * gap, np.log(gap), np.log(tstar), np.log(tstar)], dtype=float))
        if not guesses:
            gap = min(max(0.5 * duration, min_gap), max_gap)
            center = 0.5 * (lo_t + hi_t)
            guesses.append(np.asarray([center - 0.5 * gap, np.log(gap), np.log(min(max(width, min_tstar), max_tstar)), np.log(min(max(width, min_tstar), max_tstar))], dtype=float))
        return guesses

    @staticmethod
    def _warnings(segment: SegmentData, features: dict[str, float]) -> tuple[str, ...]:
        warnings = []
        if segment.component.signal_type not in {"whole_event_anomaly", "caustic_crossing", "complex"}:
            warnings.append("full caustic crossing atom fitted outside a whole-event/caustic-like segment")
        if int(features.get("n_points", 0)) < 20:
            warnings.append("full caustic crossing has limited point support")
        return tuple(warnings)


class SignedTwoFoldCausticAtom(ResidualAtom):
    atom_name = "signed_two_fold_caustic"
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
        min_gap = max(2.0 * float(features.get("cadence", 0.0)), 0.25 * width, min_tstar, 1e-6)
        max_gap = max(hi_t - lo_t, min_gap)
        sep0 = min(max(0.5 * width, min_gap), max_gap)
        best: AtomFitResult | None = None
        for signs in ((1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (-1.0, -1.0)):
            left0 = min(max(t_peak - 0.5 * sep0, lo_t), hi_t - min_gap)
            left1 = min(max(t_peak - width, lo_t), hi_t - min_gap)
            gap1 = min(max(2.0 * width, min_gap), max_gap)
            guesses = [
                np.asarray([left0, np.log(sep0), np.log(width)], dtype=float),
                np.asarray([left1, np.log(gap1), np.log(max(0.5 * width, min_tstar))], dtype=float),
            ]
            fit = self._fit_profiled(
                segment=segment,
                features=features,
                theta0_list=guesses,
                bounds=[(lo_t, hi_t - min_gap), (np.log(min_gap), np.log(max_gap)), (np.log(min_tstar), np.log(max_tstar))],
                shape_from_theta=lambda theta, time, ss=signs: np.column_stack(
                    (
                        fold_g0(ss[0] * (time - theta[0]) / np.exp(theta[2])),
                        fold_g0(ss[1] * (time - (theta[0] + np.exp(theta[1]))) / np.exp(theta[2])),
                    )
                ),
                params_from_theta=lambda theta, ss=signs: {
                    "tc1": float(theta[0]),
                    "tc2": float(theta[0] + np.exp(theta[1])),
                    "caustic_inside_duration": float(np.exp(theta[1])),
                    "tstar": float(np.exp(theta[2])),
                    "tstar_1": float(np.exp(theta[2])),
                    "tstar_2": float(np.exp(theta[2])),
                    "entry_exit_sign_1": float(ss[0]),
                    "entry_exit_sign_2": float(ss[1]),
                    "rho_over_sinalpha": float(np.exp(theta[2]) / max(segment.pspl.tE, 1e-12)),
                    "rho_over_sinalpha_1": float(np.exp(theta[2]) / max(segment.pspl.tE, 1e-12)),
                    "rho_over_sinalpha_2": float(np.exp(theta[2]) / max(segment.pspl.tE, 1e-12)),
                    "contact_separation_over_2tstar": float(np.exp(theta[1]) / (2.0 * np.exp(theta[2]))),
                },
                expected_amplitude_sign=None,
                extra_warnings=(),
            )
            if best is None or fit.bic < best.bic:
                best = fit
        if best is None:
            raise RuntimeError("Signed two-fold fit did not produce any candidate.")
        return best
