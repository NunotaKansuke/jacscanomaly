from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np

from .base import ResidualAtom
from ..pspl import pspl_magnification_from_u, u_abs, u_vec
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
    atom_name = "chang_refsdal_lookup_perturbation"
    class_label = "chang_refsdal"
    estimation_role = "physical_local"

    def fit(self, segment: SegmentData, features: dict[str, float]) -> AtomFitResult:
        t = np.asarray(segment.time, dtype=float)
        if t.size == 0:
            raise ValueError("Cannot fit an empty segment.")
        t_peak = float(features.get("t_peak", t[int(np.argmax(np.abs(segment.residual)))]))
        width_t = max(float(features.get("fwhm", 0.0)), float(features.get("cadence", 0.0)), 1e-6)
        sqrt_q0 = max(width_t / max(segment.pspl.tE, 1e-12), 2e-3)
        best: AtomFitResult | None = None
        candidates: list[AtomFitResult] = []
        sqrt_q_factors = tuple(float(v) for v in self.config.cr_lookup_sqrt_q_factors)
        if not sqrt_q_factors:
            sqrt_q_factors = (1.0,)
        source_radius_grid = tuple(float(v) for v in self.config.cr_lookup_source_radius_grid)
        if not source_radius_grid:
            source_radius_grid = (0.0,)
        for branch in ("major", "minor"):
            x, y = pspl_image_position(t, segment.pspl, branch=branch)
            x_peak, y_peak = pspl_image_position(np.asarray([t_peak]), segment.pspl, branch=branch)
            candidate_times = {
                float(np.clip(value, t[0], t[-1]))
                for value in (
                    t_peak,
                    float(features.get("t_positive_peak", t_peak)),
                    float(features.get("t_negative_peak", t_peak)),
                    t_peak - 0.5 * width_t,
                    t_peak + 0.5 * width_t,
                )
            }
            guesses = []
            for candidate_time in sorted(candidate_times):
                x0, y0 = pspl_image_position(np.asarray([candidate_time]), segment.pspl, branch=branch)
                guesses.extend(
                    np.asarray(
                        [float(x0[0]), float(y0[0]), np.log(max(factor * sqrt_q0, np.sqrt(self.config.q_floor)))],
                        dtype=float,
                    )
                    for factor in sqrt_q_factors
                )
            span = max(float(np.max(np.hypot(x - x_peak[0], y - y_peak[0]))), sqrt_q0, 0.1)
            for source_radius_hat in source_radius_grid:
                fit = self._fit_profiled(
                    segment=segment,
                    features=features,
                    theta0_list=guesses,
                    bounds=[
                        (float(x_peak[0]) - 2.0 * span, float(x_peak[0]) + 2.0 * span),
                        (float(y_peak[0]) - 2.0 * span, float(y_peak[0]) + 2.0 * span),
                        (
                            0.5 * np.log(max(float(self.config.q_floor), 1e-12)),
                            0.5 * np.log(max(float(self.config.q_ceil), float(self.config.q_floor) * 1.0001)),
                        ),
                    ],
                    shape_from_theta=lambda theta, time, br=branch, rho_hat=source_radius_hat: self._shape(
                        theta,
                        time,
                        segment,
                        branch=br,
                        source_radius_hat=rho_hat,
                    ),
                    params_from_theta=lambda theta, br=branch, rho_hat=source_radius_hat: self._params(
                        theta, segment, branch=br, source_radius_hat=rho_hat
                    ),
                    expected_amplitude_sign=None,
                    extra_warnings=("physical local Chang-Refsdal approximation; validity requires an isolated planetary perturbation",),
                    fixed_physical_model=True,
                )
                candidates.append(fit)
                if best is None or fit.bic < best.bic:
                    best = fit
        if best is None:
            raise RuntimeError("Chang-Refsdal fit did not produce any candidate.")
        physical = {
            key: float(best.params[key])
            for key in (
                "image_branch", "x_planet", "y_planet", "sqrt_q",
                "s", "q", "alpha", "rho", "rho_over_sqrt_q", "gamma",
            )
            if key in best.params and np.isfinite(best.params[key])
        }
        rho_values = sorted(set(source_radius_grid))
        rho_hat = float(best.params.get("rho_over_sqrt_q", 0.0))
        sqrt_q = float(best.params.get("sqrt_q", np.nan))
        rho_index = min(range(len(rho_values)), key=lambda i: abs(rho_values[i] - rho_hat))
        finite_source_relation: str
        if rho_index == 0 and rho_values[rho_index] == 0.0:
            upper_hat = 0.5 * rho_values[1] if len(rho_values) > 1 else 0.0
            physical.pop("rho_over_sqrt_q", None)
            physical.pop("rho", None)
            if upper_hat > 0.0:
                physical["rho_over_sqrt_q_upper"] = upper_hat
                physical["rho_upper"] = upper_hat * sqrt_q
                finite_source_relation = "finite source is unresolved on the lookup grid; reported rho quantities are upper limits"
            else:
                finite_source_relation = "only a point-source lookup was evaluated; rho is unconstrained"
        elif rho_index == len(rho_values) - 1:
            lower_hat = 0.5 * (rho_values[rho_index - 1] + rho_hat) if rho_index else rho_hat
            physical.pop("rho_over_sqrt_q", None)
            physical.pop("rho", None)
            physical["rho_over_sqrt_q_lower"] = lower_hat
            physical["rho_lower"] = lower_hat * sqrt_q
            finite_source_relation = "finite-source optimum is at the largest lookup radius; reported rho quantities are lower limits"
        else:
            low_hat = 0.5 * (rho_values[rho_index - 1] + rho_hat)
            high_hat = 0.5 * (rho_hat + rho_values[rho_index + 1])
            physical["rho_over_sqrt_q_low"] = low_hat
            physical["rho_over_sqrt_q_high"] = high_hat
            physical["rho_low"] = low_hat * sqrt_q
            physical["rho_high"] = high_hat * sqrt_q
            finite_source_relation = "rho/sqrt(q) is grid-resolved; neighboring lookup midpoints give its resolution interval"
        finite_candidates = sorted((fit for fit in candidates if np.isfinite(fit.bic)), key=lambda fit: fit.bic)
        modes = []
        for fit in finite_candidates[:8]:
            mode = {
                key: float(fit.params[key])
                for key in (
                    "image_branch", "x_planet", "y_planet", "sqrt_q",
                    "s", "q", "alpha", "rho", "rho_over_sqrt_q", "gamma",
                )
                if key in fit.params and np.isfinite(fit.params[key])
            }
            mode["bic"] = float(fit.bic)
            mode["delta_bic"] = float(fit.bic - best.bic)
            modes.append(mode)
        diagnostics = dict(best.fit_diagnostics or {})
        diagnostics["physical_modes"] = modes
        invalid_reasons = [
            reason for reason in best.physical_invalid_reasons
            if reason != "no identifiable physical quantity"
        ]
        if not best.success:
            invalid_reasons.append("fit was not successful")
        if best.delta_chi2 < self.config.physical_delta_chi2_threshold:
            invalid_reasons.append("insufficient delta_chi2")
        if "optimizer parameter is near bound" in best.warnings:
            invalid_reasons.append("physical solution is on a fit boundary")
        if float(best.params.get("q", np.inf)) > float(self.config.cr_physical_q_max):
            invalid_reasons.append("q is too large for the configured planetary Chang-Refsdal approximation")
        invalid_reasons = list(dict.fromkeys(invalid_reasons))
        return replace(
            best,
            estimation_role="physical_local",
            physical_params=physical,
            constraint_relations=(
                "x_planet and y_planet are native Chang-Refsdal fit coordinates in the PSPL lens frame",
                "q=sqrt_q^2 and s, alpha are deterministic coordinate transforms, not grid assumptions",
                finite_source_relation,
            ),
            fit_diagnostics=diagnostics,
            physical_valid=not invalid_reasons,
            physical_invalid_reasons=tuple(invalid_reasons),
        )

    def _params(
        self,
        theta: np.ndarray,
        segment: SegmentData,
        *,
        branch: str,
        source_radius_hat: float,
    ) -> dict[str, float]:
        x_planet, y_planet = float(theta[0]), float(theta[1])
        sqrt_q = float(np.exp(theta[2]))
        s = float(np.hypot(x_planet, y_planet))
        binary_axis_angle = float(np.arctan2(y_planet, x_planet))
        alpha = float(np.arctan2(np.sin(-binary_axis_angle), np.cos(-binary_axis_angle)))
        rho_hat = float(max(source_radius_hat, 0.0))
        return {
            "image_branch": 1.0 if branch == "major" else -1.0,
            "x_planet": x_planet,
            "y_planet": y_planet,
            "s": s,
            "alpha": alpha,
            "binary_axis_angle": binary_axis_angle,
            "q": sqrt_q * sqrt_q,
            "sqrt_q": sqrt_q,
            "rho_over_sqrt_q": rho_hat,
            "rho": rho_hat * sqrt_q,
            "gamma": float(self._local_shear(x_planet, y_planet)),
            "lookup_grid": float(self.config.cr_lookup_grid_size),
            "lookup_gamma_step": float(self.config.cr_lookup_gamma_step),
        }

    def _shape(
        self,
        theta: np.ndarray,
        time: np.ndarray,
        segment: SegmentData,
        *,
        branch: str,
        source_radius_hat: float,
    ) -> np.ndarray:
        return chang_refsdal_flux_residual(
            time,
            segment.pspl,
            x_planet=float(theta[0]),
            y_planet=float(theta[1]),
            q=float(np.exp(2.0 * theta[2])),
            rho_over_sqrt_q=float(source_radius_hat),
            branch=branch,
            grid_size=int(self.config.cr_lookup_grid_size),
            extent=float(self.config.cr_lookup_extent),
            gamma_step=float(self.config.cr_lookup_gamma_step),
        )

    @staticmethod
    def _local_shear(x: float, y: float) -> float:
        r = max(float(np.hypot(x, y)), 1e-6)
        return 1.0 / (r * r)


def chang_refsdal_flux_residual(
    time: np.ndarray,
    pspl,
    *,
    x_planet: float,
    y_planet: float,
    q: float,
    rho_over_sqrt_q: float = 0.0,
    branch: str,
    grid_size: int = 192,
    extent: float = 4.5,
    gamma_step: float = 0.025,
) -> np.ndarray:
    """Physical local flux perturbation of one PSPL image in the CR limit."""
    x, y = pspl_image_position(np.asarray(time, dtype=float), pspl, branch=branch)
    sqrt_q = np.sqrt(max(float(q), 1e-16))
    radius = max(float(np.hypot(x_planet, y_planet)), 1e-12)
    erx, ery = float(x_planet) / radius, float(y_planet) / radius
    etx, ety = -ery, erx
    dx, dy = x - float(x_planet), y - float(y_planet)
    xi = (dx * erx + dy * ery) / sqrt_q
    eta = (dx * etx + dy * ety) / sqrt_q
    relative = cr_relative_magnification(
        xi,
        eta,
        1.0 / (radius * radius),
        source_radius_hat=float(rho_over_sqrt_q),
        grid_size=int(grid_size),
        extent=float(extent),
        gamma_step=float(gamma_step),
    )
    total = pspl_magnification_from_u(u_abs(time, pspl))
    image_magnification = 0.5 * (total + 1.0) if branch == "major" else 0.5 * (total - 1.0)
    return float(pspl.Fs) * image_magnification * (relative - 1.0)


_CR_GRID_SIZE = 192
_CR_EXTENT = 4.5
_CR_GAMMA_STEP = 0.025
_CR_SOURCE_RADIUS_STEP = 0.025


def cr_relative_magnification(
    xi: np.ndarray,
    eta: np.ndarray,
    gamma: float,
    *,
    source_radius_hat: float = 0.0,
    grid_size: int = _CR_GRID_SIZE,
    extent: float = _CR_EXTENT,
    gamma_step: float = _CR_GAMMA_STEP,
) -> np.ndarray:
    """
    Bilinear lookup of a ray-shot Chang-Refsdal relative magnification map.

    The map is normalized by its outer source-plane density, so values tend to
    one away from the local caustic.  This is intentionally a local morphology
    lookup for local constraint measurement, not a full binary-lens fit.
    """
    x = np.asarray(xi, dtype=float)
    y = np.asarray(eta, dtype=float)
    gamma_key = _gamma_key(gamma, gamma_step)
    rho_key = _source_radius_key(source_radius_hat)
    grid = _cr_map(gamma_key, rho_key, int(grid_size), float(extent))
    return _bilinear(grid, x, y, float(extent))


def warm_chang_refsdal_lookup_cache(
    *,
    gamma_values: tuple[float, ...] | None = None,
    source_radius_hat_values: tuple[float, ...] | None = None,
    grid_size: int = _CR_GRID_SIZE,
    extent: float = _CR_EXTENT,
    gamma_step: float = _CR_GAMMA_STEP,
) -> None:
    if gamma_values is None:
        gamma_values = tuple(np.arange(0.0, 0.951, float(gamma_step)))
    if source_radius_hat_values is None:
        source_radius_hat_values = (0.0, 0.03, 0.1, 0.3, 1.0)
    for gamma in gamma_values:
        gamma_key = _gamma_key(float(gamma), gamma_step)
        for rho_hat in source_radius_hat_values:
            _cr_map(gamma_key, _source_radius_key(float(rho_hat)), int(grid_size), float(extent))


def _gamma_key(gamma: float, gamma_step: float = _CR_GAMMA_STEP) -> float:
    value = float(np.clip(abs(float(gamma)), 0.0, 4.0))
    step = max(float(gamma_step), 1e-6)
    return round(value / step) * step


def _source_radius_key(source_radius_hat: float) -> float:
    value = float(np.clip(max(float(source_radius_hat), 0.0), 0.0, 3.0))
    if value == 0.0:
        return 0.0
    return round(value / _CR_SOURCE_RADIUS_STEP) * _CR_SOURCE_RADIUS_STEP


@lru_cache(maxsize=256)
def _cr_map(gamma: float, source_radius_hat: float, grid_size: int, extent: float) -> np.ndarray:
    n = int(grid_size)
    extent = float(extent)
    image_extent = 2.2 * extent
    axis = np.linspace(-image_extent, image_extent, 2 * n)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    r2 = xx * xx + yy * yy
    finite = r2 > 1e-8

    # Chang-Refsdal equation in local coordinates:
    # In radial/tangential coordinates around the planet, expansion of the
    # host deflection gives diag(1+gamma, 1-gamma).
    sx = xx - np.where(finite, xx / r2, 0.0) + gamma * xx
    sy = yy - np.where(finite, yy / r2, 0.0) - gamma * yy
    good = finite & np.isfinite(sx) & np.isfinite(sy)
    hist, _, _ = np.histogram2d(
        sy[good].ravel(),
        sx[good].ravel(),
        bins=n,
        range=[[-extent, extent], [-extent, extent]],
    )
    hist = hist.astype(float)
    border = max(2, n // 12)
    outer = np.concatenate(
        (
            hist[:border, :].ravel(),
            hist[-border:, :].ravel(),
            hist[:, :border].ravel(),
            hist[:, -border:].ravel(),
        )
    )
    norm = float(np.median(outer[outer > 0.0])) if np.any(outer > 0.0) else 1.0
    rel = hist / max(norm, 1e-12)
    rel = np.clip(rel, 0.0, 50.0)
    if source_radius_hat > 0.0:
        rel = _convolve_with_uniform_source(rel, float(source_radius_hat), extent)
    return rel


def _convolve_with_uniform_source(grid: np.ndarray, source_radius_hat: float, extent: float) -> np.ndarray:
    n = int(grid.shape[0])
    pixel = 2.0 * float(extent) / max(n - 1, 1)
    radius_pix = float(source_radius_hat) / max(pixel, 1e-12)
    if radius_pix < 0.5:
        return grid
    yy, xx = np.indices((n, n), dtype=float)
    center = 0.5 * (n - 1)
    rr = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    kernel = (rr <= radius_pix).astype(float)
    total = float(np.sum(kernel))
    if total <= 0.0:
        return grid
    kernel /= total
    shape = (2 * n, 2 * n)
    f_grid = np.fft.rfft2(grid, shape)
    f_kernel = np.fft.rfft2(np.fft.ifftshift(kernel), shape)
    conv = np.fft.irfft2(f_grid * f_kernel, shape)
    start = n // 2
    smoothed = conv[start : start + n, start : start + n]
    if smoothed.shape != grid.shape:
        return grid
    return np.clip(smoothed, 0.0, 50.0)


def _bilinear(grid: np.ndarray, x: np.ndarray, y: np.ndarray, extent: float) -> np.ndarray:
    n = grid.shape[0]
    gx = (np.asarray(x, dtype=float) + extent) * (n - 1) / (2.0 * extent)
    gy = (np.asarray(y, dtype=float) + extent) * (n - 1) / (2.0 * extent)
    outside = (gx < 0.0) | (gx > n - 1) | (gy < 0.0) | (gy > n - 1)
    gx = np.clip(gx, 0.0, n - 1.000001)
    gy = np.clip(gy, 0.0, n - 1.000001)
    ix = np.floor(gx).astype(int)
    iy = np.floor(gy).astype(int)
    fx = gx - ix
    fy = gy - iy
    ix1 = np.minimum(ix + 1, n - 1)
    iy1 = np.minimum(iy + 1, n - 1)
    value = (
        (1.0 - fx) * (1.0 - fy) * grid[iy, ix]
        + fx * (1.0 - fy) * grid[iy, ix1]
        + (1.0 - fx) * fy * grid[iy1, ix]
        + fx * fy * grid[iy1, ix1]
    )
    return np.where(outside, 1.0, value)
