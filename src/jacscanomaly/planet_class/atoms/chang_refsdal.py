from __future__ import annotations

from functools import lru_cache

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
    atom_name = "chang_refsdal_lookup_perturbation"
    class_label = "chang_refsdal"

    def fit(self, segment: SegmentData, features: dict[str, float]) -> AtomFitResult:
        t = np.asarray(segment.time, dtype=float)
        if t.size == 0:
            raise ValueError("Cannot fit an empty segment.")
        t_peak = float(features.get("t_peak", t[int(np.argmax(np.abs(segment.residual)))]))
        width_t = max(float(features.get("fwhm", 0.0)), float(features.get("cadence", 0.0)), 1e-6)
        sqrt_q0 = max(width_t / max(segment.pspl.tE, 1e-12), 2e-3)
        best: AtomFitResult | None = None
        sqrt_q_factors = tuple(float(v) for v in self.config.cr_lookup_sqrt_q_factors)
        if not sqrt_q_factors:
            sqrt_q_factors = (1.0,)
        source_radius_grid = tuple(float(v) for v in self.config.cr_lookup_source_radius_grid)
        if not source_radius_grid:
            source_radius_grid = (0.0,)
        for branch in ("major", "minor"):
            x, y = pspl_image_position(t, segment.pspl, branch=branch)
            x_peak, y_peak = pspl_image_position(np.asarray([t_peak]), segment.pspl, branch=branch)
            guesses = [
                np.asarray([float(x_peak[0]), float(y_peak[0]), np.log(max(factor * sqrt_q0, 1e-4))], dtype=float)
                for factor in sqrt_q_factors
            ]
            span = max(float(np.max(np.hypot(x - x_peak[0], y - y_peak[0]))), sqrt_q0, 0.1)
            for source_radius_hat in source_radius_grid:
                fit = self._fit_profiled(
                    segment=segment,
                    features=features,
                    theta0_list=guesses,
                    bounds=[
                        (float(x_peak[0]) - 2.0 * span, float(x_peak[0]) + 2.0 * span),
                        (float(y_peak[0]) - 2.0 * span, float(y_peak[0]) + 2.0 * span),
                        (np.log(1e-4), np.log(max(5.0 * span, 1e-3))),
                    ],
                    shape_from_theta=lambda theta, time, br=branch, rho_hat=source_radius_hat: self._shape(
                        theta,
                        time,
                        segment,
                        branch=br,
                        source_radius_hat=rho_hat,
                    ),
                    params_from_theta=lambda theta, br=branch, rho_hat=source_radius_hat: {
                        "image_branch": 1.0 if br == "major" else -1.0,
                        "x_planet": float(theta[0]),
                        "y_planet": float(theta[1]),
                        "sqrt_q_local": float(np.exp(theta[2])),
                        "q_local": float(np.exp(2.0 * theta[2])),
                        "image_width": float(np.exp(theta[2])),
                        "source_radius_hat": float(max(rho_hat, 0.0)),
                        "rho_over_sqrt_q": float(max(rho_hat, 0.0)),
                        "rho_local": float(max(rho_hat, 0.0) * np.exp(theta[2])),
                        "gamma_local": float(self._local_shear(theta[0], theta[1])),
                        "lookup_grid": float(self.config.cr_lookup_grid_size),
                        "lookup_gamma_step": float(self.config.cr_lookup_gamma_step),
                    },
                    expected_amplitude_sign=None,
                    extra_warnings=("Chang-Refsdal finite-source lookup; local q and rho are approximate",),
                )
                if best is None or fit.bic < best.bic:
                    best = fit
        if best is None:
            raise RuntimeError("Chang-Refsdal fit did not produce any candidate.")
        return best

    def _shape(
        self,
        theta: np.ndarray,
        time: np.ndarray,
        segment: SegmentData,
        *,
        branch: str,
        source_radius_hat: float,
    ) -> np.ndarray:
        x, y = pspl_image_position(time, segment.pspl, branch=branch)
        sqrt_q = max(float(np.exp(theta[2])), 1e-8)
        gamma = ChangRefsdalPerturbationAtom._local_shear(theta[0], theta[1])
        xi = (x - theta[0]) / sqrt_q
        eta = (y - theta[1]) / sqrt_q
        return (
            cr_relative_magnification(
                xi,
                eta,
                gamma,
                source_radius_hat=source_radius_hat,
                grid_size=int(self.config.cr_lookup_grid_size),
                extent=float(self.config.cr_lookup_extent),
                gamma_step=float(self.config.cr_lookup_gamma_step),
            )
            - 1.0
        )

    @staticmethod
    def _local_shear(x: float, y: float) -> float:
        r = max(float(np.hypot(x, y)), 1e-6)
        return 1.0 / (r * r)


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
    lookup for seed generation, not a replacement for a full binary-lens fit.
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
    value = float(np.clip(abs(float(gamma)), 0.0, 0.95))
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
    # y = x - x/|x|^2 - Gamma * (x, -y).  The sign convention only rotates the
    # shear axes; the lookup is used as a morphology family.
    sx = xx - np.where(finite, xx / r2, 0.0) - gamma * xx
    sy = yy - np.where(finite, yy / r2, 0.0) + gamma * yy
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
