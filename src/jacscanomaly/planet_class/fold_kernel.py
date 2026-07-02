from __future__ import annotations

import functools
import math

import numpy as np


class FoldKernelLookup:
    """
    Lookup-table evaluator for the uniform finite-source fold kernel.

    The table is built in two coordinates: a dense linear grid near the caustic
    edge and a logarithmic grid for the long positive tail.  Interpolation is
    linear in kernel value near the edge and log-log in the tail, preserving the
    z^-1/2 asymptotic behavior better than a single linear grid.
    """

    def __init__(
        self,
        *,
        z_edge_max: float = 4.0,
        z_tail_max: float = 1.0e4,
        n_edge: int = 1600,
        n_tail: int = 800,
        n_quad: int = 96,
    ):
        self.z_edge_max = float(z_edge_max)
        self.z_tail_max = float(z_tail_max)
        self.n_edge = int(n_edge)
        self.n_tail = int(n_tail)
        self.n_quad = int(n_quad)

        edge = np.linspace(-1.0, self.z_edge_max, self.n_edge)
        tail = np.geomspace(self.z_edge_max, self.z_tail_max, self.n_tail)
        self.z_grid = np.unique(np.r_[edge, tail])
        self.g_grid = fold_g0_integral(self.z_grid, n_quad=self.n_quad)
        positive = self.z_grid > self.z_edge_max
        self._tail_z = self.z_grid[positive]
        self._tail_g = np.maximum(self.g_grid[positive], 1e-300)

    def __call__(self, z) -> np.ndarray:
        z_arr = np.asarray(z, dtype=float)
        out = np.zeros_like(z_arr, dtype=float)
        finite = np.isfinite(z_arr)
        active = finite & (z_arr > -1.0)
        if not np.any(active):
            return out

        za = z_arr[active]
        direct = za <= self.z_edge_max
        vals = np.zeros_like(za, dtype=float)
        if np.any(direct):
            vals[direct] = np.interp(za[direct], self.z_grid, self.g_grid)
        if np.any(~direct):
            tail_z = np.minimum(za[~direct], self.z_tail_max)
            vals[~direct] = np.exp(
                np.interp(
                    np.log(tail_z),
                    np.log(self._tail_z),
                    np.log(self._tail_g),
                )
            )
            beyond = za[~direct] > self.z_tail_max
            if np.any(beyond):
                vals_tail = vals[~direct]
                vals_tail[beyond] = vals_tail[beyond] * np.sqrt(self.z_tail_max / za[~direct][beyond])
                vals[~direct] = vals_tail
        out[active] = vals
        return out


@functools.lru_cache(maxsize=8)
def default_fold_kernel_lookup() -> FoldKernelLookup:
    return FoldKernelLookup()


@functools.lru_cache(maxsize=16)
def _legendre_nodes(n: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(int(n))
    return np.asarray(x, dtype=float), np.asarray(w, dtype=float)


def fold_g0(z, *, lookup: FoldKernelLookup | None = None) -> np.ndarray:
    """
    Evaluate the uniform-source straight-fold kernel using a lookup table.
    """

    table = default_fold_kernel_lookup() if lookup is None else lookup
    return table(z)


def fold_g0_integral(z, *, n_quad: int = 96) -> np.ndarray:
    """
    Uniform-source straight-fold kernel.

    The normalization follows the finite-source fold integral with n=0:

        G0(z) = C * int_{max(-z, -1)}^1 (1 - x^2)^(1/2) / sqrt(x + z) dx

    with C = Gamma(2) / (sqrt(pi) Gamma(3/2)) = 2 / pi.
    The contribution is zero for z <= -1. This low-level function performs
    quadrature directly and is mainly used to build lookup tables and tests.
    """

    z_arr = np.asarray(z, dtype=float)
    out = np.zeros_like(z_arr, dtype=float)
    active = np.isfinite(z_arr) & (z_arr > -1.0)
    if not np.any(active):
        return out

    nodes, weights = _legendre_nodes(int(n_quad))
    za = z_arr[active]
    vals = np.zeros_like(za, dtype=float)
    regular = za >= 1.0
    if np.any(regular):
        vals[regular] = _fold_g0_regular(za[regular], nodes, weights)
    if np.any(~regular):
        vals[~regular] = _fold_g0_edge(za[~regular], nodes, weights)
    out[active] = vals
    return out


class LimbDarkenedFoldKernelLookup:
    """
    Lookup-table evaluator for G0 + Gamma * (G1/2 - G0).
    """

    def __init__(
        self,
        *,
        z_edge_max: float = 4.0,
        z_tail_max: float = 1.0e4,
        n_edge: int = 1600,
        n_tail: int = 800,
        n_quad: int = 96,
    ):
        self.z_edge_max = float(z_edge_max)
        self.z_tail_max = float(z_tail_max)
        edge = np.linspace(-1.0, self.z_edge_max, int(n_edge))
        tail = np.geomspace(self.z_edge_max, self.z_tail_max, int(n_tail))
        self.z_grid = np.unique(np.r_[edge, tail])
        self.g0_grid = fold_gn_integral(self.z_grid, n=0.0, n_quad=n_quad)
        self.g_half_grid = fold_gn_integral(self.z_grid, n=0.5, n_quad=n_quad)
        positive = self.z_grid > self.z_edge_max
        self._tail_z = self.z_grid[positive]
        self._tail_g0 = np.maximum(self.g0_grid[positive], 1e-300)
        self._tail_g_half = np.maximum(self.g_half_grid[positive], 1e-300)

    def __call__(self, z, gamma: float = 0.0) -> np.ndarray:
        gamma = float(np.clip(gamma, 0.0, 1.0))
        return self.g0(z) + gamma * (self.g_half(z) - self.g0(z))

    def g0(self, z) -> np.ndarray:
        return self._interp(z, self.g0_grid, self._tail_g0)

    def g_half(self, z) -> np.ndarray:
        return self._interp(z, self.g_half_grid, self._tail_g_half)

    def _interp(self, z, grid_values: np.ndarray, tail_values: np.ndarray) -> np.ndarray:
        z_arr = np.asarray(z, dtype=float)
        out = np.zeros_like(z_arr, dtype=float)
        active = np.isfinite(z_arr) & (z_arr > -1.0)
        if not np.any(active):
            return out
        za = z_arr[active]
        direct = za <= self.z_edge_max
        vals = np.zeros_like(za)
        if np.any(direct):
            vals[direct] = np.interp(za[direct], self.z_grid, grid_values)
        if np.any(~direct):
            tail_z = np.minimum(za[~direct], self.z_tail_max)
            vals[~direct] = np.exp(
                np.interp(np.log(tail_z), np.log(self._tail_z), np.log(tail_values))
            )
            beyond = za[~direct] > self.z_tail_max
            if np.any(beyond):
                vals_tail = vals[~direct]
                vals_tail[beyond] = vals_tail[beyond] * np.sqrt(self.z_tail_max / za[~direct][beyond])
                vals[~direct] = vals_tail
        out[active] = vals
        return out


@functools.lru_cache(maxsize=4)
def default_limb_darkened_fold_kernel_lookup() -> LimbDarkenedFoldKernelLookup:
    return LimbDarkenedFoldKernelLookup()


def fold_limb_darkened(z, gamma: float = 0.0, *, lookup: LimbDarkenedFoldKernelLookup | None = None) -> np.ndarray:
    table = default_limb_darkened_fold_kernel_lookup() if lookup is None else lookup
    return table(z, gamma=gamma)


def fold_gn_integral(z, *, n: float, n_quad: int = 96) -> np.ndarray:
    z_arr = np.asarray(z, dtype=float)
    out = np.zeros_like(z_arr, dtype=float)
    active = np.isfinite(z_arr) & (z_arr > -1.0)
    if not np.any(active):
        return out
    nodes, weights = _legendre_nodes(int(n_quad))
    za = z_arr[active]
    vals = np.zeros_like(za)
    regular = za >= 1.0
    if np.any(regular):
        vals[regular] = _fold_gn_regular(za[regular], nodes, weights, n=float(n))
    if np.any(~regular):
        vals[~regular] = _fold_gn_edge(za[~regular], nodes, weights, n=float(n))
    coeff = math.gamma(float(n) + 2.0) / (math.sqrt(math.pi) * math.gamma(float(n) + 1.5))
    out[active] = coeff * vals
    return out


def _fold_g0_regular(z: np.ndarray, nodes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x = nodes[None, :]
    w = weights[None, :]
    zz = z[:, None]
    integrand = np.sqrt(np.maximum(1.0 - x * x, 0.0)) / np.sqrt(np.maximum(x + zz, 1e-300))
    return (2.0 / np.pi) * np.sum(w * integrand, axis=1)


def _fold_g0_edge(z: np.ndarray, nodes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # Transform x = -z + y^2. This removes the 1/sqrt(x+z) endpoint singularity.
    ymax = np.sqrt(np.maximum(1.0 + z, 0.0))
    y = 0.5 * ymax[:, None] * (nodes[None, :] + 1.0)
    w = 0.5 * ymax[:, None] * weights[None, :]
    x = -z[:, None] + y * y
    integrand = 2.0 * np.sqrt(np.maximum(1.0 - x * x, 0.0))
    return (2.0 / np.pi) * np.sum(w * integrand, axis=1)


def _fold_gn_regular(z: np.ndarray, nodes: np.ndarray, weights: np.ndarray, *, n: float) -> np.ndarray:
    x = nodes[None, :]
    w = weights[None, :]
    zz = z[:, None]
    integrand = np.maximum(1.0 - x * x, 0.0) ** (n + 0.5) / np.sqrt(np.maximum(x + zz, 1e-300))
    return np.sum(w * integrand, axis=1)


def _fold_gn_edge(z: np.ndarray, nodes: np.ndarray, weights: np.ndarray, *, n: float) -> np.ndarray:
    ymax = np.sqrt(np.maximum(1.0 + z, 0.0))
    y = 0.5 * ymax[:, None] * (nodes[None, :] + 1.0)
    w = 0.5 * ymax[:, None] * weights[None, :]
    x = -z[:, None] + y * y
    integrand = 2.0 * np.maximum(1.0 - x * x, 0.0) ** (n + 0.5)
    return np.sum(w * integrand, axis=1)
