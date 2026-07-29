# scanomaly/plot.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from .anomaly_models import get_flat_plot_model_masked, get_anom_plot_model_masked
from .singlelens_model import (
    A_pspl_func,
    A_fspl_func,
    A_pspl_parallax_func,
    A_fspl_parallax_func,
    A_pspl_space_parallax_func,
    A_fspl_space_parallax_func,
    A_cv_asymexp_logtau_func,
)


def _single_lens_model_flux(fit, time) -> np.ndarray:
    """
    Evaluate the fitted single-lens model at arbitrary times.
    """
    names = tuple(getattr(fit, "param_names", ()))
    projector = getattr(fit, "parallax_projector", None)
    raw_params = getattr(fit, "raw_params", None)
    if (
        projector is not None
        and hasattr(projector, "magnification_at")
        and raw_params is not None
    ):
        magnification = projector.magnification_at(
            np.asarray(time, dtype=float),
            np.asarray(raw_params, dtype=float),
        )
        return (
            float(np.asarray(fit.fs)) * magnification
            + float(np.asarray(fit.fb))
        )
    params = jnp.asarray(fit.params)
    time_j = jnp.asarray(time)

    if names == ("t0", "tE", "u0", "rho", "piEN", "piEE"):
        P = getattr(fit, "parallax_projector", None)
        if P is not None and hasattr(P, "NE_ref"):
            return _gulls_vbm_model_flux(fit, time)

    if names == ("t0", "tE", "u0"):
        A = A_pspl_func(params, time_j)
    elif names == ("t0", "tE", "u0", "rho"):
        A = A_fspl_func(params, time_j)
    elif names == ("t0", "tE", "u0", "piEN", "piEE"):
        P = getattr(fit, "parallax_projector", None)
        if P is None:
            raise ValueError("Cannot plot parallax model without fit.parallax_projector.")
        if hasattr(P, "earth") or hasattr(P, "NE_ref"):
            A = A_pspl_space_parallax_func(params, time_j, P)
        else:
            A = A_pspl_parallax_func(params, time_j, P)
    elif names == ("t0", "tE", "u0", "rho", "piEN", "piEE"):
        P = getattr(fit, "parallax_projector", None)
        if P is None:
            raise ValueError("Cannot plot parallax model without fit.parallax_projector.")
        if hasattr(P, "earth") or hasattr(P, "NE_ref"):
            A = A_fspl_space_parallax_func(params, time_j, P)
        else:
            A = A_fspl_parallax_func(params, time_j, P)
    elif names == ("t0", "tau_rise", "tau_decay"):
        t0, tau_rise, tau_decay = params
        q = jnp.array([t0, jnp.log(tau_rise), jnp.log(tau_decay)])
        A = A_cv_asymexp_logtau_func(q, time_j)
    else:
        raise ValueError(f"Unsupported single-lens parameter set: {names}")

    flux = fit.fs * A + fit.fb
    return np.asarray(jax.device_get(flux), dtype=float)


def _gulls_vbm_model_flux(fit, time) -> np.ndarray:
    try:
        import VBMicrolensing
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise ImportError("VBMicrolensing is required to plot GULLS FSPL fits.") from exc

    P = getattr(fit, "parallax_projector", None)
    if P is None:
        raise ValueError("Cannot plot GULLS parallax model without fit.parallax_projector.")

    t0, tE, u0, rho, piEN, piEE = map(float, np.asarray(fit.params, dtype=float)[:6])
    t = np.asarray(time, dtype=float)
    t_eval = t + float(np.asarray(P.time_add))
    t_grid = np.asarray(P.t, dtype=float)
    r_grid = np.asarray(P.r, dtype=float)
    r_t = np.column_stack([np.interp(t_eval, t_grid, r_grid[:, j]) for j in range(3)])
    north = np.asarray(P.sky_north, dtype=float)
    east = np.asarray(P.sky_east, dtype=float)
    ne_t = np.column_stack([r_t @ north, r_t @ east])
    d_ne = (
        ne_t
        - np.asarray(P.NE_ref, dtype=float)[None, :]
        - (t_eval - float(np.asarray(P.tref)))[:, None] * np.asarray(P.NE_vref, dtype=float)[None, :]
    )
    d_n = d_ne[:, 0]
    d_e = d_ne[:, 1]
    d_tau = -(piEN * d_n + piEE * d_e)
    d_beta = piEE * d_n - piEN * d_e
    tE_safe = max(abs(tE), 1e-12)
    tau = (t - t0) / tE_safe + d_tau
    beta = u0 + d_beta
    u = np.sqrt(tau * tau + beta * beta)

    vbm = VBMicrolensing.VBMicrolensing()
    vbm.Tol = 1e-4
    vbm.RelTol = 1e-4
    rho_safe = max(float(rho), 1e-12)
    A = np.asarray([vbm.ESPLMag(float(ui), rho_safe) for ui in u], dtype=float)
    return float(np.asarray(fit.fs)) * A + float(np.asarray(fit.fb))


def _adaptive_single_lens_curve(
    fit,
    xlim: Tuple[float, float],
    *,
    base_points: int = 128,
    max_flux_step: float | None = None,
    min_time_step: float | None = None,
    max_points: int = 12000,
    max_iter: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample the model only inside xlim, densifying intervals with large flux jumps.
    """
    xmin, xmax = map(float, xlim)
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if xmax == xmin:
        xmax = xmin + 1.0

    n0 = max(2, int(base_points))
    x = np.linspace(xmin, xmax, n0)
    y = _single_lens_model_flux(fit, x)

    if max_flux_step is None:
        ferr = np.asarray(getattr(fit, "ferr", []), dtype=float)
        ferr = ferr[np.isfinite(ferr) & (ferr > 0)]
        if ferr.size:
            max_flux_step = 0.25 * float(np.median(ferr))
        else:
            flux = np.asarray(getattr(fit, "flux", []), dtype=float)
            flux = flux[np.isfinite(flux)]
            scale = np.ptp(flux) if flux.size else 1.0
            max_flux_step = 0.0025 * max(float(scale), 1.0)
    max_flux_step = max(float(max_flux_step), np.finfo(float).eps)

    if min_time_step is None:
        min_time_step = max((xmax - xmin) / max_points, 1e-6)
    min_time_step = max(float(min_time_step), np.finfo(float).eps)

    for _ in range(max_iter):
        if len(x) >= max_points:
            break

        dx = np.diff(x)
        refineable = np.flatnonzero(dx > min_time_step)
        if refineable.size == 0:
            break

        # Evaluate midpoints for all refineable intervals so we can check both
        # jump size and curvature. Curvature = |y_mid - linear_interp| catches
        # the "equal endpoints but bump in between" (trapezoid) case that a
        # pure jump criterion misses.
        mids = 0.5 * (x[refineable] + x[refineable + 1])
        y_mids = _single_lens_model_flux(fit, mids)

        dy_abs = np.abs(y[refineable + 1] - y[refineable])
        y_interp = 0.5 * (y[refineable] + y[refineable + 1])
        curvature = np.abs(y_mids - y_interp)

        needs_split = (dy_abs > max_flux_step) | (curvature > max_flux_step)
        if not needs_split.any():
            break

        ins = np.flatnonzero(needs_split)
        available = max_points - len(x)
        if available <= 0:
            break
        if ins.size > available:
            priority = np.maximum(dy_abs[ins], curvature[ins])
            order = np.argsort(priority)[-available:]
            ins = np.sort(ins[order])

        x = np.concatenate([x, mids[ins]])
        y = np.concatenate([y, y_mids[ins]])
        order = np.argsort(x)
        x = x[order]
        y = y[order]

    return x, y


@dataclass
class AnomalyPlotter:
    """
    Plot utilities for scanomaly results.

    Conventions
    -----------
    - "pspl": uses PSPL best-fit parameters to define center/width
    - "anomaly": uses the best anomaly candidate (result.best) to define center/width

    Notes
    -----
    - All plot methods return (fig, ax/axes). If show=False, nothing is displayed
      and you can adjust limits/styles outside.
    """

    # ----------------------------
    # helpers
    # ----------------------------
    def _compute_xlim(
        self,
        result,
        *,
        center: str = "pspl",          # "pspl" or "anomaly"
        width_mode: str = "pspl",      # "pspl" or "anomaly" or "custom"
        a: float = 3.0,
        xlim: Tuple[float, float] | None = None,
        half_width: float | None = None,
        min_hw: float | None = None,
    ) -> Tuple[float, float]:
        """
        Compute xlim for plots.

        Parameters
        ----------
        center : {"pspl", "anomaly"} (compat: "best")
            Center of the view window.
        width_mode : {"pspl", "anomaly", "custom"} (compat: "best")
            How to compute window half-width.
            - "pspl": half_width = a * tE * u0
            - "anomaly": half_width = a * teff (from result.best)
            - "custom": half_width must be provided.
        a : float
            Multiplier for the default half-width rule.
        xlim : tuple | None
            If provided, returned as-is.
        half_width : float | None
            Used only when width_mode="custom".
        min_hw : float | None
            Minimum half-width enforced after the normal calculation.
            Prevents degenerate (u0≈0) fits from producing a near-zero window.

        Returns
        -------
        (xmin, xmax)
        """
        if xlim is not None:
            return xlim

        # backward compat
        if center == "best":
            center = "anomaly"
        if width_mode == "best":
            width_mode = "anomaly"

        # center
        if center == "anomaly" and (getattr(result, "best", None) is not None):
            t_center = float(result.best.t0)
        else:
            t_center = float(np.asarray(result.fit.params)[0])

        # width
        if width_mode == "custom":
            if half_width is None:
                raise ValueError("When width_mode='custom', half_width must be specified.")
            hw = float(half_width)

        elif width_mode == "anomaly":
            if getattr(result, "best", None) is None:
                t0, tE, u0 = map(float, np.asarray(result.fit.params)[:3])
                hw = float(a * abs(tE * u0))
            else:
                hw = float(a * result.best.teff)

        else:  # "pspl"
            t0, tE, u0 = map(float, np.asarray(result.fit.params)[:3])
            hw = float(a * abs(tE * u0))

        # Apply minimum half-width (prevents degenerate u0≈0 from collapsing window)
        if min_hw is not None:
            hw = max(hw, float(min_hw))

        xmin = t_center - hw
        xmax = t_center + hw

        # If the best anomaly candidate is just outside the window, extend to include it.
        # Only extend when the anomaly is within 3×hw (i.e., not a completely different season).
        best = getattr(result, "best", None)
        if best is not None:
            ano_t0 = float(best.t0)
            if not (xmin <= ano_t0 <= xmax):
                dist = min(abs(ano_t0 - xmin), abs(ano_t0 - xmax))
                if dist <= 2.0 * hw:
                    buf = max(float(best.teff) * 5.0, 5.0)
                    if ano_t0 < xmin:
                        xmin = ano_t0 - buf
                    else:
                        xmax = ano_t0 + buf

        return (xmin, xmax)

    # ----------------------------
    # basic plots
    # ----------------------------
    def plot_lc(
        self,
        result,
        *,
        show: bool = True,
        ax=None,
        center: str = "pspl",          # "pspl" or "anomaly"
        width_mode: str = "pspl",      # "pspl" or "anomaly" or "custom"
        a: float = 3.0,
        xlim: Tuple[float, float] | None = None,
        half_width: float | None = None,
        model_base_points: int = 128,
        model_max_flux_step: float | None = None,
        model_min_time_step: float | None = None,
        model_max_points: int = 12000,
    ):
        """
        Plot light curve and PSPL best-fit model.

        Returns (fig, ax).
        """
        t, f, e = result.time, result.flux, result.ferr
        xl = self._compute_xlim(
            result, center=center, width_mode=width_mode, a=a, xlim=xlim, half_width=half_width
        )
        t_plot, m_plot = _adaptive_single_lens_curve(
            result.fit,
            xl,
            base_points=model_base_points,
            max_flux_step=model_max_flux_step,
            min_time_step=model_min_time_step,
            max_points=model_max_points,
        )

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        ax.errorbar(t, f, yerr=e, fmt=".", label="data", zorder=0)
        ax.plot(t_plot, m_plot, label="PSPL model", zorder=1)
        ax.set_xlabel("time")
        ax.set_ylabel("flux")
        ax.legend()

        ax.set_xlim(xl)

        if show:
            plt.show()
        return fig, ax

    def plot_residual(
        self,
        result,
        *,
        show: bool = True,
        ax=None,
        center: str = "pspl",          # "pspl" or "anomaly"
        width_mode: str = "pspl",      # "pspl" or "anomaly" or "custom"
        a: float = 50.0,
        xlim: Tuple[float, float] | None = None,
        half_width: float | None = None,
    ):
        """
        Plot PSPL residual (flux - model_flux).

        Returns (fig, ax).
        """
        t, r = result.time, result.residual

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        ax.plot(t, r, ".", zorder=0)
        ax.axhline(0.0, zorder=1, c="C1")

        ax.set_xlabel("time")
        ax.set_ylabel("residual")

        xl = self._compute_xlim(
            result, center=center, width_mode=width_mode, a=a, xlim=xlim, half_width=half_width
        )
        ax.set_xlim(xl)

        if show:
            plt.show()
        return fig, ax

    # ----------------------------
    # anomaly window plot
    # ----------------------------
    def plot_anomaly_window(
        self,
        result,
        *,
        show: bool = True,
        ax=None,
        xlim: Tuple[float, float] | None = None,
        a: float = 5.0,              # xlim = teff * a
        show_flat: bool = True,
        show_anom: bool = True,
        teff_coeff: float = 3.0,
        use_errorbar: bool = True,
    ):
        """
        Plot residual around the best anomaly candidate, and overlay template models
        ONLY inside the chi2 evaluation window.

        - Data: residual vs time (optionally errorbar)
        - Model lines are drawn only within |t - t0| <= teff_coeff * teff

        Returns (fig, ax).
        """
        if getattr(result, "best", None) is None:
            return None, None

        t0 = float(result.best.t0)
        teff = float(result.best.teff)
        w = teff_coeff * teff

        # CPU arrays
        t_np = np.asarray(result.time)
        r_np = np.asarray(result.residual)
        e_np = np.asarray(result.ferr)
        t_plot_np = np.arange(t0 - w, t0 + w + 0.001, 0.001)

        # x window for chi2 evaluation
        mask = (t_np >= (t0 - w)) & (t_np <= (t0 + w))
        mask_j = jnp.asarray(mask)

        # JAX arrays for prediction
        t = jnp.asarray(t_np)
        r = jnp.asarray(r_np)
        e = jnp.asarray(e_np)
        w_j = 1.0 / (e ** 2)
        t_plot = jnp.asarray(t_plot_np)

        y_flat = None
        y_anom = None

        if show_flat:
            # predict_flat_model should accept (data_flux, data_ferr) or (r,e) depending on your definition
            y_flat = np.asarray(jax.device_get(get_flat_plot_model_masked(t_plot, r, w_j, mask_j)))

        if show_anom:
            y_anom_j, _ = get_anom_plot_model_masked(t_plot, t0, teff, t, r, w_j, mask_j)
            y_anom = np.asarray(jax.device_get(y_anom_j))

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if use_errorbar:
            ax.errorbar(t_np, r_np, yerr=e_np, fmt=".", zorder=2)
        else:
            ax.plot(t_np, r_np, ".", zorder=2)

        # Draw model lines ONLY within the chi2 window
        if y_flat is not None:
            ax.plot(t_plot_np, y_flat, label="flat", c="C1", zorder=3)

        if y_anom is not None:
            ax.plot(t_plot_np, y_anom, label="anomaly", c="r", zorder=3)

        # range
        if xlim is None:
            hw = a * teff
            xlim = (t0 - hw, t0 + hw)
        ax.set_xlim(xlim)

        # Set ylim from the visible x-window only so out-of-window points
        # do not inflate the vertical range.
        vis = (t_np >= xlim[0]) & (t_np <= xlim[1])
        if vis.any():
            y_samples = [r_np[vis] - e_np[vis], r_np[vis] + e_np[vis]]

            line_vis = (t_plot_np >= xlim[0]) & (t_plot_np <= xlim[1])
            if line_vis.any():
                if y_flat is not None:
                    y_samples.append(y_flat[line_vis])
                if y_anom is not None:
                    y_samples.append(y_anom[line_vis])

            y_all = np.concatenate(y_samples)
            y_lo = np.percentile(y_all, 1)
            y_hi = np.percentile(y_all, 99)
            if y_hi > y_lo:
                mg = 0.15 * (y_hi - y_lo)
                ax.set_ylim(y_lo - mg, y_hi + mg)

        ax.set_xlabel("time")
        ax.set_ylabel("residual")
        ax.legend()

        if show:
            plt.show()
        return fig, ax

    # ----------------------------
    # tripanel plot
    # ----------------------------
    def plot_result(
        self,
        result,
        *,
        center: str = "pspl",          # "pspl" or "anomaly"
        width_mode: str = "pspl",      # "pspl" or "anomaly" or "custom"
        a: float = 3.0,
        xlim: Tuple[float, float] | None = None,
        half_width: float | None = None,
        min_hw: float = 30.0,
        show: bool = True,
        figsize=(10, 8),
        height_ratios=(3, 1, 1),
        show_anomaly_window: bool = False,
        teff_coeff: float = 3.0,
        model_base_points: int = 128,
        model_max_flux_step: float | None = None,
        model_min_time_step: float | None = None,
        model_max_points: int = 12000,
    ):
        """
        3-panel plot:
          1) data + PSPL model
          2) residual
          3) clusters (t0 vs dchi2)

        Range control:
          - center/width_mode/a OR xlim/half_width
          - min_hw: minimum half-width in time units (default 30 days);
            prevents degenerate u0≈0 fits from collapsing the window.
        """
        t = np.asarray(result.time)
        f = np.asarray(result.flux)
        e = np.asarray(result.ferr)
        res = np.asarray(result.residual)
        clusters = np.asarray(result.clusters_all)

        xl = self._compute_xlim(
            result, center=center, width_mode=width_mode, a=a, xlim=xlim,
            half_width=half_width, min_hw=min_hw,
        )
        t_plot, m_plot = _adaptive_single_lens_curve(
            result.fit,
            xl,
            base_points=model_base_points,
            max_flux_step=model_max_flux_step,
            min_time_step=model_min_time_step,
            max_points=model_max_points,
        )

        fig, axes = plt.subplots(
            3, 1, figsize=figsize, sharex=True, height_ratios=height_ratios
        )

        # 1) data + model
        ax = axes[0]
        ax.errorbar(t, f, yerr=e, fmt="o", markersize=2, alpha=0.7, label="data", zorder=0)
        ax.plot(t_plot, m_plot, lw=2, label="best model", zorder=1)
        ax.set_xlim(xl)
        ax.set_ylabel("flux")
        ax.minorticks_on()
        ax.legend()

        # 2) residual
        ax = axes[1]
        ax.errorbar(t, res, yerr=e, fmt="o", markersize=2, alpha=1.0, zorder=0)
        ax.axhline(0.0, lw=1, zorder=1, c="C1")
        ax.set_xlim(xl)
        ax.set_ylabel("residual")
        ax.minorticks_on()

        if show_anomaly_window and (getattr(result, "best", None) is not None):
            t0c = float(result.best.t0)
            w = float(teff_coeff * result.best.teff)
            ax.axvline(t0c, lw=1)
            ax.axvspan(t0c - w, t0c + w, alpha=0.1)

        # 3) clusters
        ax = axes[2]
        if clusters.size:
            ax.scatter(clusters[:, 0], clusters[:, 2], s=60, marker="x", c="r")
        ax.set_xlim(xl)
        ax.set_xlabel("time")
        ax.set_ylabel("dchi2")
        ax.minorticks_on()

        # Set data-based ylim for flux and residual panels.
        # Prevents the PSPL model spike (which is huge when u0≈0) from
        # dominating the y-axis and hiding all the data.
        vis = (t >= xl[0]) & (t <= xl[1])
        if vis.any():
            f_vis = f[vis]
            e_vis = e[vis]
            f_lo = np.percentile(f_vis - e_vis, 1)
            f_hi = np.percentile(f_vis + e_vis, 99)
            if f_hi > f_lo:
                mg = 0.15 * (f_hi - f_lo)
                axes[0].set_ylim(f_lo - mg, f_hi + mg)

            r_vis = res[vis]
            r_lo = np.percentile(r_vis - e_vis, 1)
            r_hi = np.percentile(r_vis + e_vis, 99)
            if r_hi > r_lo:
                mg = 0.15 * (r_hi - r_lo)
                axes[1].set_ylim(r_lo - mg, r_hi + mg)

        if show:
            plt.show()
        return fig, axes

@dataclass
class SingleLensPlotter:
    """
    Plot utilities for single lens fitting results only.

    This plotter mirrors the interface philosophy of AnomalyPlotter,
    but operates directly on SingleLensFitResult.

    Conventions
    -----------
    - center = "pspl" only (kept for API compatibility)
    - width_mode:
        - "pspl": a * tE * u0
        - "custom": use half_width
    """

    # ----------------------------
    # helpers
    # ----------------------------
    def _compute_xlim(
        self,
        fit,
        *,
        width_mode: str = "pspl",
        a: float = 50.0,
        xlim: tuple[float, float] | None = None,
        half_width: float | None = None,
    ) -> tuple[float, float]:
        """
        Compute xlim for single lens plots.
        """
        if xlim is not None:
            return xlim

        t0, tE, u0 = map(float, np.asarray(fit.params)[:3])

        if width_mode == "custom":
            if half_width is None:
                raise ValueError("width_mode='custom' requires half_width.")
            hw = float(half_width)
        else:
            hw = float(a * abs(tE * u0))

        return (t0 - hw, t0 + hw)

    # ----------------------------
    # basic plots
    # ----------------------------
    def plot_lc(
        self,
        fit,
        *,
        show: bool = True,
        ax=None,
        width_mode: str = "pspl",
        a: float = 3.0,
        xlim: tuple[float, float] | None = None,
        half_width: float | None = None,
        model_base_points: int = 128,
        model_max_flux_step: float | None = None,
        model_min_time_step: float | None = None,
        model_max_points: int = 12000,
    ):
        """
        Plot light curve with single lens best-fit model.

        API-consistent with AnomalyPlotter.plot_lc_with_model.
        """
        t = np.asarray(fit.time)
        f = np.asarray(fit.flux)
        e = np.asarray(fit.ferr)
        xl = self._compute_xlim(
            fit,
            width_mode=width_mode,
            a=a,
            xlim=xlim,
            half_width=half_width,
        )
        t_plot, m_plot = _adaptive_single_lens_curve(
            fit,
            xl,
            base_points=model_base_points,
            max_flux_step=model_max_flux_step,
            min_time_step=model_min_time_step,
            max_points=model_max_points,
        )

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        ax.errorbar(t, f, yerr=e, fmt=".", label="data", zorder=0)
        ax.plot(t_plot, m_plot, lw=2, label="model", zorder=1)

        ax.set_xlabel("time")
        ax.set_ylabel("flux")
        ax.legend()
        ax.minorticks_on()

        ax.set_xlim(xl)

        if show:
            plt.show()
        return fig, ax

    def plot_residual(
        self,
        fit,
        *,
        show: bool = True,
        ax=None,
        width_mode: str = "pspl",
        a: float = 3.0,
        xlim: tuple[float, float] | None = None,
        half_width: float | None = None,
        use_errorbar: bool = True,
    ):
        """
        Plot single lens residual (flux - model_flux).

        API-consistent with AnomalyPlotter.plot_residual.
        """
        t = np.asarray(fit.time)
        r = np.asarray(fit.residual)
        e = np.asarray(fit.ferr)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if use_errorbar:
            ax.errorbar(t, r, yerr=e, fmt=".", zorder=0)
        else:
            ax.plot(t, r, ".", label="residual", zorder=0)

        ax.axhline(0.0, zorder=1, c="C1")
        ax.set_xlabel("time")
        ax.set_ylabel("residual")
        ax.minorticks_on()

        xl = self._compute_xlim(
            fit,
            width_mode=width_mode,
            a=a,
            xlim=xlim,
            half_width=half_width,
        )
        ax.set_xlim(xl)

        if show:
            plt.show()
        return fig, ax
