# scanomaly/plot.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from .utils import calc_A_pspl
from .utils import predict_flat_model, predict_anom_model


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
                t0, tE, u0 = map(float, np.asarray(result.fit.params))
                hw = float(a * abs(tE * u0))
            else:
                hw = float(a * result.best.teff)

        else:  # "pspl"
            t0, tE, u0 = map(float, np.asarray(result.fit.params))
            hw = float(a * abs(tE * u0))

        return (t_center - hw, t_center + hw)

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
    ):
        """
        Plot light curve and PSPL best-fit model.

        Returns (fig, ax).
        """
        t, f, e = result.time, result.flux, result.ferr
        m = result.model_flux

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        ax.errorbar(t, f, yerr=e, fmt=".", label="data", zorder=0)
        ax.plot(t, m, label="PSPL model", zorder=1)
        ax.set_xlabel("time")
        ax.set_ylabel("flux")
        ax.legend()

        xl = self._compute_xlim(
            result, center=center, width_mode=width_mode, a=a, xlim=xlim, half_width=half_width
        )
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

        # CPU arrays
        t_np = np.asarray(result.time)
        r_np = np.asarray(result.residual)
        e_np = np.asarray(result.ferr)

        t0 = float(result.best.t0)
        teff = float(result.best.teff)

        # x window for chi2 evaluation
        w = teff_coeff * teff
        mask = (t_np >= (t0 - w)) & (t_np <= (t0 + w))

        # JAX arrays for prediction
        t = jnp.asarray(t_np)
        r = jnp.asarray(r_np)
        e = jnp.asarray(e_np)

        y_flat = None
        y_anom = None

        if show_flat:
            # predict_flat_model should accept (data_flux, data_ferr) or (r,e) depending on your definition
            y_flat = np.asarray(jax.device_get(predict_flat_model(r, e)))

        if show_anom:
            y_anom_j, _choose = predict_anom_model(t0, teff, t, r, e)
            y_anom = np.asarray(jax.device_get(y_anom_j))

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if use_errorbar:
            ax.errorbar(t_np, r_np, yerr=e_np, fmt=".")
        else:
            ax.plot(t_np, r_np, ".")

        # Draw model lines ONLY within the chi2 window
        if y_flat is not None:
            ax.plot(t_np[mask], y_flat[mask], label="flat", c="C1")

        if y_anom is not None:
            ax.plot(t_np[mask], y_anom[mask], label="anomaly", c="r")

        # range
        if xlim is None:
            hw = a * teff
            xlim = (t0 - hw, t0 + hw)
        ax.set_xlim(xlim)

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
        show: bool = True,
        figsize=(10, 8),
        height_ratios=(3, 1, 1),
        show_anomaly_window: bool = False,
        teff_coeff: float = 3.0,
    ):
        """
        3-panel plot:
          1) data + PSPL model
          2) residual
          3) clusters (t0 vs dchi2)

        Range control:
          - center/width_mode/a OR xlim/half_width
        """
        t = np.asarray(result.time)
        f = np.asarray(result.flux)
        e = np.asarray(result.ferr)
        res = np.asarray(result.residual)
        clusters = np.asarray(result.clusters_all)

        # PSPL best params
        t0_b, tE_b, u0_b = map(float, np.asarray(result.fit.params))
        fs_b = float(np.asarray(result.fit.fs))
        fb_b = float(np.asarray(result.fit.fb))

        xl = self._compute_xlim(
            result, center=center, width_mode=width_mode, a=a, xlim=xlim, half_width=half_width
        )

        # model curve in that x-range
        t_plot = np.linspace(xl[0], xl[1], 1000)
        A_plot = np.asarray(calc_A_pspl(t0_b, tE_b, u0_b, t_plot))
        f_plot = A_plot * fs_b + fb_b

        fig, axes = plt.subplots(
            3, 1, figsize=figsize, sharex=True, height_ratios=height_ratios
        )

        # 1) data + model
        ax = axes[0]
        ax.errorbar(t, f, yerr=e, fmt="o", markersize=2, alpha=0.7, label="data", zorder=0)
        ax.plot(t_plot, f_plot, lw=2, label="PSPL best-fit", zorder=1)
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

        if show:
            plt.show()
        return fig, axes

@dataclass
class PSPLPlotter:
    """
    Plot utilities for PSPL fitting results only.

    This plotter mirrors the interface philosophy of AnomalyPlotter,
    but operates directly on PSPLFitResult.

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
        Compute xlim for PSPL plots.
        """
        if xlim is not None:
            return xlim

        t0, tE, u0 = map(float, np.asarray(fit.params))

        if width_mode == "custom":
            if half_width is None:
                raise ValueError("width_mode='custom' requires half_width.")
            hw = float(half_width)
        else:  # "pspl"
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
    ):
        """
        Plot light curve with PSPL best-fit model.

        API-consistent with AnomalyPlotter.plot_lc_with_model.
        """
        t = np.asarray(fit.time)
        f = np.asarray(fit.flux)
        e = np.asarray(fit.ferr)
        m = np.asarray(fit.model_flux)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        ax.errorbar(t, f, yerr=e, fmt=".", label="data", zorder=0)
        ax.plot(t, m, lw=2, label="PSPL model", zorder=1)

        ax.set_xlabel("time")
        ax.set_ylabel("flux")
        ax.legend()
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
        Plot PSPL residual (flux - model_flux).

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
