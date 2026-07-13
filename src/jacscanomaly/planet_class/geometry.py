"""
Deterministic heuristic geometry for planetary anomalies on a PSPL baseline.

Given the refined single-lens parameters ``(t0, tE, u0)`` and the measured
anomaly time ``t_anom``, the following quantities are fixed by algebra alone
(no fitting, no lens-model assumption beyond "the anomaly marks the epoch at
which the source passes the planet-perturbed image"):

.. math::

   \\tau_{\\rm anom} = (t_{\\rm anom} - t_0)/t_E, \\qquad
   u_{\\rm anom} = \\sqrt{\\tau_{\\rm anom}^2 + u_0^2}, \\qquad
   \\tan\\alpha = u_0/\\tau_{\\rm anom},

   s^\\dagger_\\pm = \\frac{\\sqrt{u_{\\rm anom}^2 + 4} \\pm u_{\\rm anom}}{2}.

``alpha`` is the angle between the source trajectory and the planet-host axis
(mirror-degenerate under ``u0 -> -u0``).  ``s_dagger_plus`` applies to
major-image (bump) perturbations and ``s_dagger_minus`` to minor-image (dip)
perturbations; each is the geometric mean of the degenerate inner/outer
solutions, :math:`s^\\dagger = \\sqrt{s_{\\rm in} s_{\\rm out}}`.

The mass-ratio estimators convert the one remaining well-measured observable,
the anomaly duration relative to ``tE``, into ``q`` under an explicitly named
assumption:

* dips (Han 2006; Hwang et al. 2022):
  :math:`q = (\\Delta t_{\\rm dip}/4t_E)^2\\,(s^\\dagger_-/u_{\\rm anom})\\sin^2\\alpha`
  (equal to the often-quoted
  :math:`(\\Delta t_{\\rm dip}/4t_E)^2 (s^\\dagger_-/|u_0|)|\\sin^3\\alpha|`
  but regular as :math:`u_0 \\to 0`);
* bumps (Gould & Loeb 1992 order-of-magnitude): the perturbed region is the
  planet's Einstein ring, so :math:`q \\simeq (t_p/t_E)^2` with ``t_p`` the
  fitted bump width parameter.

References: Gould & Loeb (1992); Gaudi & Gould (1997); Han (2006);
Hwang et al. (2022), AJ 163, 43; Ryu et al. (2022).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


REGIME_PLANETARY = "planetary"
REGIME_CENTRAL_RESONANT = "central_or_resonant"

BRANCH_MAJOR = "major_image"
BRANCH_MINOR = "minor_image"
BRANCH_UNKNOWN = "unknown"


@dataclass(frozen=True)
class AnomalyGeometry:
    """
    Deterministic anomaly geometry in the PSPL trajectory frame.

    All quantities follow from ``(t_anom, t0, tE, u0)`` alone.  Errors are
    first-order propagations of ``t_anom_err`` only; the PSPL parameters are
    treated as fixed.
    """

    t_anom: float
    t_anom_err: float
    tau_anom: float
    tau_anom_err: float
    u_anom: float
    u_anom_err: float
    alpha: float
    alpha_err: float
    sin_alpha: float
    s_dagger_plus: float
    s_dagger_plus_err: float
    s_dagger_minus: float
    s_dagger_minus_err: float
    preferred_branch: str
    regime: str

    @property
    def s_dagger_preferred(self) -> float:
        if self.preferred_branch == BRANCH_MAJOR:
            return self.s_dagger_plus
        if self.preferred_branch == BRANCH_MINOR:
            return self.s_dagger_minus
        return float("nan")

    def summary_dict(self, *, prefix: str = "") -> dict[str, object]:
        row: dict[str, object] = {
            f"{prefix}t_anom": float(self.t_anom),
            f"{prefix}tau_anom": float(self.tau_anom),
            f"{prefix}u_anom": float(self.u_anom),
            f"{prefix}alpha": float(self.alpha),
            f"{prefix}sin_alpha": float(self.sin_alpha),
            f"{prefix}s_dagger_plus": float(self.s_dagger_plus),
            f"{prefix}s_dagger_minus": float(self.s_dagger_minus),
            f"{prefix}preferred_branch": self.preferred_branch,
            f"{prefix}regime": self.regime,
        }
        for key, value in (
            ("t_anom_err", self.t_anom_err),
            ("tau_anom_err", self.tau_anom_err),
            ("u_anom_err", self.u_anom_err),
            ("alpha_err", self.alpha_err),
            ("s_dagger_plus_err", self.s_dagger_plus_err),
            ("s_dagger_minus_err", self.s_dagger_minus_err),
        ):
            if np.isfinite(value):
                row[f"{prefix}{key}"] = float(value)
        return row


def anomaly_geometry(
    t_anom: float,
    *,
    t0: float,
    tE: float,
    u0: float,
    t_anom_err: float = float("nan"),
    preferred_branch: str = BRANCH_UNKNOWN,
    central_u_anom_max: float = 0.2,
) -> AnomalyGeometry:
    """
    Compute the deterministic anomaly geometry.

    Parameters
    ----------
    t_anom : float
        Anomaly center time measured by the shape fit.
    t0, tE, u0 : float
        Refined PSPL parameters (trajectory frame).
    t_anom_err : float, optional
        1-sigma uncertainty of ``t_anom``; propagated to all derived values.
    preferred_branch : str, optional
        ``"major_image"`` for bumps, ``"minor_image"`` for dips,
        ``"unknown"`` otherwise.
    central_u_anom_max : float, optional
        Below this ``u_anom`` both ``s_dagger`` branches approach 1 and a
        central-caustic origin cannot be excluded, so the regime is flagged
        ``"central_or_resonant"`` and ``q`` estimates are unreliable.
    """
    tE = float(tE)
    if not np.isfinite(tE) or tE <= 0.0:
        raise ValueError("anomaly_geometry requires a positive, finite tE.")
    u0 = abs(float(u0))
    tau = (float(t_anom) - float(t0)) / tE
    u_anom = float(np.hypot(tau, u0))
    alpha = float(np.arctan2(u0, tau))
    sin_alpha = u0 / u_anom if u_anom > 0.0 else 0.0
    # s_dagger_plus/minus equal the PSPL major/minor image radii r_+(u_anom)
    # and r_-(u_anom) = 1/r_+(u_anom).
    root = float(np.sqrt(u_anom * u_anom + 4.0))
    s_plus = 0.5 * (root + u_anom)
    s_minus = 0.5 * (root - u_anom)

    tau_err = abs(float(t_anom_err)) / tE
    if np.isfinite(tau_err) and u_anom > 0.0:
        u_anom_err = abs(tau / u_anom) * tau_err
        alpha_err = (u0 / (u_anom * u_anom)) * tau_err
        # ds_dagger_plus/du = s_plus/root, ds_dagger_minus/du = -s_minus/root.
        s_plus_err = (s_plus / root) * u_anom_err
        s_minus_err = (s_minus / root) * u_anom_err
    else:
        u_anom_err = alpha_err = s_plus_err = s_minus_err = float("nan")

    regime = (
        REGIME_CENTRAL_RESONANT
        if u_anom <= float(central_u_anom_max)
        else REGIME_PLANETARY
    )
    return AnomalyGeometry(
        t_anom=float(t_anom),
        t_anom_err=abs(float(t_anom_err)),
        tau_anom=tau,
        tau_anom_err=tau_err,
        u_anom=u_anom,
        u_anom_err=u_anom_err,
        alpha=alpha,
        alpha_err=alpha_err,
        sin_alpha=sin_alpha,
        s_dagger_plus=s_plus,
        s_dagger_plus_err=s_plus_err,
        s_dagger_minus=s_minus,
        s_dagger_minus_err=s_minus_err,
        preferred_branch=preferred_branch,
        regime=regime,
    )


def q_from_dip(
    dt_dip: float,
    *,
    tE: float,
    geometry: AnomalyGeometry,
    dt_dip_err: float = float("nan"),
) -> tuple[float, float]:
    """
    Mass ratio for a minor-image dip (Han 2006; Hwang et al. 2022).

    ``q = (dt_dip / 4 tE)^2 * (s_dagger_minus / u_anom) * sin^2(alpha)``,
    where ``dt_dip`` is the full dip duration.  Returns ``(q, q_err)``;
    ``q_err`` propagates ``dt_dip_err`` and the geometry's ``t_anom_err``
    to first order (the estimate itself is good to a factor of ~2).
    """
    tE = float(tE)
    dt_dip = float(dt_dip)
    u = geometry.u_anom
    if not (np.isfinite(dt_dip) and dt_dip > 0.0 and u > 0.0):
        return float("nan"), float("nan")
    base = (dt_dip / (4.0 * tE)) ** 2
    q = base * (geometry.s_dagger_minus / u) * geometry.sin_alpha**2

    # d ln q / d dt_dip = 2 / dt_dip; d ln q via u_anom uses
    # dq/du = q * (d ln s_minus/du - 1/u) with d ln s_minus/du = -1/root,
    # and sin_alpha = u0/u so d ln sin^2(alpha)/du = -2/u.
    terms = []
    if np.isfinite(dt_dip_err):
        terms.append(2.0 * dt_dip_err / dt_dip)
    if np.isfinite(geometry.u_anom_err):
        root = float(np.sqrt(u * u + 4.0))
        dlnq_du = -1.0 / root - 3.0 / u
        terms.append(abs(dlnq_du) * geometry.u_anom_err)
    q_err = q * float(np.sqrt(sum(t * t for t in terms))) if terms else float("nan")
    return float(q), float(q_err)


def q_from_bump(
    t_p: float,
    *,
    tE: float,
    t_p_err: float = float("nan"),
) -> tuple[float, float]:
    """
    Order-of-magnitude mass ratio for a major-image bump (Gould & Loeb 1992).

    The perturbed region is taken to be the planet's Einstein ring, whose
    crossing time is ``sqrt(q) * tE``, so ``q = (t_p / tE)^2`` with ``t_p``
    the fitted bump width parameter.
    """
    tE = float(tE)
    t_p = float(t_p)
    if not (np.isfinite(t_p) and t_p > 0.0 and tE > 0.0):
        return float("nan"), float("nan")
    q = (t_p / tE) ** 2
    q_err = 2.0 * q * (t_p_err / t_p) if np.isfinite(t_p_err) else float("nan")
    return float(q), float(q_err)
