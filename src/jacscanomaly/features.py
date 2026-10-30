"""Shape, strength, centre and timescale of each anomaly window.

:func:`measure_features` characterizes every window of a host + atoms fit
(:func:`~jacscanomaly.robust_fit`); nothing the atoms accepted is dropped.
With the host held fixed, ``r = flux - host`` in the window is described by

* the atoms (free, smooth): ``dchi2 = chi2(host) - chi2(host + atoms)`` is
  the strength,
* named shape families, each scored by the fraction of ``dchi2`` it explains,
  ``f_k = (chi2(host) - chi2(host + family k)) / dchi2``:

  ========  =============================================  ===========  ======
  family    model of ``r``                                 nonlinear    linear
  ========  =============================================  ===========  ======
  bump      one atom ``a h + b l``                         t0, teff     2
  fold      ``A G(z) + c H(z)``, ``z = +-(t - tc) / dt``   tc, dt       2
  double    two atoms                                      2 x bump     4
  pair      two folds, inside (``in``) or outside          t1, dt1,     4-5
            (``out``) between them, + floor and slope      t2, dt2
            between them
  ========  =============================================  ===========  ======

``G`` is the uniform-source fold profile and ``H`` the fraction of the source
disk inside the fold (Schneider & Weiss 1987; Albrow et al. 1999; Gaudi &
Gould 2002); ``dt`` is the time the source radius takes to cross the fold.
``in``: entry at ``t1``, exit at ``t2``, the interior above the host.
``out``: exit at ``t1``, entry at ``t2``; the outer sides are inside the
caustic and get their own floor (the host has absorbed part of it).

Each fold crossing is also described on its own (:class:`Crossing`): the
time of its peak and the half width at half maximum on each side, measured
on the fitted family near that crossing, ``y = L_out + (L_in - L_out) H(z) +
A G(z)``, half maximum taken against the level on that side (``L_out``
outside the caustic, ``L_in`` the floor inside). With the uniform-source
fold profile the inner side is fixed at a multiple of ``dt``; the outer side
also depends on ``L_in / A``. With ``dt`` near the cadence the data alone do
not resolve the outer side: the widths are those of the fitted profile.

Shape: bump/fold and double/pair are two complexity classes (2 and 4
nonlinear parameters). Within a class (same parameter count) the fold / pair
is taken if it fits better by ``dchi2_min`` (a caustic edge must be resolved
to be claimed); the second class is taken only if it explains a further
``delta`` of ``dchi2`` *and* at least ``dchi2_min`` in chi2 over the first; ``"complex"`` if the
chosen family explains less than ``f_min``. Fractions, not tests against an
exact null: a real anomaly never fits a named shape exactly, and a null test
would label S/N rather than shape.

Assumptions: the host is fixed (a host that absorbed the anomaly is not
repaired here); straight fold, uniform source, constant velocity across a
crossing (limb darkening and fold curvature stay in ``1 - f``); a cusp
approach is a bump (not separable by shape); independent noise with the
given ``ferr``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from .candidate import Candidate


@dataclass(frozen=True)
class FeatureConfig:
    """Shape decision and start controls."""

    f_min: float = 0.8
    delta: float = 0.05
    dchi2_min: float = 25.0
    coeff: float = 3.0           # atom support, as in AtomConfig
    n_edges: int = 5             # steepest rises / drops of the atoms tried as fold times
    edge_separation: int = 5     # samples between two edge starts
    keep_starts: int = 3         # starts refined after the linear-only screen


@dataclass(frozen=True)
class Crossing:
    """One fold crossing: ``entry`` or ``exit``, peak time and left / right HWHM."""

    kind: str
    time: float
    hwhm_left: float
    hwhm_right: float
    dt: float

    @property
    def fwhm(self) -> float:
        return self.hwhm_left + self.hwhm_right

    def summary_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "time": self.time, "hwhm_left": self.hwhm_left,
                "hwhm_right": self.hwhm_right, "fwhm": self.fwhm, "dt": self.dt}


@dataclass(frozen=True)
class Feature:
    """One anomaly window.

    ``kind``: ``bump``, ``dip``, ``fold_entry``, ``fold_exit``, two atoms in
    time order (``bump+dip``, ...), ``caustic_entry_exit`` (inside between the
    crossings), ``caustic_exit_entry`` (outside between them), ``complex`` or
    ``none`` (no improvement at all). ``time`` and ``timescale``:

    * bump / dip: ``t0`` of the atom and the FWHM of its fitted profile (an
      atom ``a h + b l`` can be flat-topped, so ``teff`` is not its width),
    * fold: ``tc`` and ``dt``,
    * two atoms: the mid time and separation of their peaks (largest ``|r|``
      of each), which also order them,
    * caustic: ``(t1 + t2) / 2`` and ``t2 - t1``,
    * complex / none: the time of the largest ``|atoms|`` and ``nan``.

    ``explained`` maps each family to its fraction of ``dchi2``; ``params``
    holds the chosen family's times (``t1, t2, dt1, dt2`` for a caustic);
    ``crossings`` the fold crossings in time order (fold: one, caustic: two);
    ``spans`` the half-maximum interval of each atom (bump / dip: one, two
    atoms: two), around the atom's largest ``|r|``.
    """

    candidate: Candidate
    kind: str
    time: float
    timescale: float
    dchi2: float
    explained: Mapping[str, float]
    params: Mapping[str, float]
    n_points: int
    crossings: tuple[Crossing, ...] = ()
    spans: tuple[tuple[float, float], ...] = ()

    @property
    def t_start(self) -> float:
        return float(self.candidate.t_start)

    @property
    def t_end(self) -> float:
        return float(self.candidate.t_end)

    def summary_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind, "time": self.time, "timescale": self.timescale,
            "dchi2": self.dchi2, "t_start": self.t_start, "t_end": self.t_end,
            "n_points": self.n_points,
            **{f"f_{k}": v for k, v in self.explained.items()},
            **dict(self.params),
            "crossings": [c.summary_dict() for c in self.crossings],
            "spans": [list(x) for x in self.spans],
        }


@dataclass(frozen=True)
class FeatureResult:
    features: tuple[Feature, ...] = field(default_factory=tuple)

    @property
    def strongest(self) -> Optional[Feature]:
        return max(self.features, key=lambda f: f.dchi2, default=None)

    def summary_dict(self) -> dict[str, object]:
        s = self.strongest
        return {
            "n_features": len(self.features),
            "kinds": dict(Counter(f.kind for f in self.features)),
            "dchi2_total": float(sum(f.dchi2 for f in self.features)),
            "strongest": None if s is None else s.summary_dict(),
        }

    def feature_dicts(self) -> list[dict[str, object]]:
        return [f.summary_dict() for f in self.features]

    def summary_text(self) -> str:
        if not self.features:
            return "no anomaly windows"
        return "; ".join(f"{f.kind} t={f.time:.4f} ts={f.timescale:.3g} dchi2={f.dchi2:.0f}"
                         for f in self.features)


# --- uniform-source fold profile ---------------------------------------------
# G(z) = (2/pi) int_{-1}^{min(z,1)} sqrt(1 - x^2) / sqrt(z - x) dx, z = source
# centre inside the fold in source radii; ~ 1/sqrt(z) far inside, 0 outside.
# With x = z - u^2 the integrand is smooth: (2/pi) int 2 sqrt(1 - x^2) du.

def _fold_table():
    xg, wg = np.polynomial.legendre.leggauss(64)
    z = np.concatenate([np.linspace(-1.0, 3.0, 801), np.geomspace(3.01, 60.0, 400)])
    ua = np.sqrt(np.clip(z - 1.0, 0.0, None))
    ub = np.sqrt(z + 1.0)
    u = 0.5 * (ub - ua)[:, None] * xg + 0.5 * (ub + ua)[:, None]
    x = z[:, None] - u * u
    g = (2.0 / np.pi) * 0.5 * (ub - ua) * (wg * 2.0 * np.sqrt(np.clip(1.0 - x * x, 0.0, None))).sum(1)
    return z, g


_ZT, _GT = _fold_table()


def _G(z):
    z = np.asarray(z, dtype=float)
    out = np.interp(z, _ZT, _GT, left=0.0)
    big = z > _ZT[-1]
    out[big] = z[big] ** -0.5 * (1.0 + 3.0 / (32.0 * z[big] ** 2))
    return out


def _H(z):
    x = np.clip(z, -1.0, 1.0)
    return (x * np.sqrt(1.0 - x * x) + np.arcsin(x)) / np.pi + 0.5


# --- families: design columns for given nonlinear parameters -----------------

def _atom_cols(t, t0, log_teff, coeff):
    tau = (t - t0) / np.exp(log_teff)
    inside = np.abs(tau) < coeff
    q, qe = 1.0 + tau * tau, 1.0 + coeff * coeff
    high = 1.0 / np.sqrt(q) - 1.0 / np.sqrt(qe)
    low = (q + 2.0) / np.sqrt(q * (q + 4.0)) - (qe + 2.0) / np.sqrt(qe * (qe + 4.0))
    return [np.where(inside, high, 0.0), np.where(inside, low, 0.0)]


def _bump(t, p, coeff):
    return np.array(_atom_cols(t, p[0], p[1], coeff))


def _double(t, p, coeff):
    return np.array(_atom_cols(t, p[0], p[1], coeff) + _atom_cols(t, p[2], p[3], coeff))


def _fold(t, p, sign):
    z = sign * (t - p[0]) / np.exp(p[1])
    return np.array([_G(z), _H(z)])


def _pair_times(p):
    """t1, t2, dt1, dt2 of a pair; t2 - t1 >= dt1 + dt2, so the crossings are resolved."""
    dt1, dt2 = np.exp(p[1]), np.exp(p[3])
    return p[0], p[0] + dt1 + dt2 + np.exp(p[2]), dt1, dt2


def _pair(t, p, orient):
    t1, t2, dt1, dt2 = _pair_times(p)
    z1, z2 = (t - t1) / dt1, (t2 - t) / dt2
    box = _H(z1) * _H(z2)
    if orient > 0:
        cols = [_G(z1) * _H(z2), _G(z2) * _H(z1)]
    else:
        cols = [_G(-z1), _G(-z2), 1.0 - box]
    return np.array(cols + [box, box * (t - 0.5 * (t1 + t2))])


def _residual(cols, r, w):
    x = cols * w
    coef, *_ = np.linalg.lstsq(x.T, r * w, rcond=None)
    return coef, r * w - x.T @ coef


def _fit(t, r, w, cols, starts, lower, upper, keep):
    """Best of the ``keep`` starts with the lowest linear-only chi2, refined."""
    if not starts:
        return None
    n_par = lower.size + cols(t, starts[0]).shape[0]
    if t.size <= n_par:
        return None
    starts = [np.clip(p, lower + 1e-9, upper - 1e-9) for p in starts]
    screen = [float(np.sum(_residual(cols(t, p), r, w)[1] ** 2)) for p in starts]
    best = None
    for i in np.argsort(screen)[:keep]:
        try:
            s = least_squares(lambda p: _residual(cols(t, p), r, w)[1], starts[i],
                              bounds=(lower, upper), x_scale="jac", max_nfev=200)
        except ValueError:
            continue
        if best is None or 2.0 * s.cost < best[0]:
            best = (2.0 * s.cost, s.x, _residual(cols(t, s.x), r, w)[0])
    return best


def _edges(t, m, k, sep):
    """The ``k`` steepest rises (+1) and drops (-1) of the atoms, ``sep`` samples apart."""
    dm = np.gradient(m, t)
    out = {}
    for s in (+1, -1):
        pk, _ = find_peaks(np.concatenate([[-np.inf], s * dm, [-np.inf]]), distance=sep)
        pk = pk - 1
        out[s] = [t[i] for i in pk[np.argsort(-s * dm[pk])[:k]]]
    return out


def _families(t, r, w, m, config):
    """chi2, parameters and coefficients of every family on one window."""
    coeff, keep = config.coeff, config.keep_starts
    span = t[-1] - t[0]
    cadence = float(np.median(np.diff(t)))
    lo, hi = np.log(1e-4), np.log(max(span, 2e-4))
    teffs = [np.log(x * span) for x in (0.02, 0.08, 0.25)]
    dts = [x * cadence for x in (0.3, 1.0, 3.0, 10.0)]
    two = lambda a: np.r_[a, a]
    out = {}

    bound = np.array([t[0], lo]), np.array([t[-1], hi])
    peak = t[int(np.argmax(np.abs(m)))]
    out["bump"] = _fit(t, r, w, lambda tt, p: _bump(tt, p, coeff),
                       [np.array([peak, l]) for l in teffs], *bound, keep)

    pk, _ = find_peaks(np.concatenate([[0.0], np.abs(m), [0.0]]), distance=config.edge_separation)
    pk = pk - 1
    pk = pk[np.argsort(-np.abs(m[pk]))[:4]]
    starts = [np.array([t[i], a, t[j], b]) for i in pk for j in pk if t[i] < t[j]
              for a in teffs for b in teffs]
    out["double"] = _fit(t, r, w, lambda tt, p: _double(tt, p, coeff), starts,
                         two(bound[0]), two(bound[1]), keep)

    edges = _edges(t, m, config.n_edges, config.edge_separation)
    for name, s in (("fold_entry", +1), ("fold_exit", -1)):
        starts = [np.array([tc, np.log(dt)]) for tc in edges[s] for dt in dts]
        out[name] = _fit(t, r, w, lambda tt, p, s=s: _fold(tt, p, s), starts, *bound, keep)

    lower, upper = np.array([t[0], lo, lo, lo]), np.array([t[-1], hi, hi, hi])
    for name, o in (("caustic_entry_exit", +1), ("caustic_exit_entry", -1)):
        starts = [np.array([a, np.log(dt), np.log(max(b - a - 2.0 * dt, 1e-4)), np.log(dt)])
                  for a in edges[o] for b in edges[-o] if b > a for dt in dts]
        out[name] = _fit(t, r, w, lambda tt, p, o=o: _pair(tt, p, o), starts, lower, upper, keep)
    return out


def _crossing(tc, dt, inward, amp, level_out, level_in, n=4001) -> Crossing:
    """Peak and HWHM of ``level_out + (level_in - level_out) H(z) + amp G(z)``.

    ``z = inward (t - tc) / dt`` grows into the caustic; each half maximum is
    taken against the level on its own side.
    """
    z = np.linspace(-3.0, 60.0, n)
    y = level_out + (level_in - level_out) * _H(z) + amp * _G(z)
    i = int(np.argmax(np.sign(amp) * y)) if amp != 0 else int(np.argmin(np.abs(z - 1.0)))

    def half(side, level):
        j = np.arange(i, -1, -1) if side < 0 else np.arange(i, n)
        below = np.flatnonzero(np.sign(amp) * (y[j] - 0.5 * (y[i] + level)) <= 0)
        if not below.size or below[0] == 0:
            return np.nan
        k = j[below[0]]
        k0 = k + 1 if side < 0 else k - 1  # last sample above half
        h = 0.5 * (y[i] + level)
        zk = z[k0] + (h - y[k0]) * (z[k] - z[k0]) / (y[k] - y[k0])
        return abs(zk - z[i]) * dt

    w_out, w_in = half(-1, level_out), half(+1, level_in)
    left, right = (w_out, w_in) if inward > 0 else (w_in, w_out)
    return Crossing("entry" if inward > 0 else "exit", float(tc + inward * z[i] * dt),
                    float(left), float(right), float(dt))


def _half_max(t, y) -> tuple[float, float]:
    """Interval around the largest ``|y|`` where ``|y|`` stays above half of it."""
    i = int(np.argmax(np.abs(y)))
    above = np.abs(y) >= 0.5 * abs(y[i])
    lo = i - (np.flatnonzero(~above[:i + 1][::-1])[:1].tolist() or [i + 1])[0] + 1
    hi = i + (np.flatnonzero(~above[i:])[:1].tolist() or [above.size - i])[0] - 1
    return float(t[lo]), float(t[hi])


def _sign_kind(y) -> str:
    return "bump" if y[int(np.argmax(np.abs(y)))] > 0 else "dip"


def _characterize(candidate, t, r, w, m, config) -> Feature:
    chi0 = float(np.sum((r * w) ** 2))
    dchi2 = chi0 - float(np.sum(((r - m) * w) ** 2))
    fits = _families(t, r, w, m, config)
    gain = {k: (chi0 - v[0]) if v else -np.inf for k, v in fits.items()}
    ref = dchi2 if dchi2 > 0 else max(max(gain.values()), 0.0)
    peak = float(t[int(np.argmax(np.abs(m)))])
    if ref <= 0:
        return Feature(candidate, "none", peak, np.nan, dchi2, {}, {}, t.size)

    def best(*names):
        return max(names, key=lambda k: gain[k])

    cls = {"bump": "bump", "fold": best("fold_entry", "fold_exit"),
           "double": "double", "pair": best("caustic_entry_exit", "caustic_exit_entry")}
    frac = {k: (float(gain[v] / ref) if np.isfinite(gain[v]) else np.nan) for k, v in cls.items()}

    def beats(k, j, margin):
        return gain[cls[k]] >= gain[cls[j]] + margin

    # same parameter count within a class: an edge only has to be resolved
    shape = "fold" if beats("fold", "bump", config.dchi2_min) else "bump"
    two = "pair" if beats("pair", "double", config.dchi2_min) else "double"
    if np.isfinite(gain[cls[two]]) and beats(two, shape, max(config.delta * ref, config.dchi2_min)):
        shape = two
    if not np.isfinite(gain[cls[shape]]) or frac[shape] < config.f_min:
        return Feature(candidate, "complex", peak, np.nan, dchi2, frac, {}, t.size)

    _, p, coef = fits[cls[shape]]
    tt = np.linspace(t[0], t[-1], 20001)
    if shape == "bump":
        y = coef @ _bump(tt, p, config.coeff)
        lo, hi = _half_max(tt, y)
        return Feature(candidate, _sign_kind(y), float(p[0]), hi - lo, dchi2, frac,
                       {"t0": float(p[0]), "teff": float(np.exp(p[1]))}, t.size, spans=((lo, hi),))
    if shape == "double":
        y = [coef[2 * i:2 * i + 2] @ _bump(tt, p[2 * i:2 * i + 2], config.coeff) for i in (0, 1)]
        # an atom can peak off its t0 (a ring), so its peak orders and times it
        peaks = [float(tt[int(np.argmax(np.abs(yi)))]) for yi in y]
        order = (0, 1) if peaks[0] <= peaks[1] else (1, 0)
        tp = [peaks[i] for i in order]
        return Feature(candidate, "+".join(_sign_kind(y[i]) for i in order),
                       0.5 * (tp[0] + tp[1]), tp[1] - tp[0], dchi2, frac,
                       {"t0_1": float(p[2 * order[0]]), "teff_1": float(np.exp(p[2 * order[0] + 1])),
                        "t0_2": float(p[2 * order[1]]), "teff_2": float(np.exp(p[2 * order[1] + 1]))},
                       t.size,
                       spans=tuple(_half_max(tt, y[i]) for i in order))
    if shape == "fold":
        s = +1 if cls["fold"] == "fold_entry" else -1
        tc, dt = float(p[0]), float(np.exp(p[1]))
        return Feature(candidate, cls["fold"], tc, dt, dchi2, frac, {"tc": tc, "dt": dt}, t.size,
                       (_crossing(tc, dt, s, coef[0], 0.0, coef[1]),))
    t1, t2, dt1, dt2 = (float(x) for x in _pair_times(p))
    mid = 0.5 * (t1 + t2)
    between = lambda tk: coef[-2] + coef[-1] * (tk - mid)  # box floor and slope
    if cls["pair"] == "caustic_entry_exit":  # outside the caustic: the host (0)
        crossings = (_crossing(t1, dt1, +1, coef[0], 0.0, between(t1)),
                     _crossing(t2, dt2, -1, coef[1], 0.0, between(t2)))
    else:  # inside: the outer floor (1 - box); outside: between the crossings
        crossings = (_crossing(t1, dt1, -1, coef[0], between(t1), coef[2]),
                     _crossing(t2, dt2, +1, coef[1], between(t2), coef[2]))
    return Feature(candidate, cls["pair"], mid, t2 - t1, dchi2, frac,
                   {"t1": t1, "t2": t2, "dt1": dt1, "dt2": dt2}, t.size, crossings)


def measure_features(
    time,
    flux,
    ferr,
    host_flux,
    atom_flux,
    windows: Sequence[Candidate],
    *,
    config: FeatureConfig = FeatureConfig(),
) -> FeatureResult:
    """Characterize every window: ``host_flux`` is the fixed host, ``atom_flux`` the atoms."""
    time, flux, ferr, host_flux, atom_flux = (
        np.asarray(x, dtype=float) for x in (time, flux, ferr, host_flux, atom_flux))
    out = []
    for c in windows:
        s = (time >= c.t_start) & (time <= c.t_end)
        if s.sum() < 3:  # too few samples for any shape
            r, w, m = (flux - host_flux)[s], 1.0 / ferr[s], atom_flux[s]
            dchi2 = float(np.sum((r * w) ** 2) - np.sum(((r - m) * w) ** 2))
            t = float(time[s][np.argmax(np.abs(m))]) if s.any() else 0.5 * (c.t_start + c.t_end)
            out.append(Feature(c, "complex", t, np.nan, dchi2, {}, {}, int(s.sum())))
            continue
        out.append(_characterize(c, time[s], (flux - host_flux)[s], 1.0 / ferr[s],
                                 atom_flux[s], config))
    return FeatureResult(tuple(out))


__all__ = ["Crossing", "Feature", "FeatureConfig", "FeatureResult", "measure_features"]
