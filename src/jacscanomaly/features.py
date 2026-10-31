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
  pair      two folds, inside (``in``) or outside          t1, dt1,     4
            (``out``) between them, + a linear floor       t2, dt2
            between them
  ========  =============================================  ===========  ======

``G`` is the uniform-source fold profile and ``H`` the fraction of the source
disk inside the fold (Schneider & Weiss 1987; Albrow et al. 1999; Gaudi &
Gould 2002); ``dt`` is the time the source radius takes to cross the fold.
``in``: entry at ``t1``, exit at ``t2``, the interior above the host.
``out``: exit at ``t1``, entry at ``t2``; the outer sides are inside the
caustic, the host's level (0), and the floor between the crossings is below
it.

Signs (named assumption): a fold crossing adds or removes one image pair of
positive magnification, so its spike amplitude ``A >= 0`` and the level
inside the caustic is not below the level outside at each crossing: fold
``c >= 0``; ``in`` floor ``>= 0`` (the host is outside); ``out`` floor between
the crossings ``<= 0`` (the host is inside). The linear part of fold and pair
is a non-negative least squares under these signs; bump and double stay free
(they also describe dips). A crossing is claimed by its spike: a fold or pair
whose spike term, any one alone (nonlinear parameters held), gains less than
``dchi2_min`` in chi2 is not fitted (steps alone are a dip or a bump).
A pair's two crossings must be resolved: each crossing's spike (FWHM)
is shorter than the time between the two peaks. The straight fold assumes a
source small against the caustic, so a crossing short against the chord; a
spike as long as the pair is two folds the data do not separate (a bump or a
double).

A pair crossing's FWHM (for that test, :class:`Crossing`) is measured on the
fitted family near it, ``y = L_out + (L_in - L_out) H(z) + A G(z)``, half
maximum against the level on each side.

The fractions are numbers to read, not the decision. Fractions, not tests against an
exact null: a real anomaly never fits a named shape exactly, and a null test
would label S/N rather than shape.

Parts: every window is described by the atoms model ``y`` itself, not by its
atoms (atoms superpose: two overlapping atoms of opposite sign are not two
features). Its parts are the extrema of ``y``: in each segment where the model
is seen (``|y| / ferr >= 1``; a pass through the host shorter than a seen run
next to it does not end the segment; seen by no single sample, the window
is one segment), the local maxima of ``|y|``, taken from the
strongest down, each kept if at least half of the nearest one already kept
(a weaker one next to it is its rebound; one comparable to a kept neighbour
is kept, 0140; another segment reports its own largest), two of one sign counted as two only when ``|y|``
between them falls below half of the weaker; each with its FWHM (the run
where ``y`` keeps its sign above half its extremum, stopped at the lowest
``|y|`` before a neighbouring part). The model is read on the samples.

Caustic (a number, the label a rough guide): a crossing changes the
magnification by the anomaly's size within the source crossing time, short
against the anomaly; a smooth anomaly changes by its size over its own width.
Per segment, ``S = W / tau`` with the equivalent width
``W = int |y| dt / max|y|`` and the fastest rise or fall
``tau = max|y| / max|dy/dt|``; the window's ``sharpness`` is the largest ``S``.
One smooth single-scale part has ``S ~ 1.5`` at any S/N (atom 1.43, Gaussian
1.52), two separate ones ``~ 3``; a box of length ``L`` with walls ``d`` has
``L / d``. ``caustic``: ``sharpness >= sharp_min`` (4, above two smooth
parts). A data gap (a spacing over 10 x the segment's median) carries no
area; edges are as steep as the samples show. A lone spike with nothing else
at its scale is a bump (a cusp approach and a lone crossing are not
separable by shape).

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
from scipy.optimize import least_squares, nnls
from scipy.signal import find_peaks

from .candidate import Candidate


@dataclass(frozen=True)
class FeatureConfig:
    """Shape decision and start controls."""

    dchi2_min: float = 25.0
    sharp_min: float = 4.0       # sharpness of a caustic (two smooth parts: ~3)
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
class Part:
    """One extremum of the atoms model: its sign, peak time and FWHM."""

    kind: str
    time: float
    timescale: float

    def summary_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "time": self.time, "timescale": self.timescale}


@dataclass(frozen=True)
class Feature:
    """One anomaly window.

    ``kind``: the parts' kinds in time order (``bump``, ``dip``,
    ``bump+dip``, ...), ``complex`` (no part seen) or ``none`` (no improvement
    at all). ``time`` and ``timescale``: the strongest part's peak time and
    FWHM (complex / none: the time of the largest ``|atoms|`` and ``nan``).

    ``explained`` maps each family to its fraction of ``dchi2``; ``parts`` the
    extrema of the atoms model, ``spans`` their half-maximum intervals and
    ``span_kinds`` their signs, in time order; ``sharpness`` the largest edge
    sharpness ``S`` of its segments and ``caustic`` whether it reaches
    ``sharp_min``.
    """

    candidate: Candidate
    kind: str
    time: float
    timescale: float
    dchi2: float
    explained: Mapping[str, float]
    n_points: int
    sharpness: float = np.nan
    caustic: bool = False
    spans: tuple[tuple[float, float], ...] = ()
    span_kinds: tuple[str, ...] = ()
    parts: tuple[Part, ...] = ()

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
            "n_points": self.n_points, "sharpness": self.sharpness, "caustic": self.caustic,
            **{f"f_{k}": v for k, v in self.explained.items()},
            "spans": [list(x) for x in self.spans],
            "span_kinds": list(self.span_kinds),
            "parts": [x.summary_dict() for x in self.parts],
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
            "n_caustic": sum(f.caustic for f in self.features),
            "dchi2_total": float(sum(f.dchi2 for f in self.features)),
            "strongest": None if s is None else s.summary_dict(),
        }

    def feature_dicts(self) -> list[dict[str, object]]:
        return [f.summary_dict() for f in self.features]

    def summary_text(self) -> str:
        if not self.features:
            return "no anomaly windows"
        return "; ".join(f"{f.kind}{' (caustic)' if f.caustic else ''} t={f.time:.4f} "
                         f"ts={f.timescale:.3g} S={f.sharpness:.2g} dchi2={f.dchi2:.0f}"
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


_FOLD_SIGN = np.array([1, 1])  # A, c >= 0


def _pair_times(p):
    """t1, t2, dt1, dt2 of a pair; t2 - t1 >= dt1 + dt2, so the crossings are resolved."""
    dt1, dt2 = np.exp(p[1]), np.exp(p[3])
    return p[0], p[0] + dt1 + dt2 + np.exp(p[2]), dt1, dt2


def _pair(t, p, orient):
    """Pair columns; the floor between the crossings is linear, ``e1`` at ``t1`` to ``e2`` at ``t2``.

    ``in``: ``A1, A2, e1, e2``. ``out``: ``A1, A2, g1, g2``, the floor
    ``-g_k`` below the outer sides (the host, 0).
    """
    t1, t2, dt1, dt2 = _pair_times(p)
    z1, z2 = (t - t1) / dt1, (t2 - t) / dt2
    box = _H(z1) * _H(z2)
    l2 = (t - t1) / (t2 - t1)
    floor = [box * (1.0 - l2), box * l2]
    if orient > 0:
        return np.array([_G(z1) * _H(z2), _G(z2) * _H(z1)] + floor)
    return np.array([_G(-z1), _G(-z2), -floor[0], -floor[1]])


# a fold only adds flux: A >= 0, inside not below outside at each crossing
_PAIR_SIGN = np.array([1, 1, 1, 1])


def _residual(cols, r, w, sign=None):
    """Weighted linear fit; ``sign[k] = 1`` holds coefficient ``k >= 0``, ``0`` leaves it free."""
    x = cols * w
    if sign is None:
        coef, *_ = np.linalg.lstsq(x.T, r * w, rcond=None)
    else:
        free = np.flatnonzero(sign == 0)
        c, _ = nnls(np.vstack([x, -x[free]]).T, r * w)
        coef = c[:x.shape[0]]
        coef[free] -= c[x.shape[0]:]
    return coef, r * w - x.T @ coef


def _fit(t, r, w, cols, starts, lower, upper, keep, sign=None):
    """Best of the ``keep`` starts with the lowest linear-only chi2, refined."""
    if not starts:
        return None
    n_par = lower.size + cols(t, starts[0]).shape[0]
    if t.size <= n_par:
        return None
    starts = [np.clip(p, lower + 1e-9, upper - 1e-9) for p in starts]
    screen = [float(np.sum(_residual(cols(t, p), r, w, sign)[1] ** 2)) for p in starts]
    best = None
    for i in np.argsort(screen)[:keep]:
        try:
            s = least_squares(lambda p: _residual(cols(t, p), r, w, sign)[1], starts[i],
                              bounds=(lower, upper), x_scale="jac", max_nfev=200)
        except ValueError:
            continue
        if best is None or 2.0 * s.cost < best[0]:
            best = (2.0 * s.cost, s.x, _residual(cols(t, s.x), r, w, sign)[0])
    return best


def _spikes_seen(t, r, w, cols, fit, sign, spikes, need):
    """Whether each spike column of ``fit`` alone gains ``need`` in chi2 (parameters held)."""
    if fit is None:
        return False
    x = cols(t, fit[1])
    for k in spikes:
        rest = _residual(np.delete(x, k, axis=0), r, w, np.delete(sign, k))[1]
        if float(np.sum(rest**2)) - fit[0] < need:
            return False
    return True


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
    """chi2, parameters and coefficients of every family on one window.

    A fold or pair whose spike (one per crossing) gains less than
    ``dchi2_min`` is not fitted (``None``): a crossing is claimed by its spike.
    """
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
        cols = lambda tt, p, s=s: _fold(tt, p, s)
        out[name] = _fit(t, r, w, cols, starts, *bound, keep, _FOLD_SIGN)
        if not _spikes_seen(t, r, w, cols, out[name], _FOLD_SIGN, (0,), config.dchi2_min):
            out[name] = None

    lower, upper = np.array([t[0], lo, lo, lo]), np.array([t[-1], hi, hi, hi])
    for name, o in (("caustic_entry_exit", +1), ("caustic_exit_entry", -1)):
        starts = [np.array([a, np.log(dt), np.log(max(b - a - 2.0 * dt, 1e-4)), np.log(dt)])
                  for a in edges[o] for b in edges[-o] if b > a for dt in dts]
        cols = lambda tt, p, o=o: _pair(tt, p, o)
        out[name] = _fit(t, r, w, cols, starts, lower, upper, keep, _PAIR_SIGN)
        if not _spikes_seen(t, r, w, cols, out[name], _PAIR_SIGN, (0, 1), config.dchi2_min):
            out[name] = None
        elif not _crossings_resolved(_pair_crossings(name, *out[name][1:])):
            out[name] = None
    return out


def _pair_crossings(name, p, coef) -> tuple[Crossing, Crossing]:
    """The two crossings of a fitted pair, in time order."""
    t1, t2, dt1, dt2 = (float(x) for x in _pair_times(p))
    if name == "caustic_entry_exit":  # outside the caustic: the host (0)
        return (_crossing(t1, dt1, +1, coef[0], 0.0, coef[2]),
                _crossing(t2, dt2, -1, coef[1], 0.0, coef[3]))
    # inside: the outer sides (the host, 0); outside: -g_k between the crossings
    return (_crossing(t1, dt1, -1, coef[0], -coef[2], 0.0),
            _crossing(t2, dt2, +1, coef[1], -coef[3], 0.0))


def _crossings_resolved(crossings) -> bool:
    """Whether each crossing's spike (FWHM) is shorter than the time between the two peaks."""
    a, b = crossings
    return bool(np.all(np.array([a.fwhm, b.fwhm]) < b.time - a.time))  # nan: not resolved


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


def _edge(t, a, i, j, level) -> float:
    """Time where ``a`` crosses ``level`` between samples ``i`` (above) and ``j``."""
    return float(t[i] + (level - a[i]) * (t[j] - t[i]) / (a[j] - a[i])) if a[j] != a[i] else float(t[j])


def _segments(t, y, w) -> list[tuple[int, int]]:
    """Runs where ``|y| w >= 1``, joined across a pass through the host shorter
    than a run next to it (none seen: the whole window); index bounds."""
    seen = np.flatnonzero(np.abs(y) * w >= 1.0)
    runs = [(int(r[0]), int(r[-1])) for r in np.split(seen, np.flatnonzero(np.diff(seen) > 1) + 1) if r.size]
    segs = runs[:1] or [(0, y.size - 1)]  # seen by no single sample (only all together): one segment
    for lo, hi in runs[1:]:  # a pass through the host shorter than a run next to it is inside the anomaly
        plo, phi = segs[-1]
        if t[lo] - t[phi] < max(t[phi] - t[plo], t[hi] - t[lo]):
            segs[-1] = (plo, hi)
        else:
            segs.append((lo, hi))
    return segs


def _sharpness(t, y, segs) -> float:
    """Largest ``S = W / tau`` of the segments: equivalent width over the fastest rise or fall."""
    best = np.nan
    for lo, hi in segs:
        tt, a = t[lo:hi + 1], np.abs(y[lo:hi + 1])
        if tt.size < 2 or a.max() <= 0:
            continue
        dt = np.diff(tt)
        gap = dt > 10.0 * np.median(dt)  # a data gap carries no area
        width = float(np.sum((0.5 * (a[1:] + a[:-1]) * dt)[~gap])) / a.max()
        rise = a.max() / float(np.max(np.abs(np.diff(y[lo:hi + 1])) / np.maximum(dt, 1e-12)))
        best = np.fmax(best, width / rise)
    return float(best)


def _extrema(t, y, segs) -> list[tuple[Part, tuple[float, float], float]]:
    """Kept extrema of the atoms model ``y``: part, half-maximum interval, ``|y|``; time order.

    In each segment, the local maxima of ``|y|`` from the strongest down, each
    kept if ``|y| >= 1/2`` of the nearest kept one; two of one sign count as two
    only when ``|y|`` between them falls below half of the weaker. Widths stop
    at the window's ends or the lowest ``|y|`` before a neighbouring part.
    """
    a, sg = np.abs(y), np.sign(y)
    out, kept = [], []
    for lo, hi in segs:
        aa = a[lo:hi + 1]
        pk, _ = find_peaks(np.concatenate([[-np.inf], aa, [-np.inf]]))
        seg = []
        for i in sorted((int(i) - 1 + lo for i in pk), key=lambda i: -a[i]):  # strongest first
            near = min(seg, key=lambda k: abs(t[k] - t[i])) if seg else i
            if a[i] >= 0.5 * a[near] and all(  # weaker than half its kept neighbour: its rebound
                    sg[i] != sg[k] or a[min(i, k):max(i, k) + 1].min() < 0.5 * a[i] for k in seg):
                seg.append(i)
        kept += sorted(seg)
    for n, i in enumerate(kept):
        v = y * sg[i]  # the extremum's sign up
        # stop at the window's end or at the lowest |y| before a neighbouring kept extremum
        left = 0 if n == 0 else kept[n - 1] + int(np.argmin(a[kept[n - 1]:i + 1]))
        right = y.size - 1 if n == len(kept) - 1 else i + int(np.argmin(a[i:kept[n + 1] + 1]))
        h = []
        for step, stop in ((-1, left), (+1, right)):
            j = i
            while j != stop and v[j + step] >= 0.5 * v[i]:
                j += step
            h.append(_edge(t, v, j, j + step, 0.5 * v[i]) if j != stop else float(t[j]))
        out.append((Part("bump" if y[i] > 0 else "dip", float(t[i]), h[1] - h[0]),
                    (h[0], h[1]), float(a[i])))
    return out


def _characterize(candidate, t, r, w, m, config) -> Feature:
    chi0 = float(np.sum((r * w) ** 2))
    dchi2 = chi0 - float(np.sum(((r - m) * w) ** 2))
    fits = _families(t, r, w, m, config)
    gain = {k: (chi0 - v[0]) if v else -np.inf for k, v in fits.items()}
    ref = dchi2 if dchi2 > 0 else max(max(gain.values()), 0.0)
    peak = float(t[int(np.argmax(np.abs(m)))])
    if ref <= 0:
        return Feature(candidate, "none", peak, np.nan, dchi2, {}, t.size)
    best = lambda *names: max(gain[k] for k in names)
    frac = {k: (float(g / ref) if np.isfinite(g) else np.nan) for k, g in
            (("bump", gain["bump"]), ("fold", best("fold_entry", "fold_exit")),
             ("double", gain["double"]), ("pair", best("caustic_entry_exit", "caustic_exit_entry")))}
    segs = _segments(t, m, w)
    sharp = _sharpness(t, m, segs)
    caustic = bool(sharp >= config.sharp_min)
    ext = _extrema(t, m, segs)
    if not ext:
        return Feature(candidate, "complex", peak, np.nan, dchi2, frac, t.size, sharp, caustic)
    parts = tuple(x[0] for x in ext)
    main = max(ext, key=lambda x: x[2])[0]
    return Feature(candidate, "+".join(x.kind for x in parts), main.time, main.timescale,
                   dchi2, frac, t.size, sharp, caustic, spans=tuple(x[1] for x in ext),
                   span_kinds=tuple(x.kind for x in parts), parts=parts)


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
            out.append(Feature(c, "complex", t, np.nan, dchi2, {}, int(s.sum())))
            continue
        out.append(_characterize(c, time[s], (flux - host_flux)[s], 1.0 / ferr[s],
                                 atom_flux[s], config))
    return FeatureResult(tuple(out))


__all__ = ["Crossing", "Feature", "FeatureConfig", "FeatureResult", "Part", "measure_features"]
