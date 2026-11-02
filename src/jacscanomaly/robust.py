"""Host fit that the anomalies do not pull: the host and the anomalies fitted jointly.

No sample is masked. The anomalies are modelled by the scan's own templates
("atoms") and fitted together with the host to every sample::

    F(t) = fs A(t; p) + fb + sum_k [a_k h(tau_k) + b_k l(tau_k)],
    tau_k = (t - t0_k) / teff_k

``h`` and ``l`` are the scan's high- and low-magnification templates, shifted
to zero at ``|tau| = coeff`` and zero outside, so an atom only touches its own
support. ``p`` are the host parameters in the fitter's seed convention
(``t0, tE, u0[, log rho][, piEN, piEE]``); ``fs``, ``fb``, ``a_k``, ``b_k``
are solved linearly for every ``p``.

:func:`fit_atoms` grows the atoms from one start: scan the data minus the
current atoms (host fixed), add the best accepted candidate, refit, repeat
while an atom pays its price; then drop atoms that no longer pay.
:func:`robust_fit` runs it from every start of every host model and keeps the
lowest ``J``.

Assumptions
-----------
* An atom costs ``k_atom ln n`` (BIC; ``t0, teff, a, b``). Starts and models
  are compared by ``J = chi2 + (n_host + 2 + k_atom n_atoms) ln n``.
* An atom is at most ``max_teff_fraction`` of the start host's local
  timescale ``tE max(1, u(t0))`` wide, so atoms cannot take over the host's
  curvature; in the wing (``u > 1``) the host varies on ``tE u``, so wider
  atoms are allowed there. Exempt: an atom centred on the segment
  :func:`~jacscanomaly.relock_host` moved the host off, which that test has
  already shown is not host.
* When the relocked host's peak is off the relock's anomaly segment, the
  plain fit (which sits on that segment) is not a start: ``base`` is the
  relock's masked H1 fit instead.
* The start host is held fixed (``fs, fb`` only) until the first atom is in,
  so a relocked host is not pulled back onto the spike it was moved off; with
  no atom at the end the host is refit.
* The host is what the samples outside the anomalies say. The start
  ``"fixed"`` holds the host fitted outside the best free start's atom
  windows, and it is kept unless a moved host wins by ``free_host_margin`` in
  ``J``: the ``h`` template's wings trade off against the host shape. The
  default margin is the host's own price, ``n_host ln n``: a free start fits
  the host to the samples inside the windows too, where the held host has no
  say, so a moved host must pay for its parameters once more there. The
  models are then compared by ``J`` alone (the margin is within a model).
* A host must beat a constant flux on the samples outside its atom windows
  (the relock's detection test); a start whose host does not is used only
  when none does, and the model is then flagged ``host_identified = False``.
* ``fs, fb >= 0``: they are fluxes. The atom amplitudes are free.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Optional

import numpy as np
from scipy.optimize import least_squares

from . import _cpp_grid
from .candidate import Candidate, _merged
from .features import FeatureConfig, FeatureResult, measure_features
from .fitters import FSPLFitter, PSPLFitter, _pspl_magnification
from .parallax_backend import _ParallaxFitter
from .relock import HostRelock, _detected, relock_host
from .singlelens_fit import SingleLensFitResult


@dataclass(frozen=True)
class AtomConfig:
    """Atom support (``coeff`` in units of ``teff``), acceptance and price.

    ``free_host_margin``: the ``J`` by which a fit that moved the host must beat
    the held host (start ``"fixed"``) to be selected; ``None`` is
    ``n_host ln n``, the host's own price.
    """

    coeff: float = 3.0
    min_score: float = 10.0
    max_teff_fraction: float = 0.25
    max_atoms: int = 15
    k_atom: float = 4.0
    free_host_margin: Optional[float] = None


@dataclass(frozen=True)
class Atom:
    """One anomaly term: ``a h(tau) + b l(tau)``, ``tau = (t - t0) / teff``."""

    t0: float
    teff: float
    a: float
    b: float


@dataclass(frozen=True)
class AtomFit:
    """Result of :func:`fit_atoms` from one start.

    ``chi2_host_only`` is the start host with no atoms (``fs, fb`` fitted).
    ``log`` lists the steps: ``("add", t0, teff, score, chi2)``,
    ``("stop", t0, teff, dchi2)``, ``("drop", t0, teff, dchi2)``.
    """

    params: np.ndarray
    fs: float
    fb: float
    atoms: tuple[Atom, ...]
    chi2: float
    chi2_host_only: float
    log: tuple[tuple, ...]


@dataclass(frozen=True)
class ModelFit:
    """The lowest-``J`` :class:`AtomFit` of one host model, and what follows from it.

    ``fit`` is the host alone with the atom fit's ``fs, fb``, the samples
    inside ``windows`` (the merged atom supports) masked. ``features`` characterize every
    window against the host + atoms of the atom fit (its own ``fs, fb``).
    ``sigma`` are the 1-sigma host parameter errors with fluxes and atom
    amplitudes profiled. ``base`` is the plain fit without atoms. ``rank`` is
    what the start selection compares: ``J``, less the margin for the start
    ``"fixed"``. ``host_identified``: the host (``fs A + fb``) beats a constant
    flux on the samples outside ``windows``; ``False`` when no start's host
    does (the whole light curve is anomalous).
    """

    name: str
    start: str
    atom_fit: AtomFit
    J: float
    J_starts: Mapping[str, float]
    sigma: np.ndarray
    windows: tuple[Candidate, ...]
    fit: SingleLensFitResult
    features: Optional[FeatureResult]
    base: Optional[SingleLensFitResult]
    rank: float = np.nan
    host_identified: bool = True

    @property
    def params(self) -> np.ndarray:
        return self.atom_fit.params

    @property
    def atoms(self) -> tuple[Atom, ...]:
        return self.atom_fit.atoms

    @property
    def dchi2(self) -> float:
        """``chi2(base) - chi2(host + atoms)``: what the atoms explain (``nan`` without ``base``)."""
        return float(self.base.chi2) - self.atom_fit.chi2 if self.base is not None else np.nan


@dataclass(frozen=True)
class RobustFit:
    """Result of :func:`robust_fit`: every host model, and the one selected."""

    models: Mapping[str, ModelFit]
    selected: str
    relock: HostRelock
    errors: Mapping[str, str] = field(default_factory=dict)

    @property
    def best(self) -> ModelFit:
        return self.models[self.selected]


def atom_basis(time, t0: float, teff: float, coeff: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """The two templates of an atom, zero at ``|tau| = coeff`` and outside."""
    tau = (np.asarray(time, dtype=float) - t0) / teff
    inside = np.abs(tau) < coeff
    q, qe = 1.0 + tau * tau, 1.0 + coeff * coeff
    high = 1.0 / np.sqrt(q) - 1.0 / np.sqrt(qe)
    low = (q + 2.0) / np.sqrt(q * (q + 4.0)) - (qe + 2.0) / np.sqrt(qe * (qe + 4.0))
    return np.where(inside, high, 0.0), np.where(inside, low, 0.0)


class _Host:
    """Magnification, bounds and parameter scales of one host model on fixed samples."""

    def __init__(self, fitter, time):
        self.time = time
        self.magnification = fitter.magnification_model(time)
        lo, hi, self._extra_scale = [-np.inf, 1e-3, -np.inf], [np.inf] * 3, []
        if isinstance(fitter, FSPLFitter):
            lo, hi, self._extra_scale = lo + [np.log(1e-6)], hi + [np.log(10.0)], [0.5]
        elif isinstance(fitter, _ParallaxFitter):
            if fitter.finite_source:
                raise ValueError("robust_fit supports point-source parallax only")
            m = fitter.max_piE
            lo, hi, self._extra_scale = lo + [-m, -m], hi + [m, m], [0.05, 0.05]
        elif not isinstance(fitter, PSPLFitter):
            raise TypeError(f"unsupported host fitter {type(fitter).__name__}")
        self.lower, self.upper = np.array(lo), np.array(hi)

    def clip(self, p) -> np.ndarray:
        return np.clip(np.asarray(p, dtype=float), self.lower, self.upper)

    def scale(self, p) -> list[float]:
        return [0.01 * abs(p[1]), 0.1 * abs(p[1]), max(abs(p[2]), 1e-3)] + self._extra_scale


def _linear(host, flux, ferr, p, atoms, coeff):
    """Linear coefficients (fs, fb, a_1, b_1, ...) and the normalised residual.

    ``fs, fb >= 0`` (fluxes); the atom amplitudes are free. With two bounds
    the solution is the feasible one of lowest ``chi2`` over the active sets.
    """
    magnification = host.magnification(p)
    out = _cpp_grid.atoms_linear(magnification, host.time, flux, ferr,
                                 np.asarray(atoms, dtype=float).reshape(-1, 2), coeff)
    cols = None
    if out is None:  # numerically singular normal equations: minimum-norm SVD solution
        cols = _columns(host.time, magnification, atoms, coeff)
        out = _lstsq(cols, flux, ferr, ())
    if out[0][0] >= 0.0 and out[0][1] >= 0.0:
        return out
    cols = cols if cols is not None else _columns(host.time, magnification, atoms, coeff)
    trials = [_lstsq(cols, flux, ferr, zero) for zero in ((0,), (1,), (0, 1))]
    feasible = [t for t in trials if t[0][0] >= 0.0 and t[0][1] >= 0.0]
    return min(feasible, key=lambda t: float(t[1] @ t[1]))


def _columns(time, magnification, atoms, coeff):
    cols = [magnification, np.ones_like(time)]
    for t0, teff in atoms:
        cols += atom_basis(time, t0, teff, coeff)
    return np.vstack(cols).T


def _lstsq(cols, flux, ferr, zero):
    """Least squares on ``cols`` with the coefficients ``zero`` held at 0."""
    free = np.setdiff1d(np.arange(cols.shape[1]), zero)
    design = cols[:, free] / ferr[:, None]
    coef = np.zeros(cols.shape[1])
    coef[free], *_ = np.linalg.lstsq(design, flux / ferr, rcond=None)
    return coef, flux / ferr - design @ coef[free]


def _fit_host(host, flux, ferr, p, atoms, coeff):
    p0 = host.clip(p)
    r = least_squares(lambda q: _linear(host, flux, ferr, q, atoms, coeff)[1], p0,
                      bounds=(host.lower, host.upper), x_scale=host.scale(p0))
    coef, z = _linear(host, flux, ferr, r.x, atoms, coeff)
    return r.x, coef, float(z @ z)


def _atom_flux(time, coef, atoms, coeff):
    out = np.zeros_like(time)
    for i, (t0, teff) in enumerate(atoms):
        high, low = atom_basis(time, t0, teff, coeff)
        out += coef[2 + 2 * i] * high + coef[3 + 2 * i] * low
    return out


def fit_atoms(
    fitter,
    finder,
    time,
    flux,
    ferr,
    p0,
    *,
    anomaly: Optional[np.ndarray] = None,
    config: AtomConfig = AtomConfig(),
    fix_host: bool = False,
    _host: Optional[_Host] = None,
) -> AtomFit:
    """Grow and prune atoms on the host model of ``fitter`` from the start ``p0``.

    ``finder`` scans the residual (its PSPL host is given the data minus the
    atoms minus ``fs (A_host - A_pspl)``, the PSPL-shaped remainder).
    ``anomaly`` (``True`` = anomaly) exempts atoms centred there from the
    ``tE`` cap; pass :attr:`HostRelock.mask` of a relocked host.
    ``fix_host``: the host stays at ``p0`` (``fs, fb`` and the atoms linear only).
    """
    time, flux, ferr = (np.asarray(x, dtype=float) for x in (time, flux, ferr))
    host = _host or _Host(fitter, time)
    price = config.k_atom * np.log(time.size)
    criteria = finder.config.candidate_criteria
    anomaly = np.zeros(time.size, bool) if anomaly is None else np.asarray(anomaly, bool)
    t0_host, tE_host, u0_host = (float(x) for x in p0[:3])

    def allowed(candidate) -> bool:
        if float(candidate.score) < config.min_score:
            return False
        if criteria is not None and not criteria.accepts(dchi2=float(candidate.dchi2),
                                                         quality=candidate.quality):
            return False
        u = np.hypot(u0_host, (float(candidate.t0) - t0_host) / tE_host)
        if abs(float(candidate.teff)) <= config.max_teff_fraction * abs(tE_host) * max(1.0, u):
            return True
        i = min(int(np.searchsorted(time, float(candidate.t0))), time.size - 1)
        return bool(anomaly[i])

    def refit(p, atoms):
        if not fix_host:
            return _fit_host(host, flux, ferr, p, atoms, config.coeff)
        coef, z = _linear(host, flux, ferr, p, atoms, config.coeff)
        return p, coef, float(z @ z)

    atoms, log = [], []
    p = host.clip(p0)
    coef, z = _linear(host, flux, ferr, p, atoms, config.coeff)
    chi2 = chi2_0 = float(z @ z)
    for _ in range(config.max_atoms):
        rest = (flux - _atom_flux(time, coef, atoms, config.coeff)
                - coef[0] * (host.magnification(p) - _pspl_magnification(time, *p[:3])))
        scan = finder.run(time, rest, ferr, x0=p[:3], data_kind="flux", refit=False, verbose=False)
        ok = [c for c in scan.scored_candidates if allowed(c)]
        if not ok:
            break
        best = max(ok, key=lambda c: float(c.dchi2))
        trial = atoms + [(float(best.t0), float(best.teff))]
        p1, coef1, chi2_1 = refit(p, trial)
        if chi2 - chi2_1 <= price:
            log.append(("stop", float(best.t0), float(best.teff), chi2 - chi2_1))
            break
        atoms, p, coef, chi2 = trial, p1, coef1, chi2_1
        log.append(("add", float(best.t0), float(best.teff), float(best.score), chi2))
    while atoms:
        trials = [refit(p, atoms[:i] + atoms[i + 1:])
                  for i in range(len(atoms))]
        i = int(np.argmin([r[2] for r in trials]))
        if trials[i][2] - chi2 >= price:
            break
        log.append(("drop", *atoms[i], trials[i][2] - chi2))
        atoms = atoms[:i] + atoms[i + 1:]
        p, coef, chi2 = trials[i]
    if not atoms:
        p, coef, chi2 = refit(p, atoms)
    return AtomFit(
        params=np.asarray(p, dtype=float), fs=float(coef[0]), fb=float(coef[1]),
        atoms=tuple(Atom(t0, teff, float(coef[2 + 2 * i]), float(coef[3 + 2 * i]))
                    for i, (t0, teff) in enumerate(atoms)),
        chi2=chi2, chi2_host_only=chi2_0, log=tuple(log))


def _sigma(host, flux, ferr, p, atoms, coeff) -> np.ndarray:
    """1-sigma host parameter errors, fluxes and atom amplitudes profiled."""
    z0 = _linear(host, flux, ferr, p, atoms, coeff)[1]
    jac = []
    for k, s in enumerate(host.scale(p)):
        q = np.array(p, dtype=float)
        q[k] += 1e-3 * s
        jac.append((_linear(host, flux, ferr, q, atoms, coeff)[1] - z0) / (1e-3 * s))
    jac = np.array(jac).T
    try:
        return np.sqrt(np.diag(np.linalg.inv(jac.T @ jac)))
    except np.linalg.LinAlgError:
        return np.full(len(p), np.nan)


def _inside(time, windows) -> np.ndarray:
    inside = np.zeros(time.size, bool)
    for w in windows:
        inside |= (time >= w.t_start) & (time <= w.t_end)
    return inside


def _J(r: AtomFit, n: int, config: AtomConfig) -> float:
    return r.chi2 + (r.params.size + 2 + config.k_atom * len(r.atoms)) * np.log(n)


def _host_seen(host, flux, ferr, r: AtomFit, coeff) -> bool:
    """Whether the host alone (``fs A + fb``) beats a constant outside the atom windows."""
    keep = ~_inside(host.time, atom_windows(r.atoms, coeff))
    residual = flux - (r.fs * host.magnification(r.params) + r.fb)
    return _detected(flux, ferr, residual, keep, r.params.size + 2)


def atom_windows(atoms, coeff: float = 3.0) -> tuple[Candidate, ...]:
    """The atom supports ``|t - t0| <= coeff teff``, overlapping ones merged."""
    return tuple(Candidate(lo, hi) for lo, hi in _merged(
        [Candidate(a.t0 - coeff * abs(a.teff), a.t0 + coeff * abs(a.teff)) for a in atoms]))


def robust_fit(
    fitters: Mapping[str, object],
    finder,
    time,
    flux,
    ferr,
    p_scan,
    *,
    config: AtomConfig = AtomConfig(),
) -> RobustFit:
    """Fit every host model with atoms and select the lowest ``J``.

    ``fitters`` maps a model name to its fitter (PSPL, FSPL or point-source
    parallax); the first must be a :class:`PSPLFitter`. ``p_scan`` is the
    scan's PSPL ``(t0, tE, u0)``.

    Starts per model:

    * PSPL: ``p_scan``, relocked by :func:`relock_host` when it sits on a
      short anomaly; every other model: the PSPL result nested
      (``fitter.nested_seed``).
    * every model: ``base``, the plain fit (``fitter.fit(pspl_params=p_scan)``);
      when the relock moved the host's peak off its anomaly segment, the fit
      of the relock's H1 instead (that segment masked, located by the
      relocked host): the plain fit sits on the segment the relock rejected.
      A relocked host still peaking inside the segment (e.g. a high-
      magnification peak) keeps the plain fit, which needs that peak.
    * every model: ``fixed``, the host fitted on the samples outside the atom
      windows of the best free start (the lowest ``J`` among those whose host
      is seen outside its windows), held fixed while the atoms are regrown.

    The start with the lowest ``J`` among those whose host is seen outside its
    windows is kept (all, when none is), ``fixed`` counted at
    ``J - config.free_host_margin`` (``n_host ln n`` by default). The models
    are compared by ``J``, identified hosts first.
    """
    time, flux, ferr = (np.asarray(x, dtype=float) for x in (time, flux, ferr))
    names = list(fitters)
    if not names or not isinstance(fitters[names[0]], PSPLFitter):
        raise ValueError("the first fitter must be a PSPLFitter")
    p_scan = np.asarray(p_scan, dtype=float)[:3]
    pspl = fitters[names[0]]
    relock = relock_host(pspl, pspl.evaluate_fixed(time, flux, ferr, p_scan))
    p_pspl = np.asarray(relock.fit.params, dtype=float) if relock.relocked else p_scan
    anomaly = np.asarray(relock.mask, bool) if relock.relocked else None
    n = time.size
    # the relocked host's peak off the anomaly segment: the plain fit is not a start
    off = anomaly is not None and not anomaly[min(int(np.searchsorted(time, p_pspl[0])), n - 1)]
    base_kw = {"pspl_params": p_pspl, "mask": anomaly} if off else {"pspl_params": p_scan}
    models, errors, pspl_best = {}, {}, None
    for name in names:
        fitter = fitters[name]
        host = _Host(fitter, time)
        try:
            base = fitter.fit(time, flux, ferr, **base_kw)
        except Exception as exc:  # the model still has its nested start
            base, errors[f"{name}/base"] = None, repr(exc)
        if name == names[0]:
            starts = {"scan": p_pspl}
        elif pspl_best is not None:
            starts = {"nested": fitter.nested_seed(pspl_best)}
        else:  # no PSPL result to nest
            starts = {}
        if base is not None:
            starts["base"] = fitter.seed_from_fit(base)
        runs = {}
        for key, p0 in starts.items():
            try:
                runs[key] = fit_atoms(fitter, finder, time, flux, ferr, p0, anomaly=anomaly,
                                      config=config, _host=host)
            except Exception as exc:  # one failed start leaves the others
                errors[f"{name}/{key}"] = repr(exc)
        if not runs:
            continue
        seen = {k: _host_seen(host, flux, ferr, r, config.coeff) for k, r in runs.items()}
        J = {k: _J(r, n, config) for k, r in runs.items()}
        # "fixed": the host the samples outside the best free start's windows give, held
        free = min(runs, key=lambda k: (not seen[k], J[k]))
        outside = ~_inside(time, atom_windows(runs[free].atoms, config.coeff))
        if not outside.all():
            try:
                p0 = fitter.seed_from_fit(fitter.fit(time, flux, ferr, runs[free].params,
                                                     mask=~outside))
                runs["fixed"] = fit_atoms(fitter, finder, time, flux, ferr, p0, anomaly=anomaly,
                                          config=config, fix_host=True, _host=host)
                seen["fixed"] = _host_seen(host, flux, ferr, runs["fixed"], config.coeff)
                J["fixed"] = _J(runs["fixed"], n, config)
            except Exception as exc:
                errors[f"{name}/fixed"] = repr(exc)
        margin = (config.free_host_margin if config.free_host_margin is not None
                  else runs[free].params.size * np.log(n))
        rank = {k: v - margin * (k == "fixed") for k, v in J.items()}
        # a start whose host the samples outside its windows do not see is not a host
        identified = any(seen.values())
        start = min(rank, key=lambda k: (identified and not seen[k], rank[k]))
        r = runs[start]
        if name == names[0]:
            pspl_best = r.params[:3]
        windows = atom_windows(r.atoms, config.coeff)
        inside = _inside(time, windows)
        # windows over (nearly) all samples: nothing outside to score, evaluate on all
        few = np.count_nonzero(~inside) < r.params.size + 3
        fit = fitter.evaluate_fixed(time, flux, ferr, r.params,
                                    mask=None if few or not inside.any() else inside)
        # the host the atoms sit on: the atom fit's fs, fb, not refit outside the windows
        model = r.fs * host.magnification(r.params) + r.fb
        z = ((flux - model) / ferr)[~inside]
        fit = replace(fit, fs=r.fs, fb=r.fb, model_flux=model, residual=flux - model,
                      chi2=float(z @ z), chi2_dof=float(z @ z) / max(z.size - r.params.size, 1))
        models[name] = ModelFit(
            name=name, start=start, atom_fit=r, J=float(J[start]),
            rank=float(rank[start]),
            J_starts={k: float(v) for k, v in J.items()},
            sigma=_sigma(host, flux, ferr, r.params, [(a.t0, a.teff) for a in r.atoms], config.coeff),
            windows=windows, fit=fit,
            features=measure_features(
                time, flux, ferr, r.fs * host.magnification(r.params) + r.fb,
                _atom_flux(time, [0.0, 0.0] + [x for a in r.atoms for x in (a.a, a.b)],
                           [(a.t0, a.teff) for a in r.atoms], config.coeff),
                windows, config=FeatureConfig(coeff=config.coeff)) if windows else None,
            base=base, host_identified=identified)
    if not models:
        raise RuntimeError(f"no host model could be fitted: {errors}")
    selected = min(models, key=lambda k: (not models[k].host_identified, models[k].J))
    return RobustFit(models=models, selected=selected, relock=relock, errors=errors)


__all__ = ["Atom", "AtomConfig", "AtomFit", "ModelFit", "RobustFit", "atom_basis",
           "atom_windows", "fit_atoms", "robust_fit"]
