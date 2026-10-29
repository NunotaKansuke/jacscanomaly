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
* An atom is at most ``max_teff_fraction`` of the start host's ``tE`` wide, so
  atoms cannot take over the host -- except an atom centred on the segment
  :func:`~jacscanomaly.relock_host` moved the host off, which that test has
  already shown is not host.
* The start host is held fixed (``fs, fb`` only) until the first atom is in,
  so a relocked host is not pulled back onto the spike it was moved off; with
  no atom at the end the host is refit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np
from scipy.optimize import least_squares

from .candidate import Candidate, _merged
from .features import FeatureResult, measure_features
from .fitters import FSPLFitter, PSPLFitter, _pspl_magnification
from .parallax_backend import _ParallaxFitter
from .relock import HostRelock, relock_host
from .singlelens_fit import SingleLensFitResult


@dataclass(frozen=True)
class AtomConfig:
    """Atom support (``coeff`` in units of ``teff``), acceptance and price."""

    coeff: float = 3.0
    min_score: float = 10.0
    max_teff_fraction: float = 0.25
    max_atoms: int = 15
    k_atom: float = 4.0


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

    ``fit`` is the host alone, with ``fs, fb`` from the samples outside
    ``windows`` (the merged atom supports); ``features`` are measured on it.
    ``sigma`` are the 1-sigma host parameter errors with fluxes and atom
    amplitudes profiled. ``base`` is the plain fit without atoms.
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

    @property
    def params(self) -> np.ndarray:
        return self.atom_fit.params

    @property
    def atoms(self) -> tuple[Atom, ...]:
        return self.atom_fit.atoms


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
    """Linear coefficients (fs, fb, a_1, b_1, ...) and the normalised residual."""
    cols = [host.magnification(p), np.ones_like(host.time)]
    for t0, teff in atoms:
        cols += atom_basis(host.time, t0, teff, coeff)
    design = np.vstack(cols).T / ferr[:, None]
    coef, *_ = np.linalg.lstsq(design, flux / ferr, rcond=None)
    return coef, flux / ferr - design @ coef


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
    _host: Optional[_Host] = None,
) -> AtomFit:
    """Grow and prune atoms on the host model of ``fitter`` from the start ``p0``.

    ``finder`` scans the residual (its PSPL host is given the data minus the
    atoms minus ``fs (A_host - A_pspl)``, the PSPL-shaped remainder).
    ``anomaly`` (``True`` = anomaly) exempts atoms centred there from the
    ``tE`` cap; pass :attr:`HostRelock.mask` of a relocked host.
    """
    time, flux, ferr = (np.asarray(x, dtype=float) for x in (time, flux, ferr))
    host = _host or _Host(fitter, time)
    price = config.k_atom * np.log(time.size)
    criteria = finder.config.candidate_criteria
    anomaly = np.zeros(time.size, bool) if anomaly is None else np.asarray(anomaly, bool)
    max_teff = config.max_teff_fraction * abs(float(p0[1]))

    def allowed(candidate) -> bool:
        if float(candidate.score) < config.min_score:
            return False
        if criteria is not None and not criteria.accepts(dchi2=float(candidate.dchi2),
                                                         quality=candidate.quality):
            return False
        if abs(float(candidate.teff)) <= max_teff:
            return True
        i = min(int(np.searchsorted(time, float(candidate.t0))), time.size - 1)
        return bool(anomaly[i])

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
        p1, coef1, chi2_1 = _fit_host(host, flux, ferr, p, trial, config.coeff)
        if chi2 - chi2_1 <= price:
            log.append(("stop", float(best.t0), float(best.teff), chi2 - chi2_1))
            break
        atoms, p, coef, chi2 = trial, p1, coef1, chi2_1
        log.append(("add", float(best.t0), float(best.teff), float(best.score), chi2))
    while atoms:
        trials = [_fit_host(host, flux, ferr, p, atoms[:i] + atoms[i + 1:], config.coeff)
                  for i in range(len(atoms))]
        i = int(np.argmin([r[2] for r in trials]))
        if trials[i][2] - chi2 >= price:
            break
        log.append(("drop", *atoms[i], trials[i][2] - chi2))
        atoms = atoms[:i] + atoms[i + 1:]
        p, coef, chi2 = trials[i]
    if not atoms:
        p, coef, chi2 = _fit_host(host, flux, ferr, p, atoms, config.coeff)
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
    * every model: ``base``, the plain fit (``fitter.fit(pspl_params=p_scan)``).
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
    models, errors, pspl_best = {}, {}, None
    for name in names:
        fitter = fitters[name]
        host = _Host(fitter, time)
        try:
            base = fitter.fit(time, flux, ferr, pspl_params=p_scan)
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
        J = {key: r.chi2 + (r.params.size + 2 + config.k_atom * len(r.atoms)) * np.log(n)
             for key, r in runs.items()}
        start = min(J, key=J.get)
        r = runs[start]
        if name == names[0]:
            pspl_best = r.params[:3]
        windows = atom_windows(r.atoms, config.coeff)
        inside = np.zeros(n, bool)
        for w in windows:
            inside |= (time >= w.t_start) & (time <= w.t_end)
        fit = fitter.evaluate_fixed(time, flux, ferr, r.params, mask=inside if inside.any() else None)
        models[name] = ModelFit(
            name=name, start=start, atom_fit=r, J=float(J[start]),
            J_starts={k: float(v) for k, v in J.items()},
            sigma=_sigma(host, flux, ferr, r.params, [(a.t0, a.teff) for a in r.atoms], config.coeff),
            windows=windows, fit=fit,
            features=measure_features(fit, windows) if windows else None, base=base)
    if not models:
        raise RuntimeError(f"no host model could be fitted: {errors}")
    selected = min(models, key=lambda k: models[k].J)
    return RobustFit(models=models, selected=selected, relock=relock, errors=errors)


__all__ = ["Atom", "AtomConfig", "AtomFit", "ModelFit", "RobustFit", "atom_basis",
           "atom_windows", "fit_atoms", "robust_fit"]
