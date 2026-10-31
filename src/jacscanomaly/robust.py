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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np
from scipy.optimize import least_squares

from . import _cpp_grid
from .candidate import Candidate, _merged
from .features import FeatureConfig, FeatureResult, measure_features
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
    ``windows`` (the merged atom supports). ``features`` characterize every
    window against the host + atoms of the atom fit (its own ``fs, fb``).
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

    @property
    def dchi2(self) -> float:
        """``chi2(base) - chi2(host + atoms)``: what the atoms explain (``nan`` without ``base``)."""
        return float(self.base.chi2) - self.atom_fit.chi2 if self.base is not None else np.nan

    def to_dict(self) -> dict:
        """JSON-ready record (``nan`` stays ``nan``; :func:`write_event_json` writes ``null``)."""
        a = self.atom_fit
        n = int(self.fit.time.size)
        feats = self.features.features if self.features is not None else ()
        # the fit's own convention (rho, not the seed's log rho); sigma follows
        params = np.asarray(self.fit.params, float).reshape(-1)[:a.params.size]
        sigma = np.asarray(self.sigma, float) * np.where(params != a.params, params, 1.0)
        return {
            "param_names": list(self.fit.param_names)[:a.params.size],
            "params": params.tolist(), "sigma": sigma.tolist(),
            "fs": float(a.fs), "fb": float(a.fb),
            "J": self.J, "start": self.start, "J_starts": dict(self.J_starts),
            "n_points": n, "n_free": int(a.params.size) + 2 + 4 * len(a.atoms),
            "chi2": float(a.chi2),
            "chi2_masked": float(self.fit.chi2),
            "n_masked": int(np.count_nonzero(self.fit.mask)) if self.fit.mask is not None else 0,
            "dchi2_atoms": self.dchi2,  # chi2 of the scan stage's fit (base) - chi2
            "atoms": [{"t0": x.t0, "teff": x.teff, "a": x.a, "b": x.b} for x in a.atoms],
            "features": [{**{k: v for k, v in f.summary_dict().items() if k != "parts"},
                          "parts": [{"kind": q.kind, "time": q.time, "fwhm": q.timescale}
                                    for q in f.parts]} for f in feats],
        }


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

    def to_dict(self, event_id=None, scan=None, score_threshold: float = 10.0) -> dict:
        """One record of the event, in its two stages.

        ``scan``: the Finder's ``AnomalyResult`` (anomaly = its best candidate
        scores ``>= score_threshold``) and the plain fit of every model
        (``base``: the PSPL fit and the FSPL / parallax fallbacks), the one of
        lowest BIC selected. ``atoms``: the host + atoms fit of every model,
        the one of lowest ``J`` selected, with its features.
        """
        bases = {k: m.base for k, m in self.models.items() if m.base is not None}
        bic = {k: _bic(b) for k, b in bases.items()}
        sel = min(bic, key=bic.get) if bic else None
        return {
            "event_id": event_id,
            "scan": {
                "anomaly": scan_anomaly(scan, score_threshold),
                "candidates": [{"t0": c.t0, "teff": c.teff, "dchi2": c.dchi2, "score": c.score}
                               for c in getattr(scan, "scored_candidates", ())
                               if c.score >= score_threshold],
                "selected": sel,
                "models": {k: {"selected": k == sel, **_fit_dict(b), "bic": bic[k]}
                           for k, b in bases.items()},
            },
            "atoms": {
                "selected": self.selected,
                "relocked": bool(self.relock.relocked),
                "models": {k: {"selected": k == self.selected, **m.to_dict()}
                           for k, m in self.models.items()},
            },
            "errors": dict(self.errors),
        }


def _fit_dict(fit: SingleLensFitResult) -> dict:
    """A plain single-lens fit: parameters, fluxes, chi2 on its kept samples."""
    params = np.asarray(fit.params, float).reshape(-1)
    return {"param_names": list(fit.param_names)[:params.size], "params": params.tolist(),
            "fs": float(fit.fs), "fb": float(fit.fb), "chi2": float(fit.chi2),
            "n_points": int(fit.time.size),
            "n_masked": int(np.count_nonzero(fit.mask)) if fit.mask is not None else 0}


def _bic(fit: SingleLensFitResult) -> float:
    """``chi2 + (n_params + 2) ln n`` on the kept samples (``fs, fb`` counted)."""
    n = int(fit.time.size) - (int(np.count_nonzero(fit.mask)) if fit.mask is not None else 0)
    return float(fit.chi2) + (np.asarray(fit.params).size + 2) * np.log(n)


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
    magnification = host.magnification(p)
    out = _cpp_grid.atoms_linear(magnification, host.time, flux, ferr,
                                 np.asarray(atoms, dtype=float).reshape(-1, 2), coeff)
    if out is not None:
        return out
    # numerically singular normal equations: minimum-norm SVD solution
    cols = [magnification, np.ones_like(host.time)]
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
    * every model: ``base``, the plain fit (``fitter.fit(pspl_params=p_scan)``);
      when the relock moved the host's peak off its anomaly segment, the fit
      of the relock's H1 instead (that segment masked, located by the
      relocked host): the plain fit sits on the segment the relock rejected.
      A relocked host still peaking inside the segment (e.g. a high-
      magnification peak) keeps the plain fit, which needs that peak.

    The start with the lowest ``J`` is kept.
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
            features=measure_features(
                time, flux, ferr, r.fs * host.magnification(r.params) + r.fb,
                _atom_flux(time, [0.0, 0.0] + [x for a in r.atoms for x in (a.a, a.b)],
                           [(a.t0, a.teff) for a in r.atoms], config.coeff),
                windows, config=FeatureConfig(coeff=config.coeff)) if windows else None,
            base=base)
    if not models:
        raise RuntimeError(f"no host model could be fitted: {errors}")
    selected = min(models, key=lambda k: models[k].J)
    return RobustFit(models=models, selected=selected, relock=relock, errors=errors)


def scan_anomaly(scan, score_threshold: float = 10.0) -> dict:
    """The scan's anomaly decision: its best candidate scores ``>= score_threshold``."""
    best = getattr(scan, "best", None)
    get = lambda k: float(getattr(best, k)) if best is not None else np.nan
    return {"detected": bool(best is not None and best.score >= score_threshold) if scan is not None
            else None,
            "score": get("score"), "score_threshold": float(score_threshold),
            "dchi2": get("dchi2"), "t0": get("t0"), "teff": get("teff")}


def _json_safe(x):
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)) or getattr(x, "shape", None) == ():
        x = float(x)
        return x if np.isfinite(x) else None
    return x


def write_event_json(path, fits: RobustFit, event_id=None, scan=None,
                     score_threshold: float = 10.0) -> dict:
    """Write ``fits.to_dict(...)`` to ``path`` (non-finite numbers as ``null``)."""
    import json

    record = _json_safe(fits.to_dict(event_id=event_id, scan=scan, score_threshold=score_threshold))
    with open(path, "w") as fh:
        json.dump(record, fh, indent=1)
    return record


__all__ = ["Atom", "AtomConfig", "AtomFit", "ModelFit", "RobustFit", "atom_basis",
           "atom_windows", "fit_atoms", "robust_fit", "scan_anomaly", "write_event_json"]
