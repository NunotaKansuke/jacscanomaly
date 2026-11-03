"""One light curve and every step run on it: ``fit``, ``scan``, ``fit_atoms``.

::

    ev = Event(time, flux, ferr, ra_deg=268.4, dec_deg=-29.2, ephemeris="roman.txt")
    ev.fit("pspl")
    if ev.scan().detected:
        ev.fit("fspl", start="pspl")
        fspl_scan = ev.scan()
        ...
    ev.write_json("event.json")

The data are kept once, on a scale of order one (``flux / median|flux|``; a
magnitude light curve as flux relative to its median magnitude). Fluxes in
the results (``fs``, ``fb``, plots) are given back on the input scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional

import numpy as np

from .config import FinderConfig
from .finder import Finder
from .models import ScoredCandidate, named_params
from .relock import relock_host
from .robust import AtomConfig, _fit_model

# model name -> FinderConfig.fitter_kind
_KINDS = {"pspl": "pspl", "fspl": "fspl", "parallax": "pspl_parallax",
          "fspl_parallax": "fspl_parallax"}
_EXTRA = {"pspl": (), "fspl": ("rho",), "parallax": ("piEN", "piEE"),
          "fspl_parallax": ("rho", "piEN", "piEE")}


def _sig(x) -> float:
    """6 significant digits: shapes and fluxes."""
    return float(f"{float(x):.6g}")


def _t(x) -> float:
    """Times to 1e-5 d (about 1 s), whatever the epoch."""
    return round(float(x), 5)


def _chi2(x) -> float:
    """chi2, J and dchi2 to 0.01: their differences stay readable."""
    return round(float(x), 2)


def _params(p: Mapping[str, float]) -> dict:
    return {k: _t(v) if k == "t0" else _sig(v) for k, v in p.items()}


@dataclass(frozen=True)
class Fit:
    """A single-lens fit: named physical parameters, ``fs, fb`` on the input scale."""

    model: str
    params: Mapping[str, float]
    fs: float
    fb: float
    chi2: float
    J: float

    def to_dict(self) -> dict:
        return {"params": _params(self.params), "fs": _sig(self.fs), "fb": _sig(self.fb),
                "chi2": _chi2(self.chi2), "J": _chi2(self.J)}


@dataclass(frozen=True)
class Scan:
    """The residual scan of one fit.

    ``detections``: every candidate with ``score >= detection_score`` that
    ``candidate_criteria`` accepts, by score. ``relocked``: the PSPL host sat
    on the anomaly and was moved before this scan.
    """

    model: str
    detections: tuple[ScoredCandidate, ...]
    relocked: bool = False

    @property
    def detected(self) -> bool:
        return bool(self.detections)

    def to_dict(self) -> dict:
        return {"detected": self.detected, "relocked": self.relocked,
                "detections": [{"t0": _t(c.t0), "teff": _sig(c.teff),
                                "dchi2": _chi2(c.dchi2), "score": round(float(c.score), 1)}
                               for c in self.detections]}


@dataclass(frozen=True)
class Atoms:
    """Host + atoms of one model.

    ``features``: one per anomaly window (``kind``, ``caustic``, ``time``,
    ``timescale`` = FWHM, ``dchi2``). ``host_identified``: the host beats a
    constant flux outside the windows. ``dchi2``: what the atoms explain
    over the plain fit.
    """

    model: str
    params: Mapping[str, float]
    fs: float
    fb: float
    chi2: float
    J: float
    host_identified: bool
    n_atoms: int
    dchi2: float
    features: tuple = ()

    def to_dict(self) -> dict:
        return {"params": _params(self.params), "fs": _sig(self.fs), "fb": _sig(self.fb),
                "chi2": _chi2(self.chi2), "J": _chi2(self.J),
                "host_identified": self.host_identified,
                "n_atoms": self.n_atoms, "dchi2": _chi2(self.dchi2),
                "anomalies": [{"kind": f.kind, "caustic": bool(f.caustic),
                               "t_center": _t(f.time), "fwhm": _sig(f.timescale),
                               "dchi2": _chi2(f.dchi2)} for f in self.features]}


def _scaled(time, flux, ferr, kind):
    """Data on a scale of order one, and the factor back to the input flux."""
    time, flux, ferr = (np.asarray(x, dtype=float) for x in (time, flux, ferr))
    if kind not in ("flux", "mag"):
        raise ValueError("kind must be 'flux' or 'mag'.")
    if not (time.ndim == flux.ndim == ferr.ndim == 1 and time.size == flux.size == ferr.size):
        raise ValueError("time, flux and ferr must be 1D arrays of one length.")
    if not (np.isfinite(time).all() and np.isfinite(flux).all() and np.isfinite(ferr).all()):
        raise ValueError("time, flux and ferr must be finite.")
    if np.any(ferr <= 0):
        raise ValueError("ferr must be positive.")
    order = np.argsort(time, kind="stable")
    time, flux, ferr = time[order], flux[order], ferr[order]
    if kind == "mag":
        mag0 = float(np.median(flux))
        f = 10.0 ** (-0.4 * (flux - mag0))
        return time, f, 0.4 * np.log(10.0) * f * ferr, 1.0, mag0
    unit = float(np.median(np.abs(flux)))
    unit = unit if unit > 0 else 1.0
    return time, flux / unit, ferr / unit, unit, None


class Event:
    """One light curve; ``fit``, ``scan`` and ``fit_atoms`` keep their results here.

    ``kind="mag"`` takes magnitudes and their errors. ``ra_deg``, ``dec_deg``
    and ``ephemeris`` (a spacecraft ephemeris table) are needed for the
    parallax models only. ``config`` holds the search settings.

    Results are stored by model name in :attr:`fits`, :attr:`scans` and
    :attr:`atoms`. ``scan`` and ``fit_atoms`` act on the last fit unless a
    model is named; a model not fitted yet is an error. Refitting a model
    drops its own scan and atoms.
    """

    def __init__(self, time, flux, ferr, *, kind: str = "flux", ra_deg=None, dec_deg=None,
                 ephemeris=None, config: Optional[FinderConfig] = None):
        self.kind = kind
        self.time, self._flux, self._ferr, self._unit, self._mag0 = _scaled(time, flux, ferr, kind)
        self.config = config if config is not None else FinderConfig()
        self._sky = dict(ra_deg=ra_deg, dec_deg=dec_deg, satellite_ephemeris_path=ephemeris)
        self.fits: dict[str, Fit] = {}
        self.scans: dict[str, Scan] = {}
        self.atoms: dict[str, Atoms] = {}
        self.errors: dict[str, str] = {}
        self._raw: dict[str, Any] = {}        # model -> SingleLensFitResult (scaled)
        self._scan_raw: dict[str, Any] = {}   # model -> AnomalyResult, for the plots
        self._atoms_raw: dict[str, Any] = {}  # model -> ModelFit
        self._finders: dict[tuple, Finder] = {}
        self._relock = None
        self._last: Optional[str] = None

    @property
    def n_points(self) -> int:
        return int(self.time.size)

    # ------------------------------------------------------------- steps
    def fit(self, model: str, *, start: Optional[str] = None, x0=None,
            tref: Optional[float] = None) -> Fit:
        """Fit ``model`` (``"pspl"``, ``"fspl"``, ``"parallax"``, ``"fspl_parallax"``).

        The start is ``x0`` (the fitter's parameters), else the fit named by
        ``start``, else the last fit: its ``t0, tE, u0`` and, when both models
        have them, ``rho`` and ``piE``. With no fit yet the fitter's own
        multistart is used. ``tref``: the parallax reference time (default:
        the time of maximum flux).
        """
        fitter = self._fitter(model, tref)
        t, f, e = self.time, self._flux, self._ferr
        if start is not None:
            self._require(start)
        src = self._raw[start] if start is not None else self._raw.get(self._last)
        if x0 is not None:
            raw = fitter.fit(t, f, e, np.asarray(x0, dtype=float))
        elif src is None:
            raw = fitter.fit(t, f, e)
        elif model == "pspl":
            raw = fitter.fit(t, f, e, np.asarray(src.params, dtype=float)[:3])
        elif tuple(src.param_names) == tuple(self._raw_names(model)):
            raw = fitter.fit(t, f, e, fitter.seed_from_fit(src))
        else:
            # the multistart with the start's PSPL nested in it
            raw = fitter.fit(t, f, e, pspl_params=np.asarray(src.params, dtype=float)[:3])
            shared = [k for k in _EXTRA[model] if k in tuple(src.param_names)]
            if shared:  # and the start's own rho / piE
                other = fitter.fit(t, f, e, _mapped(fitter, src, model))
                raw = min((raw, other), key=lambda r: float(r.chi2))
        self._store(model, raw)
        return self.fits[model]

    def scan(self, model: Optional[str] = None) -> Scan:
        """Scan the residual of ``model`` (default: the last fit).

        A PSPL scan that detects something also checks whether the PSPL host
        sits on the anomaly (the relock); a moved host replaces the PSPL fit
        and is scanned again.
        """
        model = self._require(model)
        scanner = self._scanner()
        raw = self._raw_fit(model)
        res = scanner._scan(raw, self.time, self._flux, self._ferr)
        relocked = False
        if model == "pspl" and res.detections and self._relock is None:
            self._relock = relock_host(self._fitter("pspl"), raw)
            if self._relock.relocked:
                self._store("pspl", self._relock.fit, keep_relock=True)
                res = scanner._scan(self._raw["pspl"], self.time, self._flux, self._ferr)
                relocked = True
        self._scan_raw[model] = res
        self.scans[model] = Scan(model, tuple(res.detections), relocked)
        return self.scans[model]

    def fit_atoms(self, model: Optional[str] = None, *,
                  config: Optional[AtomConfig] = None) -> Atoms:
        """Fit the host of ``model`` (default: the last fit) jointly with atoms.

        Each atom is a residual-scan detection (``score >= detection_score``)
        that lowers ``J = chi2 + (n_host + 2 + 4 n_atoms) ln n``. Starts: the
        plain fit, the PSPL host nested, the fit with the relock's anomaly
        masked (when the relock moved the host), and the host fitted outside
        the atom windows (held fixed).
        """
        model = self._require(model)
        config = config if config is not None else AtomConfig(
            min_score=float(self.config.detection_score))
        fitter = self._fitter(model)
        t, f, e = self.time, self._flux, self._ferr
        base = self._raw_fit(model)
        starts = {"fit": fitter.seed_from_fit(base)}
        pspl = (self._atoms_raw_params("pspl") if model != "pspl" else None)
        if pspl is not None:
            starts["nested"] = fitter.nested_seed(pspl[:3])
        anomaly = None
        if self._relock is not None and self._relock.relocked:
            anomaly = np.asarray(self._relock.mask, bool)
            p = np.asarray(self._raw["pspl"].params, dtype=float)
            # the host's peak off the anomaly: the fit with it masked is a start
            if not anomaly[min(int(np.searchsorted(t, p[0])), t.size - 1)]:
                try:
                    starts["masked"] = fitter.seed_from_fit(
                        fitter.fit(t, f, e, pspl_params=p[:3], mask=anomaly))
                except Exception as exc:
                    self.errors[f"{model}/masked"] = repr(exc)
        errors: dict[str, str] = {}
        m = _fit_model(model, fitter, self._scanner(), t, f, e, starts, base, anomaly,
                       config, errors)
        self.errors.update(errors)
        if m is None:
            raise RuntimeError(f"atoms on {model!r} could not be fitted: {errors}")
        self._atoms_raw[model] = m
        self.atoms[model] = Atoms(
            model=model, params=named_params(m.fit), fs=float(m.atom_fit.fs) * self._unit,
            fb=float(m.atom_fit.fb) * self._unit, chi2=float(m.atom_fit.chi2), J=float(m.J),
            host_identified=bool(m.host_identified), n_atoms=len(m.atoms),
            dchi2=float(base.chi2) - float(m.atom_fit.chi2),
            features=tuple(m.features.features) if m.features is not None else ())
        return self.atoms[model]

    # ------------------------------------------------------------- output
    def to_dict(self) -> dict:
        """Every stored result; fluxes on the input scale."""
        return {"n_points": self.n_points, "kind": self.kind,
                "fits": {k: v.to_dict() for k, v in self.fits.items()},
                "scans": {k: v.to_dict() for k, v in self.scans.items()},
                "atoms": {k: v.to_dict() for k, v in self.atoms.items()}}

    def write_json(self, path) -> dict:
        from .output import write_json

        return write_json(path, self)

    def summary(self):
        """One row per model: the fit, its scan and its atoms (a DataFrame with pandas)."""
        rows = []
        for m, fit in self.fits.items():
            s, a = self.scans.get(m), self.atoms.get(m)
            rows.append({
                "model": m, "chi2": fit.chi2, "J": fit.J,
                "detected": None if s is None else s.detected,
                "n_detections": None if s is None else len(s.detections),
                "best_score": (max((float(c.score) for c in s.detections), default=np.nan)
                               if s is not None else np.nan),
                "atoms_J": np.nan if a is None else a.J,
                "n_atoms": None if a is None else a.n_atoms,
                "host_identified": None if a is None else a.host_identified,
            })
        try:
            import pandas as pd
            return pd.DataFrame(rows)
        except Exception:
            return rows

    def release(self) -> None:
        """Drop the scan grids and the model arrays; the results stay.

        A later ``scan`` or ``fit_atoms`` evaluates the model again; the plots
        need a new ``scan``.
        """
        self._scan_raw.clear()
        self._atoms_raw.clear()
        self._raw = {k: replace(r, model_flux=None, residual=None) for k, r in self._raw.items()}

    # ------------------------------------------------------------- plots
    def plot_lc(self, model: Optional[str] = None, **kwargs):
        return self._plotter().plot_lc(self._display(model), **kwargs)

    def plot_residual(self, model: Optional[str] = None, **kwargs):
        return self._plotter().plot_residual(self._display(model), **kwargs)

    def plot_anomaly_window(self, model: Optional[str] = None, **kwargs):
        return self._plotter().plot_anomaly_window(self._display(model), **kwargs)

    def plot_result(self, model: Optional[str] = None, **kwargs):
        return self._plotter().plot_result(self._display(model), **kwargs)

    # ------------------------------------------------------------- internals
    def _require(self, model: Optional[str] = None) -> str:
        model = self._last if model is None else model
        if model is None:
            raise RuntimeError("nothing is fitted yet: call ev.fit(model) first")
        if model not in _KINDS:
            raise ValueError(f"unknown model {model!r}; one of {sorted(_KINDS)}")
        if model not in self._raw:
            raise RuntimeError(f"{model!r} is not fitted yet: call ev.fit({model!r}) first")
        return model

    def _finder(self, model: str, tref: Optional[float] = None) -> Finder:
        if model not in _KINDS:
            raise ValueError(f"unknown model {model!r}; one of {sorted(_KINDS)}")
        tref = tref if tref is not None else self.config.tref
        key = (model, tref)
        if key not in self._finders:
            finder = Finder(replace(self.config, fitter_kind=_KINDS[model], tref=tref,
                                    **self._sky))
            finder._ensure_fitter(finder._resolve_parallax_tref(self.time, self._flux, x0=None))
            self._finders[key] = finder
        return self._finders[key]

    def _fitter(self, model: str, tref: Optional[float] = None):
        return self._finder(model, tref).fitter

    def _scanner(self) -> Finder:
        return self._finder("pspl")

    def _store(self, model: str, raw, *, keep_relock: bool = False) -> None:
        self._raw[model] = raw
        for d in (self.scans, self._scan_raw, self.atoms, self._atoms_raw):
            d.pop(model, None)
        if model == "pspl" and not keep_relock:
            self._relock = None
        n = self.n_points
        p = np.asarray(raw.params, dtype=float).reshape(-1)
        chi2 = float(raw.chi2)
        self.fits[model] = Fit(model=model, params=named_params(raw),
                               fs=float(raw.fs) * self._unit, fb=float(raw.fb) * self._unit,
                               chi2=chi2, J=chi2 + (p.size + 2) * np.log(n))
        self._last = model

    def _raw_fit(self, model: str):
        raw = self._raw[model]
        if raw.model_flux is None:  # released: evaluate again
            fitter = self._fitter(model)
            raw = fitter.evaluate_fixed(self.time, self._flux, self._ferr,
                                        fitter.seed_from_fit(raw))
            self._raw[model] = raw
        return raw

    @staticmethod
    def _raw_names(model: str) -> tuple:
        return ("t0", "tE", "u0") + _EXTRA[model]

    def _atoms_raw_params(self, model: str):
        m = self._atoms_raw.get(model)
        if m is not None:
            return np.asarray(m.params, dtype=float)
        raw = self._raw.get(model)
        return None if raw is None else np.asarray(raw.params, dtype=float)

    def _plotter(self):
        from .plot import AnomalyPlotter

        return AnomalyPlotter()

    def _display(self, model: Optional[str]):
        """The scan result of ``model`` on the input flux scale, for the plotters."""
        model = self._require(model)
        if model not in self._scan_raw:
            raise RuntimeError(f"no scan of {model!r} to plot: call ev.scan({model!r}) first")
        r, u = self._scan_raw[model], self._unit
        fit = r.fit
        fit = replace(fit, flux=np.asarray(fit.flux) * u, ferr=np.asarray(fit.ferr) * u,
                      fs=float(fit.fs) * u, fb=float(fit.fb) * u,
                      model_flux=np.asarray(fit.model_flux) * u,
                      residual=np.asarray(fit.residual) * u)
        return replace(r, flux=r.flux * u, ferr=r.ferr * u, residual=r.residual * u,
                       model_flux=r.model_flux * u, fit=fit)


def _mapped(fitter, src, model: str) -> np.ndarray:
    """``fitter``'s start from ``src``: its ``t0, tE, u0`` and the shared ``rho`` / ``piE``."""
    q = np.array(fitter.nested_seed(np.asarray(src.params, dtype=float)[:3]), dtype=float)
    have = dict(zip(src.param_names, np.asarray(src.params, dtype=float).reshape(-1)))
    for i, name in enumerate(("t0", "tE", "u0") + _EXTRA[model]):
        if i >= 3 and name in have:
            q[i] = np.log(max(abs(have[name]), 1e-12)) if name == "rho" else have[name]
    return q


__all__ = ["Atoms", "Event", "Fit", "Scan"]
