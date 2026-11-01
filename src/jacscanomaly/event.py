"""One event: its light curve in, one record out.

The stages are the module's (:meth:`Finder.run <jacscanomaly.Finder.run>`,
:func:`~jacscanomaly.robust_fit`); the pipeline runs them in order on a
:class:`LightCurve` and collects them in an :class:`EventResult`::

    lc = LightCurve(t, f, e, id="2_675_129")
    scan = finder.run(lc.time, lc.flux, lc.ferr, x0=p0)
    res = EventResult(lc, scan)
    if res.detected:
        res = EventResult(lc, scan, robust_fit(fitters, finder, lc.time, lc.flux, lc.ferr, p0))
    write_results(path, [res])

Assumptions
-----------
* "Anomaly" is the scan's decision: its best candidate scores
  ``>= score_threshold``. The atoms describe the anomaly; they do not decide it.
* ``without_atoms`` of a model is its plain fit (:attr:`ModelFit.base`; with
  no anomaly, the scan's PSPL fit), the best of lowest BIC
  ``chi2 + (n_params + 2) ln n``. ``with_atoms`` is the host + atoms fit, the
  best as :func:`~jacscanomaly.robust_fit` selects it.
* Parameters in the fit's convention (``rho``, not the seed's ``log rho``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import numpy as np

from .robust import ModelFit, RobustFit
from .singlelens_fit import SingleLensFitResult


@dataclass(frozen=True)
class LightCurve:
    """The data of one event, sorted by time; non-finite and ``ferr <= 0`` samples dropped."""

    time: np.ndarray
    flux: np.ndarray
    ferr: np.ndarray
    id: Optional[str] = None

    def __post_init__(self):
        t, f, e = (np.asarray(x, float).reshape(-1) for x in (self.time, self.flux, self.ferr))
        keep = np.isfinite(t) & np.isfinite(f) & np.isfinite(e) & (e > 0)
        order = np.argsort(t[keep], kind="stable")
        for k, x in (("time", t), ("flux", f), ("ferr", e)):
            object.__setattr__(self, k, x[keep][order])

    def __len__(self) -> int:
        return int(self.time.size)


@dataclass(frozen=True)
class EventResult:
    """The scan of one event and, when it found an anomaly, the robust fit."""

    lightcurve: LightCurve
    scan: object  # AnomalyResult
    fits: Optional[RobustFit] = None
    score_threshold: float = 10.0

    @property
    def detected(self) -> bool:
        best = getattr(self.scan, "best", None)
        return bool(best is not None and best.score >= self.score_threshold)

    def plain_fits(self) -> dict[str, SingleLensFitResult]:
        """The fit of every model without atoms (the scan's PSPL fit with no robust fit)."""
        if self.fits is None:
            return {"pspl": self.scan.fit}
        return {k: m.base for k, m in self.fits.models.items() if m.base is not None}

    def to_dict(self) -> dict:
        """JSON-ready record (``nan`` stays ``nan``; :func:`write_results` writes ``null``)."""
        best = getattr(self.scan, "best", None)
        get = lambda k: float(getattr(best, k)) if best is not None else np.nan
        plain = self.plain_fits()
        bic = {k: _bic(f) for k, f in plain.items()}
        atoms = self.fits.models if self.fits is not None else {}
        models = {}
        for k, f in plain.items():
            n = np.asarray(f.params).size
            models[k] = {"param_names": list(f.param_names)[:n],
                         "without_atoms": {"params": _floats(f.params), "chi2": float(f.chi2)}}
            if k in atoms:
                models[k]["with_atoms"] = _with_atoms(atoms[k])
        return {
            "anomaly": {"detected": self.detected, "score": get("score"), "t0": get("t0"),
                        "teff": get("teff"), "dchi2": get("dchi2")},
            "best_model": {"without_atoms": min(bic, key=bic.get) if bic else None,
                           "with_atoms": self.fits.selected if self.fits is not None else None},
            "n_points": len(self.lightcurve),
            "models": models,
        }


def _with_atoms(m: ModelFit) -> dict:
    feats = m.features.features if m.features is not None else ()
    return {"params": _floats(m.fit.params)[:m.params.size], "chi2": float(m.atom_fit.chi2),
            "features": [{"kind": f.kind, "caustic": bool(f.caustic), "t_center": float(f.time),
                          "fwhm": float(f.timescale), "dchi2": float(f.dchi2)} for f in feats]}


def _floats(x) -> list[float]:
    return np.asarray(x, float).reshape(-1).tolist()


def _bic(fit: SingleLensFitResult) -> float:
    """``chi2 + (n_params + 2) ln n`` on the kept samples (``fs, fb`` counted)."""
    n = int(fit.time.size) - (int(np.count_nonzero(fit.mask)) if fit.mask is not None else 0)
    return float(fit.chi2) + (np.asarray(fit.params).size + 2) * np.log(n)


def _json_safe(x):
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def _dump(x, ind: int = 0) -> str:
    """Indented JSON; number lists and short flat dicts (a feature) on one line."""
    pad, sub = "  " * ind, "  " * (ind + 1)
    flat = lambda v: not isinstance(v, (dict, list)) or (
        isinstance(v, list) and all(not isinstance(y, (dict, list)) for y in v))
    if isinstance(x, list) and x and not all(flat(y) for y in x):
        return "[\n" + ",\n".join(sub + _dump(y, ind + 1) for y in x) + "\n" + pad + "]"
    if isinstance(x, dict) and (not all(flat(v) for v in x.values()) or len(json.dumps(x)) > 200):
        sep = ",\n\n" if ind == 0 else ",\n"  # a blank line between events
        return "{\n" + sep.join(f"{sub}{json.dumps(k)}: {_dump(v, ind + 1)}"
                                for k, v in x.items()) + "\n" + pad + "}"
    return json.dumps(x, separators=(", ", ": "))


def write_results(path, results: Iterable[EventResult] | Mapping[str, dict]) -> dict:
    """Write ``{lightcurve.id: to_dict()}`` of every event to ``path``; returns it.

    ``results`` may also be such a mapping already made (records read back, to merge).
    """
    if not isinstance(results, Mapping):
        results = {r.lightcurve.id: r.to_dict() for r in results}
    doc = _json_safe(results)
    with open(path, "w") as fh:
        fh.write(_dump(doc) + "\n")
    return doc


__all__ = ["EventResult", "LightCurve", "write_results"]
