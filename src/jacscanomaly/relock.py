"""Move a single-lens fit off a short anomaly and onto its longer host event.

An automatic single-lens fit can lock onto a short, bright anomaly (e.g. a
planetary spike) instead of the host event around it. :func:`relock_host`
tests that and, when it holds, returns the host fit::

    r = relock_host(finder.fitter, result.fit)
    if r.relocked:  # rescan around the host; an unmasked refit slides back
        result = finder.run(time, flux, ferr, x0=r.fit.params, refit=False)

This is a layer composed of the post-fit components (``fitter.fit(mask=)``,
:class:`~jacscanomaly.Candidate`, :func:`~jacscanomaly.signal_mask`); it does
not change them.

Test, for a fit ``fit0``
------------------------
1. core = samples where ``fit0``'s event is detectable,
   ``(model - min(model)) / ferr > 1``.
2. Refit with the core excluded (the fitter's automatic init) -> ``rest``.
   ``dchi2 = chi2(fit0) - chi2(rest)`` on the samples outside the core is how
   much better another event explains everything except ``fit0``'s.
3. Shrink the exclusion to the anomaly itself: alternate
   ``signal_mask(host, Candidate(core time range))`` and the masked refit from
   ``rest``. Each step does not increase ``J = chi2_kept + penalty * n_masked``.
4. Relock when ``dchi2 >= min_dchi2`` and the anomaly is short against the
   host, ``span(mask) <= max_span * |tE u0|_host``.

Assumptions
-----------
* A planetary anomaly is much shorter than its host event (``max_span``).
  This rejects events whose whole light curve is anomalous: there the
  anomalous part stays a large fraction of any refit event.
* The event width is ``|tE u0|``, not ``tE``: a spike fit can reach
  ``tE ~ 1e3`` d with ``u0 ~ 1e-5``. The fit parameters must start with
  ``(t0, tE, u0)``, as in every single-lens fitter here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .candidate import Candidate
from .signal_mask import signal_mask
from .singlelens_fit import SingleLensFitResult


@dataclass(frozen=True)
class HostRelock:
    """Outcome of :func:`relock_host`.

    ``fit`` is the fit to use: the host fit when ``relocked``, else ``fit0``.
    ``host`` is the masked host refit (``None`` when ``fit0`` has no
    detectable core). ``mask`` marks the anomaly (``True`` = excluded).
    """

    relocked: bool
    fit: SingleLensFitResult
    host: Optional[SingleLensFitResult]
    mask: np.ndarray
    dchi2: float
    span_ratio: float


def _width(params) -> float:
    p = np.asarray(params, dtype=float)
    return abs(float(p[1] * p[2]))


def relock_host(
    fitter,
    fit0: SingleLensFitResult,
    *,
    min_dchi2: float = 100.0,
    max_span: float = 0.15,
    penalty: float = 2.0,
    n_iter: int = 4,
) -> HostRelock:
    """Return the host-event fit when ``fit0`` sits on a short anomaly.

    ``fitter`` is the single-lens fitter that produced ``fit0`` (any fitter
    with ``fit(time, flux, ferr, p0=None, *, mask=None)``).
    """
    names = tuple(getattr(fit0, "param_names", ()) or ())
    if names and names[:3] != ("t0", "tE", "u0"):
        raise ValueError(f"relock_host needs (t0, tE, u0, ...) parameters, got {names}")
    time = np.asarray(fit0.time, dtype=float)
    flux = np.asarray(fit0.flux, dtype=float)
    ferr = np.asarray(fit0.ferr, dtype=float)
    model0 = np.asarray(fit0.model_flux, dtype=float)
    core = (model0 - model0.min()) / ferr > 1.0
    if not core.any():
        return HostRelock(False, fit0, None, core, 0.0, float("inf"))

    rest = fitter.fit(time, flux, ferr, mask=core)
    keep = ~core
    chi0 = float(np.sum((np.asarray(fit0.residual)[keep] / ferr[keep]) ** 2))
    chi1 = float(np.sum((np.asarray(rest.residual)[keep] / ferr[keep]) ** 2))

    window = [Candidate(time[core].min(), time[core].max())]
    host, mask = rest, core
    for _ in range(n_iter):
        new_mask = signal_mask(host, window, penalty=penalty).mask
        if np.array_equal(new_mask, mask) or not new_mask.any():
            break
        mask = new_mask
        host = fitter.fit(time, flux, ferr, p0=np.asarray(host.params), mask=mask)

    width = _width(host.params)
    span_ratio = float(np.ptp(time[mask])) / width if width > 0.0 else float("inf")
    relocked = bool(chi0 - chi1 >= min_dchi2 and span_ratio <= max_span)
    return HostRelock(relocked, host if relocked else fit0, host, mask, chi0 - chi1, span_ratio)


__all__ = ["HostRelock", "relock_host"]
