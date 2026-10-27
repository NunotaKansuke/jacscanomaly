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
Two hypotheses, each "one single-lens event + one masked anomaly segment",
scored by ``J = chi2_kept + penalty * n_masked``:

* H0, ``fit0`` is the host: the anomaly may be anywhere.
* H1, ``fit0`` sits on the anomaly: the anomaly is inside ``fit0``'s core,
  the samples where its event is detectable,
  ``(model - min(model)) / ferr > 1``, and the host is another event, the
  fit with the core excluded (the fitter's automatic init).

Each is minimised from its start by alternating ``signal_mask(fit, window)``
and the masked refit; each step does not increase ``J``. Relock when
``J1 <= J0`` (equal when both starts reach the same host) and the H1 host
is seen by the samples it keeps (:func:`~jacscanomaly.host_detected`):
when the mask covers the event, the "host" is a fit to the baseline.

Assumptions
-----------
* Masking a sample frees one parameter (``penalty = 2``, AIC), the same
  price :func:`~jacscanomaly.signal_mask` uses. Both hypotheses have the
  same parameters otherwise, so no other term enters.
* One contiguous anomaly segment per hypothesis.
* The fit parameters must start with ``(t0, tE, u0)``, as in every
  single-lens fitter here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .candidate import Candidate
from .masked_refit import host_detected
from .signal_mask import signal_mask
from .singlelens_fit import SingleLensFitResult


@dataclass(frozen=True)
class HostRelock:
    """Outcome of :func:`relock_host`.

    ``fit`` is the fit to use: the host fit when ``relocked``, else ``fit0``.
    ``host`` is the masked host refit under H1 (``None`` when ``fit0`` has no
    detectable core). ``mask`` marks its anomaly (``True`` = excluded).
    ``dJ = J0 - J1``; relocked iff ``dJ >= 0``.
    """

    relocked: bool
    fit: SingleLensFitResult
    host: Optional[SingleLensFitResult]
    mask: np.ndarray
    dJ: float


def _minimise_j(fitter, fit, mask, window, penalty, n_iter):
    """Alternate signal_mask and the masked refit from ``(fit, mask)``."""
    time = np.asarray(fit.time, dtype=float)
    flux = np.asarray(fit.flux, dtype=float)
    ferr = np.asarray(fit.ferr, dtype=float)
    for _ in range(n_iter):
        new_mask = signal_mask(fit, [window], penalty=penalty).mask
        if np.array_equal(new_mask, mask) or not new_mask.any():
            break
        mask = new_mask
        try:
            fit = fitter.fit(time, flux, ferr, fitter.seed_from_fit(fit), mask=mask)
        except ValueError:  # too few samples kept to refit: J is scored as is
            break
    z = np.asarray(fit.residual)[~mask] / ferr[~mask]
    return fit, mask, float(np.sum(z**2)) + penalty * int(np.count_nonzero(mask))


def relock_host(
    fitter,
    fit0: SingleLensFitResult,
    *,
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
        return HostRelock(False, fit0, None, core, float("-inf"))

    _, _, j0 = _minimise_j(fitter, fit0, np.zeros_like(core),
                           Candidate(time.min(), time.max()), penalty, n_iter)
    rest = fitter.fit(time, flux, ferr, mask=core)
    host, mask, j1 = _minimise_j(fitter, rest, core,
                                 Candidate(time[core].min(), time[core].max()), penalty, n_iter)
    # H1 needs a host: one the kept samples see (not a fit to the baseline)
    relocked = j1 <= j0 and host_detected(host, mask)  # equal: same host from both starts
    return HostRelock(relocked, host if relocked else fit0, host, mask, j0 - j1)


__all__ = ["HostRelock", "relock_host"]
