"""Move a single-lens fit off a short anomaly and onto its longer host event.

An automatic single-lens fit can lock onto a short, bright anomaly (e.g. a
planetary spike) instead of the host event around it. :func:`relock_host`
tests that and, when it holds, returns the host fit::

    r = relock_host(finder.fitter, result.fit)
    if r.relocked:  # rescan around the host; an unmasked refit slides back
        result = finder.run(time, flux, ferr, x0=r.fit.params, refit=False)

It uses only ``fitter.fit(..., mask=)``; the anomaly segment and the host
test below are private to this module.

Test, for a fit ``fit0``
------------------------
Two hypotheses, each "one single-lens event + one masked anomaly segment",
scored by ``J = chi2_kept + penalty * n_masked``:

* H0, ``fit0`` is the host: the anomaly may be anywhere.
* H1, ``fit0`` sits on the anomaly: the anomaly is inside ``fit0``'s core,
  the samples where its event is detectable,
  ``(model - min(model)) / ferr > 1``, and the host is another event, the
  fit with the core excluded (the fitter's automatic init).

The anomaly segment of a fit, for its parameters held fixed, is the
contiguous run overlapping the window that maximises ``sum(z_i^2 - penalty)``
(``z`` = residual / ferr); it does not cross a data gap longer than the
window. Each hypothesis is minimised from its start by alternating that
segment and the masked refit; each step does not increase ``J``. Relock when
``J1 <= J0`` (equal when both starts reach the same host) and the H1 host
is seen by the samples it keeps (it beats a constant flux on them by
``chi2 + p ln n``): when the mask covers the event, the "host" is a fit to
the baseline.

Assumptions
-----------
* Masking a sample frees one parameter (``penalty = 2``, AIC), the same
  Both hypotheses have the same parameters otherwise, so no other term
  enters.
* One contiguous anomaly segment per hypothesis.
* The fit parameters must start with ``(t0, tE, u0)``, as in every
  single-lens fitter here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .candidate import Candidate
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


def _best_segment(
    time: np.ndarray,
    z: np.ndarray,
    support: np.ndarray,
    *,
    penalty: float,
    max_gap: float,
    segment_cost: float = 0.0,
    inside_only: bool = False,
) -> tuple[np.ndarray, float]:
    """Maximum-gain segment of ``z**2 - penalty`` that overlaps ``support``."""
    mask = np.zeros(time.shape, dtype=bool)
    inside = np.flatnonzero(support)
    if inside.size == 0:
        return mask, 0.0
    # The data run (no gap longer than max_gap) that holds the window.
    breaks = np.flatnonzero(np.diff(time) > max_gap)
    lo = int(breaks[breaks < inside[0]][-1] + 1) if np.any(breaks < inside[0]) else 0
    hi = int(breaks[breaks >= inside[-1]][0]) if np.any(breaks >= inside[-1]) else time.size - 1
    if inside_only:
        lo, hi = max(lo, int(inside[0])), min(hi, int(inside[-1]))
    gain = np.square(z[lo : hi + 1]) - float(penalty)
    gain[~np.isfinite(gain)] = -float(penalty)
    n = gain.size
    ending = np.empty(n)    # best sum of a segment ending at i
    starting = np.empty(n)  # best sum of a segment starting at i
    ending[0] = gain[0]
    for i in range(1, n):
        ending[i] = gain[i] + max(ending[i - 1], 0.0)
    starting[-1] = gain[-1]
    for i in range(n - 2, -1, -1):
        starting[i] = gain[i] + max(starting[i + 1], 0.0)
    through = ending + starting - gain
    window = inside[(inside >= lo) & (inside <= hi)] - lo
    best = int(window[np.argmax(through[window])])
    total = float(through[best])
    if total <= segment_cost:
        return mask, total - segment_cost
    start = best
    while start > 0 and ending[start - 1] > 0.0:
        start -= 1
    end = best
    while end < n - 1 and starting[end + 1] > 0.0:
        end += 1
    # The window must carry evidence of its own; otherwise the segment only
    # reaches through noise to a signal that belongs to another window.
    if float(np.sum(gain[start : end + 1][support[lo + start : lo + end + 1]])) <= segment_cost:
        return mask, 0.0
    mask[lo + start : lo + end + 1] = True
    return mask, total - segment_cost


def _segment_mask(fit, window: Candidate, penalty: float) -> np.ndarray:
    """The anomaly segment of ``fit`` overlapping ``window`` (``True`` = anomaly)."""
    time = np.asarray(fit.time, dtype=float).reshape(-1)
    z = (np.asarray(fit.residual, dtype=float).reshape(-1)
         / np.maximum(np.asarray(fit.ferr, dtype=float).reshape(-1), 1.0e-12))
    return _best_segment(time, z, window.support(time), penalty=penalty,
                         max_gap=window.t_end - window.t_start)[0]


def _host_detected(fit: SingleLensFitResult, mask: np.ndarray) -> bool:
    """Whether ``fit`` beats a constant flux on the samples ``mask`` keeps.

    Both are scored by ``chi2 + p ln n`` over the kept samples, with ``p`` =
    the nonlinear parameters + ``fs``/``fb`` for the fit and 1 for the
    constant.
    """
    keep = ~np.asarray(mask, dtype=bool)
    n = max(int(np.count_nonzero(keep)), 1)
    flux = np.asarray(fit.flux, dtype=float)[keep]
    ferr = np.asarray(fit.ferr, dtype=float)[keep]
    chi2 = float(np.sum((np.asarray(fit.residual, dtype=float)[keep] / ferr) ** 2))
    w = 1.0 / ferr**2
    flat = float(np.sum(((flux - np.sum(w * flux) / np.sum(w)) / ferr) ** 2))
    p = int(np.asarray(fit.params).size) + 2
    return chi2 + p * np.log(n) < flat + np.log(n)


def _minimise_j(fitter, fit, mask, window, penalty, n_iter):
    """Alternate the anomaly segment and the masked refit from ``(fit, mask)``."""
    time = np.asarray(fit.time, dtype=float)
    flux = np.asarray(fit.flux, dtype=float)
    ferr = np.asarray(fit.ferr, dtype=float)
    for _ in range(n_iter):
        new_mask = _segment_mask(fit, window, penalty)
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
    relocked = j1 <= j0 and _host_detected(host, mask)  # equal: same host from both starts
    return HostRelock(relocked, host if relocked else fit0, host, mask, j0 - j1)


__all__ = ["HostRelock", "relock_host"]
