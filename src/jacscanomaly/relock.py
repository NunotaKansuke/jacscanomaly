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
``J1 <= J0`` (equal when both starts reach the same host), or when H0's
minimisation moved the host's ``t0`` out of ``fit0``'s core and by more than
``fit0``'s ``tE`` (H0 then left
``fit0`` for another event, so ``fit0`` itself is no host; the host is then
the better of H0's and H1's), and the host is seen by the samples it keeps (it beats a constant flux on them by
``chi2 + p ln n``): when the mask covers the event, the "host" is a fit to
the baseline; and, when its peak falls in a data gap longer than its ``tE``,
the samples bordering the gap are kept: a segment masked up to a season's
end otherwise frees the host to peak in the gap, where only the mask lets it
go (a peak just past the data, the data next to it kept, is constrained by
them; an anomaly masked on a sampled peak is the usual case).

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
    ``host`` is the masked host refit under H1, or H0's when H0 left ``fit0``
    for a better host (``None`` when ``fit0`` has no detectable core).
    ``mask`` marks its anomaly (``True`` = excluded). ``dJ = J0 - J1``.
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


def _peak_unmasked(fit: SingleLensFitResult, mask: np.ndarray) -> bool:
    """Whether a peak in a data gap (longer than ``tE``) has its bordering samples kept."""
    time = np.asarray(fit.time, dtype=float)
    t0, tE = float(np.asarray(fit.params)[0]), abs(float(np.asarray(fit.params)[1]))
    mask = np.asarray(mask, dtype=bool)
    before, after = np.flatnonzero(time <= t0), np.flatnonzero(time >= t0)
    near = ([before[np.argmax(time[before])]] if before.size else []) + \
           ([after[np.argmin(time[after])]] if after.size else [])
    if len(near) == 2 and time[near[1]] - time[near[0]] <= tE:
        return True  # the peak is sampled: a masked anomaly on it is the usual case
    return not mask[near].any()


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

    host0, mask0, j0 = _minimise_j(fitter, fit0, np.zeros_like(core),
                                   Candidate(time.min(), time.max()), penalty, n_iter)
    t0 = float(np.asarray(host0.params)[0])
    # H0 slid off fit0's event: its t0 left fit0's core and moved more than
    # fit0's tE (fit0 itself may peak in a gap, outside its core's samples)
    t00, tE0 = (float(x) for x in np.asarray(fit0.params)[:2])
    left = (not time[core].min() <= t0 <= time[core].max()) and abs(t0 - t00) > abs(tE0)
    rest = fitter.fit(time, flux, ferr, mask=core)
    host, mask, j1 = _minimise_j(fitter, rest, core,
                                 Candidate(time[core].min(), time[core].max()), penalty, n_iter)
    if left and j0 < j1:  # H0's own host is the better one off fit0
        host, mask = host0, mask0
    # the host must be one the kept samples see (not a fit to the baseline)
    relocked = ((j1 <= j0 or left)  # equal: same host from both starts
                and _host_detected(host, mask) and _peak_unmasked(host, mask))
    return HostRelock(relocked, host if relocked else fit0, host, mask, j0 - j1)


__all__ = ["HostRelock", "relock_host"]
