"""Fit a single-lens model with the anomalies in the scan's windows masked.

:func:`masked_refit` refits ``fit`` with the anomaly inside each candidate
window excluded::

    m = masked_refit(fitter, fit, candidates)
    m.fit, m.mask, m.candidates, m.chi2, m.criterion

The windows (``t0 +- teff_coeff * teff`` of a scan candidate) say *where*
an anomaly is; they are the scan's search region, not its extent. How much
of each window is masked is decided by the one objective every model is
compared with,

    J = chi2_kept + penalty * n_masked + segment_cost * n_segments,

a masked sample being one free parameter (its own value) at the AIC price
``penalty = 2`` (as in :func:`~jacscanomaly.relock_host`), and a masked
segment costing ``segment_cost = 2 ln n`` for its start and end, located
among ``n`` samples (BIC). The window was already detected; a sample
inside it is masked when it is more anomalous than ``z^2 = 2`` on average
over a segment, and pure noise rarely pays a segment's ``2 ln n``. The mask is the segment inside each window
that pays for itself (:func:`~jacscanomaly.signal_mask`), alternated
with the masked refit until it repeats; each step does not increase ``J``.
Three starts are minimised and the lowest ``J`` is kept: ``fit`` as given,
the refit with the whole windows masked (a host not pulled by the anomaly),
and the fitter's automatic start (or the nested PSPL, :func:`masked_refit`). So a host
misfit around the anomaly is not masked when a host that fits it exists,
and a weak, broad window over the host peak masks nothing.

Models refitted on different masks (what is anomalous depends on the model)
are compared on all ``n`` samples by
``criterion = J + k ln n``, ``k`` counting the nonlinear parameters
(``fs``/``fb`` are common to every model and left out).

:func:`host_detected` asks whether the kept samples see the host event at
all: when the mask covers the event, a constant flux explains the kept
samples as well, and the host parameters are not measured. Such an event
is not "a single-lens event with a local anomaly";
:func:`constant_fit` gives the baseline to measure its features against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .candidate import Candidate, CandidateLike, as_candidates
from .signal_mask import signal_mask
from .singlelens_fit import SingleLensFitResult


@dataclass(frozen=True)
class MaskedRefit:
    """Outcome of :func:`masked_refit`.

    ``fit`` was fitted with ``mask`` excluded (unmasked when nothing is
    masked). ``candidates`` are the masked segments, one per window that
    masks anything; ``windows`` the windows searched. ``n_fit`` is the
    number of fits run.
    """

    fit: SingleLensFitResult
    mask: np.ndarray
    candidates: tuple[Candidate, ...]
    chi2: float
    criterion: float
    n_fit: int
    windows: tuple[Candidate, ...] = ()

    @property
    def n_masked(self) -> int:
        return int(np.count_nonzero(self.mask))


def window_mask(time, candidates: CandidateLike) -> np.ndarray:
    """Union of the candidate windows on ``time`` (``True`` = masked)."""
    t = np.asarray(time, dtype=float).reshape(-1)
    mask = np.zeros(t.shape, dtype=bool)
    for candidate in as_candidates(candidates):
        mask |= candidate.support(t)
    return mask


def _fit_mask(fit) -> np.ndarray:
    mask = getattr(fit, "mask", None)
    n = np.asarray(fit.time).size
    return np.zeros(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)


def _chi2_kept(fit, mask) -> float:
    z = np.asarray(fit.residual, dtype=float) / np.asarray(fit.ferr, dtype=float)
    return float(np.sum(z[~mask] ** 2))


def _minimise_j(fitter, fit, windows, penalty, segment_cost, n_iter):
    """Alternate signal_mask in ``windows`` and the masked refit from ``fit``.

    Returns ``(J, fit, segments, n_fit)``: ``segments`` are the masks (one
    per window that masks anything) whose union ``fit`` was fitted under.
    """
    time = np.asarray(fit.time, dtype=float)
    data = (time, np.asarray(fit.flux, dtype=float), np.asarray(fit.ferr, dtype=float))
    mask = _fit_mask(fit)
    segments, n_fit = ((mask,) if mask.any() else ()), 0
    for _ in range(n_iter):
        new = tuple(m for m in signal_mask(fit, windows, penalty=penalty,
                                           segment_cost=segment_cost,
                                           inside_only=True).per_candidate if m.any())
        new_mask = np.logical_or.reduce(new) if new else np.zeros(time.shape, dtype=bool)
        if np.array_equal(new_mask, mask):
            segments = new
            break
        try:
            fit = fitter.fit(*data, fitter.seed_from_fit(fit), mask=new_mask if new else None)
        except ValueError:  # too few samples kept to refit: J is scored as is
            break
        mask, segments, n_fit = new_mask, new, n_fit + 1
    return _j(fit, mask, segments, penalty, segment_cost), fit, segments, n_fit


def _j(fit, mask, segments, penalty, segment_cost) -> float:
    return _chi2_kept(fit, mask) + penalty * int(np.count_nonzero(mask)) + segment_cost * len(segments)


def _as_segments(time, masks) -> tuple[Candidate, ...]:
    out = []
    for m in masks:
        idx = np.flatnonzero(m)
        out.append(Candidate(float(time[idx[0]]), float(time[idx[-1]])))
    return tuple(out)


def masked_refit(
    fitter,
    fit: SingleLensFitResult,
    candidates: CandidateLike,
    *,
    nested=None,
    penalty: float = 2.0,
    segment_cost: float | None = None,
    n_iter: int = 4,
) -> MaskedRefit:
    """Refit ``fit`` with the anomaly in each candidate window masked.

    ``fitter`` produced ``fit`` (or fits the same model). ``penalty`` is the
    cost of a masked sample, ``segment_cost`` of a masked segment (default
    ``2 ln n``). From each start the mask and
    the fit are alternated (see the module doc); the lowest ``J`` is kept.

    Starts: ``fit``; ``fitter.seed_from_fit(fit)`` refitted with the whole
    windows masked; and either the fitter's automatic start under that mask
    (it leaves a basin ``fit`` may sit in, e.g. a host locked on the anomaly
    with ``u0 >> 1``) or, given ``nested`` (a PSPL fit or its
    ``(t0, tE, u0)``), the point of this model that is that PSPL
    (``fitter.nested_seed``: ``rho -> 0``, ``piE = 0``) under the PSPL's own
    mask. A model that contains the PSPL starts there at the PSPL's ``J`` or
    lower, so it ends no worse than the PSPL.
    """
    time = np.asarray(fit.time, dtype=float)
    values = as_candidates(candidates)
    full = window_mask(time, values)
    penalty = float(penalty)
    segment_cost = 2.0 * np.log(max(time.size, 2)) if segment_cost is None else float(segment_cost)
    data = (time, np.asarray(fit.flux, dtype=float), np.asarray(fit.ferr, dtype=float))
    kept = full if full.any() else None
    starts, n_fit = [fit], 0
    if full.any() and not np.array_equal(full, _fit_mask(fit)):
        starts.append(fitter.fit(*data, fitter.seed_from_fit(fit), mask=kept))
        n_fit += 1
    if nested is not None:
        params = np.asarray(getattr(nested, "params", nested), dtype=float)
        if hasattr(nested, "params"):  # the PSPL's own mask (none when it masks nothing)
            under = _fit_mask(nested)
            under = under if under.any() else None
        else:
            under = kept
        starts.append(fitter.fit(*data, fitter.nested_seed(params), mask=under))
        n_fit += 1
    elif len(starts) > 1:
        starts.append(fitter.fit(*data, mask=kept))
        n_fit += 1
    results = []
    for start in starts:
        j, f, segments, n = _minimise_j(fitter, start, values, penalty, segment_cost, n_iter)
        n_fit += n
        results.append((j, f, segments))
    j, fit, segments = min(results, key=lambda r: r[0])
    mask = np.logical_or.reduce(segments) if segments else np.zeros(time.shape, bool)
    chi2 = _chi2_kept(fit, mask)
    k = int(np.asarray(fit.params).size)
    return MaskedRefit(
        fit=fit,
        mask=mask,
        candidates=_as_segments(time, segments),
        chi2=chi2,
        criterion=j + k * np.log(max(mask.size, 1)),
        n_fit=n_fit,
        windows=values,
    )


def window_refit(
    fitter,
    fit: SingleLensFitResult,
    windows: CandidateLike,
    *,
    nested=None,
    penalty: float = 2.0,
    segment_cost: float | None = None,
) -> MaskedRefit:
    """Refit ``fit`` with the whole ``windows`` masked (their extent is given).

    Scored as :func:`masked_refit`, each window that masks a sample one
    segment. Starts: ``fitter.seed_from_fit(fit)`` and, given ``nested``,
    ``fitter.nested_seed`` of that PSPL; the lower ``chi2_kept`` is kept.
    """
    time = np.asarray(fit.time, dtype=float)
    values = as_candidates(windows)
    mask = window_mask(time, values)
    segment_cost = 2.0 * np.log(max(time.size, 2)) if segment_cost is None else float(segment_cost)
    data = (time, np.asarray(fit.flux, dtype=float), np.asarray(fit.ferr, dtype=float))
    seeds = [fitter.seed_from_fit(fit)]
    if nested is not None:
        seeds.append(fitter.nested_seed(np.asarray(getattr(nested, "params", nested), dtype=float)))
    fits = [fitter.fit(*data, seed, mask=mask if mask.any() else None) for seed in seeds]
    best = min(fits, key=lambda f: _chi2_kept(f, mask))
    n_segments = sum(bool(window_mask(time, [w]).any()) for w in values)
    chi2 = _chi2_kept(best, mask)
    k = int(np.asarray(best.params).size)
    return MaskedRefit(
        fit=best,
        mask=mask,
        candidates=values,
        chi2=chi2,
        criterion=chi2 + penalty * int(mask.sum()) + segment_cost * n_segments + k * np.log(max(mask.size, 1)),
        n_fit=len(fits),
        windows=values,
    )


def constant_fit(time, flux, ferr, mask=None) -> SingleLensFitResult:
    """Constant flux (weighted mean of the kept samples) as a fit result.

    It has no nonlinear parameters; ``fs`` is NaN, so features report
    ``flux / model`` ratios. Any component that takes a fit accepts it.
    """
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    e = np.asarray(ferr, dtype=float)
    keep = np.ones(t.shape, dtype=bool) if mask is None else ~np.asarray(mask, dtype=bool)
    w = 1.0 / e[keep] ** 2
    level = float(np.sum(w * f[keep]) / np.sum(w))
    model = np.full(t.shape, level)
    chi2 = float(np.sum(((f - model)[keep] / e[keep]) ** 2))
    return SingleLensFitResult(
        time=t, flux=f, ferr=e, params=np.zeros(0), param_names=(),
        chi2=chi2, chi2_dof=chi2 / max(int(keep.sum()) - 1, 1),
        fs=float("nan"), fb=level, model_flux=model, residual=f - model,
        model_kind="constant", mask=None if mask is None else ~keep,
    )


def host_detected(fit: SingleLensFitResult, mask=None) -> bool:
    """Whether ``fit`` beats a constant flux on the samples ``mask`` keeps.

    Both are scored by ``chi2 + p ln n`` over the kept samples, with ``p`` =
    the nonlinear parameters + ``fs``/``fb`` for the fit and 1 for the
    constant. ``mask`` defaults to the fit's own mask.
    """
    mask = _fit_mask(fit) if mask is None else np.asarray(mask, dtype=bool)
    keep = ~mask
    n = max(int(np.count_nonzero(keep)), 1)
    ferr = np.asarray(fit.ferr, dtype=float)
    chi2 = float(np.sum((np.asarray(fit.residual, dtype=float)[keep] / ferr[keep]) ** 2))
    flat = constant_fit(fit.time, fit.flux, ferr, mask)
    p = int(np.asarray(fit.params).size) + 2
    return chi2 + p * np.log(n) < float(flat.chi2) + np.log(n)


__all__ = ["MaskedRefit", "constant_fit", "host_detected", "masked_refit", "window_mask", "window_refit"]
