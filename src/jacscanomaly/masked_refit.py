"""Fit a single-lens model with the scan's anomaly windows masked.

:func:`masked_refit` refits ``fit`` with every sample inside the candidate
windows excluded::

    m = masked_refit(fitter, fit, candidates)
    m.fit, m.mask, m.chi2, m.criterion

The mask is the windows themselves, the extent the anomaly scan measured
with its template (``t0 +- teff_coeff * teff``). It is not re-derived from
the residual point by point: a per-point test cannot tell the anomaly from
the host's own misfit around it (a host fitted with the anomaly in it is off
over the whole peak, and masking that removes what constrains ``u0``), and
it misses a coherent anomaly whose samples are each weak.

Models refitted on different windows (what is anomalous depends on the
model) are compared on all ``n`` samples by
``criterion = chi2_kept + penalty * n_masked + k ln n``: a masked sample
costs ``penalty`` (2: one free parameter, AIC, the price
:func:`~jacscanomaly.signal_mask` and :func:`~jacscanomaly.relock_host` use),
and ``k`` counts the nonlinear parameters (``fs``/``fb`` are common to every
model and left out). With the same windows the mask term cancels.

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
from .singlelens_fit import SingleLensFitResult


@dataclass(frozen=True)
class MaskedRefit:
    """Outcome of :func:`masked_refit`.

    ``fit`` was fitted with ``mask`` excluded (unmasked when no window holds a
    sample). ``n_fit`` is the number of fits run: 0 when ``fit`` already had
    that mask.
    """

    fit: SingleLensFitResult
    mask: np.ndarray
    candidates: tuple[Candidate, ...]
    chi2: float
    criterion: float
    n_fit: int

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


def masked_refit(
    fitter,
    fit: SingleLensFitResult,
    candidates: CandidateLike,
    *,
    nested=None,
    penalty: float = 2.0,
) -> MaskedRefit:
    """Refit ``fit`` with the candidate windows masked.

    ``fitter`` produced ``fit`` (or fits the same model). Two refits run
    under the mask and the one with the lower ``chi2_kept`` is kept: one
    continues from ``fitter.seed_from_fit(fit)``; the other leaves the basin
    ``fit`` may have been pulled into by the samples now masked (e.g. a host
    locked on the anomaly with ``u0 >> 1``), which a local refit cannot.

    That second start is the fitter's own automatic one, or, given
    ``nested`` (a PSPL fit or its ``(t0, tE, u0)``), the point of this model
    that is that PSPL (``fitter.nested_seed``: ``rho -> 0``, ``piE = 0``). A
    model that contains the PSPL fits the kept samples at least as well as
    the PSPL does there, and the nested start makes the refit reach that,
    where a start elsewhere can end on a worse local optimum (an FSPL stuck
    at ``rho ~ u0``). When ``fit`` was already fitted with the same mask it
    is kept as is, and only the nested start (if given) is fitted against it.
    """
    time = np.asarray(fit.time, dtype=float)
    values = as_candidates(candidates)
    mask = window_mask(time, values)
    data = (time, np.asarray(fit.flux, dtype=float), np.asarray(fit.ferr, dtype=float))
    kept = mask if mask.any() else None
    fits = [fit] if np.array_equal(mask, _fit_mask(fit)) else [
        fitter.fit(*data, fitter.seed_from_fit(fit), mask=kept)
    ]
    if nested is not None:
        params = np.asarray(getattr(nested, "params", nested), dtype=float)
        fits.append(fitter.fit(*data, fitter.nested_seed(params), mask=kept))
    elif fits[0] is not fit:
        fits.append(fitter.fit(*data, mask=kept))
    n_fit = sum(f is not fit for f in fits)
    fit = min(fits, key=lambda f: _chi2_kept(f, mask))
    chi2 = _chi2_kept(fit, mask)
    k = int(np.asarray(fit.params).size)
    return MaskedRefit(
        fit=fit,
        mask=mask,
        candidates=values,
        chi2=chi2,
        criterion=chi2 + penalty * int(np.count_nonzero(mask)) + k * np.log(max(mask.size, 1)),
        n_fit=n_fit,
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


__all__ = ["MaskedRefit", "constant_fit", "host_detected", "masked_refit", "window_mask"]
