"""Signal masks as the residual segment that pays for its own exclusion.

:func:`signal_mask` marks the samples of a single-lens fit that belong to an
anomaly, one contiguous segment per :class:`~jacscanomaly.Candidate`. It does
not refit; pass the mask to a fitter's ``mask=`` to refit without the signal.

Criterion
---------
Excluding a sample from the fit is the same as giving it one free parameter
(its own value), which removes its ``z_i^2 = (residual_i / ferr_i)^2`` from
chi-square. With a cost ``penalty`` per freed parameter (2 = AIC), the
objective

    J(mask, theta) = chi2_kept(theta) + penalty * n_masked

is minimised over masks, for the fit parameters ``theta`` held fixed, by the
segment that maximises ``sum(z_i^2 - penalty)``. Per candidate, the segment
is the best one that overlaps the candidate window; it is empty unless both
the segment and its part inside the window have a positive gain. It does not
cross a data gap longer than the window itself.

A masked refit minimises ``chi2_kept`` with the mask held fixed, so
alternating ``signal_mask`` and ``fitter.fit(..., mask=...)`` never increases
``J`` and stops when the mask repeats.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .candidate import Candidate, CandidateLike, as_candidates


@dataclass(frozen=True)
class SignalMask:
    """Signal points of one fit (``True`` = signal), per candidate and in union.

    ``gains[k]`` is ``sum(z_i^2 - penalty)`` over candidate ``k``'s segment,
    i.e. the drop in ``J`` from masking it (``<= 0`` means no segment).
    """

    mask: np.ndarray
    per_candidate: tuple[np.ndarray, ...]
    candidates: tuple[Candidate, ...]
    gains: tuple[float, ...]
    penalty: float

    @property
    def n_masked(self) -> int:
        return int(np.count_nonzero(self.mask))


def _best_segment(
    time: np.ndarray,
    z: np.ndarray,
    support: np.ndarray,
    *,
    penalty: float,
    max_gap: float,
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
    if total <= 0.0:
        return mask, total
    start = best
    while start > 0 and ending[start - 1] > 0.0:
        start -= 1
    end = best
    while end < n - 1 and starting[end + 1] > 0.0:
        end += 1
    # The window must carry evidence of its own; otherwise the segment only
    # reaches through noise to a signal that belongs to another window.
    if float(np.sum(gain[start : end + 1][support[lo + start : lo + end + 1]])) <= 0.0:
        return mask, 0.0
    mask[lo + start : lo + end + 1] = True
    return mask, total


def signal_mask(
    fit,
    candidates: CandidateLike,
    *,
    penalty: float = 2.0,
) -> SignalMask:
    """Mark the anomaly segment of ``fit`` for each candidate window.

    ``fit`` is any single-lens fit result (``time``, ``ferr`` and
    ``residual = flux - model_flux`` are used). ``penalty`` is the chi-square
    cost of excluding one sample (2 = AIC). Each candidate is processed on its
    own; the union is ``SignalMask.mask``.
    """
    if not penalty > 0.0:
        raise ValueError("penalty must be positive")
    time = np.asarray(fit.time, dtype=float).reshape(-1)
    residual = np.asarray(fit.residual, dtype=float).reshape(-1)
    ferr = np.asarray(fit.ferr, dtype=float).reshape(-1)
    z = residual / np.maximum(ferr, 1.0e-12)
    values = as_candidates(candidates)
    segments = [
        _best_segment(
            time,
            z,
            candidate.support(time),
            penalty=penalty,
            max_gap=candidate.t_end - candidate.t_start,
        )
        for candidate in values
    ]
    per_candidate = tuple(mask for mask, _ in segments)
    union = np.zeros(time.shape, dtype=bool)
    for mask in per_candidate:
        union |= mask
    return SignalMask(
        mask=union,
        per_candidate=per_candidate,
        candidates=values,
        gains=tuple(gain for _, gain in segments),
        penalty=float(penalty),
    )


__all__ = ["SignalMask", "signal_mask"]
