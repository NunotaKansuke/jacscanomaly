"""Scan-independent anomaly candidates.

A :class:`Candidate` only says *where* to look: a time window and, optionally,
an anchor time.  The post-fit components (:func:`~jacscanomaly.signal_mask`,
:func:`~jacscanomaly.measure_features`) act on candidates and never on a scan
result, so a candidate written by hand -- including one the scan rejected --
drives them exactly like one converted from :meth:`Finder.run`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union

import numpy as np


@dataclass(frozen=True)
class Candidate:
    """Time window ``t_start <= t <= t_end`` that may contain an anomaly.

    ``t0`` is an optional anchor hint (e.g. the scan's best-fit center); it
    defaults to the window center.  ``label`` is free text carried through to
    downstream results.
    """

    t_start: float
    t_end: float
    t0: Optional[float] = None
    label: Optional[str] = None

    def __post_init__(self) -> None:
        start, end = float(self.t_start), float(self.t_end)
        if not (np.isfinite(start) and np.isfinite(end)):
            raise ValueError("Candidate window must be finite.")
        if end < start:
            raise ValueError("Candidate requires t_start <= t_end.")
        object.__setattr__(self, "t_start", start)
        object.__setattr__(self, "t_end", end)
        if self.t0 is not None:
            object.__setattr__(self, "t0", float(self.t0))

    @property
    def anchor(self) -> float:
        return 0.5 * (self.t_start + self.t_end) if self.t0 is None else float(self.t0)

    @classmethod
    def from_t0_teff(
        cls,
        t0: float,
        teff: float,
        *,
        teff_coeff: float = 3.0,
        label: Optional[str] = None,
    ) -> "Candidate":
        """Window ``|t - t0| <= teff_coeff * |teff|`` (the scan's support rule)."""
        half_width = max(float(teff_coeff), 0.0) * abs(float(teff))
        return cls(float(t0) - half_width, float(t0) + half_width, t0=float(t0), label=label)

    @classmethod
    def from_scan(
        cls,
        candidate,
        *,
        teff_coeff: float = 3.0,
        label: Optional[str] = None,
    ) -> "Candidate":
        """Copy ``t0``/``teff`` from a scan candidate (e.g. ``BestCandidate``).

        Only the numbers are copied; whether the scan accepted the candidate
        plays no role.
        """
        return cls.from_t0_teff(
            float(candidate.t0), float(candidate.teff), teff_coeff=teff_coeff, label=label
        )

    def support(self, time) -> np.ndarray:
        """Boolean mask of the samples inside the window."""
        t = np.asarray(time, dtype=float).reshape(-1)
        return (t >= self.t_start) & (t <= self.t_end)

    def contains(self, value: float) -> bool:
        return self.t_start <= float(value) <= self.t_end


CandidateLike = Union[Candidate, Sequence[Candidate]]


def as_candidates(candidates: CandidateLike) -> tuple[Candidate, ...]:
    """Normalize one candidate or a sequence of candidates to a tuple."""
    if isinstance(candidates, Candidate):
        return (candidates,)
    values = tuple(candidates)
    if not all(isinstance(value, Candidate) for value in values):
        raise TypeError("candidates must be Candidate objects.")
    return values


def _merged(windows: CandidateLike) -> list[tuple[float, float]]:
    spans = sorted((c.t_start, c.t_end) for c in as_candidates(windows))
    merged: list[list[float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def agreed_windows(*window_sets: CandidateLike) -> tuple[tuple[Candidate, ...], ...]:
    """Each set's windows, grown to the union of every window they overlap.

    With one set of masked segments per model: where segments of several
    models overlap, the models see the same anomaly and differ only in its
    extent, and each gets the union. A window no other set overlaps stays
    as it is (what is anomalous depends on the model).
    """
    spans = _merged([w for windows in window_sets for w in as_candidates(windows)])
    out = []
    for windows in window_sets:
        own = _merged(windows)
        out.append(tuple(
            Candidate(a, b) for a, b in spans
            if any(max(a, c) <= min(b, d) for c, d in own)
        ))
    return tuple(out)


def candidates_from_scan(
    result,
    *,
    which: Literal["best", "all"] = "best",
    teff_coeff: Optional[float] = None,
) -> list[Candidate]:
    """Convert a :class:`~jacscanomaly.models.AnomalyResult` to candidates.

    ``which="best"`` returns the selected best candidate (empty when there is
    none); ``"all"`` returns every scored cluster, accepted by the criteria or
    not.  ``teff_coeff`` defaults to 3.0, the default scan support width.
    """
    coeff = 3.0 if teff_coeff is None else float(teff_coeff)
    if which == "best":
        rows = [] if result.best is None else [result.best]
    elif which == "all":
        rows = list(result.scored_candidates)
    else:
        raise ValueError("which must be 'best' or 'all'.")
    return [Candidate.from_scan(row, teff_coeff=coeff) for row in rows]


__all__ = ["Candidate", "CandidateLike", "as_candidates", "candidates_from_scan"]
