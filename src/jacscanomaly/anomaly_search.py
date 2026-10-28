"""Find every anomaly of an event, strongest first.

:func:`search_anomalies` alternates the scan and the masked host refit::

    s = search_anomalies(finder, time, flux, ferr, x0, min_score=10.0)
    s.candidates, s.windows, s.refit

Round ``i``: scan the samples outside the windows found so far, with the
host held at its current parameters (``refit=False``); take the scan's best
candidate; refit the host with the anomaly in every window masked
(:func:`~jacscanomaly.masked_refit`: the window is where to look, the mask
is the part of it that pays its price). Stop when the best candidate is
missing or scores below ``min_score``.

Why rescan instead of taking every candidate of one scan: a strong anomaly
pulls the host fit, and the host's misfit then shows up as extra residual
structure (or hides a weak anomaly). Each rescan sees the residual of a
host fitted without the anomalies found so far.

This is a layer composed of ``Finder.run``, :class:`~jacscanomaly.Candidate`
and :func:`~jacscanomaly.masked_refit`; it does not change them.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .candidate import Candidate
from .masked_refit import MaskedRefit, masked_refit, window_mask


@dataclass(frozen=True)
class AnomalySearch:
    """Outcome of :func:`search_anomalies`.

    ``candidates`` are the scans' best candidates in the order found, and
    ``windows`` their ``t0 +- teff_coeff * teff`` windows. ``refit`` is the
    host refitted with all windows masked (the host as is when nothing was
    found), so its ``criterion`` compares searches of different models.
    ``scans`` holds every scan; the last one found nothing more.
    """

    candidates: tuple
    windows: tuple[Candidate, ...]
    refit: MaskedRefit
    scans: tuple


def search_anomalies(
    finder,
    time,
    flux,
    ferr,
    x0,
    *,
    min_score: float = -np.inf,
    max_rounds: int = 10,
    scan=None,
    **run_kwargs,
) -> AnomalySearch:
    """Scan, mask the best candidate, refit the host, and rescan.

    ``x0`` is the host parameters for the first scan (held fixed).
    ``scan`` is that first scan when already run (``finder.run`` at ``x0``
    with ``refit=False``); it is then not repeated.
    ``run_kwargs`` go to every ``finder.run`` call (e.g. ``data_kind``).
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    ferr = np.asarray(ferr, dtype=float)
    coeff = float(getattr(finder.config, "teff_coeff", 3.0))
    run_kwargs.setdefault("verbose", False)

    if scan is None:
        scan = finder.run(time, flux, ferr, x0=np.asarray(x0, dtype=float), refit=False, **run_kwargs)
    host = scan.fit
    scans, candidates, windows = [scan], [], []
    refit = masked_refit(finder.fitter, host, windows)
    for _ in range(max_rounds):
        best = scan.best
        if best is None or not float(best.score) >= min_score:
            break
        candidates.append(best)
        windows.append(Candidate.from_scan(best))
        refit = masked_refit(finder.fitter, host, windows)
        keep = ~window_mask(time, windows)  # searched, masked or not
        scan = finder.run(time[keep], flux[keep], ferr[keep],
                          x0=finder.fitter.seed_from_fit(refit.fit), refit=False, **run_kwargs)
        scans.append(scan)
    return AnomalySearch(tuple(candidates), tuple(windows), refit, tuple(scans))


__all__ = ["AnomalySearch", "search_anomalies"]
