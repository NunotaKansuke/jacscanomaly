from __future__ import annotations

from dataclasses import dataclass

from .models import CandidateQuality


@dataclass(frozen=True)
class CandidateCriteria:
    """
    Quality criteria used to accept/reject anomaly candidates.

    All thresholds are optional. Any criterion set to ``None`` is ignored.
    """

    min_dchi2: float | None = None
    min_n_eff: float | None = None
    min_n_contrib: int | None = None
    min_n_window: int | None = None
    min_longest_run: int | None = None
    max_peak_frac: float | None = None

    def accepts(self, *, dchi2: float, quality: CandidateQuality) -> bool:
        if self.min_dchi2 is not None and dchi2 < float(self.min_dchi2):
            return False
        if self.min_n_eff is not None and quality.n_eff < float(self.min_n_eff):
            return False
        if self.min_n_contrib is not None and quality.n_contrib < int(self.min_n_contrib):
            return False
        if self.min_n_window is not None and quality.n_window < int(self.min_n_window):
            return False
        if self.min_longest_run is not None and quality.longest_run < int(self.min_longest_run):
            return False
        if self.max_peak_frac is not None and quality.peak_frac > float(self.max_peak_frac):
            return False
        return True
