from types import SimpleNamespace

import numpy as np
import pytest

from jacscanomaly import Candidate, candidates_from_scan


def test_from_t0_teff_uses_the_scan_support_rule():
    time = np.linspace(0.0, 20.0, 201)
    candidate = Candidate.from_t0_teff(10.0, -0.5, teff_coeff=3.0)

    assert candidate.t_start == pytest.approx(8.5)
    assert candidate.t_end == pytest.approx(11.5)
    assert candidate.anchor == pytest.approx(10.0)
    assert np.array_equal(candidate.support(time), np.abs(time - 10.0) <= 1.5)


def test_hand_written_candidate_defaults_anchor_to_window_center():
    candidate = Candidate(2.0, 4.0, label="by eye")

    assert candidate.anchor == pytest.approx(3.0)
    assert candidate.contains(2.0) and candidate.contains(4.0)
    assert not candidate.contains(4.01)


@pytest.mark.parametrize("window", [(1.0, 0.0), (np.nan, 1.0), (0.0, np.inf)])
def test_invalid_windows_are_rejected(window):
    with pytest.raises(ValueError):
        Candidate(*window)


def test_candidates_from_scan_ignores_acceptance():
    rejected = SimpleNamespace(t0=5.0, teff=0.2, passes=False)
    accepted = SimpleNamespace(t0=12.0, teff=1.0, passes=True)
    result = SimpleNamespace(best=accepted, scored_candidates=[accepted, rejected])

    best = candidates_from_scan(result)
    every = candidates_from_scan(result, which="all", teff_coeff=2.0)

    assert best == [Candidate.from_t0_teff(12.0, 1.0)]
    assert every[1] == Candidate.from_t0_teff(5.0, 0.2, teff_coeff=2.0)
    assert candidates_from_scan(SimpleNamespace(best=None, scored_candidates=[])) == []

