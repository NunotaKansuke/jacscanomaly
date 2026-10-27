from types import SimpleNamespace

import numpy as np
import pytest

from jacscanomaly import Candidate, signal_mask


def _fit(time, residual):
    return SimpleNamespace(
        time=time, residual=residual, ferr=np.ones(time.shape, dtype=float)
    )


def test_segment_spans_a_sign_change_inside_one_anomaly():
    time = np.arange(60, dtype=float)
    z = np.zeros(time.shape)
    z[20:25] = 3.0
    z[22] = -0.5  # crossing zero inside the signal costs less than it gains

    result = signal_mask(_fit(time, z), Candidate(21.0, 22.0))

    assert np.flatnonzero(result.mask).tolist() == [20, 21, 22, 23, 24]
    assert result.gains[0] == pytest.approx(4 * (9.0 - 2.0) + (0.25 - 2.0))


def test_segment_may_extend_beyond_the_window_but_must_overlap_it():
    time = np.arange(60, dtype=float)
    z = np.zeros(time.shape)
    z[20:30] = 4.0

    touching = signal_mask(_fit(time, z), Candidate(28.0, 35.0))
    elsewhere = signal_mask(_fit(time, z), Candidate(45.0, 50.0))

    assert np.flatnonzero(touching.mask).tolist() == list(range(20, 30))
    assert elsewhere.n_masked == 0 and elsewhere.gains[0] <= 0.0


def test_penalty_sets_the_per_point_break_even():
    time = np.arange(40, dtype=float)
    z = np.zeros(time.shape)
    z[10:20] = 1.5  # z^2 = 2.25

    assert signal_mask(_fit(time, z), Candidate(12.0, 16.0), penalty=2.0).n_masked == 10
    assert signal_mask(_fit(time, z), Candidate(12.0, 16.0), penalty=3.0).n_masked == 0


def test_segment_does_not_cross_a_gap_longer_than_the_window():
    time = np.r_[np.arange(30), np.arange(60, 90)].astype(float)
    z = np.zeros(time.shape)
    z[25:35] = 4.0  # signal on both sides of the gap

    result = signal_mask(_fit(time, z), Candidate(26.0, 29.0))

    assert np.flatnonzero(result.mask).tolist() == list(range(25, 30))


def test_signal_mask_is_the_union_of_independent_candidates():
    time = np.arange(200, dtype=float)
    z = np.zeros(time.shape)
    z[30:40] = 4.0
    z[130:140] = -4.0
    fit = _fit(time, z)
    first, second = Candidate(32.0, 36.0), Candidate(132.0, 136.0)

    both = signal_mask(fit, [first, second])

    assert both.candidates == (first, second)
    assert np.array_equal(both.per_candidate[0], signal_mask(fit, first).mask)
    assert np.array_equal(both.mask, both.per_candidate[0] | both.per_candidate[1])
    assert np.flatnonzero(both.mask).tolist() == list(range(30, 40)) + list(range(130, 140))
