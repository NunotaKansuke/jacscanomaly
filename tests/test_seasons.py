import numpy as np
import pytest

from jacscanomaly.seasons import Season, SeasonSplitter


def test_split_preserves_original_indices_for_unsorted_time():
    splitter = SeasonSplitter(gap=5.0)
    seasons = splitter.split(np.array([10.0, 1.0, 2.0, 20.0]))

    assert [(s.start, s.end) for s in seasons] == [(1.0, 2.0), (10.0, 10.0), (20.0, 20.0)]
    assert [s.indices.tolist() for s in seasons] == [[1, 2], [0], [3]]


def test_split_empty_input_returns_empty_list():
    assert SeasonSplitter().split([]) == []


def test_split_rejects_non_1d_time():
    with pytest.raises(ValueError, match="time must be 1D"):
        SeasonSplitter().split(np.zeros((2, 2)))


def test_split_rejects_non_finite_time():
    with pytest.raises(ValueError, match="finite"):
        SeasonSplitter().split([1.0, np.nan])


def test_build_t0_grid_uses_half_open_interval():
    season = Season(start=0.0, end=1.0, indices=np.array([0, 1]))

    np.testing.assert_allclose(SeasonSplitter.build_t0_grid(season, 0.4), [0.0, 0.4, 0.8])


def test_build_t0_grid_rejects_non_positive_spacing():
    season = Season(start=0.0, end=1.0, indices=np.array([0, 1]))

    with pytest.raises(ValueError, match="dt0 must be positive"):
        SeasonSplitter.build_t0_grid(season, 0.0)
