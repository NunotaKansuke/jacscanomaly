import numpy as np
import pytest

from jacscanomaly import Finder, FinderConfig
from jacscanomaly.extract import ResultExtractor
from jacscanomaly.runner import _cpp_grid


@pytest.mark.parametrize("seed", range(6))
def test_indexed_extraction_preserves_greedy_selection(seed):
    rng = np.random.default_rng(seed)
    t0 = np.tile(np.arange(96.0), 4)
    teff = np.repeat([0.01, 0.03, 0.1, 0.5], 96)
    dchi2 = rng.integers(-2, 20, t0.size).astype(float)
    order = rng.permutation(t0.size)
    t0, teff, dchi2 = t0[order], teff[order], dchi2[order]
    reference = ResultExtractor(sigma_overlap=3.0, min_points=3).iterative_anomaly_extraction(
        t0, teff, dchi2
    )
    actual = _cpp_grid.extract_clusters(t0, teff, dchi2, sigma_overlap=3.0, min_points=3)
    np.testing.assert_array_equal(actual, reference)


def test_indexed_extraction_strict_floating_point_boundaries():
    # Put points exactly at, immediately inside, and immediately outside the
    # strict overlap boundary. Repetitions exercise tie order and active sets.
    radius = 3.0 * (0.01 + 0.01)
    boundary = np.array([np.nextafter(radius, 0.0), radius, np.nextafter(radius, np.inf)])
    t0 = np.tile(np.r_[0.0, -boundary, boundary, np.arange(1.0, 70.0)], 3)
    teff = np.full(t0.shape, 0.01)
    dchi2 = np.arange(t0.size, 0, -1, dtype=float)
    expected = ResultExtractor().iterative_anomaly_extraction(t0, teff, dchi2)
    actual = _cpp_grid.extract_clusters(t0, teff, dchi2)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("seed", range(6))
def test_sorted_leave_one_out_matches_numpy_clipping(seed):
    rng = np.random.default_rng(seed)
    values = np.r_[rng.chisquare(3.0, 180 + seed), [50.0, 100.0, 200.0]]
    if seed == 1:
        values = np.round(values, 1)  # Tied ranks must keep the same median/MAD.
    excluded = np.arange(values.size, dtype=np.int64)
    finder = Finder(FinderConfig())
    for maxiters in (0, 1, 5):
        actual = _cpp_grid.score_backgrounds(values, excluded, clip_sigma=5.0, maxiters=maxiters)
        finder = Finder(FinderConfig(best_score_clip_maxiters=maxiters))
        for index, row in enumerate(actual):
            if not row[3]:
                continue  # Degenerate MAD deliberately uses the scalar path.
            bulk = finder._upper_clip_score_background(np.delete(values, index))
            expected = [np.median(bulk), finder._robust_scale(bulk), bulk.size]
            np.testing.assert_array_equal(row[:3], expected)


def test_sorted_leave_one_out_zero_mad_requests_scalar_fallback():
    values = np.ones(16)
    result = _cpp_grid.score_backgrounds(values, np.arange(16, dtype=np.int64))
    assert np.all(result[:, 3] == 0.0)


def test_grouped_scoring_matches_scalar_for_sparse_and_degenerate_backgrounds(monkeypatch):
    rng = np.random.default_rng(73)
    n = 180
    clusters = np.column_stack((np.arange(n, dtype=float), rng.choice([0.01, 0.03, 0.1, 10.0], n), rng.chisquare(3.0, n)))
    metrics = np.column_stack((clusters, np.tile([20, 5, 4.0, 0.25, 0.1, 4], (n, 1))))
    finder = Finder(FinderConfig(best_score_min_reference_clusters=8))
    actual = finder._score_candidates(clusters, metrics)
    monkeypatch.setattr(finder, "_indexed_score_backgrounds", lambda rows: [None] * len(rows))
    expected = finder._score_candidates(clusters, metrics)
    assert actual == expected


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 16, 17])
def test_leave_one_out_small_odd_even_samples(n):
    values = np.random.default_rng(18 + n).normal(size=n)
    exclusions = np.r_[-1, np.arange(n)].astype(np.int64)
    finder = Finder(FinderConfig(best_score_clip_maxiters=0))
    result = _cpp_grid.score_backgrounds(values, exclusions, maxiters=0)
    for excluded, row in zip(exclusions, result):
        bulk = values if excluded == -1 else np.delete(values, excluded)
        if not row[3]:
            continue
        assert row[2] == bulk.size
        if bulk.size < 2:
            assert np.isnan(row[:2]).all()
        else:
            np.testing.assert_array_equal(row[:2], [np.median(bulk), finder._robust_scale(bulk)])


def test_grouped_scoring_retains_sparse_and_zero_mad_fallbacks(monkeypatch):
    clusters = np.column_stack((np.arange(64.), np.r_[np.full(63, 0.1), 100.], np.r_[np.ones(60), [2., 3., 80., 90.]]))
    metrics = np.column_stack((clusters, np.tile([20, 5, 4.0, 0.25, 0.1, 4], (64, 1))))
    finder = Finder(FinderConfig(best_score_min_reference_clusters=8))
    actual = finder._score_candidates(clusters, metrics)
    monkeypatch.setattr(finder, "_indexed_score_backgrounds", lambda rows: [None] * len(rows))
    expected = finder._score_candidates(clusters, metrics)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(
            [a.t0, a.teff, a.dchi2, a.med_others, a.std_others, a.score, a.n_score_reference],
            [b.t0, b.teff, b.dchi2, b.med_others, b.std_others, b.score, b.n_score_reference],
        )
        assert a.quality == b.quality
