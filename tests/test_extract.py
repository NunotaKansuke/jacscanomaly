import numpy as np
import pytest

from jacscanomaly.extract import ResultExtractor


def test_iterative_anomaly_extraction_returns_best_representatives():
    extractor = ResultExtractor(sigma_overlap=1.0, min_points=1)
    t0 = np.array([0.0, 0.2, 10.0, 10.1])
    teff = np.ones_like(t0)
    dchi2 = np.array([5.0, 9.0, 4.0, 7.0])

    clusters = extractor.iterative_anomaly_extraction(t0, teff, dchi2)

    np.testing.assert_allclose(clusters, [[0.2, 1.0, 9.0], [10.1, 1.0, 7.0]])


def test_iterative_anomaly_extraction_empty_input_has_stable_shape():
    clusters = ResultExtractor().iterative_anomaly_extraction([], [], [])

    assert clusters.shape == (0, 3)


def test_iterative_anomaly_extraction_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        ResultExtractor().iterative_anomaly_extraction([1.0], [1.0, 2.0], [3.0])


def test_iterative_anomaly_extraction_stops_on_non_finite_best():
    clusters = ResultExtractor(min_points=1).iterative_anomaly_extraction(
        [1.0, 2.0],
        [1.0, 1.0],
        [np.nan, -np.inf],
    )

    assert clusters.shape == (0,)
