import numpy as np

from jacscanomaly import CandidateCriteria, Finder, FinderConfig
from jacscanomaly.models import SeasonSummary


def _metrics(clusters: np.ndarray) -> np.ndarray:
    rows = []
    for t0, teff, dchi2 in clusters:
        rows.append([t0, teff, dchi2, 20, 5, 4.0, 0.25, 0.1, 4])
    return np.asarray(rows, dtype=float)


def _season(start: float, end: float, clusters: np.ndarray) -> SeasonSummary:
    return SeasonSummary(
        season_idx=0,
        t_start=start,
        t_end=end,
        n_grid=len(clusters),
        clusters=clusters,
        grid_metrics=_metrics(clusters),
    )


def test_quality_criteria_select_after_raw_background_is_built():
    clusters = np.asarray(
        [
            [10.0, 1.0, 100.0],
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 2.0],
            [3.0, 1.0, 3.0],
            [4.0, 1.0, 4.0],
            [5.0, 1.0, 5.0],
        ]
    )
    metrics = _metrics(clusters)
    metrics[1:, 5] = 1.0
    finder = Finder(
        FinderConfig(
            candidate_criteria=CandidateCriteria(min_n_eff=2.0),
            best_score_min_reference_clusters=2,
        )
    )

    best = finder._pick_best_candidate(
        clusters,
        metrics,
        seasons=[_season(0.0, 20.0, clusters)],
    )

    assert best is not None
    assert best.dchi2 == 100.0
    assert best.n_score_reference == 5
    assert np.isfinite(best.score)


def test_score_uses_only_same_season_and_nearby_timescales():
    local = np.asarray(
        [
            [10.0, 1.0, 100.0],
            [1.0, 0.75, 1.0],
            [2.0, 1.0, 2.0],
            [3.0, 1.5, 3.0],
            [4.0, 8.0, 90.0],
        ]
    )
    other_season = np.asarray(
        [
            [110.0, 1.0, 80.0],
            [111.0, 1.0, 85.0],
        ]
    )
    clusters = np.concatenate([local, other_season])
    seasons = [
        _season(0.0, 20.0, local),
        _season(100.0, 120.0, other_season),
    ]
    finder = Finder(
        FinderConfig(
            best_score_teff_ratio=2.0,
            best_score_min_reference_clusters=3,
        )
    )

    best = finder._pick_best_candidate(clusters, _metrics(clusters), seasons=seasons)

    assert best is not None
    assert best.n_score_reference == 3
    assert best.med_others == 2.0


def test_score_adaptively_clips_strong_secondary_cluster():
    background = np.arange(1.0, 9.0)
    clusters = np.asarray(
        [[10.0, 1.0, 100.0]]
        + [[float(i), 1.0, value] for i, value in enumerate(background, start=1)]
        + [[9.0, 1.0, 40.0]]
    )
    finder = Finder(
        FinderConfig(
            best_score_min_reference_clusters=8,
            best_score_upper_clip_sigma=5.0,
        )
    )

    best = finder._pick_best_candidate(
        clusters,
        _metrics(clusters),
        seasons=[_season(0.0, 20.0, clusters)],
    )

    assert best is not None
    assert best.n_score_reference == 8
    assert best.med_others == 4.5
    expected_scale = 1.482602218505602 * 2.0
    assert np.isclose(best.std_others, expected_scale)
    assert np.isclose(best.score, (100.0 - 4.5) / expected_scale)
