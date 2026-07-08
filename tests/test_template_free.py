import numpy as np

from jacscanomaly.template_free import TemplateFreeScanner, TemplateFreeSearchConfig


def _run(z, *, time=None, **config):
    z = np.asarray(z, dtype=float)
    if time is None:
        time = np.arange(z.size, dtype=float)
    ferr = np.ones_like(z)
    cfg = {
        "seed_z_threshold": 3.0,
        "candidate_chi2_threshold": 10.0,
        "zero_crossings_each_side": 1,
    }
    cfg.update(config)
    return TemplateFreeScanner(
        TemplateFreeSearchConfig(**cfg)
    ).run(time, z, ferr)


def test_zero_crossing_scan_splits_at_exact_zeros():
    result = _run([0, 6, 5, 0, -5, -5, 0, 5, 5, 0])

    spans = [(cand.start_index, cand.end_index) for cand in result.candidates]
    assert spans == [(0, 3), (4, 6), (7, 9)]
    assert {cand.kind for cand in result.candidates} == {"zero_crossing"}


def test_zero_crossing_scan_splits_direct_sign_changes():
    result = _run([0.1, 6, 5, -0.1, -5, -5, 0.1, 5, 5, 0.1])

    spans = [(cand.start_index, cand.end_index) for cand in result.candidates]
    assert spans == [(0, 2), (6, 9), (3, 5)]


def test_zero_crossing_scan_respects_season_boundaries():
    time = np.array([0, 1, 2, 300, 301, 302], dtype=float)
    result = _run([6, 6, 6, -6, -6, -6], time=time)

    assert [(cand.start_index, cand.end_index) for cand in result.candidates] == [
        (0, 2),
        (3, 5),
    ]
    assert [cand.season_idx for cand in result.candidates] == [0, 1]


def test_duplicate_seed_windows_keep_strongest_seed():
    result = _run([0, 7, 6, 0])

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert (candidate.start_index, candidate.end_index) == (0, 3)
    assert candidate.seed_start_index == 1


def test_legacy_result_buckets_remain_empty():
    result = _run(
        [0, 6, 5, 0],
        fixed_window_points=2,
        fixed_chi2_threshold=1.0,
        hybrid_seed_chi2_threshold=1.0,
        run_blind_reduced_chi2=True,
    )

    assert result.candidates
    assert result.fixed_window_candidates == ()
    assert result.hybrid_candidates == ()
    assert result.blind_reduced_candidates == ()


def test_sigma_clipped_renormalization_is_per_season():
    time = np.array([0, 1, 2, 300, 301, 302], dtype=float)
    result = _run(
        [10, 11, 12, 20, 21, 22],
        time=time,
        renormalize_z=True,
        candidate_chi2_threshold=1000.0,
    )

    assert np.allclose(result.z[:3], [-1.22474487, 0.0, 1.22474487])
    assert np.allclose(result.z[3:], [-1.22474487, 0.0, 1.22474487])
