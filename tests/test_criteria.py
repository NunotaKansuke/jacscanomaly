from jacscanomaly.criteria import CandidateCriteria
from jacscanomaly.models import CandidateQuality


def quality(**overrides):
    values = {
        "n_window": 10,
        "n_contrib": 5,
        "n_eff": 4.0,
        "peak_frac": 0.3,
        "rho1": 0.1,
        "longest_run": 3,
    }
    values.update(overrides)
    return CandidateQuality(**values)


def test_candidate_criteria_accepts_when_all_thresholds_pass():
    criteria = CandidateCriteria(
        min_dchi2=20.0,
        min_n_eff=3.0,
        min_n_contrib=4,
        min_n_window=8,
        min_longest_run=2,
        max_peak_frac=0.5,
    )

    assert criteria.accepts(dchi2=25.0, quality=quality())


def test_candidate_criteria_rejects_each_failed_threshold():
    assert not CandidateCriteria(min_dchi2=20.0).accepts(dchi2=19.0, quality=quality())
    assert not CandidateCriteria(min_n_eff=3.0).accepts(dchi2=25.0, quality=quality(n_eff=2.9))
    assert not CandidateCriteria(min_n_contrib=4).accepts(dchi2=25.0, quality=quality(n_contrib=3))
    assert not CandidateCriteria(min_n_window=8).accepts(dchi2=25.0, quality=quality(n_window=7))
    assert not CandidateCriteria(min_longest_run=2).accepts(dchi2=25.0, quality=quality(longest_run=1))
    assert not CandidateCriteria(max_peak_frac=0.5).accepts(dchi2=25.0, quality=quality(peak_frac=0.6))


def test_candidate_criteria_ignores_none_thresholds():
    assert CandidateCriteria().accepts(dchi2=-1.0, quality=quality(n_eff=0.0, peak_frac=1.0))
