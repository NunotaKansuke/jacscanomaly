import numpy as np

from jacscanomaly.models import AnomalyResult, BestCandidate, CandidateQuality, SeasonSummary


def make_result(best=None):
    season = SeasonSummary(
        season_idx=0,
        t_start=0.0,
        t_end=10.0,
        n_grid=7,
        clusters=np.array([[5.0, 1.0, 10.0]]),
        grid_metrics=np.zeros((2, 9)),
    )
    return AnomalyResult(
        time=np.array([0.0, 1.0]),
        flux=np.array([1.0, 2.0]),
        ferr=np.array([0.1, 0.2]),
        fit=None,
        residual=np.array([0.0, 0.1]),
        model_flux=np.array([1.0, 1.9]),
        chi2_dof=1.25,
        seasons=[season],
        clusters_all=np.array([[5.0, 1.0, 10.0]]),
        grid_metrics_all=np.zeros((2, 9)),
        best=best,
    )


def test_summary_dict_without_best_candidate():
    summary = make_result().summary_dict()

    assert summary["n_points"] == 2
    assert summary["n_seasons"] == 1
    assert summary["n_clusters"] == 1
    assert summary["n_grid_total"] == 7
    assert summary["chi2_dof"] == 1.25
    assert summary["has_best"] is False
    assert "best_t0" not in summary


def test_summary_dict_with_best_candidate_includes_quality():
    best = BestCandidate(
        t0=5.0,
        teff=1.5,
        dchi2=42.0,
        med_others=3.0,
        std_others=2.0,
        score=19.5,
        quality=CandidateQuality(
            n_window=8,
            n_contrib=4,
            n_eff=3.5,
            peak_frac=0.4,
            rho1=0.2,
            longest_run=3,
        ),
    )

    summary = make_result(best=best).summary_dict()

    assert summary["has_best"] is True
    assert summary["best_t0"] == 5.0
    assert summary["best_teff"] == 1.5
    assert summary["best_dchi2"] == 42.0
    assert summary["best_score"] == 19.5
    assert summary["best_n_eff"] == 3.5
    assert summary["best_longest_run"] == 3


def test_summary_text_and_str_are_cli_friendly():
    result = make_result()

    text = result.summary_text()

    assert "jacscanomaly summary" in text
    assert "best        : None" in text
    assert str(result) == text


def test_summary_table_falls_back_or_returns_dataframe():
    table = make_result().summary_table()

    if isinstance(table, list):
        assert table[0]["n_points"] == 2
    else:
        assert table.loc[0, "n_points"] == 2
