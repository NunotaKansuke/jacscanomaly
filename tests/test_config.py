import inspect

from jacscanomaly import CandidateCriteria, FinderConfig


def test_default_config_uses_cpp_backends_for_pspl_workflow():
    config = FinderConfig()

    assert config.fitter_kind == "pspl"
    assert config.grid_backend == "cpp"
    assert config.single_fit_backend == "cpp"
    assert config.common_ratio == 4.0 / 3.0


def test_common_ratio_default_has_readable_repr():
    config = FinderConfig()
    signature = inspect.signature(FinderConfig)

    assert "common_ratio=4.0 / 3.0" in repr(config)
    assert str(signature.parameters["common_ratio"]) == "common_ratio: 'float' = 4.0 / 3.0"


def test_config_accepts_candidate_criteria():
    criteria = CandidateCriteria(min_dchi2=20.0, min_n_eff=2.0)
    config = FinderConfig(candidate_criteria=criteria)

    assert config.candidate_criteria is criteria
