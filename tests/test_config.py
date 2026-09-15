import inspect

from jacscanomaly import CandidateCriteria, FinderConfig


def test_default_config_uses_canonical_fitter_contract():
    config = FinderConfig()

    assert config.fitter_kind == "pspl"
    assert config.grid_backend == "cpp"
    assert config.fitter_maxiter == 1000
    assert config.fitter_tol == 1.0e-6
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


def test_config_exposes_geometry_and_magnification_options():
    config = FinderConfig(
        fitter_kind="fspl_parallax",
        parallax_geometry="space",
        parallax_observer_convention="gulls",
        magnification_tol=1.0e-5,
        magnification_reltol=2.0e-5,
    )

    assert config.parallax_geometry == "space"
    assert config.parallax_observer_convention == "gulls"
    assert config.magnification_tol == 1.0e-5
    assert config.magnification_reltol == 2.0e-5
