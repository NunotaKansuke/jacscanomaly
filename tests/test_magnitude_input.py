import numpy as np
import pytest

from jacscanomaly import CandidateCriteria, Finder, FinderConfig
from jacscanomaly.singlelens_model import A_pspl_func


def _finder():
    return Finder(
        FinderConfig(
            grid_backend="jax",
            teff_init=1.0,
            teff_grid_n=1,
            candidate_criteria=CandidateCriteria(min_dchi2=1.0e9),
        )
    )


def test_magnitude_input_is_converted_to_relative_flux_for_run():
    time = np.linspace(0.0, 20.0, 60)
    x0 = np.array([10.0, 5.0, 0.2])
    raw_flux = 2.0 * np.asarray(A_pspl_func(x0, time)) + 0.3
    raw_ferr = np.full_like(time, 0.05)
    mag_zero_point = 25.0
    mag = mag_zero_point - 2.5 * np.log10(raw_flux)
    magerr = raw_ferr / ((np.log(10.0) / 2.5) * raw_flux)

    result = _finder().run(
        time,
        mag,
        magerr,
        x0=x0,
        data_kind="mag",
        refit=False,
        verbose=False,
    )

    scale = 10.0 ** ((mag_zero_point - np.median(mag)) / 2.5)
    np.testing.assert_allclose(result.flux, raw_flux / scale, rtol=1e-6)
    np.testing.assert_allclose(result.ferr, raw_ferr / scale, rtol=1e-6)
    np.testing.assert_allclose(result.model_flux, raw_flux / scale, rtol=1e-6)
    np.testing.assert_allclose(result.residual, 0.0, atol=1e-6)


def test_flux_is_the_default_input_representation():
    time = np.array([1.0, 2.0])
    flux = np.array([2.0, 3.0])
    ferr = np.array([0.1, 0.2])

    _, _, _, _, _, converted_flux, converted_ferr = _finder()._to_arrays(time, flux, ferr, None)

    np.testing.assert_array_equal(converted_flux, flux)
    np.testing.assert_array_equal(converted_ferr, ferr)


def test_magnitude_conversion_is_stable_for_large_zero_points():
    _, flux_j, ferr_j, _, _, flux, ferr = _finder()._to_arrays(
        [1.0, 2.0, 3.0],
        [1000.0, 1000.5, 1001.0],
        [0.1, 0.1, 0.1],
        None,
        data_kind="mag",
    )

    assert np.all(np.isfinite(flux))
    assert np.all(np.isfinite(ferr))
    assert np.all(flux > 0)
    assert np.all(ferr > 0)
    assert np.all(np.isfinite(np.asarray(flux_j)))
    assert np.all(np.isfinite(np.asarray(ferr_j)))


def test_magnitude_input_rejects_unknown_data_kind():
    with pytest.raises(ValueError, match="data_kind"):
        _finder()._to_arrays([1.0], [20.0], [0.1], None, data_kind="counts")
