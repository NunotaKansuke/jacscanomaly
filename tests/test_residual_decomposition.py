import numpy as np

from jacscanomaly import Finder, FinderConfig
from jacscanomaly.planet_signal import PlanetSignalConfig, PlanetSignalExtractor
from jacscanomaly.residual_decomposition import decompose_binned_residual


def test_binned_decomposition_preserves_broad_smooth_residual():
    time = np.linspace(-5.0, 5.0, 1001)
    smooth_z = 12.0 * np.exp(-0.5 * (time / 1.5) ** 2)

    result = decompose_binned_residual(
        time,
        smooth_z,
        characteristic_scale=0.2,
    )

    assert np.max(np.abs(result.local_z)) < 1.0
    assert np.sum(result.local_z**2) < 0.01 * np.sum(smooth_z**2)
    np.testing.assert_allclose(
        result.binned_z + result.local_z,
        smooth_z,
        atol=1e-12,
    )


def test_binned_decomposition_recovers_caustics_on_smooth_mismatch():
    time = np.linspace(-5.0, 5.0, 1001)
    smooth_z = 9.0 * np.exp(-0.5 * (time / 1.8) ** 2)
    caustics = (
        18.0 * np.exp(-0.5 * ((time + 0.7) / 0.04) ** 2)
        - 15.0 * np.exp(-0.5 * ((time - 0.8) / 0.05) ** 2)
    )

    result = decompose_binned_residual(
        time,
        smooth_z + caustics,
        characteristic_scale=0.2,
    )

    quiet = (np.abs(time + 0.7) > 0.2) & (np.abs(time - 0.8) > 0.2)
    assert np.median(np.abs(result.local_z[quiet])) < 0.2
    assert np.max(result.local_z[np.abs(time + 0.7) < 0.1]) > 8.0
    assert np.min(result.local_z[np.abs(time - 0.8) < 0.1]) < -5.0


def test_mask_core_keeps_separate_caustics_but_not_smooth_envelope():
    time = np.linspace(-5.0, 5.0, 1001)
    smooth_z = 10.0 * np.exp(-0.5 * (time / 1.5) ** 2)
    caustics = (
        16.0 * np.exp(-0.5 * ((time + 0.7) / 0.04) ** 2)
        + 14.0 * np.exp(-0.5 * ((time - 0.8) / 0.05) ** 2)
    )
    z = smooth_z + caustics
    decomposition = decompose_binned_residual(
        time,
        z,
        characteristic_scale=0.2,
    )
    extractor = PlanetSignalExtractor(
        Finder(FinderConfig()),
        PlanetSignalConfig(mask_local_group_peak_frac=0.3),
    )

    mask = extractor._coherent_residual_core_mask(
        time=time,
        abs_z=np.abs(z),
        structure_abs_z=np.abs(decomposition.local_z),
        window=np.ones(time.shape, dtype=bool),
        pad=decomposition.bin_width,
    )

    assert np.any(mask & (np.abs(time + 0.7) < 0.1))
    assert np.any(mask & (np.abs(time - 0.8) < 0.1))
    assert not np.any(mask & (np.abs(time) < 0.3))
    assert np.mean(mask) < 0.1


def test_physical_diagnostic_support_cannot_enter_planet_mask():
    time = np.linspace(-2.0, 2.0, 401)
    z = 12.0 * np.exp(-0.5 * (time / 0.08) ** 2)
    protection = np.abs(time) < 0.15
    extractor = PlanetSignalExtractor(Finder(FinderConfig()))
    extractor._mask_protection = protection

    mask = extractor._coherent_residual_core_mask(
        time=time,
        abs_z=np.abs(z),
        structure_abs_z=np.abs(z),
        window=np.ones(time.shape, dtype=bool),
        pad=0.05,
    )

    assert not np.any(mask & protection)


def test_sign_coherent_halo_recovers_caustic_wings_without_crossing_sign():
    time = np.linspace(-0.2, 0.2, 81)
    z = 40.0 * np.exp(-0.5 * (time / 0.045) ** 2)
    z[time > 0.11] = -12.0
    core = np.abs(time) <= 0.015
    extractor = PlanetSignalExtractor(
        Finder(FinderConfig()),
        PlanetSignalConfig(
            mask_halo_min_abs_z=5.0,
            mask_halo_peak_frac=0.1,
            mask_halo_max_gap_points=1,
        ),
    )
    extractor._time_cadence = float(np.median(np.diff(time)))

    grown = extractor._grow_sign_coherent_halo(
        time=time,
        z=z,
        core=core,
        window=np.ones(time.size, dtype=bool),
        max_distance=0.15,
    )

    assert np.any(grown & (np.abs(time) > 0.05))
    assert not np.any(grown & (time > 0.11))
