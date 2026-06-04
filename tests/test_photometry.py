import numpy as np

from jacscanomaly.photometry import linear_fit, solve_fs_fb


def test_linear_fit_recovers_weighted_line_parameters():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x + 1.5
    w = np.array([1.0, 2.0, 1.0, 3.0])

    a, b = linear_fit(x, y, w)

    np.testing.assert_allclose(np.asarray(a), 2.0, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(b), 1.5, rtol=1e-12)


def test_solve_fs_fb_recovers_flux_scale_and_blend():
    amp = np.array([1.0, 1.5, 2.0, 2.5])
    flux = 3.0 * amp + 0.7
    ferr = np.array([0.1, 0.2, 0.1, 0.3])

    fs, fb = solve_fs_fb(amp, flux, ferr)

    np.testing.assert_allclose(np.asarray(fs), 3.0, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(fb), 0.7, rtol=1e-12)
