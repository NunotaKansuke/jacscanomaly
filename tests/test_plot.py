import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from jacscanomaly import Event
from test_pipeline_parts import CONFIG
from test_robust import _event


def _scanned():
    ev = Event(*_event([(-8.0, 0.3)], n=3000), config=CONFIG)
    ev.fit("pspl")
    assert ev.scan().detected
    return ev


def test_the_anomaly_window_is_the_candidate_window():
    ev = _scanned()
    best = ev._display("pspl").best

    _, ax = ev.plot_anomaly_window("pspl", show=False, a=5.0, teff_coeff=3.0)

    assert np.allclose(ax.get_xlim(), (best.t0 - 5.0 * best.teff, best.t0 + 5.0 * best.teff))
    for line in ax.get_lines():
        if line.get_label() in ("flat", "anomaly"):
            x = line.get_xdata()
            assert x.min() >= best.t0 - 3.0 * best.teff - 1e-9
            assert x.max() <= best.t0 + 3.0 * best.teff + 1e-3
    plt.close("all")


def test_anomaly_width_mode_uses_the_candidate_teff():
    ev = _scanned()
    result = ev._display("pspl")
    best = result.best

    xlim = ev._plotter()._compute_xlim(result, center="anomaly", width_mode="anomaly", a=4.0)

    assert np.allclose(xlim, (best.t0 - 4.0 * best.teff, best.t0 + 4.0 * best.teff))


def test_plot_result_marks_the_candidate_window():
    ev = _scanned()
    best = ev._display("pspl").best

    fig, _ = ev.plot_result("pspl", show=False, show_anomaly_window=True, teff_coeff=3.0)

    spans = [p for ax in fig.axes for p in ax.patches]
    assert spans
    for span in spans:
        x0, x1 = span.get_x(), span.get_x() + span.get_width()
        assert np.isclose(x0, best.t0 - 3.0 * best.teff) and np.isclose(x1, best.t0 + 3.0 * best.teff)
    plt.close("all")
