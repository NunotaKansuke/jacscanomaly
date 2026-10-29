import numpy as np

from jacscanomaly import Candidate, measure_features
from jacscanomaly.features import _G, _H

HOST = 10.0


def _measure(signal_fn, window, err=0.02, seed=1, step=0.005):
    time = np.arange(8.0, 12.0, step)
    signal = signal_fn(time)
    flux = HOST + signal + np.random.default_rng(seed).normal(0.0, err, time.size)
    # the noiseless signal stands in for the atoms
    return measure_features(time, flux, np.full_like(time, err), np.full_like(time, HOST),
                            signal, [window])


def _gauss(center, width, amplitude):
    return lambda t: amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)


def _pair(t1, t2, dt, a, floor, inside):
    def signal(t):
        z1, z2 = (t - t1) / dt, (t2 - t) / dt
        box = _H(z1) * _H(z2)
        if inside:
            return a * (_G(z1) * _H(z2) + _G(z2) * _H(z1)) + floor * box
        return a * (_G(-z1) + _G(-z2)) + floor * (1.0 - box) - floor * box
    return signal


def test_bump_and_dip():
    (bump,) = _measure(_gauss(10.0, 0.1, 1.0), Candidate(9.5, 10.5)).features
    assert bump.kind == "bump" and abs(bump.time - 10.0) < 0.02 and bump.explained["bump"] > 0.95
    (dip,) = _measure(_gauss(10.0, 0.1, -1.0), Candidate(9.5, 10.5)).features
    assert dip.kind == "dip" and abs(dip.time - 10.0) < 0.02


def test_caustic_crossing_pair_inside_between():
    (f,) = _measure(_pair(9.6, 10.4, 0.01, 0.3, 0.5, True), Candidate(9.2, 10.8)).features
    assert f.kind == "caustic_entry_exit"
    assert abs(f.params["t1"] - 9.6) < 0.01 and abs(f.params["t2"] - 10.4) < 0.01
    assert abs(f.time - 10.0) < 0.01 and abs(f.timescale - 0.8) < 0.02
    assert f.explained["pair"] > f.explained["double"] + 0.05
    entry, exit_ = f.crossings
    assert (entry.kind, exit_.kind) == ("entry", "exit")
    # sharp outside, slow 1/sqrt decay inside: entry is narrow on the left
    assert entry.hwhm_left < entry.hwhm_right and exit_.hwhm_right < exit_.hwhm_left
    assert abs(entry.time - 9.6) < 0.03 and abs(exit_.time - 10.4) < 0.03


def test_caustic_pair_outside_between():
    (f,) = _measure(_pair(9.6, 10.4, 0.01, 0.3, 0.5, False), Candidate(9.2, 10.8)).features
    assert f.kind == "caustic_exit_entry"
    assert abs(f.params["t1"] - 9.6) < 0.01 and abs(f.params["t2"] - 10.4) < 0.01


def test_weak_window_is_kept_and_not_called_a_caustic():
    (f,) = _measure(_gauss(10.0, 0.2, 0.01), Candidate(9.0, 11.0)).features
    assert 0.0 < f.dchi2 < 300.0
    assert f.kind in ("bump", "complex")


def test_two_separate_bumps_are_a_double():
    two = lambda t: _gauss(9.6, 0.08, 1.0)(t) + _gauss(10.4, 0.08, -0.7)(t)
    (f,) = _measure(two, Candidate(9.2, 10.8)).features
    assert f.kind == "bump+dip"
    assert abs(f.params["t0_1"] - 9.6) < 0.02 and abs(f.params["t0_2"] - 10.4) < 0.02


def test_summary_and_no_windows():
    res = _measure(_gauss(10.0, 0.1, 1.0), Candidate(9.5, 10.5))
    s = res.summary_dict()
    assert s["n_features"] == 1 and s["kinds"] == {"bump": 1}
    assert s["strongest"]["kind"] == "bump" and res.feature_dicts()[0]["dchi2"] > 0
    time = np.linspace(8.0, 12.0, 100)
    empty = measure_features(time, time * 0 + HOST, time * 0 + 0.02, time * 0 + HOST, time * 0, [])
    assert empty.features == () and empty.strongest is None
