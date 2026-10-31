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
        return a * (_G(-z1) + _G(-z2)) - floor * box
    return signal


def test_bump_and_dip():
    (bump,) = _measure(_gauss(10.0, 0.1, 1.0), Candidate(9.5, 10.5)).features
    assert bump.kind == "bump" and abs(bump.time - 10.0) < 0.02 and bump.explained["bump"] > 0.95
    (dip,) = _measure(_gauss(10.0, 0.1, -1.0), Candidate(9.5, 10.5)).features
    assert dip.kind == "dip" and abs(dip.time - 10.0) < 0.02


def test_caustic_pair_inside_between_is_sharp():
    (f,) = _measure(_pair(9.6, 10.4, 0.01, 0.3, 0.5, True), Candidate(9.2, 10.8)).features
    assert f.caustic and f.sharpness > 10.0
    assert f.explained["pair"] > f.explained["double"] + 0.05


def test_caustic_pair_outside_between_is_sharp():
    (f,) = _measure(_pair(9.6, 10.4, 0.01, 0.3, 0.5, False), Candidate(9.2, 10.8)).features
    assert f.caustic and f.sharpness > 10.0


def test_smooth_parts_are_not_sharp():
    # one smooth part ~1.5, two separate ones ~3: both below sharp_min
    (g,) = _measure(_gauss(10.0, 0.1, 1.0), Candidate(9.0, 11.0)).features
    assert not g.caustic and abs(g.sharpness - 1.5) < 0.2
    two = lambda t: _gauss(9.6, 0.08, 1.0)(t) + _gauss(10.4, 0.08, 1.0)(t)
    (f,) = _measure(two, Candidate(9.2, 10.8)).features
    assert not f.caustic and f.sharpness < 3.5


def test_a_gap_carries_no_area():
    from jacscanomaly.features import _sharpness
    t = np.r_[np.arange(0.0, 1.0, 0.01), np.arange(5.0, 6.0, 0.01)]
    y = np.exp(-0.5 * ((t - 0.5) / 0.1) ** 2) + np.exp(-0.5 * ((t - 5.5) / 0.1) ** 2)
    assert _sharpness(t, y, [(0, t.size - 1)]) < 3.5  # not the area of a bridge over the gap


def test_weak_window_is_kept_and_not_called_a_caustic():
    (f,) = _measure(_gauss(10.0, 0.2, 0.01), Candidate(9.0, 11.0)).features
    assert 0.0 < f.dchi2 < 300.0
    assert f.kind in ("bump", "complex")


def test_two_separate_bumps_are_a_double():
    two = lambda t: _gauss(9.6, 0.08, 1.0)(t) + _gauss(10.4, 0.08, -0.7)(t)
    (f,) = _measure(two, Candidate(9.2, 10.8)).features
    assert f.kind == "bump+dip"
    assert abs(f.parts[0].time - 9.6) < 0.02 and abs(f.parts[1].time - 10.4) < 0.02


def test_summary_and_no_windows():
    res = _measure(_gauss(10.0, 0.1, 1.0), Candidate(9.5, 10.5))
    s = res.summary_dict()
    assert s["n_features"] == 1 and s["kinds"] == {"bump": 1}
    assert s["strongest"]["kind"] == "bump" and res.feature_dicts()[0]["dchi2"] > 0
    time = np.linspace(8.0, 12.0, 100)
    empty = measure_features(time, time * 0 + HOST, time * 0 + 0.02, time * 0 + HOST, time * 0, [])
    assert empty.features == () and empty.strongest is None


def test_bump_timescale_is_its_width():
    (f,) = _measure(_gauss(10.0, 0.1, 1.0), Candidate(9.5, 10.5)).features
    assert f.kind == "bump" and abs(f.timescale - 2.355 * 0.1) < 0.03


def test_double_spans_cover_each_atom():
    two = lambda t: _gauss(9.6, 0.08, 1.0)(t) + _gauss(10.4, 0.08, -0.7)(t)
    (f,) = _measure(two, Candidate(9.2, 10.8)).features
    (a1, b1), (a2, b2) = f.spans
    assert abs(a1 - 9.506) < 0.03 and abs(b1 - 9.694) < 0.03
    assert abs(a2 - 10.306) < 0.03 and abs(b2 - 10.494) < 0.03
    (p1, p2) = f.parts
    assert (p1.kind, p2.kind) == ("bump", "dip")
    assert abs(p1.time - 9.6) < 0.01 and abs(p2.time - 10.4) < 0.01
    assert abs(p1.timescale - 2.355 * 0.08) < 0.03 and abs(p2.timescale - 2.355 * 0.08) < 0.03


def test_pair_needs_resolved_crossings():
    from jacscanomaly.features import Crossing, _crossings_resolved
    a, b = Crossing("exit", 10.0, 1.0, 1.0, 0.5), Crossing("entry", 13.0, 0.3, 0.3, 0.1)
    assert _crossings_resolved((a, b))
    wide = Crossing("exit", 10.0, 2.0, 1.5, 0.5)  # FWHM 3.5 > 3.0 between the peaks
    assert not _crossings_resolved((wide, b))
    assert not _crossings_resolved((Crossing("exit", 10.0, float("nan"), 1.0, 0.5), b))


def test_rebound_next_to_a_strong_bump_is_not_a_part():
    one = lambda t: _gauss(10.0, 0.1, 1.0)(t) + _gauss(10.25, 0.08, -0.15)(t)
    (f,) = _measure(one, Candidate(9.4, 10.8)).features
    assert f.kind == "bump" and abs(f.time - 10.0) < 0.02


def test_part_comparable_to_its_kept_neighbour_is_kept():
    # bump 0.35, dip -0.55, bump 1.0 (0140): the first bump is half the dip next to it
    sig = lambda t: _gauss(9.6, 0.05, 0.35)(t) + _gauss(9.85, 0.07, -0.55)(t) + _gauss(10.1, 0.05, 1.0)(t)
    (f,) = _measure(sig, Candidate(9.2, 10.5)).features
    assert f.kind == "bump+dip+bump"
