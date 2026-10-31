import numpy as np

from jacscanomaly import (
    AtomConfig, FSPLFitter, Finder, FinderConfig, PSPLFitter, atom_basis, atom_windows, fit_atoms,
    robust_fit, write_event_json,
)
from jacscanomaly.fitters import _pspl_magnification
from jacscanomaly.robust import Atom


def _event(anomalies, *, n=6000, seed=7):
    rng = np.random.default_rng(seed)
    time = np.linspace(-30.0, 30.0, n)
    flux = _pspl_magnification(time, 0.0, 20.0, 0.3) + 0.5
    for t0, amp in anomalies:
        flux = flux + amp * np.exp(-0.5 * ((time - t0) / 0.1) ** 2)
    ferr = np.full_like(time, 0.01)
    return time, flux + rng.normal(0.0, 0.01, time.size), ferr


def _finder():
    return Finder(FinderConfig(gap=100.0, grid_backend="cpp"))


def test_atom_basis_is_zero_at_and_outside_its_support():
    time = np.linspace(-10.0, 10.0, 2001)
    high, low = atom_basis(time, 1.0, 2.0, coeff=3.0)
    outside = np.abs(time - 1.0) >= 6.0
    assert np.all(high[outside] == 0.0) and np.all(low[outside] == 0.0)
    assert high[np.argmin(np.abs(time - 1.0))] > 0.6 and low.max() > 0.0


def test_atom_windows_merge_overlapping_supports():
    atoms = [Atom(0.0, 1.0, 1.0, 0.0), Atom(4.0, 1.0, 1.0, 0.0), Atom(20.0, 0.5, 1.0, 0.0)]
    windows = atom_windows(atoms, coeff=3.0)
    assert [(w.t_start, w.t_end) for w in windows] == [(-3.0, 7.0), (18.5, 21.5)]


def test_atoms_take_the_anomalies_and_leave_the_host():
    time, flux, ferr = _event([(-8.0, 0.3), (6.0, 0.1)])

    r = fit_atoms(PSPLFitter(), _finder(), time, flux, ferr, [0.0, 20.0, 0.3])

    centres = np.array([a.t0 for a in r.atoms])
    near = (np.abs(centres + 8.0) < 0.5) | (np.abs(centres - 6.0) < 0.5)
    assert near.all() and np.any(centres < 0) and np.any(centres > 0)
    t0, tE, u0 = r.params
    assert abs(t0) < 0.05 and abs(tE - 20.0) < 0.3 and abs(u0 - 0.3) < 0.01
    assert r.chi2 < r.chi2_host_only
    assert abs(r.chi2 / time.size - 1.0) < 0.1


def test_no_anomaly_takes_no_atom():
    time, flux, ferr = _event([])

    r = fit_atoms(PSPLFitter(), _finder(), time, flux, ferr, [0.0, 20.0, 0.3])

    assert r.atoms == () and abs(r.params[1] - 20.0) < 0.5


def test_a_wide_candidate_is_capped_unless_it_is_on_the_anomaly():
    time, flux, ferr = _event([(-8.0, 0.3)])
    finder = _finder()
    narrow = AtomConfig(max_teff_fraction=1e-4)  # no scan candidate fits under the cap

    capped = fit_atoms(PSPLFitter(), finder, time, flux, ferr, [0.0, 20.0, 0.3], config=narrow)
    anomaly = np.abs(time + 8.0) < 0.5
    exempt = fit_atoms(PSPLFitter(), finder, time, flux, ferr, [0.0, 20.0, 0.3],
                       anomaly=anomaly, config=narrow)

    assert capped.atoms == ()
    assert exempt.atoms and all(abs(a.t0 + 8.0) < 0.5 for a in exempt.atoms)


def test_robust_fit_selects_by_J_and_nests_the_extended_model():
    time, flux, ferr = _event([(-8.0, 0.3)], n=3000)
    fitters = {"pspl": PSPLFitter(), "fspl": FSPLFitter()}

    r = robust_fit(fitters, _finder(), time, flux, ferr, [0.0, 20.0, 0.3])

    assert set(r.models) == {"pspl", "fspl"} and not r.relock.relocked
    assert set(r.models["fspl"].J_starts) <= {"nested", "base"}
    assert r.selected == min(r.models, key=lambda k: r.models[k].J)
    best = r.best
    assert best.atoms and len(best.windows) == 1
    assert best.fit.mask is not None and best.fit.mask[np.argmin(np.abs(time + 8.0))]
    assert np.all(np.isfinite(best.sigma))


def test_event_json_records_every_model_and_the_anomaly(tmp_path):
    import json

    time, flux, ferr = _event([(-8.0, 0.3)], n=3000)
    r = robust_fit({"pspl": PSPLFitter(), "fspl": FSPLFitter()}, _finder(), time, flux, ferr,
                   [0.0, 20.0, 0.3])

    rec = write_event_json(tmp_path / "e.json", r, event_id=7)
    assert json.load(open(tmp_path / "e.json")) == rec  # strict JSON: no NaN
    assert rec["event_id"] == 7
    sc, at = rec["scan"], rec["atoms"]
    assert sc["anomaly"]["detected"] is None and sc["candidates"] == []  # no scan given
    assert set(sc["models"]) == {"pspl", "fspl"}
    assert [k for k, m in sc["models"].items() if m["selected"]] == [sc["selected"]]
    assert at["selected"] == r.selected
    assert [k for k, m in at["models"].items() if m["selected"]] == [r.selected]
    m = at["models"][r.selected]
    assert len(m["params"]) == len(m["param_names"]) == len(m["sigma"])
    assert sc["models"][r.selected]["chi2"] - m["chi2"] == m["dchi2_atoms"] > 100
    assert m["n_masked"] > 0
    (f,) = m["features"]
    assert f["kind"] == "bump" and abs(f["time"] + 8.0) < 0.05
    assert abs(f["parts"][0]["fwhm"] - 2.3548 * 0.1) < 0.05


def test_the_cap_grows_with_u_in_the_wing():
    rng = np.random.default_rng(3)
    time = np.linspace(-30.0, 30.0, 6000)
    # a wide dip at u ~ 1.33: wider than 0.13 tE, narrower than 0.13 tE u
    flux = (_pspl_magnification(time, 0.0, 20.0, 0.3) + 0.5
            - 0.05 * np.exp(-0.5 * ((time + 26.0) / 1.5) ** 2) + rng.normal(0.0, 0.01, time.size))
    ferr = np.full_like(time, 0.01)

    r = fit_atoms(PSPLFitter(), _finder(), time, flux, ferr, [0.0, 20.0, 0.3],
                  config=AtomConfig(max_teff_fraction=0.13))

    assert any(abs(a.t0 + 26.0) < 1.0 and a.teff > 0.13 * 20.0 for a in r.atoms)

