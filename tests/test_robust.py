import numpy as np

from jacscanomaly import (
    AtomConfig, FSPLFitter, Finder, FinderConfig, PSPLFitter, atom_basis, atom_windows, fit_atoms,
    robust_fit,
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
    assert set(r.models["fspl"].J_starts) <= {"nested", "base", "fixed"}
    assert r.selected == min(r.models, key=lambda k: r.models[k].rank)
    best = r.best
    assert best.atoms and len(best.windows) == 1
    assert best.fit.mask is not None and best.fit.mask[np.argmin(np.abs(time + 8.0))]
    assert np.all(np.isfinite(best.sigma))


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



def test_the_plain_host_is_kept_unless_a_moved_one_wins_by_the_margin():
    time, flux, ferr = _event([(-8.0, 0.3)], n=3000)
    finder, fitters = _finder(), {"pspl": PSPLFitter()}

    fixed = fit_atoms(PSPLFitter(), finder, time, flux, ferr, [0.1, 19.0, 0.32], fix_host=True)
    assert fixed.atoms and np.allclose(fixed.params, [0.1, 19.0, 0.32])

    for margin in (0.0, 1e9):
        m = robust_fit(fitters, finder, time, flux, ferr, [0.0, 20.0, 0.3],
                       config=AtomConfig(free_host_margin=margin)).best
        J = m.J_starts
        assert m.start == ("fixed" if margin else min(J, key=J.get))
        assert m.rank == m.J - margin * (m.start == "fixed")


def test_fs_and_fb_are_not_negative():
    from jacscanomaly.robust import _Host, _linear
    time = np.linspace(-30.0, 30.0, 2000)
    ferr = np.full_like(time, 0.01)
    flux = 2.0 - 0.5 * _pspl_magnification(time, 0.0, 20.0, 1.0)  # fs < 0 unconstrained
    coef, z = _linear(_Host(PSPLFitter(), time), flux, ferr, np.array([0.0, 20.0, 1.0]), [], 3.0)
    assert coef[0] >= 0.0 and coef[1] >= 0.0
    assert coef[0] == 0.0 and abs(coef[1] - np.mean(flux)) < 1e-6


def test_fixed_holds_the_host_of_the_samples_outside_the_windows():
    time, flux, ferr = _event([(-8.0, 0.3)], n=3000)
    m = robust_fit({"pspl": PSPLFitter()}, _finder(), time, flux, ferr, [0.0, 20.0, 0.3],
                   config=AtomConfig(free_host_margin=1e9)).best
    t0, tE, u0 = m.params
    assert m.start == "fixed" and m.host_identified
    assert abs(t0) < 0.05 and abs(tE - 20.0) < 0.5 and abs(u0 - 0.3) < 0.02


def test_a_needle_host_under_atoms_is_not_a_host():
    rng = np.random.default_rng(3)
    time = np.linspace(-30.0, 30.0, 3000)
    ferr = np.full_like(time, 0.01)
    bump = 0.5 * np.exp(-0.5 * (time / 0.3) ** 2) * (1.0 + 0.5 * np.sin(time / 0.05))
    flux = 1.0 + bump + rng.normal(0.0, 0.01, time.size)
    # the scan start keeps a needle host (fs ~ 0) and lets the atoms build the
    # bump: the lowest J, but the samples outside the windows see no host
    m = robust_fit({"pspl": PSPLFitter()}, _finder(), time, flux, ferr, [0.0, 1000.0, 1e-4]).best
    assert m.J_starts["scan"] < m.J and m.start != "scan" and m.host_identified

