import json

import numpy as np
import pytest

from jacscanomaly import Event, FinderConfig, write_json
from test_robust import _event, _finder

CONFIG = FinderConfig(gap=100.0, grid_backend="cpp")


def test_every_candidate_from_the_score_is_a_detection():
    finder = _finder()
    result = finder.run(*_event([(-8.0, 0.3), (12.0, 0.2)], n=3000), x0=[0.0, 20.0, 0.3],
                        verbose=False)
    t0s = sorted(round(c.t0) for c in result.detections)
    assert -8 in t0s and 12 in t0s


def test_scan_and_atoms_need_a_fit():
    ev = Event(*_event([], n=500), config=CONFIG)
    with pytest.raises(RuntimeError, match="fit"):
        ev.scan()
    ev.fit("pspl")
    with pytest.raises(RuntimeError, match="fspl"):
        ev.scan("fspl")
    with pytest.raises(RuntimeError, match="fspl"):
        ev.fit_atoms("fspl")
    with pytest.raises(ValueError):
        ev.fit("binary")


def test_pipeline_fit_scan_refit_atoms_json(tmp_path):
    time, flux, ferr = _event([(-8.0, 0.3)], n=3000)
    ev = Event(time, 1000.0 * flux, 1000.0 * ferr, config=CONFIG)  # scaled inside
    fit = ev.fit("pspl")
    assert list(fit.params) == ["t0", "tE", "u0"] and abs(fit.params["tE"] - 20.0) < 1.0
    assert abs(fit.fs - 1000.0) < 50.0 and abs(fit.fb - 500.0) < 50.0  # input units
    scan = ev.scan()
    assert scan.detected and abs(scan.detections[0].t0 + 8.0) < 0.5
    assert all(c.score >= 10.0 for c in scan.detections)

    fspl = ev.fit("fspl", start="pspl")
    assert fspl.chi2 <= fit.chi2 + 1e-6  # nests the PSPL
    assert ev.scan().model == "fspl"     # the last fit
    atoms = ev.fit_atoms("pspl")
    assert atoms.dchi2 > 100 and atoms.host_identified
    (f,) = atoms.features
    assert f.kind == "bump" and abs(f.time + 8.0) < 0.05

    # refitting drops that model's scan and atoms only
    ev.fit("fspl")
    assert "fspl" not in ev.scans and "pspl" in ev.scans and "pspl" in ev.atoms

    doc = ev.write_json(tmp_path / "e.json")
    assert json.load(open(tmp_path / "e.json")) == doc  # strict JSON: no NaN
    assert set(doc["fits"]) == {"pspl", "fspl"} and set(doc["scans"]) == {"pspl"}
    assert doc["atoms"]["pspl"]["anomalies"][0]["kind"] == "bump"
    assert write_json(tmp_path / "all.json", {"evt": ev})["evt"] == doc

    rows = ev.summary()
    rows = rows.to_dict("records") if hasattr(rows, "to_dict") else rows
    assert [r["model"] for r in rows] == ["pspl", "fspl"] and rows[0]["n_atoms"] >= 1

    # released: the records stay, a later scan evaluates the model again
    ev.release()
    assert ev.write_json(tmp_path / "e2.json") == doc
    assert ev.scan("pspl").detected


def test_magnitudes_give_the_fit_of_the_flux():
    time, flux, ferr = _event([], n=800)
    ev_f = Event(time, flux, ferr, config=CONFIG)
    mag = 22.0 - 2.5 * np.log10(flux)
    ev_m = Event(time, mag, 2.5 / np.log(10.0) * ferr / flux, kind="mag", config=CONFIG)
    a, b = ev_f.fit("pspl"), ev_m.fit("pspl")
    assert np.allclose([a.params[k] for k in a.params], [b.params[k] for k in b.params],
                       rtol=1e-4, atol=1e-4)
    assert np.isclose(a.chi2, b.chi2, rtol=1e-4)
    assert not ev_f.scan().detected and ev_f.scan().to_dict()["detections"] == []
