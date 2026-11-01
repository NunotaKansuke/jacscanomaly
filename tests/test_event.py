import json

import numpy as np

from jacscanomaly import AtomConfig, EventResult, FSPLFitter, LightCurve, PSPLFitter, robust_fit, write_results
from test_robust import _event, _finder


def test_light_curve_sorts_and_drops_bad_samples():
    lc = LightCurve([3.0, 1.0, 2.0, np.nan], [1.0, 2.0, 3.0, 4.0], [0.1, 0.1, 0.0, 0.1], id="a")
    assert lc.time.tolist() == [1.0, 3.0] and lc.flux.tolist() == [2.0, 1.0] and len(lc) == 2


def test_results_record_the_scan_and_both_stages(tmp_path):
    finder, p0 = _finder(), [0.0, 20.0, 0.3]
    lc = LightCurve(*_event([(-8.0, 0.3)], n=3000), id="2_675_129")
    scan = finder.run(lc.time, lc.flux, lc.ferr, x0=p0, verbose=False)
    fits = robust_fit({"pspl": PSPLFitter(), "fspl": FSPLFitter()}, finder,
                      lc.time, lc.flux, lc.ferr, p0, config=AtomConfig(free_host_margin=0.0))
    quiet = LightCurve(*_event([], n=3000, seed=8), id="quiet")
    res = [EventResult(lc, scan, fits),
           EventResult(quiet, finder.run(quiet.time, quiet.flux, quiet.ferr, x0=p0, verbose=False))]

    doc = write_results(tmp_path / "r.json", res)
    assert json.load(open(tmp_path / "r.json")) == doc  # strict JSON: no NaN
    e, q = doc["2_675_129"], doc["quiet"]
    assert e["anomaly"]["detected"] and abs(e["anomaly"]["t0"] + 8.0) < 0.5
    assert e["best_model"]["with_atoms"] == fits.selected and e["n_points"] == 3000
    m = e["models"][fits.selected]
    assert m["without_atoms"]["chi2"] - m["with_atoms"]["chi2"] > 100
    assert len(m["with_atoms"]["params"]) == len(m["param_names"])
    f_ = e["models"]["fspl"]  # rho itself, not the fit's log rho
    assert f_["param_names"][3] == "rho"
    assert np.isclose(f_["with_atoms"]["params"][3], np.exp(fits.models["fspl"].params[3]))
    (f,) = m["with_atoms"]["features"]
    assert f["kind"] == "bump" and abs(f["t_center"] + 8.0) < 0.05
    assert abs(f["fwhm"] - 2.3548 * 0.1) < 0.05
    # no anomaly: the scan's PSPL fit, no atoms stage
    assert q["anomaly"]["detected"] is False and q["best_model"] == {"without_atoms": "pspl",
                                                                     "with_atoms": None}
    assert list(q["models"]) == ["pspl"] and "with_atoms" not in q["models"]["pspl"]
