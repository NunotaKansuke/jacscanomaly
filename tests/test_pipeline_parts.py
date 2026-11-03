import json

import numpy as np

from jacscanomaly import AtomConfig, FSPLFitter, write_json
from test_robust import _event, _finder


def test_every_candidate_from_the_score_is_a_detection():
    finder = _finder()
    result = finder.run(*_event([(-8.0, 0.3), (12.0, 0.2)], n=3000), x0=[0.0, 20.0, 0.3],
                        verbose=False)
    t0s = sorted(round(c.t0) for c in result.detections)
    assert -8 in t0s and 12 in t0s


def test_run_then_fit_host_then_one_json_record(tmp_path):
    finder = _finder()
    result = finder.run(*_event([(-8.0, 0.3)], n=3000), x0=[0.0, 20.0, 0.3], verbose=False)
    assert result.detected
    # a PSPL is put first when only another model is asked for
    host = finder.fit_host(result, {"fspl": FSPLFitter()}, config=AtomConfig(free_host_margin=0.0))
    assert list(host.models) == ["pspl", "fspl"]

    quiet = finder.run(*_event([], n=3000, seed=8), x0=[0.0, 20.0, 0.3], verbose=False)
    assert not quiet.detected
    doc = write_json(tmp_path / "r.json", {"evt": {"scan": result, "host": host},
                                           "quiet": {"scan": quiet}})
    assert json.load(open(tmp_path / "r.json")) == doc  # strict JSON: no NaN

    s, h = doc["evt"]["scan"], doc["evt"]["host"]
    assert s["detected"] and abs(s["detections"][0]["t0"] + 8.0) < 0.5 and s["n_points"] == 3000
    assert all(d["score"] >= 10.0 for d in s["detections"])
    assert list(s["fit"]["params"]) == ["t0", "tE", "u0"]
    m = h["models"][h["selected"]]
    assert m["without_atoms"]["chi2"] - m["chi2"] > 100 and m["host_identified"]
    (f,) = m["anomalies"]
    assert f["kind"] == "bump" and abs(f["t_center"] + 8.0) < 0.05
    # rho itself, not the fit's log rho
    assert np.isclose(h["models"]["fspl"]["params"]["rho"], np.exp(host.models["fspl"].params[3]))

    q = doc["quiet"]["scan"]
    assert q["detected"] is False and q["detections"] == []
