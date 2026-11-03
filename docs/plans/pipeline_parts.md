# Parts for a pipeline after `Finder.run`

## Problem

`Finder(config).run(time, flux, ferr)` is the entry point users know. What
follows it (host + atoms fit, the choice of models, one JSON record) needs
`robust_fit(fitters, finder, time, flux, ferr, p_scan)` plus `LightCurve` and
`EventResult` wrappers: the user repeats the data, builds a PSPL finder for
the residual scans, and gets a record whose layout the wrapper fixes.

## Change

The package gives parts; the user writes the branching.

```python
finder = Finder(FinderConfig())
result = finder.run(time, flux, ferr)          # unchanged
if result.detected:                            # result.detections is not empty
    host = finder.fit_host(result, {"fspl": FSPLFitter()})   # PSPL is always fitted first
    if host.selected == "fspl":
        ...
write_json("event.json", {"scan": result, "host": host})
```

* One rule lists every anomaly, as the atoms do: a detection is a candidate
  with `score >= FinderConfig.detection_score` (10) that `candidate_criteria`
  accepts. `AnomalyResult.detections` holds all of them, by score;
  `detected` is `bool(detections)`. `fit_host` accepts atoms by the same
  score (`AtomConfig(min_score=detection_score)` by default).
* `Finder.fit_host(result, fitters=None, *, config=AtomConfig()) -> RobustFit`:
  `robust_fit` on the data `result` holds, from its PSPL `(t0, tE, u0)`.
  A PSPL fitter is put first when `fitters` has none (the relock and the
  nested starts need it); the residual scans use this finder's settings with
  a PSPL host.
* `AnomalyResult.to_dict()`: `detected`, every detection (`t0, teff, dchi2,
  score`), `n_points`, the fit's named parameters and `chi2_dof`.
* `RobustFit.to_dict()`: `selected`, and per model the named parameters,
  `chi2`, `J`, `host_identified`, the anomalies (`kind, caustic, t_center,
  fwhm, dchi2`) and the plain fit without atoms (named parameters, `chi2`).
* `write_json(path, obj)`: any nesting of dicts/lists and objects with
  `to_dict()`; `nan` as `null`; number lists on one line.
* Removed: `LightCurve`, `EventResult`, `write_results` (one user, aethra's
  campaign runner, which pins its own source copy).

## Not changed

`Finder.run`, `robust_fit` (still public for full control), the fitters.
