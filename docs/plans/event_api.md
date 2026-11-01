# Light curve in, one record out

## Problem

The module has the stages (`Finder.run`, `robust_fit`, `measure_features`)
but no object for the light curve they run on, and no result that holds an
event from scan to features. Each pipeline glues them with its own arrays,
ids and JSON (aethra: `run_model_check_campaign`, `rmdc_physical_stage`); the
first JSON attempt (`RobustFit.to_dict`) wrote every internal quantity
(``J``, relock, starts), which readers do not use. Events whose scan finds no
anomaly got no record at all.

## Change

* `LightCurve(time, flux, ferr, id=None)`: the data of one event. Sorted by
  time, non-finite and ``ferr <= 0`` samples dropped.
* The stages stay as they are; the pipeline decides the order:
  ```python
  lc = LightCurve(t, f, e, id="2_675_129")
  scan = Finder(...).run(lc.time, lc.flux, lc.ferr, ...)
  fits = robust_fit(fitters, finder, lc.time, lc.flux, lc.ferr, p_scan) if anomaly else None
  rec = EventResult(lc, scan, fits).to_dict()
  ```
* `EventResult(lightcurve, scan, fits=None, score_threshold=10)`:
  `detected` is the scan's decision (best score >= threshold); `fits` is
  `None` when there is no anomaly. `to_dict()`:
  ```json
  {"anomaly": {"detected", "score", "t0", "teff", "dchi2"},
   "best_model": {"without_atoms": "pspl", "with_atoms": "pspl" | null},
   "n_points": 40540,
   "models": {"pspl": {"param_names": [...],
                       "without_atoms": {"params": [...], "chi2": ...},
                       "with_atoms": {"params": [...], "chi2": ...,
                                      "features": [{"kind", "caustic", "t_center", "fwhm", "dchi2"}]}}}}
  ```
  With no anomaly: the scan's PSPL fit only, `with_atoms` absent.
* `write_results(path, results)`: `{id: to_dict()}` for many events (or an
  `{id: record}` mapping read back, to merge per-event files), one
  line per number list / feature, a blank line between events, `nan` as
  `null`.
* Removed: `RobustFit.to_dict`, `ModelFit.to_dict`, `scan_anomaly`,
  `write_event_json` (superseded; no user yet).

## Assumptions

* `without_atoms`: the plain fit of each model (`ModelFit.base`), best by
  BIC ``chi2 + (n_params + 2) ln n``; `with_atoms`: the host + atoms fit,
  best by `robust_fit`'s selection. Neither criterion is written: readers
  get the choice and the chi2.
* Parameters in the fit's convention (`rho`, not `log rho`); times as given.
