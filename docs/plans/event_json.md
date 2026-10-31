# One JSON record per event

## Problem

The end product of an event is spread over objects (`AnomalyResult` of the
scan, `RobustFit` with a `ModelFit` per host model, `FeatureResult` per
model) and is turned into JSON only by the campaign scripts in aethra
(`rmdc_physical_stage._fit_record`, `run_model_check_campaign.run_event`),
each with its own fields. A user of the library has no single record to
write or read back.

## Change

`RobustFit.to_dict(event_id=None, scan=None)` returns a JSON-safe dict
(floats, lists; `nan` -> `null`), and `write_event_json(path, ...)` writes
it. Nothing new is computed: every number is already in the objects.

```json
{
  "event_id": 2000979,
  "anomaly": {
    "detected": true,            // the selected model keeps >= 1 atom
    "scan_score": 312.4,         // scan's best candidate score (when scan given)
    "scan_dchi2": 58000.0,       //   and its dchi2
    "dchi2": 58316.5             // selected model: chi2(base) - chi2(host + atoms)
  },
  "selected": "pspl",            // lowest J
  "relocked": false,
  "models": {
    "pspl": {
      "selected": true,
      "param_names": ["t0", "tE", "u0"],
      "params": [...], "sigma": [...], "fs": 1.0, "fb": 0.0,
      "J": ..., "start": "scan",
      "n_points": 2400, "n_free": 9,
      "chi2": ...,               // host + atoms, every sample
      "chi2_base": ...,          // plain fit, no atoms, every sample (before)
      "chi2_masked": ...,        // host alone on the samples outside the windows (after)
      "n_masked": 120,
      "atoms": [{"t0":..,"teff":..,"a":..,"b":..}],
      "features": [              // one per anomaly window
        {"t_start":..,"t_end":..,"kind":"bump","time":..,"timescale":..,  // FWHM
         "dchi2":..,"sharpness":7.2,"caustic":true,"n_points":..,
         "f_bump":..,"f_fold":..,"f_double":..,"f_pair":..,
         "parts":[{"kind":"bump","time":..,"fwhm":..}]}
      ]
    },
    "fspl": {...}, "pspl_space_parallax": {...}
  },
  "errors": {}
}
```

## Assumptions

* "Anomaly present" is the atoms' own decision: the selected model kept an
  atom, i.e. it paid its BIC price (`k_atom ln n`). The scan score is
  reported, not re-thresholded.
* `dchi2` uses `base` (the plain fit) as the no-anomaly model; when `base`
  failed it is `null`.
* `Part.timescale` is written as `fwhm` (it is the FWHM); the Python field
  is unchanged.
