# One JSON record per event

## Problem

The end product of an event is spread over objects (`AnomalyResult` of the
scan, `RobustFit` with a `ModelFit` per host model, `FeatureResult` per
model) and is turned into JSON only by the campaign scripts in aethra
(`rmdc_physical_stage._fit_record`, `run_model_check_campaign.run_event`),
each with its own fields. A user of the library has no single record to
write or read back.

## Change

`RobustFit.to_dict(event_id=None, scan=None, score_threshold=10)` returns a JSON-safe dict
(floats, lists; `nan` -> `null`), and `write_event_json(path, ...)` writes
it. Nothing new is computed: every number is already in the objects.

Two stages, kept apart:

* `scan`: the PSPL scan and its decision (best candidate score >=
  `score_threshold`), the candidates above it, and the plain fit of every
  model (`base`: PSPL and the FSPL / parallax fallbacks), lowest BIC
  (`chi2 + (n_params + 2) ln n`) selected.
* `atoms`: the host + atoms fit of every model, lowest `J` selected, with
  the window features. It describes the anomaly; it does not decide it.

```json
{
  "event_id": 2000979,
  "scan": {
    "anomaly": {"detected": true, "score": 312.4, "score_threshold": 10.0,
                "dchi2": 58000.0, "t0": 8734.70, "teff": 0.05},
    "candidates": [{"t0":..,"teff":..,"dchi2":..,"score":..}],
    "selected": "pspl",
    "models": {"pspl": {"selected": true, "param_names": [...], "params": [...],
                        "fs":..,"fb":..,"chi2":..,"n_points":..,"n_masked":0,"bic":..},
               "fspl": {...}, "pspl_space_parallax": {...}}
  },
  "atoms": {
    "selected": "pspl", "relocked": false,
    "models": {
      "pspl": {
        "selected": true, "param_names": [...], "params": [...], "sigma": [...],
        "fs":..,"fb":..,"J":..,"start":"scan","J_starts":{..},
        "n_points":..,"n_free":..,
        "chi2": ..,          // host + atoms, every sample
        "chi2_masked": ..,   // host alone outside the windows
        "n_masked": ..,
        "dchi2_atoms": ..,   // scan-stage chi2 of this model - chi2
        "atoms": [{"t0":..,"teff":..,"a":..,"b":..}],
        "features": [{"t_start":..,"t_end":..,"kind":"bump","time":..,"timescale":..,
                      "dchi2":..,"sharpness":7.2,"caustic":true,"n_points":..,
                      "f_bump":..,"f_fold":..,"f_double":..,"f_pair":..,
                      "spans":[..],"span_kinds":[..],
                      "parts":[{"kind":"bump","time":..,"fwhm":..}]}]
      }, ...
    }
  },
  "errors": {}
}
```

An event whose scan has no candidate above the threshold has `atoms: null`.

## Assumptions

* "Anomaly present" is the scan's decision (best score >= threshold, 10 by
  default, as the campaigns publish), not the atoms': the atoms and features
  describe the anomaly, they do not decide it.
* `dchi2_atoms` uses the scan stage's fit of the same model (`base`) as the
  no-anomaly model; when `base` failed it is `null`. With the relock off its
  anomaly, `base` is the masked H1 fit (`n_masked > 0` in `scan.models`).
* `Part.timescale` is written as `fwhm` (it is the FWHM); the Python field
  is unchanged.
