# v1.0 API: one `Event`, steps the user calls in order

## Problem

`Finder.run` does the fit and the scan in one call. The next steps
(`robust_fit`, `fit_host`) need the fitters, the data again and a PSPL
finder. The user cannot write "fit, scan, if found refit with FSPL, scan
again, if still found fit atoms" as it reads.

## Use

Every step is a method of `Event`; the results live on the `Event`.

```python
from jacscanomaly import Event, FinderConfig

ev = Event(time, flux, ferr, ra_deg=268.4, dec_deg=-29.2,
           ephemeris="roman.txt", config=FinderConfig())

ev.fit("pspl")
if ev.scan().detected:            # the residual of the last fit
    ev.fit("fspl")                # starts from the PSPL fit
    ev.fit("parallax")
    if ev.scan("fspl").detected:  # a model by name
        ev.fit_atoms("fspl")

ev.fits["fspl"].chi2
ev.scans["pspl"].detections       # every candidate with score >= 10
ev.atoms["fspl"].features
ev.write_json("event.json")       # every step that was run, nothing else
ev.release()                      # drop the scan grids and model arrays
```

## Objects

* `Event(time, flux, ferr, *, kind="flux", ra_deg=None, dec_deg=None, ephemeris=None,
  config=FinderConfig())`. The sky position and the observer ephemeris are
  the event's, not the finder's: they leave `FinderConfig`, which keeps the
  search settings (grid, criteria, `detection_score`).
* `ev.fit(model, *, start=None, x0=None, tref=None)`, `model` in `"pspl"`,
  `"fspl"`, `"parallax"`, `"fspl_parallax"`. The fitters are built inside;
  they are not public. Start: `x0`; else the fit named by `start`; else the
  last fit. The shared parameters (`t0, tE, u0`, and `rho` or `piE` when both
  have them) are taken from it, the rest start at the fitter's defaults. With
  no fit yet: the PSPL FFT start. Parallax `tref`: the time of maximum flux
  unless given.
* `ev.scan(model=None)`: the residual of `model`, or of the last fit. A model
  not fitted yet is an error ("fit it first"). Uses the fit's stored
  `model_flux`, so the model is not evaluated again. Result: `detections`
  (every candidate with `score >= detection_score` that `candidate_criteria`
  accepts, by score), `detected`, `relocked`.
* `ev.fit_atoms(model=None)`: today's `robust_fit` for one model: named
  `params`, `fs`, `fb`, `chi2`, `J`, `host_identified`, `n_atoms`, `features`,
  `dchi2` (what the atoms explain over the plain fit).
  Atoms are accepted by the scan's rule (`score >= detection_score`).
* Each method returns its result and stores it in `ev.fits` / `ev.scans` /
  `ev.atoms` under the model name.
* `ev.fit_atoms` of a model not fitted yet is an error, as for `scan`.

## Order

Calls may come in any order; the results stay consistent:

* "The last fit" is the last `ev.fit` call (a relock counts as a refit of
  PSPL). `scan` and `fit_atoms` do not change it.
* Refitting a model drops its own scan and atoms (they belonged to the old
  fit). Other models' results stay: each is a fit of the same data.
* `write_json` writes what is stored, so a dropped result is not written.

## Full pipeline (one event)

```python
from jacscanomaly import Event, FinderConfig

ev = Event(time, flux, ferr, ra_deg=268.4, dec_deg=-29.2,
           ephemeris="roman.txt", config=FinderConfig())

ev.fit("pspl")
if ev.scan().detected:                      # relock inside
    ev.fit("fspl", start="pspl")
    fspl_scan = ev.scan()
    ev.fit("parallax", start="pspl")
    parallax_scan = ev.scan()

    if fspl_scan.detected and parallax_scan.detected:  # neither effect explains it
        ev.fit_atoms("pspl")
        ev.fit_atoms("fspl")
        ev.fit_atoms("parallax")

ev.write_json("event.json")
```

## Relock

The relock asks whether the PSPL fit sits on the anomaly instead of the
event. It runs inside `scan()` of the PSPL fit, only when the scan detects
something: if the host moves, `ev.fits["pspl"]` is replaced and scanned
again, `relocked = True`. Later fits start from the corrected PSPL. The user
never calls it.

## Memory

The data are stored once on the `Event`. A `Fit` keeps `model_flux`
(`n` floats); a `Scan` keeps its grid (the large part). `ev.release()` drops
both and keeps the records (`write_json` still works).

## Kept, not in the main docs

`Finder`, `Finder.run`, `AnomalyResult`, the fitters, `robust_fit` stay
importable for existing scripts, without a deprecation warning. `robust_fit` becomes the
engine of `Event.fit_atoms`.

## Assumptions

* One `Event` is one light curve, one band.
* A model name maps to one fitter built from the config; a user who needs
  another fitter uses the kept API.
