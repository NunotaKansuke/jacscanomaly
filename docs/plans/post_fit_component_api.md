# Post-fit component API plan

## Context

After the single-lens fitter unification, the layer that runs *after* the
single-lens fit is the remaining tangle. Today it is reached through nested
wrappers:

```text
Finder.run_anomaly_pipeline   (286 lines, AnomalyPipelineConfig)
  -> Finder.run_effect_aware  (368 lines, effect_routing / effect_aware)
    -> PlanetSignalExtractor.run (307 lines, PlanetSignalConfig: 55 fields)
```

The same jobs are implemented several times:

| job | implementations |
| --- | --- |
| refit with some points excluded | `planet_signal._fit_masked_* / _fit_weighted_* / _fit_flat_*`, `contamination.robust_refine_with_fitter`, `singlelens_fallback` |
| decide which points are signal | `beam_interval` / `mask` / `robust` modes, `LocalTrendChangeMasker`, `residual_decomposition.decompose_binned_residual`, contamination DP |
| where the anomaly is | `Finder._finder_support_from_scan`, `AnomalyCandidate`, `EffectCandidate`, scan clusters |

Each wrapper reads the scan result directly, so no component can run on
information that did not come out of the scan, and no component can be
tuned without going through the pipeline config.

## Design principle

**Each component runs on its own. Components are linked only by passing
explicit information.**

* The scan *produces* candidates; it is not *required* to obtain one. A
  candidate written by hand (e.g. a window the scan rejected) drives every
  downstream component in exactly the same way.
* A single-lens fit result is the carrier of the data and the model
  (`time, flux, ferr, model_flux, residual = flux - model_flux, fs, fb`).
  Any fit works, including `fitter.evaluate_fixed(...)` at arbitrary
  parameters.
* No component calls another component internally. Iteration
  (mask -> refit -> re-mask) is written by the user, not hidden in a config.

## Goals

1. `mask=` on every single-lens fitter.
2. A scan-independent `Candidate` type.
3. `signal_mask(fit, candidates)`: the residual segment that pays for its own exclusion.
4. `measure_features(fit, candidates)` classifying each window's residual
   structure into **bump / dip / caustic** (feature detector + fold detector).
5. Delete the wrappers and the duplicated masking / refit paths.

## Non-goals

* `compute_effect_morphology_scores` / effect diagnostics (`effect_detection`,
  `exact_probe`, `parallax_score_test`). Kept as-is, simply no longer wired
  into `Finder`. Separate topic.
* Changing the scan (`GridRunner`, scoring, `CandidateCriteria`).
* Changing the masking / feature / fold algorithms themselves. This plan only
  moves them behind clean inputs; their numerics must be unchanged so they can
  be optimized independently later.
* aethra/RMDC scripts. The running job imports the main checkout; it is not
  touched. aethra only uses `Finder.run`, `PSPLFitter`,
  `compute_effect_morphology_scores` and `plot`, which stay.

## Components

### 1. Fitter `mask=`

```python
fitter.fit(time, flux, ferr, p0=None, *, pspl_params=None, mask=None)
fitter.evaluate_fixed(time, flux, ferr, p0, *, mask=None)
fitter.fit_fixed_model(time, flux, ferr, p0, *, model_kind=None, mask=None)
```

For `PSPLFitter`, `FSPLFitter`, `PSPLParallaxFitter`, `FSPLParallaxFitter`.

* `mask`: boolean array, `True` = **excluded** (numpy.ma convention; matches a
  signal mask, so `fitter.fit(..., mask=sm.mask)` reads naturally).
* Profiled `(fs, fb)`, LM residuals, Jacobian, `chi2`, `chi2_dof`
  (dof = kept points − n_params − 2) and the automatic seeds / multistart use
  kept points only.
* `model_flux` and `residual` in the returned `SingleLensFitResult` are
  evaluated on the **full** data, so the excluded signal is visible in the
  residual. The result gains a `mask` field (`None` or the bool array).
* Implementation: select kept points once at the top of `fit` and pass the
  subset to the existing code path, then one full-data `evaluate` at the end.
  No new optimizer path. Parallax projector is built from the full time array
  (so `tref` does not move when points are masked).

This replaces `_fit_masked_single_lens_and_evaluate_full`, `_fit_weighted_*`,
`_fit_flat_*`, `robust_refine_with_fitter` and `singlelens_fallback`.

### 2. `Candidate`

```python
@dataclass(frozen=True)
class Candidate:
    t_start: float
    t_end: float
    t0: float | None = None      # anchor hint only; None -> window centre
    label: str | None = None

    @classmethod
    def from_t0_teff(cls, t0, teff, *, teff_coeff=3.0): ...
    @classmethod
    def from_scan(cls, scored_candidate, *, teff_coeff=3.0): ...  # BestCandidate / ScoredCandidate

    def support(self, time) -> np.ndarray: ...  # t_start <= t <= t_end

candidates_from_scan(result: AnomalyResult, *, which="best" | "all",
                     teff_coeff=3.0) -> list[Candidate]
```

* The only thing downstream needs is *where* to look, so the window is the
  stored quantity; `(t0, teff)` is just one way to build it
  (`|t − t0| ≤ teff_coeff·|teff|`, same rule as `_finder_support_from_scan`).
* `from_scan` copies numbers; nothing keeps a reference to the scan, its
  score, or whether it passed criteria.
* Components accept `Candidate | Sequence[Candidate]`.

Replaces `AnomalyCandidate`, `finder_support`, `_finder_support_from_scan`.

### 3. `signal_mask`

```python
signal_mask(fit, candidates, *, penalty=2.0) -> SignalMask

@dataclass(frozen=True)
class SignalMask:
    mask: np.ndarray                 # True = signal point (union)
    per_candidate: tuple[np.ndarray, ...]
    candidates: tuple[Candidate, ...]
    gains: tuple[float, ...]         # sum(z^2 - penalty) over each segment
    penalty: float
```

* Masking a point frees one parameter, so the mask minimises
  `J = chi2_kept + penalty * n_masked` (penalty 2 = AIC) at the current fit:
  per candidate, the contiguous segment maximising `sum(z_i^2 - penalty)`
  that overlaps the window, with a positive gain also inside the window, and
  not crossing a data gap longer than the window.
* Replaces `LocalTrendChangeMasker` (2026-10-27): its fixed 7-point slope
  regressions missed smooth anomaly wings, and a running-sum mask tried
  before it needed a hand-set width and crept on central anomalies.
* No refit inside. A masked refit lowers the same `J`, so the user's loop
  converges:

```python
fit  = fitter.fit(t, f, e)
scan = Finder(cfg).run(t, f, e)               # or: skip and write candidates
cands = candidates_from_scan(scan, which="best")
sm   = signal_mask(fit, cands)
fit2 = fitter.fit(t, f, e, p0=fit.params, mask=sm.mask)
feat = measure_features(fit2, cands)       # bumps / dips / caustics
```

Replaces the `beam_interval` / `mask` / `robust` modes,
`residual_decomposition`, and the contamination DP.

### 4. `measure_features` (bump / dip / caustic)

```python
measure_features(fit, signal: SignalMask | CandidateLike, *,
                 config=FeatureConfig(),
                 require_closed_peaks=False) -> FeatureResult

@dataclass(frozen=True)
class Feature:
    kind: Literal["bump", "dip", "caustic"]
    candidate: Candidate         # which window it came from
    time: float                  # extremum
    t_start: float
    t_end: float
    timescale: float
    strength: float              # sigma
    signed_z: float
    residual: float
    fractional_deviation: float
    magnification_ratio: float
    snr: float                   # lobe sum(z)/sqrt(n), the detection statistic
    caustic: CausticDetail | None   # only for kind == "caustic"

@dataclass(frozen=True)
class CausticDetail:
    orientation: Literal["entry", "exit"]  # fast-rise/slow-decay = entry
    peak_time: float
    left_width: float
    right_width: float
    delta_chi2: float            # chi2(sym Lorentzian) - chi2(split Lorentzian)
    width_ratio: float           # property, max(w)/min(w)

@dataclass(frozen=True)
class FeatureResult:
    features: tuple[Feature, ...]   # time-ordered
    bumps / dips / caustics          # filtered views
```

One call classifies the residual structure in each candidate window into
three kinds. Internally, in this order (algorithms unchanged):

1. `PlanetFeatureDetector.run_arrays(...)` on the union of candidate supports
   -> peaks and dips (peaks are renamed **bump**).
2. Each bump gets the asymmetry test (split vs symmetric Lorentzian on raw
   z, see revision 2); a significant one becomes `kind="caustic"`.

Revision (2026-10-27), from the six 2371 events of the signal_mask study:

* **Support.** With a `SignalMask`, each candidate is measured on its window
  united with its signal segment, padded by `smooth_points` samples. The
  segment reached beyond the window on 1379 (the pre-dip started 0.2 d
  before the window, so the dip never closed and was dropped) and on 2502
  (the bump was still +4 sigma at the window edge, so its prominence floor
  was 4.1 and it was dropped).
* **Detection.** `min_abs_z` (smoothed height >= 5) is replaced by
  `min_snr`: the lobe's `sum(z)/sqrt(n) >= 5`, i.e. a constant offset over
  the lobe gains `chi2 >= 25`. The smoothed height is the S/N only for a
  7-point-wide feature; 2502's bump (about 100 points at 4-5 sigma, lobe
  S/N 26) peaked at 4.9 smoothed. Null check: about 7000 lobes outside the
  mask per event, maximum S/N 3.6-3.9.
* **Duration.** The absolute 3-sigma floor on the duration threshold is
  dropped (half prominence only); a broad feature with a smoothed height
  below 3 otherwise had zero duration. The 3-sigma level survives only as
  `noise_floor_z` in the caustic interior test (was `duration_min_abs_z`).

Revision 2 (2026-10-27): caustic = asymmetric bump. The entry/exit pairing
(`_FoldCausticPairing`, `FoldCausticEdge`, `interior`, `interval_mask`,
`noise_floor_z`, `min_asymmetry_cadences`) is removed. A caustic includes a
near-caustic passage, so a crossing pair is not required; on 1705 (source
between two triangular caustics, no crossing in the static RTModel best) both
peaks are caustics. Detection is a nested likelihood-ratio test: split vs
symmetric Lorentzian (+ local linear baseline) on raw z, `delta_chi2 >= 25`
(chi2_1, 5 sigma) and width ratio >= 1.5. The 7-point-smoothed half-widths
used before were as wide as the peaks and symmetrised them. The fit window
is cut half way to a neighbouring bump *or dip*: with only bumps as
neighbours, the slope into an adjacent dip made symmetric bumps look
asymmetric (1379, 2502: delta_chi2 199/156 with the dip, 1.6/2.1 without).
A finite-source fold template (x^-1/2 (*) disk) was also tried; it lost to
the symmetric model on non-fold shapes, so it is not used. On the six
events: 1705 exit (287) + entry (196); every other bump stays a bump.

`FeatureConfig` = the current `PlanetFeatureConfig` fields + the asymmetry
test fields; the three unused `dip_bracket_*` fields are dropped.

### 6. `Finder`

`Finder.run` stays the scan entry point (fit + scan + scoring -> `AnomalyResult`),
unchanged in behaviour. Everything after it moves out.

## Removals

Modules deleted:

* `anomaly_pipeline.py`, `effect_aware.py`
* `singlelens_fallback.py`, `contamination.py` (+ `_contamination_dp` C++
  extension), `residual_decomposition.py`, `planet_signal.py`
* `tools/robust_fspl_parallax_benchmark.py`

`effect_routing.py` is kept with `effect_detection` (its tests use
`route_candidate`). The parallax-fitter factory `make_effect_fitter` moves to
`parallax_backend.parallax_fitter_from_config`, which `Finder` uses.
`tools/rges_anomaly_pipeline.py::_run_one` still calls the removed
`run_anomaly_pipeline` and needs a rewrite on the new components.

`Finder` methods deleted: `run_anomaly_pipeline`, `run_effect_aware`,
`robust_fallback`, `detect_effects`, `run_template_free` (and their private helpers).

`planet_signal.py` (4002 lines) is reduced to the pieces above:
`LocalTrendChangeMasker`, `PlanetFeatureDetector` (+ config/result types),
`FoldCausticDetector` (+ types). `PlanetSignalExtractor`, `PlanetSignalConfig`,
`PlanetSignalResult` and all multi-mode / refit machinery are deleted. New
public functions live in `candidate.py`, `signal_mask.py`, `features.py`
(split out of `planet_signal.py`).

Kept but no longer called by `Finder`: `effect_detection`, `exact_probe`,
`template_free` (independent tools; out of scope here).

`__init__` exports shrink accordingly (~100 -> target ~50 names).

## Tests

* Fitter `mask`: `mask=None` reproduces current results bit-for-bit; masking
  an injected bump recovers the true PSPL params and the bump in `residual`;
  `chi2_dof` uses kept dof; all four fitters; mask all-False == None.
* `Candidate`: `from_t0_teff` window rule equals old `_finder_support_from_scan`;
  `from_scan` works on a candidate that failed criteria.
* `signal_mask` / `measure_features`: regression
  against current `LocalTrendChangeMasker.run_arrays`, `_PlanetFeatureDetector`,
  `FoldCausticDetector` on the existing fixtures in `test_planet_signal.py`
  (same masks / peaks / dips / pairs; caustic absorbs exactly the paired
  peaks and interior features); hand-written candidate with no scan.
* Delete `test_anomaly_pipeline.py`, `test_effect_aware_pipeline.py`,
  `test_contamination.py`, `test_residual_decomposition.py`,
  `test_robust_benchmark.py`; prune `test_planet_signal.py` to the kept parts.
* Docs: replace the pipeline workflow pages with the short composition above.

## Order of work

1. Fitter `mask=` + tests (independent, lands first).
2. `Candidate` + `signal_mask` + `measure_features`,
   with regression tests against the current implementations *before* deletion.
3. Deletions, `__init__` / docs / tests cleanup.
4. Smoke run on a few RMDC events from `/rogue1_8/nunota/aethra_rmdc_v2/out`
   (scratch under `/rogue1_8`), thread-pinned.
