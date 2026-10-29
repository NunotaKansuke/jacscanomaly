# Robust host fit with anomaly atoms (API consolidation)

## Context

The masking experiments (`masked_refit`, `signal_mask` windows, `search_anomalies`,
`iterate_mask`, `agreed_windows`) are superseded by the atoms method, accepted on
the 2371-event campaign (2026-10-29, `model_check_campaign/fixtest/atoms_models.py`):
no mask; the host and the scan's own templates ("atoms") are fitted jointly to every
sample. The method lives only in a campaign script today. This plan moves it into
the package as the one robust-fit API and deletes the experiments.

## Model (what is fitted, and the assumptions)

```text
F(t) = fs A_host(t; p) + fb + sum_k [a_k h(tau_k) + b_k l(tau_k)],   tau_k = (t - t0_k) / teff_k
```

* `h`, `l`: the scan's high/low-magnification templates, shifted to zero at
  `|tau| = 3` and zero outside (compact support).
* `p` natural host parameters: `(t0, tE, u0[, log rho][, piEN, piEE])`.
  `fs, fb, a_k, b_k` linear (variable projection); `p` by bounded least squares.
* Atom positions/widths come from the scan of the current residual (host fixed).
* Price per atom `4 ln n` (BIC, 4 parameters). Model/start comparison:
  `J = chi2 + (n_host + 2 + 4 k) ln n`.
* Atom acceptance: scan score >= 10, criteria accept, and `teff <= 0.25 tE_start`
  unless the atom's centre lies in the segment `relock_host` moved the host off
  (that segment is already shown not to be host).
* The start host stays fixed (fs, fb only) until the first atom is in; with no atom
  at the end the host is refit.

## Public API after the change

New module `robust.py`:

| name | role |
|---|---|
| `AtomConfig` | `coeff=3, min_score=10, max_teff_fraction=0.25, max_atoms=15, k_atom=4` |
| `atom_basis(t, t0, teff, coeff)` | the two compact templates |
| `fit_atoms(fitter, finder, time, flux, ferr, p0, *, anomaly=None, config)` -> `AtomFit` | greedy add + backward prune from one start |
| `robust_fit(fitters, finder, time, flux, ferr, p_scan, *, config)` -> `RobustFit` | relock, starts (scan/nested/base) per model, J selection |
| `atom_windows(atoms, coeff)` | merged atom supports (feature windows) |
| `Atom`, `AtomFit`, `ModelFit`, `RobustFit` | frozen dataclasses: host params, sigma, fs, fb, atoms, chi2, J, start, windows, log |

Fitters get one public method `magnification_model(time)` -> `p -> A(time; p)` in
seed parameters (PSPL, FSPL, parallax; the parallax evaluator is built once),
replacing the campaign's use of `_magnification`, `_raw_seed` and a hand-built
`ParallaxEvaluator`.

Kept: `Finder`, fitters, `Candidate`, `measure_features` (candidates only),
`relock_host` (its masked-J test keeps `signal_mask` and `host_detected` as private
helpers of `relock.py`).

Removed: `masked_refit.py` (`MaskedRefit`, `masked_refit`, `window_refit`,
`window_mask`, `constant_fit`, public `host_detected`), `anomaly_search.py`,
`iterative_mask.py`, `agreed_windows`, public `signal_mask`/`SignalMask`, and their tests.

## Consumers

* `fixtest/atoms_models.py` -> thin wrapper over `robust_fit` (I/O and batching only).
* aethra `rmdc_physical_stage.post_scan` and `run_rmdc_anomaly_scan_preview`
  (`search_anomalies`, `MaskedRefit`, `window_mask`, `constant_fit`, `host_detected`)
  -> `robust_fit`; the page's model-check data come from `RobustFit`. Deferred:
  these tools are outside git and run against their own src snapshot
  (`JACSCANOMALY_SRC`), so they keep working there until they are moved.

## Verification

1. Package tests pass (new `tests/test_robust.py`: atom basis support, linear
   solution on a synthetic host + bump, prune removes a useless atom, relock
   exemption).
2. Wrapper reproduces the campaign script (with the relock exemption) on ~30 events
   incl. 0015, 0956, 1111, 1285, 1767: same selected model, J within 1e-6 relative.

## Out of scope

Very strong caustics; 0956 (spike lock with no relock); C++ hot paths.
