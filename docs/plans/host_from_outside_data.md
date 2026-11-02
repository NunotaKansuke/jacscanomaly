# The host is what the data outside the anomalies say

## Back to the model

The atoms plan has one model and one rule:

```text
F(t) = fs A_host(t; p) + fb + sum_k atom_k(t)        J = chi2 + (n_host + 2 + k_atom n_atoms) ln n
```

Everything else in `robust_fit` gets a start into the right basin: the relock,
the `base`/`off` choice, the host held until the first atom, the `teff` cap.
The one exception is the `"fixed"` start plus `free_host_margin`. It encodes a
second rule: "keep the plain host unless a moved one wins by a margin".

Every flagged event that the relock tie fix did not repair breaks on that
second rule, or on a host that the model leaves unidentified. The 21 flagged
events were rerun with the tie fix and the display fix.

| failure | events | cause |
|---|---|---|
| the plain host is held, but it is pulled by the anomaly | 0242, 0815, 0498, 1391, 0834, 0687, 0181 | `"fixed"` holds `fitter.fit(...)` on all samples, which fits through the anomaly. It wins by the margin with `J` worse by 59–149. |
| spike host (`fs ~ 0`, `tE` ~ 1000 d) | 0404 (and 2259 before the tie fix) | The atoms build the event and the "host" is a needle on the anomaly. `J` can prefer it: 2259's spike scored 250 below its real host. |
| `fs, fb` of opposite sign, `fs ~ -fb` | 128 / 2332 fits | `fs, fb` are unconstrained, so a large-`u0` host with `fs (A-1)` tiny becomes a constant. |
| whole light curve anomalous | 1202, 0963, 1904, 1827, … | No host is identifiable outside the atoms. This should be flagged, not fixed. |

## What "host" means

The host is the single-lens event the samples outside the anomalies
determine. That has three consequences, one per change below.

### 1. `"fixed"` holds the host fitted outside the atom windows

* Today `"fixed"` holds the host of the plain fit on all samples.
* New: take the free start with the lowest `J`. Fit the host on the samples
  outside its atom windows (`fitter.fit(..., mask=inside)`, seeded from that
  start's host). Hold that host and grow the atoms again (`fix_host=True`).
* The margin keeps its role: the atoms' `1/|tau|` wings must not move a host
  that the outside data already fix. The held host is now one that the atoms
  did not pull, and the anomaly did not pull either.
* One pass, with no iteration between the windows and the host.

### 2. A host must be detected outside its windows

* A start's result counts only if its host beats a constant flux on the
  samples outside its windows. The test is `_host_detected` from `relock.py`:
  `chi2 + p ln n` versus `chi2_flat + ln n`, scored with the host alone
  (`fs A + fb`).
* The spike host fails this test: outside the windows `fs A + fb` is about
  `fb`.
* If no start passes, the lowest `J` is kept and
  `ModelFit.host_identified = False`. That flags the whole-anomaly class. It is
  the same quantity as the `F` used on the review pages, as a test instead of
  a threshold.

### 3. `fs >= 0`, `fb >= 0`

* These are fluxes. Only `fs` and `fb` are bounded; the atom amplitudes stay
  free.
* With two bounds the exact solution is a small enumeration of active sets.
  Solve the unconstrained problem. If `fs` or `fb` is negative, solve it again
  with that one (then both) fixed to 0, and keep the feasible solution with
  the lowest `chi2`.
* Scope: `robust._linear` only. That covers the atoms fit, the published
  `fs, fb` and the host-only score in 2. The single-lens fitters (the starts,
  the relock) stay unconstrained for now: they only seed, and their `fs, fb`
  are not published.

## Not changed

The relock (including the tie fix), the `base`/`off` choice, the `teff` cap
and its exemption, and `k_atom`.

## Decided after the reruns

* The margin is the host's own price, `n_host ln n` (`free_host_margin=None`),
  not a tuned number. It ranks the starts of one model only.
* Models are compared by `J` alone (identified hosts first). A per-model
  margin in the model choice biased it toward parallax.
* Needle hosts (`u0 -> 0`, `tE` beyond the data, 0404 and 1202) are not fixed.
  Inside the Einstein ring `fs A ~ fs tE / |t - t0|` is the atoms' `h`
  template, so the data do not separate host from anomaly there. A rule
  rejecting hosts with `u < 1` at both data edges was tried and dropped: the
  `u = 1` cut is a convention, and it moved 1202 to a start worse by 23000 in
  `J`. These events are flagged instead (below).

## Screening (no thresholds in the fit)

`aethra/tools/rmdc_physical_stage.py` writes per model `host_identified`,
`J_excess`, `tE_over_span`, `signal_host`, `signal_atoms`;
`model_check_campaign/screening/build_db.py` loads them into SQLite with
`atoms_over_host` and `tE_over_pspl`. The event pages show the same numbers.

## Full set (2371 events, against the previous run)

* 616 selections changed. Against the simulated `tE`: 274 closer, 80 farther,
  1973 within 10%; `tE` off by more than x2: 173 -> 79.
* Of the 80 farther, 15 are a parallax `tE` that runs away from its PSPL
  (`tE_over_pspl > 3` catches 9), and most of the rest are small.
* 105 events carry a screen flag.

## Assumptions

* The atom windows (`|t - t0| <= coeff teff`, `coeff = 3`) bound where the
  anomalies act. Outside them the data are host + noise.
* `fb >= 0` assumes the baseline flux is not over-subtracted. For real
  difference photometry this may need relaxing; the simulated set has none.

## Verification

1. Package tests pass, plus new tests:
   * a synthetic host + bump: `"fixed"` recovers the true host;
   * a spike-only light curve: `host_identified` is `False`;
   * `fs, fb >= 0` on a blended, large-`u0` case.
2. The 21 flagged events: figures before and after, side by side.
3. No regression on the events where the margin was set: the 23 eye-judged
   events and the 188 margin-500 events, side by side.
4. The full 2371-event set: the number of selection changes, and the
   `host_identified = False` list against the user's whole-anomaly picks
   (34, 86, 89, 110 on the R page) and good events (30, 31, 93, 102).
