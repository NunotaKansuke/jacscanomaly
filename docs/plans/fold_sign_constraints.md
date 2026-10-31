# Fold and pair families: physical signs on the linear coefficients

## Problem (2371: 0006, 0794)

Both are labelled `caustic_exit_entry` (outside the caustic between the
crossings), but the fitted coefficients say the opposite:

* 0006: between the crossings the level is +1.16, the outer sides (inside the
  caustic) -0.08; the entry amplitude is 0.07. The pair is used as a bright
  box with a fold spike on its left edge.
* 0794 (one sharp peak with long wings): the between level is +1.14, the outer
  sides -0.37, the gap ~1e-4 d. The pair imitates an asymmetric single peak.

The linear coefficients of the fold / pair families are free in sign, so the
families also fit shapes no fold crossing can make.

## Physics (named assumption)

At a fold the source gains (entry) or loses (exit) one image pair. The pair
only adds flux:

1. the fold spike amplitude `A >= 0` (the `G` term);
2. the level inside the caustic is not below the level outside, at each
   crossing (`H` term / floors).

Both hold for any lens (the extra images have positive magnification);
limb darkening and fold curvature do not change the signs.

## Change

The linear part of `fold`, `caustic_entry_exit`, `caustic_exit_entry` is a
bounded least squares (NNLS with free columns split):

* fold: `A G(z) + c H(z)`, `A, c >= 0`.
* pair, the floor between the crossings as its values at `t1` and `t2`
  (`e1, e2`, linear in between; same span as floor + slope):
  * `in` (entry, exit; inside between): `A1, A2 >= 0`, `e1, e2 >= 0`
    (the host is outside);
  * `out` (exit, entry; outside between): outer level `c` free (the host has
    absorbed part of it), written `c - g1 box l1 - g2 box l2` with
    `g1 = c - e1 >= 0`, `g2 = c - e2 >= 0`, and `A1, A2 >= 0`.

Bump and double (atoms) keep free signs: they describe dips too.

Expected: 0006 and 0794 lose the illegal pair; 0006 -> two bumps or an
entry-exit pair, 0794 -> bump (or fold). Correct pairs are unchanged (their
coefficients already have these signs).

## Check

* features-only rerun on the saved PSPL host + atoms of all 2371 events
  (no refit): kind changes listed, every change looked at by class.
* unit tests: an illegal-sign pair is not selected; a clean entry-exit pair is
  still found.

## 0969, 0595 (2371): a pair without crossings

Both are `caustic_exit_entry` with no fold spike: 0969 fits a constant offset
(`c ~ -7`) and no steps; 0595 (fspl) a broad dip with one weak spike.

* `out` drops the free outer level: the outer sides are inside the caustic,
  and the host there is the level (0); the floor between is `-g_k`, `g >= 0`.
  `in` and `out` share the signs (all four columns `>= 0`).
* A crossing is claimed by its spike (named): each `G` column alone, the
  nonlinear parameters held, must gain `dchi2_min` in chi2, or the fold /
  pair is not fitted. Steps without spikes are a dip or a bump.
* Rejected: a threshold relative to `dchi2` (`delta * ref`): a textbook
  pair's spike gains only ~0.5% of `dchi2` (the steps carry the rest).

Features-only rerun, 330 caustic-labelled events (fspl): `caustic_exit_entry`
119 -> 53, `caustic_entry_exit` 65 -> 61, folds unchanged; 0969 -> dip+bump.
0595 fspl stays `exit_entry` (exit spike gains ~84 > 25): left to the rerun on
its (unrelocked) host.

## 0595 (fix7): resolved crossings

After the spike rule 0595 stayed `caustic_exit_entry` (f_pair 0.994 vs
f_double 0.993): exit at 8580.16 with FWHM 8.47 d (`dt` 2.96), entry at
8587.56, 7.40 d apart; the data are a broad dip, then a rise and a bump.
Named rule: a pair's crossings must be resolved, each crossing's FWHM
shorter than the time between the two peaks (the straight fold assumes a
source small against the caustic, so a crossing short against the chord;
`nan` width = not resolved). 329 caustic-candidate events: 110 -> 107 pairs;
284 -> bump (one unresolved spike), 1213 -> dip, 1865 (a sub-pair 0.07 d
apart inside a complex caustic) -> bump+bump; 0595 -> dip+bump.

## 0969, 0979: the host inside the fold, and the local trend

**Rejected.** On 1949 events the host-inside fold with a trend made ~136
windows newly ``fold``, most of them smooth anomalies the trend let a fold
absorb. The caustic decision moved to the edge sharpness of the atoms model
(``caustic_sharpness.md``); the text below is kept as the record.

Both are single fold crossings with the host's level *inside* the caustic: a
spike that falls (0979, exit) or rises from (0969, entry) a demagnified
trough below the host, the trough recovering over days. The fold family only
had the host outside (``A G + c H``, outer level 0), so the trough was a
second atom and the window a double (0979 ``f_fold`` 0.37, 0969 0.65).

Change:

1. ``h``, the host's side: ``A G(z) + c (H(z) - h)``, ``h = 0`` (host
   outside, inside ``c`` above) or ``1`` (host inside, outside ``c`` below);
   the better of the two per crossing direction. Same signs (``A, c >= 0``),
   same linear count; the pair already has both (``in`` / ``out``).
2. ``b (t - tc)``, ``b`` free: the other images' magnification is not constant
   across a window of days; the fold formula of Albrow et al. (1999) / Gaudi &
   Gould (2002) carries this linear term. Zero at ``tc``, so the host (0)
   still anchors the levels at the crossing. One more linear coefficient (3
   against the bump's 2); the crossing widths ignore it (flat over a spike).

``h`` alone: 0969 ``f_fold`` 0.77, 0979 0.64 (doubles still win). With the
trend: 0969 0.93 → ``fold_entry``, 0979 0.85 → ``fold_exit``.
