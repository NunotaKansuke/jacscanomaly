# Atom width cap in the wings; no atom may eat the host's own peak

Two failures in the 2371 run (features snapshot), both inside `fit_atoms`.

## 0037: a broad dip in the wing is refused

The scan finds a dip at 8359.9 (`teff` 3.15 d, dchi2 564, score 477),
1.6 `tE` before the peak. The cap `teff <= 0.25 tE` (2.1 d here) refuses it
in every start; the atoms then go to the peak (PSPL) or parallax takes
`piE ~ 7.5` to bend the host through the dip. With the cap at 0.5 every model
takes the dip and the PSPL `J` drops by ~860.

Why the cap exists: an atom as wide as the host can replace the host's
curvature. That is a statement about the *local* timescale of the host, not
its global `tE`. The host changes on the scale of `u`:
`dA/dt ~ A' (u) du/dt`, with `u(t) = sqrt(u0^2 + tau^2)`, so near the peak
the scale is `tE * u0` and in the wing `tE * |tau|`.

Change: `max_teff = f * tE * max(1, u(t0_atom))` with `(t0, tE, u0)` of the
start host and `f = max_teff_fraction` (0.25, unchanged).

* `u <= 1` (within the Einstein ring crossing): the cap is unchanged, so no
  current near-peak decision moves.
* `u > 1`: the cap grows with the distance from the peak, where a wide atom
  cannot imitate the peak. 0037: `u = 1.65`, cap 3.44 d >= 3.15 d.

Assumption: the start host's `(t0, tE, u0)` stand for the host's timescale
(parallax / FSPL: their first three parameters, as the current cap already
assumes).

## 0042: the host slides onto the spike and an atom cancels it

Relock works here: the H1 host is `t0 9814.47, tE 15.8, u0 0.13`, off the
spike at 9814.10 (truth `tE 3.2, u0 -1.18, fs 0.1`: a weak host). But in
`fit_atoms` the host refit, once the spike atom is in, moves the host peak
*onto* the spike (`tE ~ 3500, u0 ~ 0, fs ~ 0`) and a *negative* atom of
about -25 flux cancels the host's peak. Its `J` is 29 lower than the honest
host + positive spike atom: the host's `1/|t - t0|` wings fit the spike's
wings. The same happens from the `base` start (the plain fit sits on the
spike).

Physical bound (named assumption): a planetary perturbation can at most
remove the minor image, `A_- = (A - 1) / 2` for a point lens. So the host +
anomaly flux cannot fall below `fb + fs (A + 1) / 2`, i.e. the atoms'
deficit is bounded,

    -atoms(t) <= fs (A(t) - 1) / 2 + k ferr(t),   k = 3.

`k ferr` absorbs the atoms' imperfect shape where a real minor-image dip
reaches the bound. Major-image dips and finite-source / binary-source
deficits are assumed smaller than this (the bound is loose for them).

Use in `fit_atoms`:

1. a host refit (`_fit_host` with atoms) whose result breaks the bound is not
   taken; the trial is redone with the host held at its current parameters
   (`fs, fb` and atom amplitudes still linear);
2. a run whose final model still breaks the bound (e.g. the start host
   itself sits on the spike, so the atom must be negative) is marked
   `violates`; `robust_fit` ranks starts by `(violates, J)` so any valid
   start wins, and keeps the best invalid one only when none is valid.

The bound is checked on the data times (the atoms only act there).

## Check

* 0037: all models take the 8360 dip.
* 0042: host off the spike (peak not inside the spike's atom support),
  positive spike atom.
* The 8 earlier examples (1000, 1021, 1896, 860, 1583, 529, 1625, 2302):
  unchanged selected model and features, or a change that is explained.
* Unit tests: the cap in the wing; a host-on-spike start is not selected
  when an honest start exists.

## Result (2371, 0037 / 0042)

* 0037: the cap change works; PSPL and parallax take the 8360 dip
  (PSPL `J` -857).
* 0042: the deficit bound did not change anything. The final spike host is
  `tE ~ 3500, u0 ~ 1e-6, fs ~ 3e-6`, whose own `1/|t - t0|` peak is the
  spike; its atoms are small and keep the bound. It came from the `base`
  start, the plain fit, which sits on the spike, while the relocked `scan`
  start ends at the honest host (`tE 17.8, u0 0.10`, `J` 29 higher).

  The relock had already rejected the plain fit (H0) for the masked H1
  host, so `base` contradicted it. Change: when relocked, `base` is the
  plain fit under H1 — `fitter.fit(pspl_params=p_relocked, mask=relock.mask)`.
  Then both PSPL starts end at the honest host (`J` 40492.7), FSPL too;
  parallax ends at `tE ~ 1000, u0 ~ -0.009` with positive spike atoms
  (`J` +17, not selected).
