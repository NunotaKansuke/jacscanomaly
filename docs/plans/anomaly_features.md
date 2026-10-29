# Anomaly features: shape, strength, centre, timescale

Goal: for every anomaly window of a `robust_fit` model, return four numbers
that a person reading the light curve would agree with -- including
"caustic" for fold crossings, which `features.py` never labels today
(1021 and 1583 come out as bump/dip/bump and bump/bump).

## What we have

* the host `F_host(t)` (fixed, from the selected `ModelFit`),
* the atoms `m_atoms(t)` (a free, smooth description of the anomaly),
* the windows (merged atom supports).

Everything below is measured on `r = flux - F_host` with the host fixed, so
all models share one reference and their chi2 values are comparable.

## Strength: chi2, not S/N

Per window `w`

    dchi2_w = chi2_w(host only) - chi2_w(host + atoms)

(already in the fit). It is the evidence the anomaly carries and adds up over
windows. S/N and fractional deviation stay as secondary columns.

## Shape: which family explains the dchi2

Fit a few physically named families to `r` inside the window (amplitudes
linear, a few nonlinear times) and report the explained fraction

    f_k = (chi2_w(host only) - chi2_w(host + family k)) / dchi2_w .

`shape = argmax_k f_k`; `"complex"` when `max f_k < f_min` (0.8 to start).
All `f_k` are returned.

Why fractions: an exact-null test ("is it a symmetric bump?") rejects every
real anomaly at high S/N, so it labels S/N, not shape (the reason
`features.py` has no caustic label). A fraction compares two imperfect
descriptions of the same signal and does not grow with S/N.

Families (minimal set):

| family | model of `r` | nonlinear | linear |
|---|---|---|---|
| bump / dip | one atom `a h + b l` | `t0, teff` | `a, b` (sign of peak -> bump or dip) |
| fold pair (caustic) | `A1 G(+(t-t1)/dt1)/sqrt(dt1) + A2 G(-(t-t2)/dt2)/sqrt(dt2) + c 1[t1<t<t2]` | `t1, dt1, t2, dt2` | `A1, A2, c` |
| single fold | first or second term above alone, plus `c` on its inner side | `tc, dt` | `A, c` |

`G` is the uniform-source fold profile (Schneider & Weiss 1987; Albrow et
al. 1999; Gaudi & Gould 2002): 0 outside, `~1/sqrt(z)` inside, smoothed over
one source radius. Tabulated once, interpolated in C++. The box term `c`
absorbs the interior floor *and* whatever the host already took (1021's
interior sits below the host, so `c < 0` there).

Assumptions, named:
1. Host fixed. The families describe the residual the host left; a host that
   absorbed the anomaly (1904: no atoms at all) is not repaired here.
2. Straight fold, uniform source, constant source velocity across a
   crossing. Limb darkening and fold curvature are absorbed by the atoms'
   residual share (`1 - f`), not modelled.
3. Cusp approaches look like bumps; we do not separate them (not
   identifiable from shape alone).

## Centre and timescale

| shape | centre | timescale |
|---|---|---|
| bump / dip | `t0` of the atom | `teff` (also FWHM of `r`) |
| fold pair | `(t1 + t2) / 2` (and `t1, t2`) | `t2 - t1` (and `dt1, dt2`, the source crossing times) |
| single fold | `tc` | `dt` |

## Speed

The atoms curve is noiseless, so it gives the starts:
* fold starts: the K = 5 steepest rises and K = 5 steepest drops of
  `m_atoms(t)` in the window; pairs with `t1 < t2` (<= 25) plus singles;
  `dt` from the edge width.
* bump start: the strongest atom.

Each start: linear amplitudes by a small normal-equation solve (3x3), then
`least_squares` on the <= 4 times with the amplitudes projected out. The
window holds ~10^2 to 10^3 points: milliseconds per window in numpy; the
fold profile and the projected residual move to `_cpp_grid` only if the
profile says so.

## Output

`AnomalyFeature(window, shape, dchi2, f: dict[str, float], center,
timescale, times: dict)` per window, on `ModelFit.features`, replacing the
extremum detector's bump/dip list as the primary output (the extremum list
can stay as a detail field).

## Check

* 1021, 1583: fold pair, `f_pair >~ 0.95`, `f_bump` well below.
* A set of clean bumps (planet bumps from the 2327 run): bump, `f_bump >~ 0.9`.
* Distribution of `shape` vs truth `rho`/`u0`/caustic-crossing flag on the
  2327 run; look at the disagreements, do not tune to truth.

## Prototype findings (fixtest/features/proto.py, 43 events)

* The fold pair needs both orientations: `in` (entry then exit, interior
  above the host: 1583) and `out` (exit then entry, the source *leaves* the
  caustic in between: 1021, 1896). For `out` the outer sides are inside the
  caustic and get their own free floor (`1 - box`).
* With 5 linear terms the `out` pair also fits smooth bump+dip wiggles
  (811, 1841, 1590). Fix: a "double" family (two atoms) of the same
  complexity; a caustic is claimed only if it beats the double. Families
  are compared in two complexity classes, {bump, fold} and {double, pair}.
* Rule: take the simpler class unless the richer one explains a further
  `DELTA = 0.05` of dchi2 *and* `>= 25` in chi2 (the edge must be resolved
  for a caustic claim); `complex` below `F_MIN = 0.8`. All three numbers are
  starting guesses.
* Starts: steepest rises/drops of the atoms curve at >= 5 samples apart,
  each with dt in {0.3, 1, 3, 10} x cadence; all starts are screened by the
  linear solve alone and only the best 3 go to `least_squares`.
  Result: 0.03-0.6 s per window in numpy (median 0.14 s), small next to
  the fit itself; no C++ needed yet.
* Shapes on the 43: bump 16, dip 15, two-atom 8, caustic 4 (1021 out,
  1583 in, 1896 out, 0860 out). f can exceed 1 (a family beats the atoms).
* 1904 (no atoms at all: the host absorbed the caustic) is out of reach of
  this layer.
