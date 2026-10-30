# Feature shape decision and spans (2371: 0043, 0046, 0049)

## Shape decision

The fold and pair families have the same nonlinear parameter count as the
bump and double (2 and 4). They were still treated as "richer" and had to
beat them by `max(delta * dchi2, dchi2_min)`, and the decision was sequential
(fold vs bump, pair vs double, then the second class against the first), so
the pair was never compared with the bump.

* 0049 (sharp drop, flat trough, sharp rise): pair 1.000 vs double 0.965 of
  `dchi2 = 4627`. That is 162 better in chi2, but under 5 % of dchi2, so it
  was labelled `bump+dip`.
* 0046: bump 0.963, double 0.995, pair 1.018. The pair is +0.055 over the
  bump, but it only met the double (+0.022), so the result was `bump`.

Change: within a class the fold / pair is taken when it is better by
`dchi2_min` (the edge is resolved; the counts are equal). Going from the
first class to the second (twice the nonlinear parameters) still needs
`delta * dchi2` as well.

## Timescale and span

* bump: `timescale` was the atom's `teff`. The atom `a h + b l` can be flat
  topped (0046: a 0.17 d plateau was reported as 0.036). It is now the FWHM
  of the fitted atom profile.
* `spans`: each atom's half-maximum interval (bump: one, double: two). The
  plot shades these, not `time +- timescale / 2`. For a double that range
  used to run from one atom's centre to the other's, which covers neither
  atom.
