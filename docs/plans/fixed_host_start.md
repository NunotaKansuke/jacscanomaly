# The plain host, kept unless a moved host wins by a margin

## Problem

The joint host + atoms fit can move the host (`tE`, `u0`) and add one broad
bump atom where the eye sees the plain host plus a small atom with shallow
side dips. The `h` template has broad `1/|tau|` wings, which trade off
against the host shape, so the moved host pays off in `chi2`. The gain is
real for the likelihood but small. It is within what model mismatch and
correlated noise give, so `J` alone picks it.

## Change

* `fit_atoms(..., fix_host=True)`: the host stays at `p0`. Only `fs`, `fb`
  and the atom amplitudes are solved, linearly. The add, stop and drop steps
  are unchanged.
* `robust_fit`: every model with a plain fit (`base`) gets one more start,
  `"fixed"`: `base`, with the host fixed.
* A start that moves the host is chosen over `"fixed"` only when its `J` is
  lower by more than `AtomConfig.free_host_margin` (default 150). The
  selection uses `J - margin` for `"fixed"`, both between starts and between
  models. `ModelFit.J` stays the plain `J`.

## Assumptions

* The margin is a decision rule, not part of the likelihood. It is the
  tolerance for model mismatch and correlated noise. It was set on 23 events
  judged by eye: those preferring the plain host had `dJ` <= 137 and those
  preferring the moved host had `dJ` >= 140 (one borderline case at 140).
  No single cut separates 137 from 140, so 150 was chosen.
* `free_host_margin = 0` gives the old selection (`"fixed"` competes as a
  plain start), apart from ties.
