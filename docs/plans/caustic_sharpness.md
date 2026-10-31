# Caustic or not: the edge sharpness of the atoms model

## Problem (2371, fix7)

fix7 calls a window a caustic when a fold or pair family fits it better than
a bump or double and explains ``f_min`` of ``dchi2``. That misses caustics the
straight-fold families cannot fit: a spike out of a demagnified trough (0969,
0979), a box with a spike on one wall (142, 1500), dips with a spike (920,
2132); and it calls host misfits caustics (1013, 1129). Each fix so far was
a new family or rule (the host-inside fold with a trend made ~136 false
folds). The extrema of the atoms model already describe bumps and dips well.

## Change

One number per window, from the atoms model ``y`` (relative to the host) on
the samples, in each segment of :func:`_extrema` (``|y| / ferr >= 1``,
joined across short passes through the host):

    S = W / tau,   W = int |y| dt / max|y|,   tau = max|y| / max|dy/dt|

``W`` is the equivalent width of the segment, ``tau`` its fastest rise or
fall by its own height. The window's ``sharpness`` is the largest ``S`` of its
segments; the window is labelled caustic when ``sharpness >= sharp_min = 4``.

* Every window's ``kind`` is its parts (``bump``, ``dip+bump``, ...), as for
  bumps and dips now; ``caustic`` is a flag next to it, the number is the
  result and the flag a rough guide.
* The fold / pair families stay as fractions ``f_k`` (numbers, not the
  decision); their crossings are no longer reported.
* ``Part.asymmetry`` / ``tail`` go (they did not separate caustics: 0979 tail
  2.5, smooth bumps 1.6-1.8).

## Why S

A caustic crossing changes the magnification by the anomaly's size within the
source crossing time, short against the anomaly; a smooth anomaly changes by
its size over its own width. ``S`` is that ratio:

* one smooth single-scale part: ``S ~ 1.5`` (atom 1.43, ``l`` atom 1.51,
  Gaussian 1.52), whatever its S/N;
* two separate smooth parts of one width: ``~ 3``;
* a box of length ``L`` with walls ``d``: ``L / d``; a spike of width ``d`` on a
  longer anomaly: grows with the anomaly's area against the spike's.

``sharp_min = 4`` is above two single-scale parts. The equivalent width (not
the seen extent, which grows with S/N: median non-caustic ``S`` went 1.9 ->
29 from S/N < 10 to > 100) keeps ``S`` an S/N-free shape number (1.3 -> 2.8).

## Assumptions

* The atoms model is the anomaly; read on the samples (no finer grid): an
  edge faster than the cadence is as steep as the samples show, never
  steeper (no edge inside a gap).
* A data gap carries no area: an interval longer than 10 x the segment's
  median spacing is left out of ``W``.
* A lone spike with nothing else at its scale has ``S ~ 1.5`` and is a
  ``bump``: a cusp approach and a lone crossing are not separable by shape.

## Check (fix7 atoms, 2327 events)

* fix7 caustic windows with ``S >= 4``: 103 / 115; misses sparse crossings
  (0091, 1874) and the host misfit 1129 (1.2, correctly).
* Other windows with ``S >= 4``: 311; a random 12 at ``4 <= S < 7``: 7
  caustic-like (142 box, 788, 1148, 1437, 1678, 2132, 2283), 5 smooth.
* Hand-labelled 51: caustic-like 25 / 32 at ``S >= 4`` (misses lone spikes
  0177, 1701, 2118, sparse 0091, 1874, 1698, 0445); smooth 3 / 19 (2337 4.0,
  1013 4.1, 2269 4.7).
