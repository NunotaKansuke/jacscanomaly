# Bump / dip features from the extrema of the atoms model

## Problem (2371)

Fold and caustic families work; bump / dip / two-atom descriptions do not.
They describe each *atom* (or a refit of one or two atoms), and atoms
superpose: two overlapping atoms of opposite sign (0624: a bump and a dip at
9863.8, largely cancelling) give two lobes that are not in the data, and one
atom ``a h + b l`` has a dip and a bump (0616) of which only the larger gets
a time and a width.

## Change

For a window not claimed by a fold or caustic pair (bump, double or
complex), describe the atoms model ``y = atoms(t)`` (relative to the host)
itself, not its atoms:

1. **Segments**: runs of samples where the model is seen,
   ``|y| / ferr >= 1`` (as the relock "core"), joined across a pass through
   the host shorter than a seen run next to it (a zero crossing between a
   lobe and its rebound is not a return to the host). Separate segments are
   separate anomalies of the window. A model seen by no single sample (a weak,
   broad anomaly seen only by all its samples together, 0616) is one segment.
2. **Extrema**: the local maxima of ``|y|`` in each segment, each with its
   sign (``bump`` / ``dip``).
3. **Comparable strength**: in a segment, from the strongest extremum down,
   keep one with ``|y| >= |y_near| / 2``, ``y_near`` the nearest extremum
   already kept. A weak extremum next to a strong one is its rebound, not an
   anomaly of its own; one comparable to its kept neighbour is kept even when
   that neighbour is itself weaker than the largest (0140: bump 1.4 next to
   the dip 2.1, the largest bump 3.9; first version, ``>= |y_max| / 2`` of the
   segment, dropped it); a weak one in another segment is that segment's
   strongest and is kept.
4. **Resolved at half maximum**: two kept extrema of the same sign are two
   only if ``|y|`` between them falls below half of the weaker; otherwise
   they are one (the stronger).
5. **Width**: each kept extremum's FWHM, the run around it where ``y`` keeps
   its sign and ``|y| >= |y_k| / 2``, stopped at the minimum of ``|y|``
   between it and a neighbouring kept extremum (linear interpolation at the
   ends).

Feature: ``parts`` = the kept extrema (kind, peak time, FWHM) in time order;
``kind`` = their kinds joined by ``+`` (one part: ``bump`` / ``dip``);
``time`` / ``timescale`` = the strongest part's; ``spans`` = the parts'
half-max intervals. Fold / caustic features are unchanged; the family
fractions stay as the shape decision.

## Assumptions

* The atoms model is the description of the anomaly (the data are noisier;
  the atoms are already the smooth fit).
* ``1/2`` is the same half-maximum convention as every width here: an
  extremum below half of its segment's largest does not reach the width
  level of the anomaly.
* The model is read on the samples (no finer grid): widths shorter than the
  cadence are not resolved.

## Check

0616, 0624, 0953, 0979, 0969, 0978 and a sample of bump / dip / two-atom
events: images of the model with the parts.
