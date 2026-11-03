Method
======

Scan
----

The residual of a single-lens fit is split into seasons (at gaps longer than
``gap``). In each season, at every trial centre :math:`t_0` and timescale
:math:`t_\mathrm{eff}`, the data in the window
:math:`|t - t_0| < k\, t_\mathrm{eff}` (:math:`k` = ``teff_coeff``, 3) are
fitted with a constant, and with a constant plus an amplitude times one of
two templates,

.. math::

   A_0 = Q^{-1/2}, \qquad A_1 = \frac{Q + 2}{\sqrt{Q (Q + 4)}}, \qquad
   Q = 1 + \left(\frac{t - t_0}{t_\mathrm{eff}}\right)^2.

The better template gives

.. math::

   \Delta\chi^2 = \chi^2_\mathrm{flat} - \chi^2_\mathrm{template}.

The amplitude may be negative, so dips are found as well as bumps.
Neighbouring grid points that see one feature are merged into one candidate.

Score
-----

A candidate is compared with all other candidates of a similar timescale
(in every season):

.. math::

   \mathrm{score} = \frac{\Delta\chi^2 - \mathrm{median}(\Delta\chi^2_\mathrm{others})}
                         {\mathrm{MAD\ scale}(\Delta\chi^2_\mathrm{others})},

with the strong others clipped from above. A detection is a candidate with
``score >= detection_score``. Each candidate also carries ``n_eff``, the
number of points that carry its ``dchi2``: ``(sum u_i)^2 / sum u_i^2`` with
``u_i`` the per-point improvement. A candidate made by one point has
``n_eff`` near 1.

Relock
------

A PSPL fit may sit on the anomaly instead of the event. When a PSPL scan
detects something, the anomaly segment and a masked refit are alternated,
from the fit and from a fit with its own core masked. The masked host is kept
when it lowers ``J = chi2_kept + 2 n_masked`` and the kept samples still see
it.

Atoms
-----

The host is fitted jointly with "atoms". Each atom is a detection of the
residual scan, :math:`a A_0 + b A_1` cut to zero at
:math:`|t - t_0| = k\, t_\mathrm{eff}`. Atoms are added one at a time (the largest
``dchi2`` first) and dropped again while each lowers

.. math::

   J = \chi^2 + (n_\mathrm{host} + 2 + 4 n_\mathrm{atoms}) \ln n.

So the host is fitted with the anomalies modelled, not pulled by them.
Several starts are tried (the plain fit, the PSPL host nested, the host fitted
outside the atom windows); the lowest ``J`` is kept.

Features
--------

Each anomaly window is described from the atoms model ``y`` (the host held):

``kind``
   The parts of ``y`` in time order: ``bump``, ``dip``, ``bump+dip``, ...
``time``, ``timescale``
   Peak time and FWHM of the strongest part.
``dchi2``
   ``chi2(host) - chi2(host + atoms)`` in the window.
``caustic``
   ``W / tau >= 4``, with ``W`` the equivalent width of ``|y|`` and ``tau``
   its fastest rise or fall. One smooth part has about 1.5, two about 3; a
   caustic crossing is sharper. It is a rough guide.

What it does not do
-------------------

No binary-lens model is fitted: ``s``, ``q`` and ``alpha`` are not
estimated. Widths shorter than the cadence are not resolved.
