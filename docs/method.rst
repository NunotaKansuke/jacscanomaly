Method overview
===============

``jacscanomaly`` searches for localized deviations from a baseline model. The
default microlensing workflow uses a PSPL single-lens model as the baseline.

Pipeline
--------

The high-level pipeline is:

1. Convert inputs to arrays and fit a single-lens model.
2. Compute residuals:

   .. math::

      r_i = f_i - f_{\mathrm{single}}(t_i)

3. Split the time series into observing seasons using large time gaps.
4. For each season, scan a grid of anomaly centers ``t0`` and effective
   durations ``teff``.
5. At each grid point, evaluate a local window around ``t0``.
6. Compare a flat residual model against an anomaly-template residual model.
7. Extract non-overlapping clusters and keep one representative per cluster.
8. Select the best candidate after optional quality criteria.

The following pages document the later stages in detail:

* :doc:`template_free_method` explains the residual-only zero-crossing search.
* :doc:`signal_extraction_method` explains iterative baseline refinement and
  signal masking.
* :doc:`planet_features` explains direct peak and dip measurements.

Detection statistic
-------------------

For each grid point, the package computes:

.. math::

   \Delta\chi^2 = \chi^2_{\mathrm{flat}} - \chi^2_{\mathrm{anom}}

Large positive ``dchi2`` means that a localized anomaly template improves the
fit relative to a flat residual model.

Local anomaly templates
-----------------------

The standard grid scan is a fast local detector, not a binary-lens fit. At
every ``(t0, teff)`` grid point it compares a constant residual level with the
better of two linear-amplitude templates:

.. math::

   A_0(t) = \left[1 + \left(\frac{t-t_0}{t_\mathrm{eff}}\right)^2\right]^{-1/2}

and

.. math::

   A_1(t) =
   \frac{Q + 2}{\sqrt{Q(Q+4)}},
   \qquad
   Q = 1 + \left(\frac{t-t_0}{t_\mathrm{eff}}\right)^2.

For each template, the amplitude and constant term are solved by weighted
linear least squares in the local window. The lower-:math:`\chi^2` template is
used to form ``dchi2``. The fitted local amplitude may be positive or negative,
so the scan can identify bumps and dips.

``t0`` is the trial anomaly center. ``teff`` is a detector timescale that sets
both the template width and, through ``teff_coeff``, the local evaluation half
window:

.. math::

   [t_0 - \mathtt{teff\_coeff}\,t_\mathrm{eff},
    t_0 + \mathtt{teff\_coeff}\,t_\mathrm{eff}].

It is useful for locating and ranking a residual feature; it is not itself a
physical planet parameter. Use the planet-signal workflow to refine a baseline
and measure local physical constraints after detection.

Candidate score
---------------

The candidate ``score`` measures how strongly the best cluster stands out
relative to other extracted clusters:

.. math::

   \mathrm{score}
   =
   \frac{\Delta\chi^2_{\mathrm{best}} - \mathrm{median}(\Delta\chi^2_{\mathrm{others}})}
        {\mathrm{MAD\ scale}(\Delta\chi^2_{\mathrm{others}})}

The background contains raw cluster peaks from the same observing season and a
comparable ``teff`` range. Candidate-quality cuts are applied after clustering
and therefore do not censor this background population. The reported
``std_others`` is a robust MAD-based scale retained under its historical field
name. A one-sided, MAD-based upper clip removes strong secondary candidates
without discarding a fixed percentile of ordinary background clusters.

This makes the score a local comparison: in validation checks, candidate
locations and ``dchi2`` values stayed fixed while the score changed only
through the background normalization. Efficiency thresholds calibrated with
older score definitions should therefore be regenerated for v0.4.0.

Effective number of points
--------------------------

``dchi2`` alone can be misleading when one point dominates the improvement. For
each candidate, ``jacscanomaly`` computes per-point positive improvements:

.. math::

   u_i = \max(0, \chi^2_{\mathrm{flat}, i} - \chi^2_{\mathrm{anom}, i})

The effective number of contributing points is:

.. math::

   n_{\mathrm{eff}} = \frac{(\sum_i u_i)^2}{\sum_i u_i^2}

This behaves like a participation ratio. A one-point-dominated candidate has
``n_eff`` close to 1. A candidate supported by many comparable points has larger
``n_eff``.

Other quality diagnostics
-------------------------

Each candidate also stores:

``n_window``
   Number of data points in the local evaluation window.

``n_contrib``
   Number of points above the configured per-point improvement threshold.

``peak_frac``
   Fraction of the total positive improvement carried by the strongest point.

``rho1``
   Lag-1 autocorrelation of signed per-point improvements.

``longest_run``
   Longest consecutive run of above-threshold contributing points.

Season splitting
----------------

The data are sorted by time and split whenever the gap between consecutive
points is larger than ``FinderConfig.gap``. The default is 100 days. For survey
light curves with yearly observing seasons, a smaller value such as 50 days can
separate seasons while still scanning all seasons.

Backends
--------

``grid_backend="cpp"``
   Uses the C++ for-loop grid backend. This is the default for PSPL survey
   scans and is useful for large light curves because it has lower peak memory
   use.

``single_fit_backend="cpp"``
   Uses the C++ PSPL fitter for ``fitter_kind="pspl"``.

``grid_backend="jax"``
   Uses JAX vectorized or chunked grid evaluation. This remains available for
   development and comparison.

Other single-lens model families continue to use the JAX fitters.

What the standard scan does not do
----------------------------------

The standard scan does not choose a 2L1S or 1L2S model, fit caustic geometry,
or estimate a planet mass ratio. It reports local residual candidates.
:doc:`planet_features` adds direct peak and dip counts, positions,
timescales, and strengths.

Method details
--------------

.. toctree::
   :maxdepth: 1

   template_free_method
   signal_extraction_method
