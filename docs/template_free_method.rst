Template-free Residual Search
=============================

The template-free scanner operates on a residual series rather than raw light
curve flux. It does not fit, modify, or validate a single-lens baseline. Its
purpose is to find coherent excursions when the two analytic local templates
used by :doc:`method` are too restrictive, or when residuals come from another
pipeline.

Inputs and normalization
------------------------

For residual :math:`r_i`, uncertainty :math:`\sigma_i`, and observation time
:math:`t_i`, the scanner forms

.. math::

   z_i = r_i / \sigma_i.

Time points are split into independent seasons when adjacent observations have
a gap greater than ``TemplateFreeSearchConfig.gap``. No candidate window crosses
a season boundary.

When ``renormalize_z=True``, each season is recalibrated independently. The
scanner repeatedly computes the median and standard deviation of retained
points, discards points outside ``sigma_clip_threshold`` standard deviations,
and stops when the retained set stabilizes or
``sigma_clip_max_iter`` is reached. The final season z-score is

.. math::

   z_i = (z_{i,\mathrm{raw}} - \mathrm{median}) / \mathrm{std}.

This is a robust normalization aid, not an uncertainty-model replacement.

Zero-crossing windows
---------------------

Within a season, the algorithm performs the following operations:

1. Split the ordered z-score series at direct sign changes.
2. Retain segments whose peak absolute z-score exceeds
   ``seed_z_threshold``.
3. For every retained segment, add its immediate neighboring segment on both
   sides. The candidate thus extends to the second sign crossing away from the
   seed peak when data are available.
4. Join proposed windows when their point-index gap is no larger than
   ``max(bridge_floor_points, bridge_fraction * wider_window_width)``.
   Joining is repeated because a newly joined window can bridge a later one.
5. Compute the candidate statistic

   .. math::

      \chi^2_\mathrm{window} = \sum_{i \in \mathrm{window}} z_i^2.

   Keep windows above ``candidate_chi2_threshold`` and rank them by this value.

The candidate seed is the time point with largest absolute z-score in the
joined window. ``max_candidates_per_season`` caps the returned windows after
ranking.

Interpretation and limits
-------------------------

``TemplateFreeCandidate.chi2`` is accumulated residual power, not a likelihood
ratio against a fitted anomaly model. A broad systematic trend, a poor baseline,
or a real anomaly can all produce a large value. Inspect the span, seed,
``reduced_chi2``, and the residual plot before using it for follow-up.

The join rule intentionally bridges shallow sign changes inside one smooth
structure. It can therefore combine nearby independent deviations when the
chosen bridge parameters are too permissive. Reduce ``bridge_floor_points`` or
``bridge_fraction`` when separation is more important than sensitivity to
multi-lobed structures.

Use :class:`jacscanomaly.TemplateFreeScanner` for externally supplied
residuals, or :meth:`jacscanomaly.Finder.run_template_free` with an existing
``SingleLensFitResult``.
