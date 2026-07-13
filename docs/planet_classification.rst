Planet Signal Classification
============================

The planetary-signal workflow separates a local residual signal from a
refined single-lens baseline, measures its shape, and derives the quantities
that the local data determine well: the anomaly position on the PSPL
magnification pattern (:math:`\tau_{\rm anom}`, :math:`u_{\rm anom}`,
:math:`\alpha`, :math:`s^\dagger_\pm`) and the anomaly timescale relative to
:math:`t_E`, with an assumption-tagged mass-ratio estimate where a published
relation applies.  It does not fit a 2L1S/1L2S model and does not resolve
the inner/outer or :math:`u_0`-mirror degeneracies; see
:doc:`morphology_classification_method` for the formalism and references.

Refine and classify a signal
----------------------------

Use :class:`jacscanomaly.PlanetSignalExtractor` to iteratively refit the
baseline while excluding localized residual signals. It reuses the supplied
finder's baseline fitter and template-grid configuration. The returned
:class:`jacscanomaly.PlanetSignalResult` exposes both a simple shape
classification and the heuristic anomaly estimator:

.. code-block:: python

   from jacscanomaly import (
       Finder,
       FinderConfig,
       PlanetSignalConfig,
       PlanetSignalExtractor,
   )

   finder = Finder(FinderConfig(fitter_kind="pspl"))
   signal = PlanetSignalExtractor(
       finder,
       PlanetSignalConfig(seed_min_dchi2=100.0),
   ).run(time, flux, ferr)

   shape = signal.classify()
   print(shape.signal_type)

   anomaly = signal.classify_anomaly()
   print(anomaly.summary_text())

``signal.classify()`` describes each connected extracted component with broad
shape labels such as peaks, dips, caustic crossings, and complex signals.
``signal.classify_anomaly()`` measures each component with a small template
set and derives the anomaly geometry and scales.

Peak and duration estimates
---------------------------

``signal.classify()`` is the light-weight first measurement stage. Each
component has a time span, a broad ``signal_type``, and lists of ``peaks`` and
``dips``. A peak records its observed time, residual, z-score, interpolated
duration, and the corresponding baseline and observed magnifications.

Set ``fit_template_timescale=True`` when a local analytic template fit should
refine the timing estimate of sufficiently sampled extrema:

.. code-block:: python

   from jacscanomaly import PlanetSignalClassificationConfig

   shape = signal.classify(
       PlanetSignalClassificationConfig(
           fit_template_timescale=True,
           fit_template_min_points=6,
           fit_template_min_teff=0.01,
           fit_template_max_teff=10.0,
       )
   )
   for component in shape.components:
       for peak in component.peaks:
           print(peak.fitted_t0, peak.fitted_teff, peak.fitted_chi2)

``fitted_t0`` and ``fitted_teff`` are local template measurements, while
``t_start``, ``t_end``, and ``timescale`` are threshold-crossing estimates.
They describe a residual feature and should not be read as physical binary-lens
parameters.

To preserve a trusted baseline geometry, pass the same fixed parameters used
by :meth:`jacscanomaly.Finder.run`:

.. code-block:: python

   x0 = np.array([t0, tE, u0])
   signal = PlanetSignalExtractor(finder).run(
       time,
       flux,
       ferr,
       x0=x0,
       refit=False,
   )

The nonlinear parameters in ``x0`` remain fixed; the source and blend fluxes
are solved for the unmasked data at each refinement step.

Baseline-refinement modes
-------------------------

``PlanetSignalConfig.baseline_mode`` determines how candidate signal points
are excluded before the baseline is refined:

``"beam_interval"`` (default)
   Builds a small beam of connected mask intervals from strong template-scan
   candidates, scores the resulting baseline fits, and keeps the best mask.
   This is the general-purpose choice when anomalies are localized.

``"mask"``
   Grows masks around successive strong scan candidates and refits after each
   accepted addition. Use it when a simple greedy sequence is easier to audit.

``"robust"``
   Iteratively downweights large residuals rather than selecting hard
   intervals. Use it when the signal is broad or interval boundaries are not
   well defined.

The important safeguards are ``max_mask_fraction`` (do not mask too much of
the light curve), ``max_unmasked_chi2_dof_increase`` (reject a mask that makes
the retained data substantially worse), and ``max_refined_chi2_dof_ratio``
(fall back from a catastrophic refinement). ``prior_signal_windows`` can seed
known intervals as ``(center_time, half_width)`` pairs.

The result retains the complete refinement history:

.. code-block:: python

   print(signal.initial_fit.chi2_dof)
   print(signal.refined_fit.chi2_dof)
   print(signal.signal_mask.sum())
   for step in signal.iterations:
       print(step.iteration, step.added_points)

``initial_residual`` is the residual before refinement;
``refined_residual`` is the residual against the final baseline;
``signal_mask`` identifies points excluded from that baseline fit; and
``candidates`` contains the contiguous extracted intervals.

The anomaly estimate
--------------------

``classify_anomaly()`` returns a
:class:`jacscanomaly.PlanetAnomalyFitResult` with one
:class:`jacscanomaly.ComponentAnomalyResult` per extracted component.  Each
component carries three levels of information:

``shape_fits`` / ``best_fit`` / ``shape``
   The fitted shape templates (``bump``, ``dip``, ``fold``,
   ``caustic_crossing``, ``null``) ranked by BIC, and the resulting label.
   A component whose best fit is the null polynomial is labeled
   ``no_coherent_shape``; a winning template below ``min_delta_chi2`` is
   ``low_significance``; both suppress the derived quantities.

``geometry``
   The deterministic anomaly geometry: ``tau_anom``, ``u_anom``, ``alpha``,
   ``sin_alpha``, both ``s_dagger`` branches with the preferred branch
   (major image for bumps, minor image for dips), and the ``regime`` flag
   (``planetary`` or ``central_or_resonant``).  First-order errors are
   propagated from the fitted ``t_anom`` uncertainty.

``scales``
   ``dt`` and ``dt_over_tE`` with the shape-specific duration definition
   (bump FWHM, full dip duration, entry-to-exit interval), the mass-ratio
   estimate ``q`` with its ``q_method`` tag where defined, and
   ``tstar*_over_tE`` for fold-type shapes.  ``notes`` spells out the
   assumption behind every derived number.

``grid_seed``
   A :class:`jacscanomaly.GridSeed` search region for a downstream 2L1S
   grid: both ``s_dagger`` branches, the four mirror-degenerate ``alpha``
   candidates, and the ``q`` range with a ``calibrated`` /
   ``order_of_magnitude`` / ``none`` quality tag.
   ``seed.contains(s=..., q=..., alpha=...)`` tests a parameter point
   against the region; widths are configured in
   :class:`jacscanomaly.PlanetClassConfig` (``seed_*`` fields).

.. code-block:: python

   anomaly = signal.classify_anomaly()
   best = anomaly.best_component
   if best is not None:
       print(best.shape)
       print(best.geometry.u_anom, best.geometry.s_dagger_plus)
       print(best.scales.q, best.scales.q_method)

Inspecting results
------------------

The result has compact dictionary and table helpers suitable for notebooks
and survey tables:

.. code-block:: python

   event_row = anomaly.summary_dict()
   component_rows = anomaly.component_summary_dicts()
   shape_rows = anomaly.shape_fit_dicts()

   display(anomaly.summary_table())
   display(anomaly.shape_fit_table())

   signal.plot_signal()
   anomaly.plot_summary(signal_result=signal)

``summary_table()`` has one row per component with its best shape fit,
geometry, and scales.  ``shape_fit_table()`` has one row per fitted template
(including the null), with ``chi2``, ``delta_chi2``, ``bic``, the fitted
parameters, and ``*_err`` uncertainty columns when the local covariance
estimate is available.

Configuration
-------------

:class:`jacscanomaly.PlanetSignalConfig` controls baseline refinement and
signal extraction; the most important controls are ``seed_min_dchi2``,
``max_iter``, and ``max_mask_fraction``.

:class:`jacscanomaly.PlanetClassConfig` controls the anomaly estimator:

``min_delta_chi2``
   Minimum improvement over the null polynomial for a shape to be considered
   significant.

``mixed_sign_power_fraction``
   Fraction of the total residual power required in each sign before the
   bump or dip template is tried.

``min_points_fold`` / ``min_points_crossing``
   Minimum component sizes for the fold and caustic-crossing templates.

``central_u_anom_max``
   ``u_anom`` threshold below which the geometry is flagged
   ``central_or_resonant`` and no ``q`` estimate is reported.

``estimate_param_errors``
   Enables the finite-difference covariance estimate for template
   parameters.

Interpretation
--------------

The derived quantities are heuristic seeds, not posteriors:

* ``s_dagger`` is the geometric mean of the degenerate inner/outer
  solutions, and ``alpha`` is defined up to the ``u0``-mirror reflection.
* ``q`` estimates are typically accurate to a factor of ~2 relative to full
  modeling; their errors propagate only the local measurement uncertainty.
* A bump measured here can also be a second source (1L2S); distinguishing
  the two requires information beyond the local shape.

Use the output to decide whether an event warrants detailed modeling and to
initialize that modeling, then evaluate full models on the complete light
curve with the appropriate likelihood and priors.
