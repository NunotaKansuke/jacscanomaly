Search Workflows
================

Choose a workflow by what is already known about the baseline and by the
output you need. All time, flux, and flux-error inputs are one-dimensional
arrays. ``x0`` always contains the nonlinear parameters required by the
selected ``fitter_kind``; for PSPL this is ``[t0, tE, u0]``.

Workflow map
------------

``Finder.run(time, flux, ferr)``
   Fit a single-lens baseline, scan residuals with local templates, and return
   :class:`jacscanomaly.AnomalyResult`. Use this as the default discovery run.

``Finder.run(..., x0=x0, refit=False)``
   Keep the nonlinear baseline geometry fixed, solve only ``fs`` and ``fb``,
   then run the same local-template scan. Use this when an external PSPL fit
   should not be optimized again.

``Finder.run_template_free(..., fit=fit)``
   Scan residuals of an existing ``SingleLensFitResult`` with zero-crossing
   windows instead of the standard templates. No baseline fit is performed.

``TemplateFreeScanner().run(time, residual, ferr)``
   Run the zero-crossing search on residuals supplied by another pipeline.
   This has no dependency on ``Finder`` or a single-lens fit.

``PlanetSignalExtractor(finder).run(...)``
   Iteratively refine a baseline while separating strong residual signal,
   measure its prominent peaks and dips. Use this after
   detection or for targeted event analysis.

Standard template scan
----------------------

The default workflow fits first and then searches the residuals:

.. code-block:: python

   from jacscanomaly import CandidateCriteria, Finder, FinderConfig

   finder = Finder(
       FinderConfig(
           fitter_kind="pspl",
           gap=50.0,
           candidate_criteria=CandidateCriteria(
               min_dchi2=20.0,
               min_n_eff=2.0,
           ),
       )
   )
   result = finder.run(time, flux, ferr)

   if result.best is not None:
       print(result.best.t0, result.best.teff, result.best.dchi2)

The scan uses two analytic local templates at every trial ``(t0, teff)`` and
keeps the better one. It is therefore fast and suitable for broad discovery,
but it does not fit a planetary caustic model. See :doc:`method` for the
templates and :doc:`results` for interpreting candidates.

Fixed-baseline scan
-------------------

Pass ``refit=False`` only when the nonlinear geometry in ``x0`` is already
trusted. ``Finder`` still solves the linear source and blend fluxes for the
input data, so residual normalization remains consistent:

.. code-block:: python

   x0 = np.array([t0, tE, u0])
   result = finder.run(time, flux, ferr, x0=x0, refit=False)

With the default ``refit=True``, the same ``x0`` is only an optimizer starting
point. Omitting ``x0`` triggers automatic single-lens initialization. A fixed
parameter scan currently supports the concrete fitter kinds documented in
:doc:`configuration`; BIC model selection needs a fitted baseline.

Template-free residual scan
---------------------------

The template-free scanner identifies high-absolute-z residual segments and joins
nearby zero-crossing windows. It is useful when a bell-like local template is
too restrictive or when another pipeline already supplied residuals.

With an existing fit:

.. code-block:: python

   from jacscanomaly import TemplateFreeSearchConfig

   search = finder.run_template_free(
       time,
       flux,
       ferr,
       fit=existing_fit,
       config=TemplateFreeSearchConfig(
           seed_z_threshold=5.0,
           candidate_chi2_threshold=150.0,
       ),
   )
   print(search.best)

With externally computed residuals:

.. code-block:: python

   from jacscanomaly import TemplateFreeScanner, TemplateFreeSearchConfig

   search = TemplateFreeScanner(
       TemplateFreeSearchConfig(gap=50.0)
   ).run(time, residual, ferr)

Each ``TemplateFreeCandidate`` reports its time span, total and reduced
chi-square, maximum absolute z-score, and the strongest seed point. This mode
does not infer or alter a baseline model.

Signal extraction and peak/dip measurements
--------------------------------------------

For the complete model-selection and anomaly workflow, prefer the high-level
API:

.. code-block:: python

   from jacscanomaly import Finder

   result = Finder().run_anomaly_pipeline(time, flux, ferr)

   print(result.has_anomaly_candidate)
   print(result.best_anomaly_candidate)
   print(result.final_detection.summary_dict())
   for candidate in result.anomaly_candidates:
       print(candidate["rank"], candidate["t_center"], candidate["max_abs_z"])
   print(result.adopted_fit.model_kind)

``result.anomaly_candidates`` is the normal reporting interface: a ranked
list of dictionaries with the location, interval, timescale, strength,
provenance, and adopted model. Overlapping final-residual features and
template-free windows are merged, with feature timing/sign and template-free
chi-square statistics retained in one row. ``best_anomaly_candidate`` is the
first row or ``None``. The deliberately cautious name
``has_anomaly_candidate`` means that a false value is not a proof that no
physical anomaly exists.

``result.final_detection`` is a separate ``PlanetScanDecision`` from the
frozen residual scan after model selection. It is the discovery decision;
``result.features`` and ``result.anomaly_candidates`` are characterization and
reporting layers and must not be used to rewrite that decision.

``result.fit_exclusion_mask`` contains only points excluded by an accepted
continuation fit. ``result.measurement_mask`` is a separate frozen-residual
measurement window and must not be used as an HTML removal/display mask.

Physical routing and observed time scale
----------------------------------------

Physical routing treats annual versus spacecraft parallax as an observer
geometry choice, not as competing models. ``FinderConfig(parallax_geometry=
"auto")`` selects spacecraft geometry when a satellite ephemeris is supplied,
and annual geometry otherwise. Use ``"annual"``, ``"space"``, or ``"none"``
to make that choice explicit.

The detector and reporting layers expose an ``ObservedSignalScale`` measured
from residual/profile support. It is based on a weighted central interval,
with ``censored=True`` when the signal reaches an observing edge. The scan-grid
``teff`` remains an internal proposal coordinate; plots, masks, and physical
routing use the observed scale when it is measurable.

For a candidate event, use the extractor to prevent the strongest anomaly
from biasing the single-lens baseline. Then measure extrema in the refined
residual:

.. code-block:: python

   from jacscanomaly import PlanetSignalConfig, PlanetSignalExtractor

   extractor = PlanetSignalExtractor(
       finder,
       PlanetSignalConfig(
           baseline_mode="beam_interval",
           seed_min_dchi2=100.0,
           max_iter=3,
       ),
   )
   signal = extractor.run(time, flux, ferr, x0=x0, refit=False)
   features = signal.measure_features()

   print(features.summary_text())
   rows = features.feature_dicts()

This stage returns candidate intervals, a refined baseline, and direct
measurements of each peak and dip. It does not assign physical caustic labels
or estimate binary-lens parameters.

For high-throughput first passes, use ``PlanetSignalConfig.fast()``.  It
performs one beam iteration with one retained interval.  For routing-only
work, ``PlanetSignalConfig.probe()`` performs the first grid scan without a
masked fit and exposes its cached seed.  ``run_effect_aware`` uses this probe
by default, then continues with the full beam search only for a credible seed
or a physical-effect fallback candidate; that full pass reuses the probe seed
instead of scanning it again.  Pass ``planet_fast_mode=False`` to always use
the full beam configuration.
