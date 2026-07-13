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
   classify its morphology, and measure local physical constraints. Use this after
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

Signal extraction and local physical constraints
------------------------------------------------

For a candidate event, use the extractor to prevent the strongest anomaly
from biasing the single-lens baseline. Then classify the refined residual and
inspect locally identifiable constraints:

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
   morphology = signal.classify_anomaly()

   print(morphology.summary_text())
   constraints = morphology.physical_constraint_dicts()

This stage is deliberately more selective and more expensive than the initial
scan. It returns candidate intervals, a refined baseline, broad shape labels,
local residual-atom fits, and identifiable local physical combinations.
Read :doc:`planet_classification` before interpreting a morphology label as a
physical conclusion.
