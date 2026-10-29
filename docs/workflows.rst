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

``TemplateFreeScanner().run(time, residual, ferr)``
   Run the zero-crossing search on residuals supplied by another pipeline.
   This has no dependency on ``Finder`` or a single-lens fit.

``robust_fit`` / ``measure_features``
   Post-fit components: fit the host jointly with the anomalies, so they do
   not pull it, and classify the residual into bumps and dips; see
   :ref:`post-fit-components`.

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

Pass the residuals of any fit (``fit.residual``) or of another pipeline:

.. code-block:: python

   from jacscanomaly import TemplateFreeScanner, TemplateFreeSearchConfig

   search = TemplateFreeScanner(
       TemplateFreeSearchConfig(gap=50.0)
   ).run(time, residual, ferr)

Each ``TemplateFreeCandidate`` reports its time span, total and reduced
chi-square, maximum absolute z-score, and the strongest seed point. This mode
does not infer or alter a baseline model.

.. _post-fit-components:

Post-fit components
-------------------

Everything after the single-lens fit works on a fit result (data, model and
residual) and :class:`jacscanomaly.Candidate` windows; none of it reads a scan
result.

.. code-block:: python

   from jacscanomaly import FSPLFitter, Finder, PSPLFitter, robust_fit

   finder = Finder()
   scan = finder.run(time, flux, ferr)
   r = robust_fit({"pspl": PSPLFitter(), "fspl": FSPLFitter()}, finder,
                  time, flux, ferr, scan.fit.params)
   best = r.best                     # lowest J over models and starts
   best.params, best.sigma, best.atoms, best.windows
   print(best.features.summary_text() if best.features else "no anomaly")

``robust_fit(fitters, finder, time, flux, ferr, p_scan)``
   No sample is masked. The host and the anomalies are fitted together::

      F(t) = fs A(t; p) + fb + sum_k [a_k h(tau_k) + b_k l(tau_k)]

   ``h``, ``l`` are the scan's two templates, zero outside ``|tau| < 3``
   ("atoms"). Atoms are added greedily from scans of the residual while each
   lowers ``chi2`` by more than ``4 ln n`` (BIC), then pruned with the same
   price. Starts: the scan's PSPL (moved off a short spike by
   :func:`jacscanomaly.relock_host`), the PSPL result nested into each
   extended model, and each model's plain fit. Models and starts are compared
   by ``J = chi2 + (n_host + 2 + 4 n_atoms) ln n``. ``fit_atoms`` is the
   single-start step; ``AtomConfig`` holds the constants.

``relock_host(fitter, fit)``
   Tests whether ``fit`` sits on a short anomaly instead of its host event
   and returns the host fit when it does.

``fitter.fit(..., mask=...)``
   Every single-lens fitter accepts ``mask`` (``True`` = excluded). The fit,
   ``fs``/``fb``, ``chi2`` and ``chi2_dof`` use the kept points only;
   ``model_flux`` and ``residual`` cover all points. ``evaluate_fixed`` and
   ``fit_fixed_model`` take the same argument.

``measure_features(fit, candidates)``
   Classifies the residual inside the candidate windows into ``bump``
   and ``dip`` features; see :doc:`planet_features`. ``robust_fit`` measures
   them on the host with ``fs``, ``fb`` from outside the atom windows.

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
