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

``signal_mask`` / ``fitter.fit(..., mask=...)`` / ``measure_features``
   Post-fit components: mask the signal inside candidate windows, refit the
   baseline without it, and classify the residual into bumps and dips.
   Each runs on its own; see :ref:`post-fit-components`.

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

Everything after the single-lens fit is a set of independent components. They
are linked only by the information passed between them: a fit result (data,
model and residual) and a list of :class:`jacscanomaly.Candidate` windows.
No component calls another one, and none of them reads a scan result.

.. code-block:: python

   from jacscanomaly import (
       Candidate, Finder, PSPLFitter, candidates_from_scan,
       measure_features, signal_mask,
   )

   fitter = PSPLFitter()
   fit = fitter.fit(time, flux, ferr)

   scan = Finder().run(time, flux, ferr)          # optional
   cands = candidates_from_scan(scan, which="best")
   # or, without a scan (e.g. a window the scan rejected):
   # cands = [Candidate(t_start=9.0, t_end=11.0)]

   sm = signal_mask(fit, cands)                    # True = signal point
   fit2 = fitter.fit(time, flux, ferr, p0=fit.params, mask=sm.mask)
   feats = measure_features(fit2, cands)           # bumps / dips

   print(feats.summary_text())

``Candidate``
   A time window ``t_start <= t <= t_end`` plus an optional anchor ``t0``.
   ``Candidate.from_t0_teff(t0, teff)`` uses the scan support rule
   ``|t - t0| <= 3 |teff|``. ``candidates_from_scan(result, which="all")``
   converts every scored cluster, accepted by the criteria or not; only the
   numbers are copied.

``signal_mask(fit, candidates, *, penalty=2.0)``
   For each candidate window, masks the contiguous residual segment that
   maximises ``sum(z_i**2 - penalty)`` with ``z = residual / ferr`` and
   overlaps the window. Masking a point frees one parameter, so this is the
   mask that minimises ``chi2_kept + penalty * n_masked`` (2 = AIC) at the
   current fit. A masked refit lowers the same quantity, so alternating the
   two converges. Returns the union (``SignalMask.mask``), the per-candidate
   masks and their gains. It never refits.

``fitter.fit(..., mask=...)``
   Every single-lens fitter accepts ``mask`` (``True`` = excluded). The fit,
   ``fs``/``fb``, ``chi2`` and ``chi2_dof`` use the kept points only;
   ``model_flux`` and ``residual`` cover all points, so the excluded signal
   stays visible in the residual. ``evaluate_fixed`` and ``fit_fixed_model``
   take the same argument.

``measure_features(fit, candidates)``
   Classifies the residual inside the candidate windows into ``bump``
   and ``dip`` features; see :doc:`planet_features`.

Iteration (mask, refit, re-mask) is written by the caller, not hidden in a
configuration object.

.. code-block:: python

   mask = None
   while True:
       sm = signal_mask(fit, cands)
       if mask is not None and (sm.mask == mask).all():
           break
       mask = sm.mask
       fit = fitter.fit(time, flux, ferr, p0=fit.params, mask=mask)

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
