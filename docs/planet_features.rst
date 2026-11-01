Anomaly features
================

:func:`jacscanomaly.measure_features` describes every anomaly window of a host
+ atoms fit (:func:`jacscanomaly.robust_fit`). The host is held fixed; the
anomaly is ``r = flux - host``, and its smooth description is the atoms model
``y`` fitted to it. Nothing the atoms accepted is dropped.
:func:`jacscanomaly.robust_fit` runs it for every model; call it directly only
with a host and atoms of your own.

Usage
-----

.. code-block:: python

   r = robust_fit(fitters, finder, time, flux, ferr, scan.fit.params)
   for f in r.best.features.features:
       print(f.kind, f.caustic, f.time, f.timescale, f.dchi2)

   # directly: the host flux, the atoms flux and the windows
   from jacscanomaly import measure_features
   feats = measure_features(time, flux, ferr, host_flux, atom_flux, windows)

Measurements
------------

Each :class:`jacscanomaly.Feature` (one window) contains:

``kind``
   The parts of the atoms model in time order, joined by ``+`` (``bump``,
   ``dip``, ``bump+dip``, ...). ``complex``: no part seen; ``none``: the atoms
   do not improve the window.

``parts``
   The extrema of ``y``, each with its sign, peak time and FWHM. In each
   segment where the model is seen (``|y| / ferr >= 1``), the local maxima of
   ``|y|`` are taken from the strongest down and kept when at least half of
   the nearest one already kept; a weaker one next to it is its rebound.

``time``, ``timescale``
   Peak time and FWHM of the strongest part.

``dchi2``
   ``chi2(host) - chi2(host + atoms)`` in the window: the strength.

``sharpness``, ``caustic``
   Per segment ``S = W / tau``, with the equivalent width
   ``W = int |y| dt / max|y|`` and the fastest rise or fall
   ``tau = max|y| / max|dy/dt|``. One smooth part has ``S ~ 1.5`` at any S/N,
   two separate ones ``~ 3``; a caustic crossing changes the magnification
   within the source crossing time and is sharper. ``caustic`` is
   ``sharpness >= FeatureConfig.sharp_min`` (4). The label is a rough guide; a
   lone spike with nothing else at its scale stays a bump (a cusp approach and
   a lone crossing are not separable by shape).

``explained``
   The fraction of ``dchi2`` explained by each named shape family: ``bump``
   (one atom), ``fold`` (a uniform-source fold crossing), ``double`` (two
   atoms) and ``pair`` (two fold crossings). These are numbers to read, not the
   decision.

Assumptions
-----------

* The host is fixed: a host that absorbed the anomaly is not repaired here.
* The model is read on the samples; widths shorter than the cadence are not
  resolved.
* No binary-lens parameters (``s``, ``q``, ``alpha``) are estimated.

Tabular output
--------------

.. code-block:: python

   event_row = feats.summary_dict()
   feature_rows = feats.feature_dicts()
   print(feats.summary_text())

For one record per event, see :ref:`event-records`.
