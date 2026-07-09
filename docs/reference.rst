Core API Reference
==================

This page expands the methods used in the normal analysis workflows. The
complete module inventory remains in :doc:`api`; start with :doc:`workflows`
when choosing an operation.

Finder
------

.. autoclass:: jacscanomaly.Finder
   :no-index:

.. automethod:: jacscanomaly.Finder.fit_single_lens

.. automethod:: jacscanomaly.Finder.run

.. automethod:: jacscanomaly.Finder.run_template_free

``Finder.run`` is the standard template-grid detector. Its returned
``AnomalyResult`` stores the baseline fit, residuals, all extracted clusters,
grid diagnostics, and the selected ``best`` candidate. ``x0`` is an initial
guess unless ``refit=False``; see :doc:`workflows` for the fixed-baseline mode.

Template-free residual API
--------------------------

.. autoclass:: jacscanomaly.TemplateFreeScanner
   :no-index:

.. automethod:: jacscanomaly.TemplateFreeScanner.run

.. autoclass:: jacscanomaly.TemplateFreeSearchConfig
   :no-index:

``TemplateFreeScanner`` accepts residuals rather than raw baseline flux. Its
configuration controls season splitting, optional sigma-clipped
renormalization, the high-z seed threshold, zero-crossing-window joining, and
the candidate chi-square threshold.

Planet signal extraction API
----------------------------

.. autoclass:: jacscanomaly.PlanetSignalExtractor
   :no-index:

.. automethod:: jacscanomaly.PlanetSignalExtractor.run

.. autoclass:: jacscanomaly.PlanetSignalResult
   :no-index:

.. automethod:: jacscanomaly.PlanetSignalResult.classify

.. automethod:: jacscanomaly.PlanetSignalResult.classify_anomaly

.. automethod:: jacscanomaly.PlanetSignalResult.plot_signal

.. autoclass:: jacscanomaly.PlanetSignalConfig
   :no-index:

.. autoclass:: jacscanomaly.PlanetSignalClassificationConfig
   :no-index:

The extractor returns both ``initial_fit`` and ``refined_fit``. Use
``signal_mask`` to identify excluded points, ``point_weight`` to inspect the
fit weighting, and ``iterations`` to audit accepted refinement steps. See
:doc:`planet_classification` for the three baseline modes and their controls.

Planet residual-atom fitting API
--------------------------------

.. autoclass:: jacscanomaly.PlanetAnomalyClassifier
   :no-index:

.. automethod:: jacscanomaly.PlanetAnomalyClassifier.fit

.. autoclass:: jacscanomaly.PlanetClassConfig
   :no-index:

.. autoclass:: jacscanomaly.PlanetAnomalyFitResult
   :no-index:

.. automethod:: jacscanomaly.PlanetAnomalyFitResult.summary_dict

.. automethod:: jacscanomaly.PlanetAnomalyFitResult.atom_table

.. automethod:: jacscanomaly.PlanetAnomalyFitResult.seed_table

``atom_table()`` has one row per retained local atom fit, including the fitted
parameters, ``*_err`` uncertainty columns when estimated, ``chi2``,
``delta_chi2``, BIC, score, validity penalty, and warnings. ``seed_table()``
contains downstream model initializations. These are local estimates and must
be validated by a global fit.
