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

.. automethod:: jacscanomaly.Finder.run_effect_aware

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

.. automethod:: jacscanomaly.PlanetSignalResult.measure_features

.. automethod:: jacscanomaly.PlanetSignalResult.plot_signal

.. autoclass:: jacscanomaly.PlanetSignalConfig
   :no-index:

The extractor returns both ``initial_fit`` and ``refined_fit``. Use
``signal_mask`` to identify excluded points, ``point_weight`` to inspect the
fit weighting, and ``iterations`` to audit accepted refinement steps. See
:doc:`planet_features` for the three baseline modes and their controls.

``timing`` records the extractor wall-clock total, time spent in residual-grid
scans, and the number of scans.  In effect-aware runs, the default routing
probe exposes a cached initial seed and a later full beam pass reuses that
seed rather than repeating the first scan.

.. autoclass:: jacscanomaly.PlanetSignalTiming
   :no-index:

FFT initialization API
----------------------

.. autoclass:: jacscanomaly.PSPLFFTScanner
   :no-index:

.. automethod:: jacscanomaly.PSPLFFTScanner.search

.. autoclass:: jacscanomaly.PSPLFFTSearchResult
   :no-index:

The FFT initializer is documented conceptually in :doc:`pspl_fft`; the
residual-grid FFT backend is documented in :doc:`fft_backend`.

Peak and dip measurement API
----------------------------

.. autoclass:: jacscanomaly.PlanetFeatureConfig
   :no-index:

.. autoclass:: jacscanomaly.PlanetFeatureResult
   :no-index:

.. automethod:: jacscanomaly.PlanetFeatureResult.summary_dict

.. automethod:: jacscanomaly.PlanetFeatureResult.feature_dicts

.. automethod:: jacscanomaly.PlanetFeatureResult.summary_table
