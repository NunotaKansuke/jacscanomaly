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

Post-fit components
-------------------

See :ref:`post-fit-components` for how they compose.

.. autoclass:: jacscanomaly.Candidate
   :no-index:

.. autofunction:: jacscanomaly.candidates_from_scan

.. autofunction:: jacscanomaly.signal_mask

.. autoclass:: jacscanomaly.SignalMask
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

Feature measurement API
-----------------------

.. autofunction:: jacscanomaly.measure_features

.. autoclass:: jacscanomaly.FeatureConfig
   :no-index:

.. autoclass:: jacscanomaly.Feature
   :no-index:

.. autoclass:: jacscanomaly.CausticDetail
   :no-index:

.. autoclass:: jacscanomaly.FeatureResult
   :no-index:

.. automethod:: jacscanomaly.FeatureResult.summary_dict

.. automethod:: jacscanomaly.FeatureResult.feature_dicts

.. automethod:: jacscanomaly.FeatureResult.summary_table
