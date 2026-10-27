API Reference
=============

This page is the reference layer for the user guide. For narrative
documentation, start with :doc:`quickstart`, :doc:`method`, and
:doc:`configuration`.

Main workflow
-------------

.. autosummary::
   :toctree: generated

   jacscanomaly.Finder
   jacscanomaly.FinderConfig
   jacscanomaly.CandidateCriteria
   jacscanomaly.Candidate
   jacscanomaly.candidates_from_scan
   jacscanomaly.signal_mask
   jacscanomaly.SignalMask
   jacscanomaly.measure_features
   jacscanomaly.FeatureConfig
   jacscanomaly.Feature
   jacscanomaly.FeatureResult
   jacscanomaly.TemplateFreeScanner
   jacscanomaly.TemplateFreeSearchConfig
   jacscanomaly.TemplateFreeSearchResult

FFT search backends
-------------------

.. autosummary::
   :toctree: generated

   jacscanomaly.PSPLFFTScanner
   jacscanomaly.PSPLFFTSearchResult
   jacscanomaly.PSPLFFTCandidate
   jacscanomaly.PSPLFFTProfile

Result containers
-----------------

.. autosummary::
   :toctree: generated

   jacscanomaly.AnomalyResult
   jacscanomaly.BestCandidate
   jacscanomaly.ScoredCandidate
   jacscanomaly.CandidateQuality
   jacscanomaly.SeasonSummary

Plotting
--------

.. autosummary::
   :toctree: generated

   jacscanomaly.AnomalyPlotter

Single-lens fitting
-------------------

.. autosummary::
   :toctree: generated

   jacscanomaly.PSPLFitter
   jacscanomaly.FSPLFitter
   jacscanomaly.PSPLParallaxFitter
   jacscanomaly.FSPLParallaxFitter
   jacscanomaly.SingleLensFitResult
   jacscanomaly.fspl_template_initial_guesses

Module reference
----------------

.. autosummary::
   :toctree: generated
   :recursive:

   jacscanomaly.config
   jacscanomaly.criteria
   jacscanomaly.finder
   jacscanomaly.models
   jacscanomaly.seasons
   jacscanomaly.extract
   jacscanomaly.runner
   jacscanomaly.anomaly_models
   jacscanomaly.photometry
   jacscanomaly.magnification
   jacscanomaly.trajectory
   jacscanomaly.singlelens_fit
   jacscanomaly.singlelens_model
   jacscanomaly.template_free
   jacscanomaly.candidate
   jacscanomaly.signal_mask
   jacscanomaly.features
   jacscanomaly.plot
