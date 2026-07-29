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
   jacscanomaly.PlanetSignalExtractor
   jacscanomaly.PlanetSignalConfig
   jacscanomaly.PlanetSignalResult
   jacscanomaly.PlanetSignalTiming
   jacscanomaly.PlanetFeatureConfig
   jacscanomaly.PlanetFeatureResult
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
   jacscanomaly.CPPPSPLFitter
   jacscanomaly.FSPLFitter
   jacscanomaly.NativePSPLAnnualParallaxFitter
   jacscanomaly.NativeFSPLAnnualParallaxFitter
   jacscanomaly.NativePSPLSpaceParallaxFitter
   jacscanomaly.NativeFSPLSpaceParallaxFitter
   jacscanomaly.BICSingleLensFitter
   jacscanomaly.CVFitter
   jacscanomaly.SingleLensFitResult

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
   jacscanomaly.planet_signal
   jacscanomaly.plot
