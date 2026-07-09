Planet Signal Classification
============================

The planetary-signal workflow separates a local residual signal from a
refined single-lens baseline, describes its shape, and produces starting
points for downstream physical fits. It is a triage and seed-generation
tool: an atom label or seed is not a final 2L1S/1L2S model comparison.

Refine and classify a signal
----------------------------

Run the normal finder first, then use :class:`jacscanomaly.PlanetSignalExtractor`
to iteratively refit the baseline while excluding localized residual signals.
The returned :class:`jacscanomaly.PlanetSignalResult` exposes both a simple
shape classification and the residual-atom classifier:

.. code-block:: python

   from jacscanomaly import (
       Finder,
       FinderConfig,
       PlanetSignalConfig,
       PlanetSignalExtractor,
   )

   finder = Finder(FinderConfig(fitter_kind="pspl"))
   signal = PlanetSignalExtractor(
       finder,
       PlanetSignalConfig(seed_min_dchi2=100.0),
   ).extract(time, flux, ferr)

   shape = signal.classify()
   print(shape.signal_type)

   anomaly = signal.classify_anomaly()
   print(anomaly.summary_text())

``signal.classify()`` describes each connected extracted component with broad
shape labels such as peaks, dips, caustic crossings, and complex signals.
``signal.classify_anomaly()`` fits a set of local residual atoms to those
components and ranks candidate physical-model seeds.

Inspecting results
------------------

The classification result has compact dictionary and table helpers suitable
for notebooks and survey tables:

.. code-block:: python

   event_row = anomaly.summary_dict()
   segment_rows = anomaly.segment_summary_dicts()
   atom_rows = anomaly.atom_summary_dicts()
   seed_rows = anomaly.seed_summary_dicts(top_n=20)

   display(anomaly.summary_table())
   display(anomaly.atom_table())
   display(anomaly.seed_table(top_n=20))

``best_label`` and ``class_probabilities`` compare local atom morphologies
using BIC weights. ``event_seeds`` contains deduplicated downstream starting
points. Each seed records its proposed ``model_type``, ``class_label``, score,
source atom, degeneracy tag, and numerical parameters.

The optional plots provide a quick visual check before sending seeds to a
global physical-model fitter:

.. code-block:: python

   signal.plot_signal()
   anomaly.plot_summary(signal_result=signal)

Configuration
-------------

:class:`jacscanomaly.PlanetSignalConfig` controls baseline refinement and
signal extraction. The default ``baseline_mode="beam_interval"`` selects a
small set of connected masked intervals; ``"mask"`` and ``"robust"`` are
available when their behavior is more appropriate for the event. The most
important controls are ``seed_min_dchi2``, ``max_iter``, and
``max_mask_fraction``.

:class:`jacscanomaly.PlanetClassConfig` controls which morphology atoms are
considered. All standard atoms are enabled by default. Set an
``enable_*`` option to ``False`` to restrict a targeted analysis, and adjust
``min_delta_chi2_for_seed`` to control which local fits produce physical-model
seeds. The configuration also exposes numerical controls for the finite-source
fold and Chang-Refsdal lookup calculations.

Interpretation
--------------

Residual atoms include positive and negative image perturbations, central
caustic structures, fold and cusp crossings, Chang-Refsdal perturbations,
second-source-like bumps, and systematics diagnostics. They are deliberately
local descriptions of the residual light curve. Use their ranking to choose
and initialize global physical models, then evaluate those models on the full
light curve with the appropriate likelihood and priors.
