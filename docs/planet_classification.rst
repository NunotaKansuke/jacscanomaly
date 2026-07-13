Planet Signal Classification
============================

The planetary-signal workflow separates a local residual signal from a
refined single-lens baseline, describes its shape, and produces starting
points for downstream physical fits. It is a triage and seed-generation
tool: an atom label or seed is not a final 2L1S/1L2S model comparison.

Refine and classify a signal
----------------------------

Use :class:`jacscanomaly.PlanetSignalExtractor` to iteratively refit the
baseline while excluding localized residual signals. It reuses the supplied
finder's baseline fitter and template-grid configuration. The returned
:class:`jacscanomaly.PlanetSignalResult` exposes both a simple shape
classification and the residual-atom classifier:

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
   ).run(time, flux, ferr)

   shape = signal.classify()
   print(shape.signal_type)

   anomaly = signal.classify_anomaly()
   print(anomaly.summary_text())

``signal.classify()`` describes each connected extracted component with broad
shape labels such as peaks, dips, caustic crossings, and complex signals.
``signal.classify_anomaly()`` fits a set of local residual atoms to those
components and ranks candidate physical-model seeds.

Peak and duration estimates
---------------------------

``signal.classify()`` is the light-weight first measurement stage. Each
component has a time span, a broad ``signal_type``, and lists of ``peaks`` and
``dips``. A peak records its observed time, residual, z-score, interpolated
duration, and the corresponding baseline and observed magnifications.

Set ``fit_template_timescale=True`` when a local analytic template fit should
refine the timing estimate of sufficiently sampled extrema:

.. code-block:: python

   from jacscanomaly import PlanetSignalClassificationConfig

   shape = signal.classify(
       PlanetSignalClassificationConfig(
           fit_template_timescale=True,
           fit_template_min_points=6,
           fit_template_min_teff=0.01,
           fit_template_max_teff=10.0,
       )
   )
   for component in shape.components:
       for peak in component.peaks:
           print(peak.fitted_t0, peak.fitted_teff, peak.fitted_chi2)

``fitted_t0`` and ``fitted_teff`` are local template measurements, while
``t_start``, ``t_end``, and ``timescale`` are threshold-crossing estimates.
They describe a residual feature and should not be read as physical binary-lens
parameters.

To preserve a trusted baseline geometry, pass the same fixed parameters used
by :meth:`jacscanomaly.Finder.run`:

.. code-block:: python

   x0 = np.array([t0, tE, u0])
   signal = PlanetSignalExtractor(finder).run(
       time,
       flux,
       ferr,
       x0=x0,
       refit=False,
   )

The nonlinear parameters in ``x0`` remain fixed; the source and blend fluxes
are solved for the unmasked data at each refinement step.

Baseline-refinement modes
-------------------------

``PlanetSignalConfig.baseline_mode`` determines how candidate signal points
are excluded before the baseline is refined:

``"beam_interval"`` (default)
   Builds a small beam of connected mask intervals from strong template-scan
   candidates, scores the resulting baseline fits, and keeps the best mask.
   This is the general-purpose choice when anomalies are localized.

``"mask"``
   Grows masks around successive strong scan candidates and refits after each
   accepted addition. Use it when a simple greedy sequence is easier to audit.

``"robust"``
   Iteratively downweights large residuals rather than selecting hard
   intervals. Use it when the signal is broad or interval boundaries are not
   well defined.

The important safeguards are ``max_mask_fraction`` (do not mask too much of
the light curve), ``max_unmasked_chi2_dof_increase`` (reject a mask that makes
the retained data substantially worse), and ``max_refined_chi2_dof_ratio``
(fall back from a catastrophic refinement). ``prior_signal_windows`` can seed
known intervals as ``(center_time, half_width)`` pairs.

The result retains the complete refinement history:

.. code-block:: python

   print(signal.initial_fit.chi2_dof)
   print(signal.refined_fit.chi2_dof)
   print(signal.signal_mask.sum())
   for step in signal.iterations:
       print(step.iteration, step.added_points)

``initial_residual`` is the residual before refinement;
``refined_residual`` is the residual against the final baseline;
``signal_mask`` identifies points excluded from that baseline fit; and
``candidates`` contains the contiguous extracted intervals.

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

Residual atoms and seeds
------------------------

The atom fits are local alternatives evaluated independently on each extracted
component. They cover positive and negative image perturbations, central
caustic structures, fold and cusp crossings, Chang-Refsdal perturbations,
second-source-like bumps, smooth PSPL misfits, and a systematics diagnostic.
The atom name and local parameters appear in ``atom_table()``; use
``delta_chi2``, ``bic``, ``score``, and warnings together rather than treating
one label as definitive.

Only successful, sufficiently strong atom fits generate ``SeedCandidate``
objects. A seed has ``model_type`` (for example ``"2L1S"`` or ``"1L2S"``),
``params``, a source atom, and an optional close/wide degeneracy tag. It is an
initialization proposal for a later global fit, not a posterior sample or a
final classification.

The atom stage performs the more detailed local parameter estimation. It uses
the refined residual in each extracted component, adds a low-order local
baseline where appropriate, fits all enabled morphology atoms, and sorts the
successful fits by BIC. For example:

.. code-block:: python

   from jacscanomaly import PlanetClassConfig

   anomaly = signal.classify_anomaly(
       PlanetClassConfig(
           estimate_param_errors=True,
           min_delta_chi2_for_seed=20.0,
       )
   )
   for segment in anomaly.segment_results:
       for atom in segment.atom_fits:
           print(atom.atom_name, atom.params, atom.param_errors, atom.bic)

``AtomFitResult.params`` is intentionally atom-specific. Bump-like atoms may
report a peak time and width; fold/cusp atoms report contact or crossing times
and finite-source scales; central and Chang-Refsdal atoms report their local
geometry parameters. ``param_errors`` is populated only when the local
covariance estimate is well conditioned. Check ``success``, ``warnings``, and
``validity_penalty`` before using any estimate.

The complete template atlas, model equations, exact parameter names, and the
distinction between direct constraints and approximate physical seeds are in
:doc:`morphology_classification_method`. In particular, do not read
``q_curv`` as a mass ratio, a generic ``width/tE`` as ``rho``, or a
``shear_quadrupole`` proxy as a measured Chang--Refsdal shear.

To inspect only finite, physically useful quantities without assuming that
all templates share the same parameter set:

.. code-block:: python

   import numpy as np

   best = anomaly.best_atom
   useful = {
       name: value
       for name, value in (best.params if best else {}).items()
       if np.isscalar(value) and np.isfinite(value)
   }
   print(best.class_label, useful)

The default enabled set includes simple positive/negative perturbations,
central perturbations, fold variants, cusp variants, Chang-Refsdal,
second-PSPL-like, smooth-misfit, shear, and systematics atoms. Disable
unneeded families with the corresponding ``enable_*`` fields in
``PlanetClassConfig`` to make a targeted fit faster and easier to inspect.

Interpreting fit tables
-----------------------

``anomaly.summary_table()`` has one row per extracted component and its best
atom. ``anomaly.atom_table()`` has one row per retained atom fit. The most
useful columns are:

``atom_name`` / ``class_label``
   Local model identity and a human-readable morphology label.

``chi2`` / ``delta_chi2`` / ``bic``
   Local fit quality. Larger ``delta_chi2`` is an improvement over the local
   baseline; lower BIC is preferred among the fitted local alternatives.

``score`` / ``validity_penalty`` / ``warnings``
   Triage diagnostics. A fit can be statistically competitive but receive a
   validity warning for cadence-limited width or a boundary solution.

``*_err``
   Finite-difference local covariance uncertainty for the matching parameter,
   when the estimate is available.

``anomaly.seed_table()`` has one row per deduplicated physical-model seed. Its
``params`` are flattened into columns, so it can be exported directly to a
global-model fitting queue.

Interpretation
--------------

Residual atoms include positive and negative image perturbations, central
caustic structures, fold and cusp crossings, Chang-Refsdal perturbations,
second-source-like bumps, and systematics diagnostics. They are deliberately
local descriptions of the residual light curve. Use their ranking to choose
and initialize global physical models, then evaluate those models on the full
light curve with the appropriate likelihood and priors.
