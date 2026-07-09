Morphology Classification and Seed Generation
==============================================

The classification stage has two layers. First,
``PlanetSignalClassifier`` summarizes the shape of hard-masked residual
components. Second, ``PlanetAnomalyClassifier`` fits a collection of local
residual atoms and converts strong fits into physical-model starting points.
Neither layer is a global 2L1S/1L2S posterior calculation.

Component shape classification
------------------------------

For each connected signal-mask interval, the classifier computes positive and
negative residual power,

.. math::

   C_+ = \sum_{z_i > 0} z_i^2, \qquad
   C_- = \sum_{z_i < 0} z_i^2,

and smooths the z-score series before finding extrema. An extremum must exceed
both ``min_peak_abs_z`` and a fraction of the strongest local extremum. Nearby
extrema are separated only when their intervening valley has sufficient
prominence. Overlapping extrema are suppressed.

The broad component label is then assigned deterministically:

``whole_event_anomaly``
   The extractor's flat-baseline diagnostic selected a flat baseline.

``dip``
   At least one dip exists and negative residual power dominates by
   ``negative_dominance``.

``caustic_crossing``
   At least two positive peaks exist and positive power is not subdominant.

``single_peak``
   One positive peak exists and positive power dominates by
   ``positive_dominance``.

``low_significance`` or ``complex``
   No prominent extrema, or a mixture not covered by the preceding rules.

This label controls which atom families are attempted. It does not determine a
physical lens topology.

Features and atom routing
-------------------------

For every component, the classifier measures peak time and sign, duration,
FWHM-like width, cadence, signed residual power, SNR, skewness, kurtosis,
edge sharpness, distance from the PSPL peak, and PSPL impact parameter at the
feature. These features route computation toward plausible atoms. Examples:

* positive-dominant components try positive bump and PSPL-bump atoms;
* negative-dominant components try dip and minor-image box-trough atoms;
* central, mixed-sign components can try central perturbation and double-cusp
  atoms;
* sharp, strong, or caustic-crossing-like components try fold, grazing,
  limb-darkened, two-fold, and cusp families;
* image-perturbation-like components can try Chang-Refsdal;
* broad smooth or sparse/cadence-limited components also try diagnostic shear,
  PSPL-misfit, or systematics atoms.

The ``enable_*`` settings in ``PlanetClassConfig`` can disable any family. The
routing tests save computation; they are not a claim that excluded atoms are
physically impossible.

Local atom fitting and ranking
------------------------------

Each enabled atom is fitted independently on the component residual with its
appropriate local baseline treatment. An ``AtomFitResult`` stores fitted
parameters, local chi-square, baseline chi-square, improvement, AIC, BIC,
score, success state, warnings, and a validity penalty. When
``estimate_param_errors=True``, a finite-difference local covariance estimate
is reported only if it is numerically well conditioned.

Atoms are ranked by BIC. Within a component, the class weight is

.. math::

   w_k = \sum_{j \in k} \exp[-(\mathrm{BIC}_j - \mathrm{BIC}_{\min})/2],
   \qquad p_k = w_k / \sum_l w_l.

At event level, each component's class probabilities are weighted by its local
residual power before normalization. These values are relative support among
the fitted local alternatives, not calibrated posterior probabilities.

From local fits to physical seeds
---------------------------------

Only successful atoms with ``delta_chi2`` at least
``min_delta_chi2_for_seed`` emit seeds. The conversion intentionally preserves
degeneracies:

* major-image bumps produce wide and close counterparts using the PSPL image
  position, a width-derived mass-ratio grid, and trajectory angle;
* minor-image dips produce close and wide counterparts;
* central perturbations scan configured separation and angle grids;
* fold and cusp fits emit local caustic timing and scale parameters;
* Chang-Refsdal fits emit local shear/image coordinates;
* second-PSPL-like fits emit a 1L2S seed and wide repeating 2L1S alternatives.

Seeds are deduplicated by model type, class, degeneracy tag, and rounded finite
parameters, then ranked by local score. They are intended to initialize a
subsequent full-light-curve physical fit. Treat warnings, cadence limitations,
and boundary solutions as reasons to broaden or reject a seed grid.
