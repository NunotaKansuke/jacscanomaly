Morphology Classification and Seed Generation
==============================================

The classification stage has two layers. First,
``PlanetSignalClassifier`` summarizes the shape of hard-masked residual
components. Second, ``PlanetAnomalyClassifier`` fits a collection of local
residual atoms and converts strong fits into physical-model starting points.
Neither layer is a global 2L1S/1L2S posterior calculation.

PSPL reference frame and residual dictionary
--------------------------------------------

The classifier works in the trajectory frame of the refined PSPL fit,

.. math::

   \boldsymbol u(t) = \left(\frac{t-t_0}{t_E}, u_0\right),
   \qquad u(t) = |\boldsymbol u(t)|,

with baseline magnification and flux

.. math::

   A_0(u) = \frac{u^2+2}{u\sqrt{u^2+4}},
   \qquad F_{\rm PSPL}(t) = F_s A_0[u(t)] + F_b.

All morphology atoms are fitted in flux residual space,

.. math::

   r_i = F_i - F_{\rm PSPL}(t_i),

not in magnitude residuals. Locally, the model is a residual dictionary,

.. math::

   r(t) = \mathcal P_m(t) + B\,K(t;\boldsymbol\theta) + \epsilon(t),

where :math:`\mathcal P_m` is a low-order nuisance polynomial, :math:`K` is
one atom, and :math:`B` and polynomial coefficients are linear parameters once
the atom's nonlinear parameters :math:`\boldsymbol\theta` are fixed. This is
why atom fits are fast enough to compare many local alternatives.

PSPL image geometry
-------------------

The host lens creates major and minor images with radii

.. math::

   r_+(u) = \frac{\sqrt{u^2+4}+u}{2},
   \qquad
   r_-(u) = \frac{\sqrt{u^2+4}-u}{2} = r_+^{-1}.

Their positions are :math:`\boldsymbol x_+=r_+\hat{\boldsymbol u}` and
:math:`\boldsymbol x_-=-r_-\hat{\boldsymbol u}`. The absolute image
magnifications are

.. math::

   A_+ = \frac{A_0+1}{2},
   \qquad A_- = \frac{A_0-1}{2}.

This provides the physical intuition for the initial routing: a short positive
bump is a plausible major-image perturbation; a short negative dip is a
plausible minor-image perturbation. The routing remains deliberately
non-exclusive because real anomalies can violate this simple picture.

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

Finite-source fold and cusp atoms
---------------------------------

Near a straight fold, the point-source contribution of the new image pair has
the local form :math:`A_{\rm fold}^{\rm pt}\propto
\Theta(u_\perp)/\sqrt{u_\perp}`. Finite-source integration replaces the
singularity by a kernel. With signed source-center distance in source-radius
units :math:`z=d_\perp/\rho`, the uniform-source kernel used by the code is

.. math::

   G_0(z) = \frac{2}{\pi}\Theta(1+z)
   \int_{\max(-z,-1)}^1
   \frac{\sqrt{1-x^2}}{\sqrt{x+z}}\,dx.

The straight-fold residual atom evaluates this kernel at

.. math::

   z(t) = s_{\rm ent}\frac{t-t_c}{t_*},
   \qquad s_{\rm ent}\in\{-1,+1\},

where :math:`t_c` is the source-center crossing time and :math:`t_*` is the
source-radius crossing time. The limb-darkened variant uses

.. math::

   \mathcal F_\Gamma(z) = G_0(z) + \Gamma[G_{1/2}(z)-G_0(z)].

Curved folds replace the straight signed distance with

.. math::

   z(t) = s_{\rm ent}\left[\frac{t-t_c}{t_*} + q_{\rm curv}
   \left(\frac{t-t_c}{t_*}\right)^2\right].

Thus a fold fit constrains :math:`t_*/t_E = \rho/|\sin\alpha|`, not
:math:`\rho` and crossing angle separately. Two-fold and full-crossing atoms
retain the local entry/exit timing and relative-strength information, but they
still do not determine a global caustic topology.

Cusp atoms use either a softened cusp-tail scaling or a canonical local cusp
map. The tail family has the form

.. math::

   K_{\rm cusp}(t) =
   \left[b^2 + \left(\frac{t-t_a}{w}\right)^2\right]^{-p/2},
   \qquad p\in\left\{1,\frac{2}{3}\right\}.

The two exponents represent the common axial and transverse local cusp
scalings. These are local asymptotic descriptions, so finite-source cusp and
canonical-cusp fits are diagnostic inputs to a global model, not replacements
for it.

Non-caustic atoms and image-based seeds
---------------------------------------

The positive-bump atom uses a PSPL-like local profile,

.. math::

   K_+(t) = A_0\left(\sqrt{b_p^2+
   \left(\frac{t-t_a}{t_p}\right)^2}\right)-1.

At the fitted peak time, its wide major-image seed is centered on

.. math::

   s_0 = r_+[u(t_a)],
   \qquad \alpha_0 = \arg\boldsymbol u(t_a).

The code uses the duration scaling

.. math::

   q_{\rm base} = \left(\frac{w}{t_E}\right)^2

and expands it by ``q_width_factors`` before clipping to ``q_floor`` and
``q_ceil``. It also emits the close counterpart
:math:`s\rightarrow1/s,\ \alpha\rightarrow\alpha+\pi`.

For a negative dip, the analogous minor-image seed is

.. math::

   s_0 = r_-[u(t_d)],
   \qquad \alpha_0 = \arg[-\boldsymbol u(t_d)],

with the same width-to-q scaling and a wide counterpart. The negative-dip and
box-trough atoms are phenomenological shape fits; the image formulas provide a
physics-informed seed center rather than a measured planet separation.

Central, second-source, and shear relations
-------------------------------------------

For a planetary central caustic away from the resonant regime, the on-axis
width has the familiar local scaling

.. math::

   \Delta\xi_c \simeq \frac{4q}{(s-s^{-1})^2}.

The central atom's fitted duration :math:`\Delta t` therefore defines the seed
relation

.. math::

   q(s) \simeq \frac{\Delta t}{4t_E}(s-s^{-1})^2.

The classifier evaluates this over ``s_central_grid`` and an angle grid rather
than choosing one separation. Near :math:`s=1`, the approximation becomes
fragile, so the generated seed carries a resonant-regime warning.

The second-PSPL atom is

.. math::

   K_{\rm 2PSPL}(t)=A_0\left(\sqrt{u_{0,2}^2+
   \left(\frac{t-t_{0,2}}{t_{E,2}}\right)^2}\right)-1.

It produces a direct 1L2S seed. It also produces wide repeating 2L1S seeds
using

.. math::

   q \simeq \left(\frac{t_{E,2}}{t_E}\right)^2,
   \qquad
   (\Delta x,\Delta y) \simeq
   \left(\frac{t_{0,2}-t_0}{t_E},
   u_0 \pm u_{0,2}\sqrt q\right),

followed by :math:`s=\sqrt{\Delta x^2+\Delta y^2}` and
:math:`\alpha=\operatorname{atan2}(\Delta y,\Delta x)`. The two signs are a
trajectory-side degeneracy.

A broad quadrupole-like residual can instead constrain a local shear. Wide
seeds use :math:`q\simeq\gamma s^2`; close seeds use
:math:`q\simeq\gamma/s^2` on a small configured separation grid. This is a
local relation only and is intentionally emitted as a diagnostic seed family.

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
