Morphology Classification and Local Physical Constraints
=========================================================

The classification stage has two layers. First,
``PlanetSignalClassifier`` summarizes the shape of hard-masked residual
components. Second, ``PlanetAnomalyClassifier`` fits a collection of local
residual atoms and reports only locally identifiable physical quantities.
Neither layer is a global 2L1S/1L2S posterior calculation, and the classifier
does not expand a morphology fit into assumed ``q``, ``s``, or ``alpha`` grids.

Within the atom layer, morphology ranking and physical inference are separate.
A flexible atom may win the BIC classification without producing any physical
parameter. Conversely, a physical local atom is exposed as an estimate only
when it is successful, away from fit boundaries, sufficiently strong, and
within ``physical_window_max_delta_bic`` of the best atom fitted to the same
local window.

Local-feature decomposition
---------------------------

Whole-component atoms are used for morphology ranking and locator generation,
not as the source of published physical constraints. The classifier extracts
candidate fold entry/exit times from crossing morphologies, contact times from
fold-family routes, and compact positive or negative extrema from the component
catalog. Derivative extrema provide fallback edge locators for complex broad
signals.

Each locator defines an independent time window containing both signal and
neighboring baseline points. A straight fold, and only when routed an
appropriate curved, limb-darkened, or grazing fold, is then fitted in that
window. Compact isolated extrema can instead receive the normalized
Chang--Refsdal fit. The output records ``window_id``, ``locator_kind``,
``locator_time``, and the actual window bounds, so every physical estimate is
traceable to the data that constrain it.

For two independently valid entry and exit folds,

.. math::

   R_i = \frac{t_{*,i}}{t_E} = \frac{\rho}{|\sin\psi_i|}

is reported for each edge. Assuming the same source radius permits only the
additional relation

.. math::

   \frac{|\sin\psi_{\rm entry}|}{|\sin\psi_{\rm exit}|}
   = \frac{R_{\rm exit}}{R_{\rm entry}}.

This relation and the independently measured center-crossing duration are
returned by ``physical_relation_dicts()``. Neither measurement yields
``rho`` without an external fold angle, and neither is taken from the six-basis
``full_caustic_crossing`` morphology model.

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

Template atlas and parameter semantics
--------------------------------------

``AtomFitResult.params`` is intentionally template-specific. Parameters fall
into three different categories:

* **fitted coordinates** are nonlinear or profiled-linear parameters of the
  local residual model;
* **derived constraints** follow algebraically from those fitted coordinates
  and the refined PSPL ``tE``;
* **deterministic reparameterizations** preserve all information in fitted
  physical coordinates, for example ``q=sqrt_q**2``.

The parameter name reflects this distinction. For example,
``rho_over_abs_sin_psi`` is the fold constraint
:math:`t_*/t_E`; :math:`\psi` is the angle between the trajectory and the
local fold tangent, not the global binary-axis angle :math:`\alpha`.
``characteristic_scale_over_tE`` is only a normalized phenomenological width.
``q_curv`` is a local curvature coefficient and is never a binary mass ratio.

The currently implemented templates and their principal outputs are:

.. list-table:: Residual-template atlas
   :header-rows: 1
   :widths: 22 25 53

   * - Class label
     - Local morphology
     - Principal reported quantities
   * - ``major_image_bump``
     - Positive Lorentzian-like image perturbation
     - ``t_peak``, ``width``
   * - ``major_image_pspl_bump``
     - Positive PSPL-shaped perturbation
     - ``t_peak``, ``tE_pert``, ``u0_pert``
   * - ``minor_image_dip``
     - Negative Lorentzian-like image perturbation
     - ``t_peak``, ``width``
   * - ``minor_image_box_trough``
     - Soft-edged negative trough
     - ``t_start``, ``t_end``, ``width``, ``edge_width``
   * - ``fold_caustic``
     - Straight uniform-source fold
     - ``tc``, ``t_limb``, ``tstar``, ``entry_exit_sign``,
       ``rho_over_abs_sin_psi``
   * - ``limb_darkened_fold_caustic``
     - Straight fold with linear limb darkening
     - Straight-fold quantities plus effective ``Gamma``
   * - ``curved_fold_caustic``
     - Quadratic local fold distance
     - Limb contacts ``t_entry/t_exit``, center crossings ``tc1/tc2``,
       local ``tstar_entry/exit`` and local
       ``rho_over_abs_sin_psi_entry/exit``
   * - ``grazing_fold_caustic``
     - Limb-only or shallow quadratic fold encounter
     - ``t_stationary``, ``z_stationary``; ``t_closest/z_closest`` for a
       convex trajectory; all real limb and center roots and their local
       ``tstar`` and ``rho_over_abs_sin_psi``
   * - ``two_fold_caustic``
     - Unresolved pair of fold contributions
     - ``tc1/tc2``, common ``tstar_1/2``, fold-strength ratio, and
       ``contact_separation_over_2tstar``
   * - ``full_caustic_crossing``
     - Entry, interior, and exit across a broad segment
     - Morphology only: ``t_entry/t_exit``, ``entry_edge_scale``,
       ``exit_edge_scale``, inside duration, and asymmetry. Its six freely
       weighted bases do not identify :math:`t_*` or :math:`\rho`.
   * - ``rim_trough_caustic``
     - Phenomenological bump--dip--bump profile
     - Rim/trough times, rim separation and asymmetry, and
       ``characteristic_scale_over_tE``; no direct ``rho`` claim
   * - ``cusp_caustic``
     - Softened one-dimensional cusp tail
     - ``ta``, ``b``, tail power ``p``, ``effective_core_duration``, and
       ``cusp_scale_over_tE``
   * - ``canonical_cusp``
     - Point-source canonical cusp map
     - Canonical coordinates ``eta1_0/eta2_0``, ``omega1/omega2``, closest
       time, cusp impact, and discriminant; canonical scales are local
   * - ``finite_source_cusp``
     - Unit-disc convolution of the canonical cusp map
     - Canonical-cusp geometry plus ``tstar_cusp_local`` and
       ``rho_over_sinalpha_cusp_local`` in the lookup normalization
   * - ``chang_refsdal``
     - Local image perturbation lookup
     - Fixed-normalization flux fit: image branch, planet coordinates, ``s``,
       ``alpha``, ``q``, ``gamma``, and a grid estimate or bound on
       ``rho_over_sqrt_q`` and ``rho``
   * - ``central_caustic`` / ``central_double_cusp``
     - Symmetric or double-cusp central morphology
     - Central times and the identifiable combination
       ``C_chord*q/(s-1/s)^2 = Delta_t/(4*tE)``; ``C_chord`` remains unknown
   * - ``second_pspl_like``
     - A second PSPL-shaped residual bump
     - ``t0_2``, ``tE_2``, ``u0_2``, flux ratio, and ``tE_2/tE``
   * - ``shear_quadrupole``
     - Broad even/odd smooth basis
     - Dimensionless ``gamma`` proxy and ``shear_basis_angle``; this is not a
       measured Chang--Refsdal shear and is not converted to ``q`` or ``s``
   * - ``pspl_misfit`` / ``systematics_candidate``
     - Baseline derivatives or sparse artifacts
     - Diagnostic parameters only; no planet parameters inferred

Every profiled atom also reports its fitted ``amplitude``. Multi-column atoms
may additionally report ``amplitude_1``, ``amplitude_2``, and later columns.
These are residual-flux coefficients. They are not mass ratios or caustic
strengths independent of source flux and the local coordinate normalization.

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

For a straight fold the first limb contact is
:math:`t_{\rm limb}=t_c-s_{\rm ent}t_*`. For curved and grazing folds, the
code solves :math:`z(t)=-1` for limb contacts and :math:`z(t)=0` for
source-center crossings. At each real root it computes

.. math::

   t_{*,\mathrm{local}} = \left|\frac{dz}{dt}\right|^{-1},
   \qquad
   \left(\frac{\rho}{|\sin\alpha|}\right)_{\mathrm{local}}
   = \frac{t_{*,\mathrm{local}}}{t_E}.

The grazing atom uses

.. math::

   z(t)=z_0+\frac{t-t_a}{w}
   +q_{\rm curv}\left(\frac{t-t_a}{w}\right)^2.

It reports every real root. A missing ``t_contact_*`` therefore means the
fitted quadratic has no real :math:`z=-1` contact, not that the calculation
was skipped. For positive curvature its vertex is a closest approach and is
also reported as ``t_closest`` and ``z_closest``; for negative curvature it is
only a stationary maximum.

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

The canonical implementation currently uses a one-direction trajectory in
canonical coordinates, :math:`\eta_1=\eta_{1,0}+(t-t_a)/w` and
:math:`\eta_2=\eta_{2,0}`. Consequently ``trajectory_angle_cusp`` is zero in
that coordinate convention and is not the binary-axis angle ``alpha``. The
point-source canonical map has an arbitrary local scale. Only the
finite-source lookup, whose convolution disc has unit source radius, exposes
``tstar_cusp_local`` and ``rho_over_sinalpha_cusp_local``; these remain local
normalization estimates until a global lens map fixes the canonical scaling.

Non-caustic morphology atoms
----------------------------

The positive-bump atom uses a PSPL-like local profile,

.. math::

   K_+(t) = A_0\left(\sqrt{b_p^2+
   \left(\frac{t-t_a}{t_p}\right)^2}\right)-1.

Its fitted peak time and width describe the residual morphology. Assigning the
feature to a particular unperturbed image and converting its width into a
planet Einstein radius introduces topology and crossing-scale assumptions.
The classifier therefore does not report ``q``, ``s``, or ``alpha`` from this
atom. The same rule applies to the negative-dip and box-trough atoms.

Normalized local Chang--Refsdal fit
-----------------------------------

For an isolated perturbation of one PSPL image, the physical-local atom uses

.. math::

   \Delta F(t) = F_s A_j(t)
   \left[R_{\rm CR}\left(
   \frac{\boldsymbol x_j(t)-\boldsymbol s}{\sqrt q};\gamma,
   \frac{\rho}{\sqrt q}\right)-1\right],
   \qquad \gamma=s^{-2}.

The radial/tangential host Jacobian is
:math:`\operatorname{diag}(1+\gamma,1-\gamma)`. Unlike morphology atoms, this
model has no free residual-amplitude coefficient: ``Fs``, the unperturbed
image magnification, and the Chang--Refsdal map set the absolute flux scale.
Consequently a successful isolated perturbation can constrain local ``s``,
``q``, and ``alpha`` rather than merely generating their grid.

Finite-source maps are evaluated on the configured
``cr_lookup_source_radius_grid``. An optimum at zero is reported as an upper
limit, an optimum at the largest radius as a lower limit, and an interior
optimum with neighboring-grid midpoint bounds. It is never converted into the
point estimate ``rho=0``. Multiple branch/radius modes are retained in
``fit_diagnostics['physical_modes']``. Boundary solutions and local CR fits
that are strongly disfavored by BIC remain auditable atom fits but are not
published as physical estimates.
The default ``cr_physical_q_max`` also rejects fitted mass ratios above 0.03,
where neglected higher-order host-lens terms make the planetary local
expansion unreliable.
For the same reason, CR routing is limited to single-peak, dip, weak, or
compact complex/caustic components whose FWHM is at most
``cr_max_fwhm_tE_fraction`` times :math:`t_E`. Broad and whole-event residuals
remain available to morphology atoms but are outside this local expansion.

Central, second-source, and shear relations
-------------------------------------------

For a planetary central caustic away from the resonant regime, the on-axis
width has the familiar local scaling

.. math::

   \Delta\xi_c \simeq \frac{4q}{(s-s^{-1})^2}.

The central atom's fitted duration :math:`\Delta t` constrains the projected
combination

.. math::

   \frac{\Delta t}{4t_E}
   = C_{\rm chord}\frac{q}{(s-s^{-1})^2}.

Here :math:`C_{\rm chord}` contains the unknown trajectory projection, chord,
and cusp-proximity factors. The classifier retains this factor in the reported
parameter name and does not set it to one. Consequently it reports neither a
unique :math:`q` nor a unique :math:`s` from this atom. Near :math:`s=1`, the
planetary central-caustic approximation itself also becomes unreliable.

The second-PSPL atom is

.. math::

   K_{\rm 2PSPL}(t)=A_0\left(\sqrt{u_{0,2}^2+
   \left(\frac{t-t_{0,2}}{t_{E,2}}\right)^2}\right)-1.

It reports only the fitted second-PSPL coordinates ``t0_2``, ``tE_2``,
``u0_2``, the flux ratio, and the deterministic ratio ``tE_2/tE``. Interpreting
that timescale ratio as a lens mass ratio or a repeating-lens geometry requires
an additional model assumption and is therefore not performed.

A broad quadrupole-like residual is fitted with generic even and odd smooth
basis functions. Their coefficients are divided by ``Fs`` to form the
dimensionless ``gamma`` proxy. It is not an exact lens-equation derivative or
a measured Chang--Refsdal shear, so it is not converted to :math:`q` or
:math:`s`.

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

Publishing local physical information
-------------------------------------

Only successful, sufficiently significant, non-boundary local physical fits
are published. Fold fits report crossing times and
``rho_over_abs_sin_psi``. Central fits report the chord-factor combination
above. Valid normalized Chang--Refsdal fits report their native local
coordinates and deterministic reparameterizations; grid-edge finite-source
solutions are reported as bounds. Morphology-only bumps, dips, smooth shear
proxies, and second-PSPL alternatives are not converted into lens parameters.
