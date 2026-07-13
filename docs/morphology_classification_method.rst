Heuristic Anomaly Estimation Method
===================================

``PlanetAnomalyClassifier`` treats each extracted anomaly component as a
local perturbation of the refined single-lens (PSPL) light curve and extracts
only quantities that the local data determine well:

1. the **local shape** of the signal (bump, dip, fold crossing, caustic
   crossing, or no coherent shape),
2. the **position of the anomaly relative to the PSPL magnification
   pattern**, expressed as :math:`(\tau_{\rm anom}, u_{\rm anom}, \alpha,
   s^\dagger_\pm)`, and
3. the **timescale of the anomaly relative to** :math:`t_E`, converted to a
   mass-ratio estimate only where a published relation exists.

The formalism is the standard heuristic analysis used throughout the
microlensing literature (Gould & Loeb 1992; Gaudi & Gould 1997; Han 2006;
Hwang et al. 2022, AJ 163, 43; Ryu et al. 2022).  It deliberately does not
fit a two-lens model: the output is a set of well-determined observables and
their deterministic transformations, intended to seed and sanity-check a
subsequent global analysis.

PSPL reference frame
--------------------

The estimator works in the trajectory frame of the refined PSPL fit,

.. math::

   \boldsymbol u(t) = \left(\frac{t-t_0}{t_E}, u_0\right),
   \qquad u(t) = |\boldsymbol u(t)|,
   \qquad A_0(u) = \frac{u^2+2}{u\sqrt{u^2+4}}.

All shape templates are fitted in flux-residual space,
:math:`r_i = F_i - F_{\rm PSPL}(t_i)`, as a residual dictionary

.. math::

   r(t) = \mathcal P_m(t) + \sum_j B_j K_j(t;\boldsymbol\theta) + \epsilon(t),

where :math:`\mathcal P_m` is a low-order nuisance polynomial and the linear
amplitudes :math:`B_j` are profiled out at fixed nonlinear parameters
:math:`\boldsymbol\theta`.

Stage 1: shape measurement
--------------------------

Five templates are compared by BIC.  Routing is minimal: the bump (dip)
template is tried when the positive (negative) residual power is at least
``mixed_sign_power_fraction`` of the total, and the fold and crossing
templates require ``min_points_fold`` and ``min_points_crossing`` points.

``bump``
   PSPL-shaped positive perturbation,
   :math:`K = A_0\!\left(\sqrt{u_p^2 + ((t-t_c)/t_p)^2}\right)-1`.
   Measures the center time ``t_anom``, the width parameter ``t_p``, and the
   profile FWHM.

``dip``
   Smoothed-box trough with center ``t_anom``, full duration ``dt_dip``, and
   edge width.  This is the expected morphology of a minor-image
   demagnification between the two triangular planetary caustics.

``fold``
   One finite-source fold crossing, :math:`K = G_0(\pm(t-t_c)/t_*)`, with the
   uniform-source fold kernel

   .. math::

      G_0(z) = \frac{2}{\pi}\,\Theta(1+z)
      \int_{\max(-z,-1)}^1 \frac{\sqrt{1-x^2}}{\sqrt{x+z}}\,dx,

   evaluated at the signed source-center distance
   :math:`z(t) = \pm(t-t_c)/t_*`.  Measures the crossing time and the
   source-radius crossing time :math:`t_*`.

``caustic_crossing``
   Entry fold + exit fold + interior plateau,
   :math:`[G_0((t-t_{\rm en})/t_{*,1}),\; G_0(-(t-t_{\rm ex})/t_{*,2}),\;
   W(t)]` with a smooth window :math:`W`.  Measures ``t_entry``, ``t_exit``,
   ``dt_cc``, and one :math:`t_*` per edge.

``null``
   Nuisance polynomial only.  It provides ``chi2_null`` for ``delta_chi2``
   and, when it wins the BIC comparison, the component is labeled
   ``no_coherent_shape``.  A winning template below ``min_delta_chi2`` is
   labeled ``low_significance``.

Parameter uncertainties come from a finite-difference Hessian of the profiled
chi-square, propagated through the parameter transformations; they are local
Gaussian estimates.

Stage 2: deterministic geometry
-------------------------------

The anomaly epoch fixes where the source was on the PSPL magnification
pattern.  With :math:`t_{\rm anom}` from the shape fit,

.. math::

   \tau_{\rm anom} = \frac{t_{\rm anom}-t_0}{t_E},
   \qquad
   u_{\rm anom} = \sqrt{\tau_{\rm anom}^2 + u_0^2},
   \qquad
   \tan\alpha = \frac{u_0}{\tau_{\rm anom}}.

:math:`\alpha` is the angle between the source trajectory and the planet-host
axis; it is mirror-degenerate under :math:`u_0 \to -u_0`.

A planet perturbs an image when it lies at that image's position.  The major
and minor image radii at :math:`u_{\rm anom}` give the projected separation
estimate

.. math::

   s^\dagger_\pm = \frac{\sqrt{u_{\rm anom}^2+4} \pm u_{\rm anom}}{2},

with the ``+`` branch for bumps (major-image perturbations) and the ``-``
branch for dips (minor-image perturbations).  Note the exact identities
:math:`s^\dagger_+ s^\dagger_- = 1` and
:math:`s^\dagger_+ - s^\dagger_- = u_{\rm anom}`.  Each branch is the
geometric mean of the two degenerate inner/outer solutions,
:math:`s^\dagger = \sqrt{s_{\rm in} s_{\rm out}}`; the estimator reports both
branches and marks the preferred one.

When :math:`u_{\rm anom} \le` ``central_u_anom_max`` both branches approach 1
and a central-caustic origin cannot be excluded.  The geometry is then
flagged ``central_or_resonant`` and no mass-ratio estimate is reported,
because the anomaly location no longer determines the caustic type.

Stage 3: timescales and mass-ratio estimates
--------------------------------------------

The duration ratio ``dt/tE`` is always reported.  Mass-ratio estimates are
attached only where a published relation applies, and each carries a
``q_method`` tag naming its assumption:

* **Dips** (``dip_han2006``; Han 2006; Hwang et al. 2022): with the full dip
  duration :math:`\Delta t_{\rm dip}`,

  .. math::

     q = \left(\frac{\Delta t_{\rm dip}}{4 t_E}\right)^2
     \frac{s^\dagger_-}{u_{\rm anom}}\,\sin^2\alpha .

  This equals the frequently quoted
  :math:`(\Delta t_{\rm dip}/4t_E)^2\,(s^\dagger_-/|u_0|)\,|\sin^3\alpha|`
  but remains regular as :math:`u_0 \to 0`.

* **Bumps** (``bump_planet_einstein_crossing``; Gould & Loeb 1992): the
  perturbed region is taken to be the planet's Einstein ring, whose crossing
  time is :math:`\sqrt{q}\,t_E`, so :math:`q \simeq (t_p/t_E)^2`.  This is an
  order-of-magnitude estimate.

* **Fold and caustic crossings**: a fold fit constrains

  .. math::

     \frac{t_*}{t_E} = \frac{\rho}{|\sin\psi|},

  where :math:`\psi` is the angle between the trajectory and the local fold
  tangent (not :math:`\alpha`).  Crossings additionally constrain
  ``dt_cc/tE``.  The mass ratio is **not** computed from a crossing: the
  local data do not determine it without a caustic model.

Both heuristic ``q`` estimators are typically accurate to a factor of ~2
compared with full modeling (Hwang et al. 2022); their reported uncertainties
propagate only the measurement errors, not this systematic limit.

What the estimator does not do
------------------------------

* It does not fit 2L1S or 1L2S models, caustic geometries, or parameter
  grids.
* It does not resolve the inner/outer or :math:`u_0`-mirror degeneracies;
  it reports the degeneracy-invariant combinations instead.
* It does not convert a central/resonant-regime anomaly into ``q`` or ``s``.
* It does not decide whether a bump is planetary or a second source (1L2S);
  the bump template parameters (``t_anom``, ``t_p``, ``u_p``, amplitude) are
  exactly the measurements such a comparison needs downstream.
