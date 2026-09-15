Configuration
=============

All high-level options are stored in :class:`jacscanomaly.FinderConfig`.
The relationship between configuration and the available search workflows is
described in :doc:`workflows`.

Single-lens model
-----------------

``fitter_kind`` selects the baseline model:

.. code-block:: python

   from jacscanomaly import FinderConfig

   config = FinderConfig(fitter_kind="pspl")

Supported values:

``"pspl"``
   Point-source point-lens model.

``"fspl"``
   Finite-source point-lens model.

``"pspl_parallax"``
   PSPL with annual or space parallax. Requires ``ra_deg`` and ``dec_deg``;
   choose the observer geometry with ``parallax_geometry``.

``"fspl_parallax"``
   FSPL with annual or space parallax. Requires ``ra_deg`` and ``dec_deg``;
   choose the observer geometry with ``parallax_geometry``.

For parallax models:

.. code-block:: python

   config = FinderConfig(
       fitter_kind="pspl_parallax",
       ra_deg=270.0,
       dec_deg=-30.0,
       tref=None,  # defaults to the observed brightening peak
       parallax_geometry="annual",
   )

For space-parallax models, pass a VBMicrolensing/RTModel satellite table:

.. code-block:: python

   config = FinderConfig(
       fitter_kind="pspl_parallax",
       ra_deg=267.623337808,
       dec_deg=-29.1164180355,
       tref=2459000.0,
       parallax_geometry="space",
       satellite_ephemeris_path="satellitedir/satellite1.txt",
   )

The satellite table is expected to contain rows of
``JD RA_deg Dec_deg distance_AU`` inside an optional ``$$SOE`` / ``$$EOE``
block, matching the VBMicrolensing satellite-table convention.

Both parallax model kinds use the same compiled trajectory/evaluator and the
same SciPy LM optimizer. Select the observer convention explicitly with
``parallax_observer_convention``; ``"gulls"`` is available for GULLS-format
simulations. Parallax components are bounded by ``max_piE``; a bounded SciPy
fallback is used only if an unconstrained LM step leaves that domain:

.. code-block:: python

   config = FinderConfig(
       fitter_kind="fspl_parallax",
       parallax_geometry="space",
       ra_deg=267.623337808,
       dec_deg=-29.1164180355,
       tref=2459000.0,
       satellite_ephemeris_path="satellitedir/satellite1.txt",
       parallax_observer_convention="gulls",
       parallax_time_scale="hjd",
       max_piE=1.0,
   )

Automatic single-lens initialization
------------------------------------

When no initial guess is passed to :meth:`jacscanomaly.Finder.run`, PSPL uses
a logarithmic ``tE`` bank and batched FFT correlations.  For FSPL and
FSPL-parallax, ``fspl_template_initial_guesses`` measures the observed
brightening width, converts it to the validated crossing-time/rho/
``u0/rho`` grid, profiles ``Fs`` and ``Fb`` for every complete FSPL model
with the compiled ``ESPLMag2`` evaluator, and passes the lowest-chi-square
seeds to the canonical fitter.  With the default grid this is 180 direct
original-flux trials.  A PSPL parameter triplet may be supplied only to
locate the observed interval; PSPL residuals are not used as a template.
Important options include:
include:

``auto_init_fspl_template_top_k``

The ``auto_init_teff_*``, ``auto_init_dt0_coeff``, and
``auto_init_min_n_eff`` options remain relevant to non-PSPL initialization;
the teff bounds also define the conservative fallback seed for a flat PSPL
light curve.

Season splitting
----------------

.. code-block:: python

   config = FinderConfig(gap=50.0)

``gap`` is the maximum allowed time difference between consecutive sorted data
points within one season. A new season starts when the gap is larger.

Anomaly grid
------------

.. code-block:: python

   config = FinderConfig(
       teff_init=0.03,
       common_ratio=4.0 / 3.0,
       teff_grid_n=24,
       dt0_coeff=0.17,
       teff_coeff=3.0,
       min_pts_in_window=4,
   )

``teff_init``, ``common_ratio``, and ``teff_grid_n`` define the geometric grid
of candidate durations. ``dt0_coeff`` sets the time-grid spacing:

.. math::

   dt0 = dt0\_coeff \times teff

``teff_coeff`` sets the half-width of the local evaluation window in units of
``teff``.

Candidate selection
-------------------

Use :class:`jacscanomaly.CandidateCriteria` to reject candidates before best
candidate selection:

.. code-block:: python

   from jacscanomaly import CandidateCriteria, FinderConfig

   config = FinderConfig(
       candidate_criteria=CandidateCriteria(
           min_dchi2=20.0,
           min_n_eff=2.0,
           min_n_contrib=2,
           max_peak_frac=0.8,
       )
   )

Any threshold set to ``None`` is ignored.

Backend selection
-----------------

``grid_backend`` controls only anomaly-grid evaluation. The continuous
single-lens fitters are selected by ``fitter_kind`` and share one optimizer
contract:

.. code-block:: python

   config = FinderConfig(
       grid_backend="cpp",
       fitter_kind="fspl",
       fitter_maxiter=1000,
       fitter_tol=1.0e-6,
   )

PSPL uses the analytic point-source kernel. FSPL uses the compiled
finite-source magnification kernel. Both use SciPy LM; parallax fitters use
the same SciPy LM orchestration around the compiled trajectory evaluator.
``magnification_tol`` and ``magnification_reltol`` control the FSPL kernel.

Use ``grid_backend="jax"`` when comparing anomaly-grid implementations:

.. code-block:: python

   config = FinderConfig(
       grid_backend="jax",
   )

For large JAX grids, set ``grid_chunked=True`` to always process the grid in
chunks, or set ``grid_chunk_auto=True`` to enable chunking only when the number
of grid points exceeds ``grid_chunk_threshold``. ``grid_chunk_size`` controls
the number of grid points in each chunk.
