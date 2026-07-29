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
   PSPL with annual parallax. Requires ``ra_deg`` and ``dec_deg``.

``"fspl_parallax"``
   FSPL with annual parallax. Requires ``ra_deg`` and ``dec_deg``.

``"pspl_space_parallax"``
   PSPL with annual parallax plus a spacecraft ephemeris. Requires
   ``ra_deg``, ``dec_deg``, and ``satellite_ephemeris_path``.

``"fspl_space_parallax"``
   FSPL with annual parallax plus a spacecraft ephemeris. Requires
   ``ra_deg``, ``dec_deg``, and ``satellite_ephemeris_path``.

``"bic_single_lens"``
   Compare PSPL and FSPL fits by BIC, then return the
   selected fit. Set ``bic_include_space_parallax=True`` to also compare the
   native C++/VBMicrolensing FSPL space-parallax model.

For parallax models:

.. code-block:: python

   config = FinderConfig(
       fitter_kind="pspl_parallax",
       ra_deg=270.0,
       dec_deg=-30.0,
       tref=None,  # defaults to median observation time
   )

For space-parallax models, pass a VBMicrolensing/RTModel satellite table:

.. code-block:: python

   config = FinderConfig(
       fitter_kind="pspl_space_parallax",
       ra_deg=267.623337808,
       dec_deg=-29.1164180355,
       tref=2459000.0,
       satellite_ephemeris_path="satellitedir/satellite1.txt",
   )

The satellite table is expected to contain rows of
``JD RA_deg Dec_deg distance_AU`` inside an optional ``$$SOE`` / ``$$EOE``
block, matching the VBMicrolensing satellite-table convention.

All parallax model kinds use the native C++ trajectory/VBMicrolensing
evaluator and SciPy trust-region optimization. Select the observer convention
explicitly with ``parallax_observer_convention``; ``"gulls"`` is available
for GULLS-format simulations. Parallax components are bounded by ``max_piE``:

.. code-block:: python

   config = FinderConfig(
       fitter_kind="bic_single_lens",
       bic_include_space_parallax=True,
       ra_deg=267.623337808,
       dec_deg=-29.1164180355,
       tref=2459000.0,
       satellite_ephemeris_path="satellitedir/satellite1.txt",
       parallax_observer_convention="gulls",
       parallax_time_scale="hjd",
       max_piE=1.0,
   )

The BIC-selection result includes ``model_kind``, ``bic``, and
``model_selection`` attributes that record the selected model and the BIC
values of successful trials.

Automatic single-lens initialization
------------------------------------

When no initial guess is passed to :meth:`jacscanomaly.Finder.run`, the finder
estimates initial values with a coarse scan. Important options include:

``auto_init_teff_min`` / ``auto_init_teff_max``
   Range of effective timescales used for initial candidate search.

``auto_init_teff_grid_n``
   Number of effective timescale grid points.

``auto_init_tE_min`` / ``auto_init_tE_max``
   Range of event timescales used to convert candidate durations into PSPL
   seeds.

``auto_init_min_n_eff``
   Minimum effective point count required for initial candidates. This helps
   avoid single-point outliers driving the PSPL initial guess.

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

The PSPL workflow uses C++ backends by default:

.. code-block:: python

   config = FinderConfig(
       grid_backend="cpp",
       single_fit_backend="cpp",
   )

Use the JAX backend when you want the original vectorized implementation or
when comparing backend behavior:

.. code-block:: python

   config = FinderConfig(
       grid_backend="jax",
       single_fit_backend="jax",
   )

``single_fit_backend="cpp"`` applies to ``fitter_kind="pspl"``. Other baseline
model families use the JAX fitters.

For large JAX grids, set ``grid_chunked=True`` to always process the grid in
chunks, or set ``grid_chunk_auto=True`` to enable chunking only when the number
of grid points exceeds ``grid_chunk_threshold``. ``grid_chunk_size`` controls
the number of grid points in each chunk.
