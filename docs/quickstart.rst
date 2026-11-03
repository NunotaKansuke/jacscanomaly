Quickstart
==========

One light curve is one :class:`jacscanomaly.Event`. Its steps are methods;
every result stays on the event.

.. code-block:: python

   import numpy as np

   from jacscanomaly import Event, FinderConfig

   time, flux, ferr = np.load("example/example_data.npy").T

   ev = Event(time, flux, ferr, config=FinderConfig(gap=50.0))
   ev.fit("pspl")
   scan = ev.scan()          # the residual of the last fit
   print(scan.detected, scan.detections)

Magnitudes are taken as they are (the zero point does not matter):

.. code-block:: python

   ev = Event(time, mag, magerr, kind="mag")

The full pipeline
-----------------

Fit and scan. If something is found, refit with a finite source and with
parallax and scan again; what neither explains is fitted with atoms.

.. code-block:: python

   ev = Event(time, flux, ferr, ra_deg=268.4, dec_deg=-29.2, ephemeris="roman.txt")

   ev.fit("pspl")
   pspl_scan = ev.scan()
   if pspl_scan.detected:
       ev.fit("fspl", start="pspl")
       fspl_scan = ev.scan()
       ev.fit("parallax", start="pspl")
       parallax_scan = ev.scan()
       if fspl_scan.detected and parallax_scan.detected:
           ev.fit_atoms("pspl")
           ev.fit_atoms("fspl")
           ev.fit_atoms("parallax")

   ev.write_json("event.json")

Models: ``"pspl"``, ``"fspl"``, ``"parallax"``, ``"fspl_parallax"``. The
parallax models need ``ra_deg`` and ``dec_deg``, and ``ephemeris`` for a
spacecraft.

Rules
-----

* ``ev.fit(model)`` starts from the last fit (``t0, tE, u0``, and ``rho`` or
  ``piE`` when both models have them). ``start="pspl"`` names the fit to start
  from; ``x0=`` gives the start itself. The first fit uses the automatic
  multistart. Parallax ``tref`` is the time of maximum flux unless given.
* ``ev.scan(model=None)`` and ``ev.fit_atoms(model=None)`` act on the last
  fit; a model not fitted yet is an error.
* Refitting a model drops its own scan and atoms; the other models stay.

Results
-------

.. code-block:: python

   ev.fits["fspl"].params        # {"t0": ..., "tE": ..., "u0": ..., "rho": ...}
   ev.scans["pspl"].detections   # every candidate with score >= detection_score
   ev.atoms["pspl"].features     # one per anomaly window: kind, time, FWHM, dchi2
   ev.summary()                  # one row per model (a DataFrame with pandas)
   ev.release()                  # drop the scan grids and model arrays

Several events go into one file:

.. code-block:: python

   from jacscanomaly import write_json

   write_json("results.json", {"evt1": ev1, "evt2": ev2})

Plots
-----

.. code-block:: python

   ev.plot_result("pspl")
   ev.plot_anomaly_window("pspl")

They show a scan (run ``ev.scan(model)`` first), in flux.

Settings
--------

:class:`jacscanomaly.FinderConfig` holds the search settings. The ones
usually set:

``gap`` (100 d)
   A time gap longer than this starts a new season; seasons are scanned
   separately.
``teff_init``, ``teff_grid_n`` (0.03 d, 24)
   The shortest anomaly timescale scanned and the number of timescales.
``detection_score`` (10)
   The score a candidate needs to count as a detection (and as an atom).
``parallax_geometry``, ``parallax_observer_convention``, ``max_piE``
   The parallax model: ``"annual"`` or ``"space"`` geometry, the ephemeris
   convention, and the largest ``piE`` allowed.

:class:`jacscanomaly.AtomConfig` (passed to ``fit_atoms``) sets the atom
search.
