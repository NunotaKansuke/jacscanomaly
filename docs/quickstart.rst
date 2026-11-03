Quickstart
==========

One light curve is one :class:`jacscanomaly.Event`. Each step is a method,
and each result stays on the event under the model's name. The numbers
below are from ``example/example_data.npy`` (a simulated Roman event); the
whole run takes a few seconds.

Load
----

.. code-block:: python

   import numpy as np
   from jacscanomaly import Event

   time, flux, ferr = np.load("example/example_data.npy").T
   ev = Event(time, flux, ferr)

For magnitudes, pass ``kind="mag"``; the zero point does not matter.

.. code-block:: python

   ev = Event(time, mag, magerr, kind="mag")

1. Fit a single lens
--------------------

.. code-block:: python

   fit = ev.fit("pspl")
   fit.params
   # {'t0': 9826.56, 'tE': 8.62, 'u0': 0.353}

The first fit searches its own start. ``fit.fs``, ``fit.fb`` and
``fit.chi2`` are also there.

2. Scan the residual
--------------------

.. code-block:: python

   pspl_scan = ev.scan()
   pspl_scan.detected
   # True
   c = pspl_scan.detections[0]
   c.t0, c.teff, c.dchi2, c.score
   # (9826.00, 0.30, 111.2, 55.0)

``scan()`` acts on the last fit. A detection is a short bump or dip with
``score >= 10`` (see :doc:`method`).

3. Try the single-lens effects
------------------------------

A finite source or parallax can make a short deviation too. Refit with each,
starting from the PSPL, and scan again:

.. code-block:: python

   ev.fit("fspl", start="pspl")
   fspl_scan = ev.scan()
   ev.fit("parallax", start="pspl")
   parallax_scan = ev.scan()

The parallax models need the sky position, and a spacecraft ephemeris for a
satellite:

.. code-block:: python

   ev = Event(time, flux, ferr, ra_deg=268.4, dec_deg=-29.2, ephemeris="roman.txt")

The models are ``"pspl"``, ``"fspl"``, ``"parallax"`` and
``"fspl_parallax"``.

4. Fit the anomalies
--------------------

If neither explains the anomaly (both scans still detect it), fit the host
together with it:

.. code-block:: python

   atoms = ev.fit_atoms("pspl")
   atoms.host_identified
   # True
   f = atoms.features[0]
   f.kind, f.time, f.timescale, f.dchi2, f.caustic
   # ('bump', 9826.00, 0.39, 150.3, False)

``atoms.params`` is the host fit that the anomaly does not pull. Each
feature is one anomaly window: its shape (``bump``, ``dip``, ``bump+dip``,
...), peak time, FWHM, the ``dchi2`` it explains, and whether it looks like a
caustic crossing.

The whole pipeline
------------------

.. code-block:: python

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

Results
-------

.. code-block:: python

   ev.fits["pspl"]       # Fit
   ev.scans["pspl"]      # Scan
   ev.atoms["pspl"]      # Atoms
   ev.summary()

Refitting a model drops its own scan and atoms; the other models stay.
``summary()`` gives one row per model (a pandas DataFrame when pandas is
installed):

.. code-block:: text

     model      chi2         J  detected  n_detections  best_score   atoms_J  n_atoms  host_identified
   0  pspl  40468.93  40521.98      True             1       55.02  40435.86      1.0             True
   1  fspl  40468.93  40532.59      True             1       55.02       NaN      NaN             None

Save to JSON, one event or many:

.. code-block:: python

   from jacscanomaly import write_json

   ev.write_json("event.json")
   write_json("results.json", {"evt1": ev1, "evt2": ev2})

For many events, ``ev.release()`` drops the model arrays after saving; the
results stay.

Plots
-----

.. code-block:: python

   ev.plot_result("pspl")
   ev.plot_anomaly_window("pspl")

Both show a scan, so run ``ev.scan("pspl")`` first.

Settings
--------

:class:`jacscanomaly.FinderConfig` holds the search settings:

.. code-block:: python

   from jacscanomaly import FinderConfig

   ev = Event(time, flux, ferr, config=FinderConfig(gap=50.0, detection_score=8.0))

``gap`` (100 d)
   A time gap longer than this starts a new season; seasons are scanned
   separately.
``teff_init``, ``teff_grid_n`` (0.03 d, 24)
   The shortest anomaly timescale scanned and the number of timescales.
``detection_score`` (10)
   The score a candidate needs to count as a detection, and to become an
   atom.
``parallax_geometry``, ``parallax_observer_convention``, ``max_piE``
   The parallax model: ``"annual"`` or ``"space"`` geometry, the ephemeris
   convention, and the largest ``piE`` allowed.

:class:`jacscanomaly.AtomConfig` (``ev.fit_atoms(config=...)``) sets the atom
search.
