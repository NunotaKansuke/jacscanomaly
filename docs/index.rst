jacscanomaly
============

Find short anomalies in microlensing light curves.

Fit a single-lens model, scan its residual for short bumps and dips, and
describe what is found: a host fit the anomalies do not pull, and each
anomaly as bump / dip / caustic with its time and FWHM. One light curve is
one :class:`jacscanomaly.Event`.

.. code-block:: python

   from jacscanomaly import Event

   ev = Event(time, flux, ferr)
   ev.fit("pspl")
   if ev.scan().detected:
       ev.fit_atoms("pspl")
   ev.write_json("event.json")

.. toctree::
   :maxdepth: 1

   installation
   quickstart
   method
   api
