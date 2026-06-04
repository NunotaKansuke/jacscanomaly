jacscanomaly
============

``jacscanomaly`` is a JAX-based package for scan-based anomaly detection in
time-series residuals, with microlensing anomaly searches as the primary use
case.

The package is organized around one high-level entry point:
:class:`jacscanomaly.Finder`. A typical workflow is:

1. fit a single-lens baseline model to a light curve,
2. scan the residuals over local anomaly templates,
3. extract non-overlapping anomaly candidates,
4. inspect candidate quality diagnostics such as ``score`` and ``n_eff``.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   quickstart
   method
   configuration
   results

.. toctree::
   :maxdepth: 2
   :caption: Reference

   readme
   api
   testing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
