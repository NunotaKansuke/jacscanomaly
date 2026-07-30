jacscanomaly
============

``jacscanomaly`` is a Python package for scan-based anomaly detection in
time-series light curves, with microlensing anomaly searches as the primary
use case.

The package is built for the common situation where a smooth baseline model
fits most of a light curve, but localized deviations may contain the signal of
interest. In microlensing applications the baseline is usually a single-lens
model and the deviations are candidate planetary anomalies.

What it does
------------

``jacscanomaly`` provides:

* single-lens fitting utilities for PSPL, FSPL, and parallax variants,
* a residual scan over anomaly center time and effective duration,
* non-overlapping candidate extraction,
* candidate support diagnostics such as ``n_eff`` and ``peak_frac``,
* plotting and summary helpers for scripts, CLIs, and notebooks,
* low-memory C++ backends for large PSPL survey scans.

The high-level entry point is :class:`jacscanomaly.Finder`. A typical workflow
is:

1. fit a single-lens baseline model to a light curve,
2. scan the residuals over local anomaly templates,
3. extract non-overlapping anomaly candidates,
4. inspect candidate quality diagnostics such as ``score`` and ``n_eff``.

Start here
----------

If you are using the package for the first time:

* read :doc:`installation` to install the package and optional dependencies,
* follow :doc:`quickstart` to run a light curve,
* use :doc:`workflows` to choose between fitted, fixed-baseline, residual-only,
  and signal-refinement searches,
* use :doc:`method` to understand ``dchi2``, ``score``, and ``n_eff``,
* use :doc:`configuration` when tuning a survey run,
* use :doc:`results` when building candidate tables or downstream analysis.
* use :doc:`planet_features` to measure peaks and dips in extracted
  planetary residual signals.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   quickstart
   workflows
   examples
   method
   configuration
   fft_backend
   pspl_fft
   results
   planet_features
   planet_signal_pipeline_handover_ja

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   reference
   testing
   development

Project indices
===============

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
