Quickstart
==========

Run the anomaly finder
----------------------

The main inputs are one-dimensional arrays of time, flux, and flux error:

.. code-block:: python

   import numpy as np

   from jacscanomaly import Finder, FinderConfig

   data = np.load("example/example_data.npy")
   time = data[:, 0]
   flux = data[:, 1]
   ferr = data[:, 2]

   config = FinderConfig(fitter_kind="pspl", gap=50.0)

   finder = Finder(config)
   result = finder.run(time, flux, ferr)
   result.print_summary()

Magnitude input
---------------

Flux is the default input representation. To provide magnitudes and magnitude
errors instead, pass them in the same positions and set ``data_kind="mag"``.
The finder converts them to a relative flux scale internally before fitting and
scanning:

.. code-block:: python

   result = finder.run(time, mag, magerr, data_kind="mag")

The returned :class:`jacscanomaly.AnomalyResult` contains the original data,
the single-lens fit, residuals, per-season grid summaries, extracted clusters,
and the best anomaly candidate.

This is the default fitted-baseline workflow. For a fixed baseline,
residual-only template-free search, or iterative signal extraction, see
:doc:`workflows`.

Use quality criteria
--------------------

Large ``dchi2`` values can be caused by a single high-weight point. Use
:class:`jacscanomaly.CandidateCriteria` to reject candidates before best
candidate selection:

.. code-block:: python

   from jacscanomaly import CandidateCriteria, FinderConfig

   config = FinderConfig(
       candidate_criteria=CandidateCriteria(
           min_dchi2=20.0,
           min_n_eff=2.0,
       )
   )

Here ``n_eff`` is the effective number of contributing points. Candidates with
``n_eff`` below the threshold are ignored by the best-candidate selection.

Inspect the result
------------------

.. code-block:: python

   summary = result.summary_dict()
   print(summary["best_score"])
   print(summary["best_n_eff"])

   if result.best is not None:
       best = result.best
       print(best.t0, best.teff, best.dchi2)
       print(best.quality.n_eff, best.quality.peak_frac)

In notebooks:

.. code-block:: python

   display(result.summary_table())

Plot the result
---------------

The high-level plotting methods use the latest result stored on the finder:

.. code-block:: python

   import matplotlib.pyplot as plt

   finder.plot_result()
   finder.plot_anomaly_window()
   plt.show()

The summary plot shows the light curve, residuals, and scan statistic. The
window plot focuses on the best anomaly candidate.

Fit only the single-lens model
------------------------------

If you only need the baseline fit:

.. code-block:: python

   fit = finder.fit_single_lens(time, flux, ferr)
   print(fit.params)
   print(fit.chi2)

You can also provide an explicit nonlinear initial guess:

.. code-block:: python

   x0 = np.array([2459000.0, 30.0, 0.1])  # t0, tE, u0
   fit = finder.fit_single_lens(time, flux, ferr, x0=x0)

Run the scan with fixed baseline parameters
-------------------------------------------

By default, :meth:`jacscanomaly.Finder.run` treats ``x0`` as an initial guess
and refits the selected single-lens model before scanning residuals. If you
already have baseline nonlinear parameters and want to scan without optimizing
them, pass ``refit=False``:

.. code-block:: python

   x0 = np.array([2459000.0, 30.0, 0.1])  # t0, tE, u0
   result = finder.run(time, flux, ferr, x0=x0, refit=False)

In this mode, the nonlinear parameters in ``x0`` are fixed. The linear flux
parameters ``fs`` and ``fb`` are still solved for the supplied light curve.
This is the intended find-only mode when the PSPL geometry is already known.
