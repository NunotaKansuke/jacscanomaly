Examples
========

This page gives small, copyable examples. They are intentionally compact; for
interactive exploration, see the notebooks in the repository's ``example/``
directory.

Minimal synthetic light curve
-----------------------------

The high-level finder expects a full light curve, not pre-computed residuals.
This example creates a simple microlensing-like event with a localized
deviation and runs the default PSPL workflow.

.. code-block:: python

   import numpy as np

   from jacscanomaly import CandidateCriteria, Finder, FinderConfig

   rng = np.random.default_rng(3)

   time = np.linspace(0.0, 100.0, 500)
   baseline = np.ones_like(time)
   event = 0.7 / np.sqrt(1.0 + ((time - 50.0) / 8.0) ** 2)
   anomaly = 0.08 * np.exp(-0.5 * ((time - 56.0) / 0.8) ** 2)
   ferr = np.full_like(time, 0.01)
   flux = baseline + event + anomaly + rng.normal(0.0, ferr)

   config = FinderConfig(
       fitter_kind="pspl",
       gap=50.0,
       candidate_criteria=CandidateCriteria(min_n_eff=2.0),
   )

   finder = Finder(config)
   result = finder.run(time, flux, ferr)
   result.print_summary()

The default PSPL workflow uses the C++ backends. You can still make that
explicit when writing survey scripts:

.. code-block:: python

   config = FinderConfig(
       fitter_kind="pspl",
       grid_backend="cpp",
       single_fit_backend="cpp",
       candidate_criteria=CandidateCriteria(min_n_eff=2.0),
   )

Running only the baseline fit
-----------------------------

Use :meth:`jacscanomaly.Finder.fit_single_lens` if you only want the fitted
single-lens model:

.. code-block:: python

   fit = finder.fit_single_lens(time, flux, ferr)

   params = dict(zip(fit.param_names, np.asarray(fit.params)))
   print(params)
   print(float(fit.chi2_dof))

If the automatic initialization is not appropriate for a difficult event, pass
an explicit initial guess:

.. code-block:: python

   x0 = np.array([50.0, 8.0, 0.2])
   fit = finder.fit_single_lens(time, flux, ferr, x0=x0)

If you already trust a set of baseline nonlinear parameters and want to scan
residuals without refitting them, pass ``refit=False`` to
:meth:`jacscanomaly.Finder.run`:

.. code-block:: python

   x0 = np.array([50.0, 8.0, 0.2])
   result = finder.run(time, flux, ferr, x0=x0, refit=False)

The nonlinear parameters are fixed, while ``fs`` and ``fb`` are still solved
analytically for the input light curve.

Using candidate criteria
------------------------

One-point artifacts can have large ``dchi2``. ``CandidateCriteria`` lets you
discard such candidates before the best candidate is selected:

.. code-block:: python

   criteria = CandidateCriteria(
       min_dchi2=20.0,
       min_n_eff=2.0,
       min_n_contrib=2,
       max_peak_frac=0.8,
   )

   config = FinderConfig(candidate_criteria=criteria)

For anomaly searches, ``min_n_eff`` is usually the first criterion to consider.
It removes candidates whose apparent improvement is dominated by one or two
points.

Plotting
--------

After :meth:`jacscanomaly.Finder.run`, plotting methods use the most recent
result:

.. code-block:: python

   import matplotlib.pyplot as plt

   finder.plot_result()
   finder.plot_anomaly_window()
   plt.show()

The summary plot is best for seeing the whole light curve and scan statistic.
The window plot is best for checking whether the candidate is supported by
multiple data points.

Example notebooks
-----------------

The repository includes:

``example/template_scan_example.ipynb``
   Standard bell-template residual scan.

``example/template_free_example.ipynb``
   Template-free residual chi-square scan.
