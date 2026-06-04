Results
=======

The main output of :meth:`jacscanomaly.Finder.run` is
:class:`jacscanomaly.AnomalyResult`.

Summary methods
---------------

``AnomalyResult`` provides CLI-friendly and notebook-friendly summaries:

.. code-block:: python

   result.print_summary()
   text = result.summary_text()
   row = result.summary_dict()
   table = result.summary_table()

``summary_table`` returns a one-row ``pandas.DataFrame`` when pandas is
installed. Otherwise it returns ``list[dict]``.

Important attributes
--------------------

``time``, ``flux``, ``ferr``
   Input arrays stored as NumPy arrays.

``fit``
   The single-lens fit result.

``model_flux``
   Baseline single-lens model flux.

``residual``
   ``flux - model_flux``.

``chi2_dof``
   Reduced chi-square of the baseline fit.

``seasons``
   List of :class:`jacscanomaly.SeasonSummary` objects.

``clusters_all``
   Flattened extracted clusters. Rows are ``[t0, teff, dchi2]``.

``grid_metrics_all``
   Flattened per-grid diagnostics. Columns are:

   ``[t0, teff, dchi2, n_window, n_contrib, n_eff, peak_frac, rho1, longest_run]``

``best``
   The best :class:`jacscanomaly.BestCandidate`, or ``None`` if no candidate
   exists.

Best candidate
--------------

.. code-block:: python

   if result.best is not None:
       best = result.best
       print(best.t0)
       print(best.teff)
       print(best.dchi2)
       print(best.score)

       quality = best.quality
       print(quality.n_eff)
       print(quality.peak_frac)

Candidate quality fields
------------------------

``n_window``
   Number of points in the local anomaly window.

``n_contrib``
   Number of points above the per-point contribution threshold.

``n_eff``
   Effective number of contributing points. This is often the most useful
   diagnostic for rejecting one-point artifacts.

``peak_frac``
   Strongest-point contribution divided by total positive contribution.

``rho1``
   Lag-1 autocorrelation of per-point improvements.

``longest_run``
   Longest run of above-threshold contributing points.

Working with all candidates
---------------------------

The finder stores non-overlapping cluster representatives in ``clusters_all``:

.. code-block:: python

   t0 = result.clusters_all[:, 0]
   teff = result.clusters_all[:, 1]
   dchi2 = result.clusters_all[:, 2]

The raw grid diagnostics are in ``grid_metrics_all``:

.. code-block:: python

   metrics = result.grid_metrics_all
   n_eff = metrics[:, 5]
   supported = metrics[n_eff > 2.0]

This is useful when building survey-level candidate tables or applying custom
post-processing criteria.
