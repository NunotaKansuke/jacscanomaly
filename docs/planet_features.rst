Bump and dip features
=====================

:func:`jacscanomaly.measure_features` reports direct measurements of the
residual inside candidate windows instead of fitting an anomaly model. Each
feature is classified as one of:

``bump``
   A positive residual extremum.

``dip``
   A closed negative residual extremum. One-sided negative wings (the tail of
   a crossing or a baseline mismatch) are not reported.

Usage
-----

.. code-block:: python

   from jacscanomaly import Candidate, measure_features

   feats = measure_features(fit, [Candidate(9.0, 11.0)])

   print(len(feats.bumps), len(feats.dips))
   for feature in feats.features:            # time ordered
       print(feature.kind, feature.time, feature.timescale, feature.strength)

``fit`` is any single-lens fit result; typically the refit with the signal
masked (see :ref:`post-fit-components`). Extrema are searched on the union of
the candidate windows, and each feature records the first candidate whose
support contains it (``feature.candidate``).

Pass the :class:`jacscanomaly.SignalMask` instead of the candidates to measure
each window united with its signal segment, padded by ``smooth_points``
samples. The segment can reach beyond the window; a feature cut at the window
edge has no closing floor, so a dip is dropped and a bump's prominence is
lost:

.. code-block:: python

   sm = signal_mask(fit, cands)          # after the mask/refit loop
   feats = measure_features(fit, sm)

Detection
---------

An extremum of the smoothed residual is kept when its lobe (the run where the
smoothed residual keeps its sign) has ``snr = sum(z) / sqrt(n) >= min_snr``:
a constant offset over the lobe improves chi-square by at least
``min_snr**2``. This holds for a broad, low-amplitude feature whose points are
each below ``min_snr``. Within a lobe, sub-extrema are separated by local
prominence and the relative-strength and separation rules.

No caustic label
----------------

A bump is not labelled a caustic. A left/right asymmetry test against an
exactly symmetric bump is rejected by almost any real bump at high S/N
(smooth near-caustic passages are skewed too), so such a label follows the
S/N, not the shape. A caustic label needs a positive signature of a fold
crossing (a jump unresolved at the cadence and ``rho tE``), which is not
measured here.

Measurements
------------

Each :class:`jacscanomaly.Feature` contains:

``time``
   Time of the strongest residual point.

``t_start``, ``t_end``, ``timescale``
   Interpolated half-prominence crossings of the smoothed residual and their
   difference. These are direct durations, not
   Einstein times.

``snr``
   Lobe S/N used for detection.

``strength``, ``signed_z``
   Absolute and signed residual significance in units of the photometric
   uncertainty.

``residual``
   Observed flux minus the single-lens model.

``fractional_deviation``
   Residual divided by the model flux at the feature time.

``magnification_ratio``
   Blend-corrected observed magnification divided by the fitted single-lens
   magnification when the fitted source flux is usable.

Configuration
-------------

:class:`jacscanomaly.FeatureConfig` controls smoothing, the minimum lobe S/N,
the minimum strength relative to the strongest local feature, local
prominence, minimum time separation, the relative height used for the
and the duration measurement.

.. code-block:: python

   from jacscanomaly import FeatureConfig

   feats = measure_features(
       fit, cands, config=FeatureConfig(smooth_points=5, min_snr=5.0)
   )

Tabular output
--------------

.. code-block:: python

   event_row = feats.summary_dict()
   feature_rows = feats.feature_dicts()
   table = feats.summary_table()

The labels are morphological classifications of the residual; no
binary-lens parameters (``s``, ``q``, ``alpha``) are estimated.
