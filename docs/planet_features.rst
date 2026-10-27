Bump, dip and caustic features
==============================

:func:`jacscanomaly.measure_features` reports direct measurements of the
residual inside candidate windows instead of fitting an anomaly model. Each
feature is classified as one of:

``bump``
   A positive residual extremum.

``dip``
   A closed negative residual extremum. One-sided negative wings (the tail of
   a crossing or a baseline mismatch) are not reported.

``caustic``
   A bump that is significantly left/right asymmetric: the shape a source
   makes when it crosses or passes close to a caustic. It replaces the bump.

Usage
-----

.. code-block:: python

   from jacscanomaly import Candidate, measure_features

   feats = measure_features(fit, [Candidate(9.0, 11.0)])

   print(len(feats.bumps), len(feats.dips), len(feats.caustics))
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

Caustic (asymmetry test)
------------------------

Around each bump, on the raw ``z = residual / ferr``, a Lorentzian
``A / (1 + ((t - tc) / w)**2)`` is compared with a split Lorentzian whose
half-width is ``w_left`` before ``tc`` and ``w_right`` after it. Both have a
local linear baseline; ``A`` and the baseline are linear. The split model adds
one parameter to the nested symmetric one, so for a symmetric bump
``delta_chi2`` follows chi-square with one degree of freedom. The bump is a
caustic when ``delta_chi2 >= min_asymmetry_dchi2`` (25, i.e. 5 sigma) and
``max(w) / min(w) >= min_width_ratio`` (1.5).

The fit window is the peak plus or minus
``max(asymmetry_window_timescales * timescale, asymmetry_window_cadences * cadence)``,
cut half way to the neighbouring bump or dip, so the slope into an adjacent
dip is not read as the bump's own edge.

``feature.caustic.orientation`` is ``"entry"`` for a fast rise and slow decay
(``w_left < w_right``) and ``"exit"`` for the reverse. It describes the shape
only; a near-caustic passage looks the same as a crossing. No pairing of
entries and exits is done.

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

``caustic``
   :class:`jacscanomaly.CausticDetail` (``orientation``, ``peak_time``,
   ``left_width``, ``right_width``, ``width_ratio``, ``delta_chi2``) for a
   caustic; ``None`` otherwise.

Configuration
-------------

:class:`jacscanomaly.FeatureConfig` controls smoothing, the minimum lobe S/N,
the minimum strength relative to the strongest local feature, local
prominence, minimum time separation, the relative height used for the
duration measurement, and the caustic asymmetry test
(``min_asymmetry_dchi2``, ``min_width_ratio`` and the fit window).

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

The caustic label is a morphological classification of the residual; no
binary-lens parameters (``s``, ``q``, ``alpha``) are estimated.
