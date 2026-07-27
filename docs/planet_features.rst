Planet peak and dip measurements
================================

The planet-signal workflow intentionally reports direct residual measurements
instead of fitting a physical anomaly template. It answers four questions:

* how many positive peaks are present,
* how many negative dips are present,
* when each feature occurs and how long it lasts,
* how strong each feature is.

Extract and measure
-------------------

First refine the single-lens baseline while excluding the localized signal,
then measure extrema in the refined residual:

.. code-block:: python

   from jacscanomaly import (
       Finder,
       FinderConfig,
       PlanetSignalExtractor,
   )

   finder = Finder(FinderConfig(fitter_kind="pspl"))
   signal = PlanetSignalExtractor(finder).run(time, flux, ferr)
   features = signal.measure_features()

   print(features.n_peaks, features.n_dips)
   for feature in features.features:
       print(
           feature.kind,
           feature.time,
           feature.timescale,
           feature.strength,
       )

``features.peaks`` and ``features.dips`` contain the same measurements split
by sign. ``features.features`` combines them in time order. Positive features
take precedence: when one or more locally prominent peaks are found, negative
wings around that perturbation are ignored. Dips are reported only when no
positive peak is present, and only when the negative excursion returns below
the duration threshold on both sides.

Measurements
------------

Each :class:`jacscanomaly.PlanetFeature` contains:

``time``
   Time of the strongest observed residual point in the feature.

``t_start``, ``t_end``, ``timescale``
   Interpolated threshold-crossing bounds and their difference. These are
   direct duration measurements, not Einstein times or caustic parameters.

``strength``, ``signed_z``
   Absolute and signed residual significance in units of the photometric
   uncertainty.

``residual``
   Observed flux minus the refined single-lens baseline.

``fractional_deviation``
   Residual divided by the baseline model flux at the feature time.

``magnification_ratio``
   Blend-corrected observed magnification divided by the fitted single-lens
   magnification when the fitted source flux is usable.

Configuration
-------------

:class:`jacscanomaly.PlanetFeatureConfig` controls smoothing, the minimum
absolute z-score, the minimum strength relative to the strongest local feature,
local prominence, minimum time separation, and the threshold used for the
duration measurement.

.. code-block:: python

   from jacscanomaly import PlanetFeatureConfig

   features = signal.measure_features(
       PlanetFeatureConfig(
           smooth_points=5,
           min_abs_z=5.0,
           min_relative_strength=0.1,
           min_prominence=3.0,
           min_separation=0.15,
       )
   )

Tabular output
--------------

Use ``feature_dicts()`` for JSON or CSV serialization and ``summary_table()``
for a pandas table:

.. code-block:: python

   event_row = features.summary_dict()
   feature_rows = features.feature_dicts()
   table = features.summary_table()

The result does not assign caustic labels and does not estimate ``s``, ``q``,
``alpha``, or other binary-lens parameters.
