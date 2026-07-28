# Changelog

All notable changes to this project will be documented in this file.

This project follows a loose interpretation of [Semantic Versioning](https://semver.org/).

---

## [0.5.1] - 2026-07-29

### Added
- Planet-signal extraction now preserves the selected PSPL/FSPL model family
  during masked and robust refits.
- Deep closed troughs in a `bump--dip--bump` pattern are retained as dip
  features, while one-sided negative wings remain suppressed.

### Changed
- FSPL fitting retries a small set of guarded initial seeds when the first
  finite-source solution is a poor local fit.
- Refined-fit plots label the selected single-lens model family.
- Version bumped to `0.5.1` because PyPI already contains an independent
  `0.5.0` release.

### Removed
- Removed the experimental HMC/NumPyro API, optional dependency, example, and
  tests.

---

## [0.4.0] - 2026-07-28

### Added
- `PlanetSignalResult.measure_features()` for direct peak/dip counts,
  positions, threshold-crossing timescales, z-score strengths, residuals, and
  fractional deviations.

### Changed
- Candidate-quality criteria are now applied after raw cluster extraction, so
  they select the reported candidate without censoring its score background.
- Best-candidate scores now use same-season, comparable-timescale cluster
  backgrounds with median/MAD normalization and adaptive one-sided clipping.
- Replaced fixed percentile trimming controls with
  `best_score_teff_ratio`, `best_score_min_reference_clusters`,
  `best_score_upper_clip_sigma`, and `best_score_clip_maxiters`.
- Planet-signal follow-up now reports only direct extrema measurements. It no
  longer assigns a physical anomaly morphology or derives binary-lens
  parameters from a local residual shape.

### Removed
- The `planet_class` template/BIC estimator and its heuristic `s`, `q`,
  `alpha`, caustic-shape, and grid-seed outputs.
- The optional local analytic template fit previously used to adjust extrema
  timescales.

---

## [0.3.4] - 2026-07-11

### Added
- ``Finder`` now accepts magnitude and magnitude-error input through
  ``data_kind="mag"``. Magnitudes are converted to numerically stable relative
  flux internally; the default input representation remains ``"flux"``.

---

## [0.3.3] - 2026-07-11

### Added
- Planet-signal refinement and residual-morphology classification, including
  physical-model seed generation for downstream 2L1S/1L2S fitting.
- BIC-based PSPL/FSPL single-lens model selection, with an optional GULLS
  FSPL space-parallax trial when VBMicrolensing is installed.
- Candidate quality diagnostics for the grid scan, including `n_window`,
  `n_contrib`, `n_eff`, `peak_frac`, `rho1`, and `longest_run`.
- `AnomalyResult.grid_metrics_all` and per-season `SeasonSummary.grid_metrics`
  for downstream inspection of grid-level diagnostics.
- `BestCandidate.quality` for direct access to support and temporal diagnostics
  of the selected candidate.
- Summary helpers on `AnomalyResult`: `summary_dict()`, `summary_text()`,
  `print_summary()`, and `summary_table()`.

### Changed
- The C++ grid backend is now built as a required extension during installation.
  Source distributions include the backend source and fail with an actionable
  error if a C++17 compiler or OpenMP runtime is unavailable.
- The GULLS finite-difference space-parallax fitter now supports bounded and
  optionally penalized parallax components.
- Replaced the old internal `n_out` metric with richer per-candidate quality
  diagnostics.
- Updated documentation to show result summaries and candidate quality metrics.

---

## [0.1.1] - 2026-01-21

### Added
- Initial public release of **jacscanomaly**
- Residual-based anomaly scanning framework implemented in JAX
- PSPL baseline fitting with JAXOpt
- Grid-based local anomaly detection using Δχ² statistics
- Built-in visualization tools for baseline fits, residuals, and anomaly scans
- Example notebook demonstrating a full workflow

### Notes
- This release represents the first research-ready public version.
- The anomaly detection strategy is inspired by Zang et al. (2021, AJ, 162, 163).
- Example light curves are provided for demonstration purposes only
  and are drawn from Roman Galactic Exoplanet Survey simulation products.

## [0.3.0] - 2026-04-29

### Added
- Chunked grid execution mode (`grid_chunked`, `grid_chunk_auto`, `grid_chunk_size`, `grid_chunk_threshold` in `FinderConfig`) to reduce JAX compilation size and peak memory usage on large grids.
- Configurable best-candidate score trimming via `best_score_trim_percentile` in `FinderConfig`.
- Parallax fitting now accepts shifted Julian dates (JD − 2450000) as timestamps.

### Changed
- Grid scan now uses masked weights instead of big-error windowing, improving numerical robustness.
- Plot resolution improved: model curves use finer time steps and theory lines are rendered above data points.
- Anomaly window plot scaling improved.

---

## [0.2.0] - 2026-02-01

### Added
- Unified single-lens fitting framework supporting:
  - PSPL
  - FSPL
  - PSPL with annual parallax
  - FSPL with annual parallax
- New `SingleLensFitResult` with explicit parameter names and optional raw optimizer parameters.
- Configurable single-lens model selection via `FinderConfig.fitter_kind`.
- Dedicated modules for trajectory, magnification, photometry, and objective functions.
- Optional storage of raw optimizer parameters (e.g. `logrho`) for debugging and reproducibility.

### Changed
- Refactored single-lens fitting code into `singlelens_fit.py` and `singlelens_model.py`.
- Reworked `Finder` initialization logic to validate model requirements explicitly.
- Improved error messages for invalid model selections and missing configuration parameters.
- Updated plotting utilities to operate on unified single-lens fit results.

### Removed
- Removed legacy `utils.py` and deprecated PSPL-only fitting paths.
- Removed old `singlelens.py` in favor of the new unified architecture.

### Notes
- This release introduces a **breaking internal refactor**, but preserves the high-level
  `Finder` workflow.
- Users providing custom initial guesses must ensure that the dimensionality of `x0`
  matches the selected single-lens model.
