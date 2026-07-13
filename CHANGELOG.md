# Changelog

All notable changes to this project will be documented in this file.

This project follows a loose interpretation of [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Changed
- Rewrote `planet_class` as a heuristic anomaly estimator based on the
  standard literature formalism (Gould & Loeb 1992; Gaudi & Gould 1997;
  Han 2006; Hwang et al. 2022; Ryu et al. 2022). Each anomaly component is
  measured with a small template set (`bump`, `dip`, `fold`,
  `caustic_crossing`, `null`), and the deterministic geometry
  (`tau_anom`, `u_anom`, `alpha`, `s_dagger_plus/minus` with the
  bump→major-image / dip→minor-image branch), the duration ratio `dt/tE`,
  and assumption-tagged mass-ratio estimates (`dip_han2006`,
  `bump_planet_einstein_crossing`) are derived from the fitted anomaly time
  and duration. Fold-type shapes report `tstar/tE = rho/|sin(psi)|` per
  crossing. Anomalies with small `u_anom` are flagged
  `central_or_resonant` and receive no `q` estimate.

### Removed
- The residual-atom template atlas (about 20 atoms including rim-trough,
  shear-quadrupole, curved/grazing/two-fold, canonical and finite-source
  cusp variants), the local-window physical refit machinery, the
  Chang--Refsdal lookup fit, warning/validity score penalties, BIC-weighted
  class probabilities, and the `physical_constraint`/`physical_relation`
  output tables. `AtomFitResult`, `LocalPhysicalFitResult`, and
  `SegmentModelResult` are replaced by `AnomalyShapeFit`, `AnomalyGeometry`,
  `AnomalyScales`, and `ComponentAnomalyResult`.

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
