# Changelog

All notable changes to this project will be documented in this file.

This project follows a loose interpretation of [Semantic Versioning](https://semver.org/).

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

