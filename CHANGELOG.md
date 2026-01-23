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
