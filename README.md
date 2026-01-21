# jacscanomaly

**jacscanomaly** is a JAX-based framework for scan-based anomaly detection
in time-series data.

The package is designed to detect **localized, transient anomalies** by
scanning residuals after fitting a baseline model (e.g. PSPL in microlensing),
while remaining fast and differentiable thanks to JAX.

---

## Features

*  **JAX-powered**: fast, vectorized grid scans with JIT compilation
*  **Scan-based anomaly detection** on residuals
*  **Built-in visualization**: PSPL fit, residuals, and anomaly scan summary
*  Designed for **research-grade workflows** (clear statistics & reproducibility)

---

## Installation

```bash
pip install jacscanomaly
```

---

## Quick Example

```python
import numpy as np
from jacscanomaly import Finder, FinderConfig

# load data (time, flux, flux_err)
data = np.load("example_data.npy")
time, flux, ferr = data[:, 0], data[:, 1], data[:, 2]

# initial guess for PSPL parameters
p0 = np.array([9826.56, 8.61, 0.353])

# run anomaly finder
config = FinderConfig()
finder = Finder(config)
result = finder.run(time, flux, ferr, p0)

print("=== PSPL fit ===")

t0_pspl, tE_pspl, u0_pspl = result.fit.params
print(f"  t0          = {float(t0_pspl):.3f}")
print(f"  tE          = {float(tE_pspl):.3f}")
print(f"  u0          = {float(u0_pspl):.3f}")
print(f"  chi2 / dof  = {result.chi2_dof:.3f}\n")

b = result.best
print("=== Anomaly candidate ===")
print(f"  t0          = {b.t0:.3f}")
print(f"  teff        = {b.teff:.3f}")
print(f"  dchi2       = {b.dchi2:.3e}")
print(f"  score       = {b.score:.2f}")
```

---

## Visualization

```python
finder.plot_result()
plt.show()

finder.plot_anomaly_window()
plt.show()
```
These commands produce two complementary visualizations:

1. **Three-panel summary plot (`finder.plot_result`)**
   - **Top:** Observed light curve with the best-fit baseline model (PSPL)
   - **Middle:** Residuals after baseline fitting
   - **Bottom:** Anomaly scan result (Δχ² vs. time), showing where localized
     deviations from the baseline model are detected

2. **Focused anomaly window plot (`finder.plot_anomaly_window`)**
   - A zoomed-in view around the best anomaly candidate
   - Residuals are shown together with the anomaly template and the flat model

---

## Method Overview

The workflow of `jacscanomaly` is:

1. **Baseline fitting**
   Fit a global model (e.g. PSPL) to the full light curve.

2. **Residual analysis**
   Compute residuals:

   ```
   residual = data − baseline_model
   ```

3. **Local anomaly scan**
   For each grid point `(t0, teff)`, compare:

   * a null (flat) model
   * an anomaly template model

   within a local time window.

4. **Detection statistic**
   The improvement is measured by:

   ```
   Δχ² = χ²_flat − χ²_anomaly
   ```

---

## Anomaly Score

To quantify how significant the best anomaly candidate is relative to others,
we define a **score**:

```
score = (Δχ²_best − median(Δχ²_others)) / std(Δχ²_others)
```

This measures how strongly the best candidate stands out from the rest of the grid.

---

## Configuration

Key parameters are controlled via `FinderConfig`:

```python
from jacscanomaly import FinderConfig

config = FinderConfig(
    teff_init=0.3,      # initial anomaly timescale
    teff_grid_n=5,      # number of teff grid points
    sigma=3.0,          # threshold for outlier counting
)
```

See `FinderConfig` for the full list of options.

---

## Citation

If you use **jacscanomaly** in academic work, including journal articles,
conference proceedings, or theses, please cite the software.

Citation metadata is provided in the `citation.cff` file in this repository,
which can be used directly by GitHub and reference managers.

---

## Requirements

* Python ≥ 3.9
* numpy
* jax
* jaxopt
* matplotlib

---
