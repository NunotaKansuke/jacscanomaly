# jacscanomaly

[![Documentation Status](https://readthedocs.org/projects/jacscanomaly/badge/?version=latest)](https://jacscanomaly.readthedocs.io/en/latest/?badge=latest)

**jacscanomaly** is a Python package for scan-based anomaly detection
in time-series light curves.

The package is designed to detect **microlensing planetary anomalies** by
scanning residuals after fitting a single lens model (e.g., PSPL),
with low-memory C++ backends for large survey light curves and a unified
single-lens fitting API.

---

## Features

* **Scan-based anomaly detection** on residuals after single-lens fitting
* **Unified fitters** for PSPL, FSPL, and their annual/space-parallax forms
* **Compiled model evaluation** for FSPL magnification and parallax trajectories
* **JAX model components** for anomaly grids and higher-order diagnostics
* **Candidate quality diagnostics**: effective contributing points,
  peak-contribution fraction, and time-correlation metrics
* **Built-in visualization**: PSPL fit, residuals, and anomaly scan summary

---

## Documentation

The full documentation is available on ReadTheDocs:

https://jacscanomaly.readthedocs.io/en/latest/

Start with:

* [Installation](https://jacscanomaly.readthedocs.io/en/latest/installation.html)
* [Quickstart](https://jacscanomaly.readthedocs.io/en/latest/quickstart.html)
* [Examples](https://jacscanomaly.readthedocs.io/en/latest/examples.html)
* [Method overview](https://jacscanomaly.readthedocs.io/en/latest/method.html)
* [API reference](https://jacscanomaly.readthedocs.io/en/latest/api.html)

---

## Installation

```bash
pip install jacscanomaly
```

---

## Quick Example

```python
import numpy as np
import matplotlib.pyplot as plt
from jacscanomaly import CandidateCriteria, Finder, FinderConfig

# load data (time, flux, flux_err)
data = np.load("example_data.npy")
time, flux, ferr = data[:, 0], data[:, 1], data[:, 2]

# run anomaly finder
config = FinderConfig(
    fitter_kind="pspl",
    candidate_criteria=CandidateCriteria(min_n_eff=2.0),
)
finder = Finder(config)
result = finder.run(time, flux, ferr)

# For magnitude data, pass mag and magerr in the same positions and select it:
# result = finder.run(time, mag, magerr, data_kind="mag")

# You can still pass an explicit initial guess if desired:
# p0 = np.array([10000, 10, 0.3])
# result = finder.run(time, flux, ferr, p0)

result.print_summary()

# For the complete adopted-model anomaly workflow:
# result = finder.run_anomaly_pipeline(time, flux, ferr)
# print(result.has_anomaly_candidate)
# print(result.best_anomaly_candidate)

# In notebooks, get a one-row table:
# display(result.summary_table())
```

---

## Visualization

```python
finder.plot_result()
finder.plot_anomaly_window()
plt.show()
```

These commands produce two complementary visualizations:

1. **Three-panel summary plot (`finder.plot_result`)**

   * **Top:** Observed light curve with the best-fit baseline model (PSPL)
   * **Middle:** Residuals after baseline fitting
   * **Bottom:** Anomaly scan result (Δχ² vs. time), showing where localized
     deviations from the baseline model are detected

2. **Focused anomaly window plot (`finder.plot_anomaly_window`)**

   * A zoomed-in view around the best anomaly candidate
   * Residuals are shown together with the anomaly template and the flat model

Example notebooks are available in `example/`:

* `template_scan_example.ipynb` for the standard bell-template scan
* `template_free_example.ipynb` for the template-free residual chi-square scan

For a refined planetary residual, measure its peaks and dips directly:

```python
features = signal.measure_features()
print(features.n_peaks, features.n_dips)
for feature in features.features:
    print(feature.kind, feature.time, feature.timescale, feature.strength)
```

This measurement does not assign a caustic shape or estimate binary-lens
parameters.

---

## Method Overview

The workflow of `jacscanomaly` is:

1. **First fitting**
   Fit a single lens model (e.g. PSPL) to the full light curve.

2. **Residual analysis**
   Compute residuals:

   ```
   residual = data − single_lens_model
   ```

3. **Local anomaly scan**
   For each grid point `(t0, teff)`, compare:

   * a flat model
   * an anomaly template model
     within a local time window.

4. **Detection statistic**
   The improvement is measured by:

   ```
   Δχ² = χ²_flat − χ²_anomaly
   ```

---

## Anomaly Score

To quantify how significant each extracted anomaly candidate is relative to
others, we define a **score**:

```
score(c) = (Δχ²_c − median(Δχ²_others(c))) / robust_scale(Δχ²_others(c))
```

The reference population contains raw clusters from all observing seasons and
a comparable `teff`. Candidate-quality cuts are applied only after raw cluster
extraction, so changing those cuts does not redefine the score background. The
center and scale are estimated robustly with the median and MAD. Strong
secondary candidates are removed with adaptive one-sided clipping rather than
an unconditional percentile cut.

The finder computes this independently for every extracted cluster and exposes
the sorted results as `result.scored_candidates`. `result.best` remains the
maximum-`dchi2` candidate accepted by the quality criteria; it is not a
score-based selection.

The score background is event-wide but remains local in timescale, with robust
median/MAD normalization. Existing efficiency maps or thresholds using the
older same-season score should be recalibrated.

---

## Candidate Quality Diagnostics

Large Δχ² values can sometimes be dominated by one or two points. To make this
visible, `jacscanomaly` stores per-candidate support diagnostics in
`result.best.quality` and per-grid diagnostics in `result.grid_metrics_all`.

For the best candidate:

```python
q = result.best.quality
print(q.n_window)     # points in the local chi2 window
print(q.n_contrib)    # points above the per-point improvement threshold
print(q.n_eff)        # effective number of contributing points
print(q.peak_frac)    # strongest-point fraction of total positive improvement
print(q.rho1)         # lag-1 autocorrelation of per-point improvements
print(q.longest_run)  # longest consecutive run of contributing points
```

The effective point count is computed from positive per-point improvements
using a participation-ratio style statistic:

```
n_eff = (sum_i u_i)^2 / sum_i u_i^2
```

where `u_i = max(0, chi2_flat_i - chi2_anomaly_i)`. A one-point-dominated
candidate has `n_eff` close to 1 and a large `peak_frac`.

`result.grid_metrics_all` is a NumPy array with columns:

```
[t0, teff, dchi2, n_window, n_contrib, n_eff, peak_frac, rho1, longest_run]
```

---

## Result Summaries

`AnomalyResult` provides both CLI-friendly and notebook-friendly summaries:

```python
result.print_summary()       # print formatted text
text = result.summary_text() # return formatted text
row = result.summary_dict()  # return a plain dictionary
table = result.summary_table()  # pandas.DataFrame when pandas is installed
```

`print(result)` also shows the formatted summary text.

---

## Configuration

Key parameters are controlled via `FinderConfig`:

```python
from jacscanomaly import CandidateCriteria, FinderConfig

config = FinderConfig(
    grid_backend="cpp",  # default for PSPL survey scans
    teff_init=0.03,      # initial anomaly timescale
    teff_grid_n=20,      # number of teff grid points
    sigma=3.0,           # per-point improvement threshold for n_contrib
    candidate_criteria=CandidateCriteria(min_n_eff=2.0),
    best_score_teff_ratio=2.0,
    best_score_upper_clip_sigma=5.0,
)
```

See `FinderConfig` for the full list of options.

For a finite-source baseline, use the same canonical fitter API:

```python
config = FinderConfig(
    fitter_kind="fspl",
    grid_backend="cpp",
)
```

For GULLS-convention spacecraft parallax, use the standard parallax model and
select the coordinate convention explicitly:

```python
config = FinderConfig(
    fitter_kind="fspl_parallax",
    grid_backend="cpp",
    parallax_geometry="space",
    ra_deg=267.3,
    dec_deg=-29.9,
    tref=2461504.0,
    satellite_ephemeris_path="gulls_orbit5_heliocentric.dat",
    parallax_observer_convention="gulls",
    parallax_time_scale="hjd",
)
```

All four fitters use SciPy LM for continuous optimization. FSPL magnification
and parallax trajectories are evaluated by the compiled backend. Space versus
annual parallax and the GULLS observer convention are options, not separate
fitter classes.

---

## Example Data

The light curves used as examples in this repository are drawn from an original set of
**2,371 simulated Roman light curves** generated by the **Roman Galactic Exoplanet Survey
Project Infrastructure Team (RGES PIT)**, **WG07 Survey Simulations and Pipeline Validation**
(Farzaneh Zohrabi, Matthew Penny, Macy Huston, Ali Crisp, et al).

This representative sample of 2,371 light curves was selected assuming the **Cassan exoplanet
mass function** and consists of simulated Roman light curves of **planetary microlensing events**,
including higher-order effects such as **parallax** and **orbital motion**.

---

## Algorithmic Background

The anomaly scan implemented in `jacscanomaly` is inspired by the
systematic anomaly search methodology developed for microlensing surveys
(e.g., the KMTNet AnomalyFinder series). In particular, the approach
of scanning residual light curves over a grid of anomaly times and
durations is based on key ideas presented in:

> Zang, W., Jung, Y., Yee, J., et al. (2021). *Systematic KMTNet Planetary
> Anomaly Search, Paper I: OGLE-2019-BLG-1053Lb, A Buried Terrestrial
> Planet*. The Astronomical Journal, **162**, 163.  
> DOI: 10.3847/1538-3881/ac12d4 :contentReference[oaicite:3]{index=3}

This work described a semi-automated search algorithm that iteratively
scans events for localized deviations relative to a baseline model and
quantifies the significance of detected signals — an idea that is central
to the grid-scan and Δχ² evaluation in `jacscanomaly`.

---

### Finite-source magnification (FSPL)

`jacscanomaly` uses the compiled finite-source magnification backend for the
canonical `FSPLFitter` and `FSPLParallaxFitter`. The same classes cover annual
and space geometry through `parallax_geometry`.

JAX remains available for the anomaly-grid and diagnostic modules; it is not a
second FSPL fitting route.

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
* scipy
* matplotlib

VBMicrolensing ≥ 5.5 is also required to build the compiled extensions.

---

## Development

Install the package with development dependencies:

```bash
pip install -e ".[dev]"
```

Run the unit tests:

```bash
pytest
```

Run the tests with coverage:

```bash
coverage run -m pytest
coverage report
```

Build the Sphinx documentation locally:

```bash
sphinx-build -W -b html docs docs/_build/html
```
