# jacscanomaly

[![Documentation Status](https://readthedocs.org/projects/jacscanomaly/badge/?version=latest)](https://jacscanomaly.readthedocs.io/en/latest/?badge=latest)

Find short anomalies in microlensing light curves.

Fit a single-lens model (PSPL, FSPL, parallax), scan its residual for short
bumps and dips, and describe what is found: a host fit the anomalies do not
pull, and each anomaly as bump / dip / caustic with its time and FWHM.

Documentation: https://jacscanomaly.readthedocs.io/en/latest/

## Installation

```bash
pip install jacscanomaly
```

## Example

One light curve is one `Event`; the steps are its methods and the results stay on it.

```python
import numpy as np
from jacscanomaly import Event

time, flux, ferr = np.load("example/example_data.npy").T
ev = Event(time, flux, ferr, ra_deg=268.4, dec_deg=-29.2, ephemeris="roman.txt")

ev.fit("pspl")
pspl_scan = ev.scan()
if pspl_scan.detected:
    ev.fit("fspl", start="pspl")
    fspl_scan = ev.scan()
    ev.fit("parallax", start="pspl")
    parallax_scan = ev.scan()
    if fspl_scan.detected and parallax_scan.detected:
        ev.fit_atoms("pspl")
        ev.fit_atoms("fspl")
        ev.fit_atoms("parallax")

ev.summary()
ev.write_json("event.json")
ev.plot_result("pspl")
```

## Example data

The example light curves are drawn from **2,371 simulated Roman light curves**
of planetary microlensing events (with parallax and orbital motion), made by
the **Roman Galactic Exoplanet Survey Project Infrastructure Team (RGES PIT)**,
**WG07 Survey Simulations and Pipeline Validation** (Farzaneh Zohrabi, Matthew
Penny, Macy Huston, Ali Crisp, et al.), assuming the Cassan exoplanet mass
function.

## Background

The residual scan over anomaly times and durations follows the systematic
KMTNet AnomalyFinder search:

> Zang, W., Jung, Y., Yee, J., et al. (2021). *Systematic KMTNet Planetary
> Anomaly Search, Paper I: OGLE-2019-BLG-1053Lb, A Buried Terrestrial
> Planet*. AJ, **162**, 163. DOI: 10.3847/1538-3881/ac12d4

## Citation

If you use **jacscanomaly** in academic work, please cite the software
(`citation.cff`).

## Development

```bash
git clone git@github.com:NunotaKansuke/jacscanomaly.git
cd jacscanomaly
pip install -e ".[dev]"
pytest
sphinx-build -W -b html docs docs/_build/html
```

This package was developed with the help of AI coding assistants: OpenAI
ChatGPT / Codex and Anthropic Claude (Claude Code).
