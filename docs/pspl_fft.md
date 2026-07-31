# PSPL FFT initialization

`PSPLFFTScanner` provides a fast global initializer for point-source,
point-lens fits. It profiles the linear source and blend fluxes analytically
and evaluates every trial peak time on a regular calculation grid with FFT
cross-correlations.

The observations do not need to be equally spaced. Irregular observations are
accumulated into weighted sufficient statistics on the regular FFT grid, and
empty grid bins receive zero weight. The result is a grid seed; the final fit
should always be evaluated on the original timestamps.

## What the FFT accelerates

For fixed Einstein timescale $t_E$, the exact excess magnification is

$$
x(t-t_0, u_0 \mid t_E)
= A\!\left[\sqrt{u_0^2 + ((t-t_0)/t_E)^2}\right] - 1.
$$

Changing $t_0$ translates a row of the radial source-plane map, while changing
$u_0$ selects another row. `search_tE` therefore loops only over $t_E$ in
Python and evaluates the complete $(u_0, t_0)$ plane with batched
one-dimensional FFTs. If $W_j$ is the sum of inverse-variance weights in
calculation-grid bin $j$ and $(WY)_j$ is the sum of weighted fluxes, the
profiled fit requires three correlations:

$$
\begin{aligned}
Q_x(k)    &= \sum_j W_j x_{j-k}, \\
Q_{xx}(k) &= \sum_j W_j x_{j-k}^2, \\
Q_{xy}(k) &= \sum_j (WY)_j x_{j-k}.
\end{aligned}
$$

Let $W$ and $Y$ be the total weight and total weighted flux. Then

$$
\begin{aligned}
S_{xx}(k)       &= Q_{xx}(k) - \frac{Q_x(k)^2}{W}, \\
S_{xy}(k)       &= Q_{xy}(k) - \frac{YQ_x(k)}{W}, \\
\Delta\chi^2(k) &= \frac{S_{xy}(k)^2}{S_{xx}(k)}.
\end{aligned}
$$

The implementation centers the flux before the correlation to avoid
subtracting large nearly equal baseline terms. This is algebraically equivalent
to the expressions above.

The fitted model is

$$
F(t) = f_0 + f_s [A(t)-1],
$$

where $f_0 = f_s + f_b$. `PSPLFFTProfile` returns `fs`, `f0`, `fb`,
`delta_chi2`, and the corresponding profiled `chi2` at every $t_0$. By default,
negative source-flux solutions are projected to the boundary $f_s = 0$. Blend
flux is not constrained.

## Basic use

Choose a modest source-plane row grid in $u_0$ and an outer scale bank in
$t_E$. The best candidate can be passed directly to the existing PSPL fitter:

```python
import jax.numpy as jnp
import numpy as np

from jacscanomaly import PSPLFFTScanner, PSPLFitter

scanner = PSPLFFTScanner(
    grid_dt=0.05,
    positive_source=True,
    max_grid_points=500_000,
    fft_workers=-1,
)

search = scanner.search_tE(
    time,
    flux,
    ferr,
    u0_grid=np.geomspace(0.01, 1.0, 8),
    tE_grid=np.geomspace(1.0, 1000.0, 24),
    top_k=8,
)

if search.best is None:
    raise RuntimeError("No non-singular positive-source PSPL seed found.")

p0 = search.best.as_pspl_params()  # (t0, tE, u0)
fit = PSPLFitter().fit(
    jnp.asarray(time),
    jnp.asarray(flux),
    jnp.asarray(ferr),
    jnp.asarray(p0),
)
```

For multistart fitting, `search.initial_guesses()` returns the ranked
$(t_0, t_E, u_0)$ rows. `peaks_per_template` can retain more than one local
$t_0$ maximum per template when a light curve contains several plausible
events.

The original `search(u0_grid=..., teff_grid=...)` API remains available for
code that needs a rectangular $(u_0, t_{\rm eff})$ bank. `Finder` now uses the
batched $t_E$-outer search for automatic PSPL initialization.

## Irregular cadence and grid spacing

The scanner assigns each observation to the nearest calculation-grid point and
accumulates

$$
W_j = \sum_{i\in j}\sigma_i^{-2}, \qquad
(WY)_j = \sum_{i\in j}\sigma_i^{-2}F_i.
$$

No interpolation of the measured flux is performed. Multiple observations in a
bin are combined with their inverse-variance weights, and gaps are represented
by bins with zero weight.

If `grid_dt` is omitted, the scanner uses

$$
\Delta t = \frac{\min(t_{\rm eff})}{\mathtt{samples\_per\_teff}}.
$$

A useful initial choice is roughly 5--10 samples across the shortest trial
$t_{\rm eff}$. A finer grid reduces binning and $t_0$ discretization error but
increases memory and runtime. `max_grid_points` prevents accidental large
allocations for a long observing baseline and a very small spacing.

## Meaning of the size symbols

It is useful to distinguish four counts:

- $M$: number of actual observations.
- $K$: number of trial peak times in a direct scan.
- $G$: number of points in the regular FFT calculation grid. This, not $M$, is
  the $N$ in the usual $N \log N$ shorthand.
- $B$: number of template shapes, equal to $N_{u0} N_{tE}$ for `search_tE`.

A direct template-bank scan costs approximately $O(BMK)$. The FFT scanner costs
$O(BG\log G)$ after two data transforms that are reused by the full bank. If the
light curve is extremely sparse compared with a very fine grid, a direct or
truncated-template method can still be faster.

## Choosing the template bank

Use logarithmic spacing for both positive parameters. In high-magnification
events, $u_0$ is partly degenerate with source flux and $t_E$; the most robust
peak observables are often $t_0$ and $t_{\rm eff}$. A small $u_0$ bank is
therefore usually sufficient for initialization, followed by continuous fitting
on the original data.

## Benchmarking

`tools/benchmark_pspl_fft.py` compares the legacy scalar-template bank and the
new batched $t_E$ bank with the same observation count, FFT-grid length, number
of $u_0$ rows, and number of outer scales. For example:

```console
python tools/benchmark_pspl_fft.py \\
    --grid-points 32768 --u0-count 8 --scale-count 24 --repeats 5
```

The two banks contain the same number of templates but use different physical
parameterizations, so this benchmark measures execution cost rather than
candidate-by-candidate scientific equivalence. Numerical equivalence is tested
separately by comparing every batched $(u_0, t_0)$ profile with scalar
exact-PSPL template scans at $t_{\rm eff} = u_0 t_E$.

`pspl_excess_magnification` evaluates $A - 1$ with a rationalized expression
instead of computing $A$ and subtracting one. This avoids cancellation in the
far wings and keeps the FFT template accurate across a long baseline.

## Scope and limitations

The first implementation covers rectilinear PSPL templates only. It does not
include finite-source effects, annual or space parallax, or a non-uniform FFT.
It is intentionally independent of the residual-anomaly grid runner and does
not change `Finder` defaults. Its output is an initializer and candidate finder,
not a replacement for the final likelihood fit.
