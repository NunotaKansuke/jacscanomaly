# PSPL FFT initialization plan

## Context

`jacscanomaly` already profiles the linear source and blend fluxes during a single-lens fit, but the nonlinear PSPL optimizer still needs a useful starting point in `(t0, tE, u0)`. A direct scan over `K` trial peak times and `M` observations costs `O(MK)` for every template shape.

For fixed `(u0, teff)`, where `teff = u0 * tE`, changing `t0` only translates the PSPL excess-magnification template

```text
x(t - t0) = A[u0 * sqrt(1 + ((t - t0) / teff)^2)] - 1.
```

The weighted profile likelihood at every translated `t0` can therefore be expressed through three cross-correlations and evaluated together with FFTs.

## Goals

1. Add a reusable, NumPy-only PSPL FFT scanner with no new runtime dependency.
2. Accept irregularly sampled observations by accumulating weighted sufficient statistics on a regular calculation grid; observations themselves do not need to be regularly spaced.
3. For a fixed `(u0, teff)`, return the full `t0` profile including:
   - `delta_chi2` relative to a weighted constant-flux model,
   - profiled `chi2`,
   - analytic `fs`, baseline flux, and `fb`.
4. Search a small `(u0, teff)` template bank and return ranked candidates that can be passed directly to the existing PSPL fitter as `(t0, tE, u0)` seeds.
5. Validate the FFT result against direct weighted regression and test recovery on irregular cadences.

## Non-goals for the first implementation

- Replacing the existing residual-anomaly grid runner.
- Changing `Finder` default behavior or automatic initialization policy.
- Performing the final continuous PSPL optimization on the binned grid.
- Supporting parallax or finite-source templates.
- Implementing a non-uniform FFT. The first version uses controlled weighted binning and recommends refinement on the original timestamps.

## Mathematical design

Let the regular calculation grid contain `G` bins. For observation `i`, define `w_i = 1 / ferr_i^2` and accumulate into grid bin `j`:

```text
W_j   = sum_i w_i
WY_j  = sum_i w_i * y_i
Y2    = sum_i w_i * y_i^2
```

Empty bins have zero weight. `Y2` is retained from the original observations, so only the template value is approximated as constant inside a bin.

For a template centered at grid index `k`, define `x_{j-k} = A_{j-k} - 1`. The three required correlations are

```text
Qx(k)  = sum_j W_j  * x_{j-k}
Qxx(k) = sum_j W_j  * x_{j-k}^2
Qxy(k) = sum_j WY_j * x_{j-k}.
```

With `W = sum_j W_j`, `Y = sum_j WY_j`, and `Syy = Y2 - Y^2 / W`, the centered regression terms are

```text
Sxx(k) = Qxx(k) - Qx(k)^2 / W
Sxy(k) = Qxy(k) - Y * Qx(k) / W.
```

The profiled quantities are

```text
fs(k)         = Sxy(k) / Sxx(k)
f0(k)         = Y / W - fs(k) * Qx(k) / W
fb(k)         = f0(k) - fs(k)
delta_chi2(k) = Sxy(k)^2 / Sxx(k)
chi2(k)       = Syy - delta_chi2(k).
```

When positive source flux is required, negative `Sxy` values are projected to the boundary `fs = 0`, giving `delta_chi2 = 0` there.

The correlations are evaluated as linear convolutions with zero padding. Data FFTs are cached once per light curve and reused across the template bank. For `B` template shapes, the expected cost is `O(B G log G)`, compared with `O(B M K)` for a direct scan. Here `M` is the number of observations, `K` is the number of trial peak times, and `G` is the regular FFT-grid length.

## Public API

Add `src/jacscanomaly/pspl_fft.py` with:

- `PSPLFFTScanner`
  - validates and sorts input data,
  - chooses or accepts a grid spacing,
  - bins weighted sufficient statistics,
  - caches data transforms,
  - exposes `scan_template(u0, teff)` and `search(u0_grid, teff_grid, top_k)`.
- `PSPLFFTProfile`
  - full arrays over the `t0` grid for one template.
- `PSPLFFTCandidate`
  - scalar candidate values and an `as_pspl_params()` helper returning `(t0, tE, u0)`.
- `PSPLFFTSearchResult`
  - ranked candidates, the best candidate, grid metadata, and the constant-model chi-square.

Export these classes from `jacscanomaly.__init__`.

## Numerical and operational safeguards

- Use a stable direct expression for `A - 1` to avoid cancellation in the wings.
- Require finite one-dimensional arrays, positive errors, positive `u0` and `teff`, and at least two observations.
- Reject templates with numerically singular `Sxx`.
- Use linear, not circular, convolution with sufficient zero padding.
- Guard automatic grid construction with `max_grid_points` so a long baseline and very small spacing cannot allocate an unexpectedly large array.
- Document that the returned parameters are grid seeds; final fitting must use the original timestamps.

## Tests

Add `tests/test_pspl_fft.py` covering:

1. Stable excess magnification against the direct PSPL formula.
2. FFT profile versus direct weighted regression at every `t0` on a regular grid with gaps and heteroscedastic errors.
3. Recovery of a synthetic event from irregularly sampled data.
4. Positive-source boundary behavior.
5. Input validation and the grid-size guard.
6. Public package exports.

## Documentation

Add `docs/pspl_fft.md` with the equations, the distinction between observation count `M` and FFT-grid length `G`, irregular-cadence behavior, an example template-bank search, and refinement through `PSPLFitter`.

Link the page from `docs/index.rst` if the existing toctree structure permits a minimal edit.

## Delivery sequence

1. Commit this plan on a new branch.
2. Add the scanner, public exports, tests, and user documentation.
3. Run focused tests and numerical comparison scripts.
4. Push the implementation commit to the same branch.

A later change can benchmark this scanner against the current automatic initializer and, if justified, add an opt-in `FinderConfig` backend before considering any default change.
