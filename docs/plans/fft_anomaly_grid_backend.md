# FFT anomaly-grid backend plan

## Context

The existing `jacscanomaly` residual grid evaluates every `(t0, teff)` point independently with either JAX or the C++ loop backend. At each point it restricts the data to

```text
|t - t0| < teff_coeff * teff,
```

fits a weighted constant model, fits both existing anomaly templates, and reports the larger chi-square improvement.

The two templates are

```text
A0(t) = 1 / sqrt(1 + ((t - t0) / teff)^2)
```

and

```text
Q(t)  = 1 + ((t - t0) / teff)^2
A1(t) = (Q + 2) / sqrt(Q * (Q + 4)).
```

For fixed `teff`, changing `t0` translates the box window, `A0`, and `A1`. The weighted constant fit and both weighted line fits can therefore be reduced to cross-correlations and evaluated together with FFTs.

## User-facing goal

Support

```python
FinderConfig(grid_backend="fft")
```

alongside the existing `"jax"` and `"cpp"` choices, without changing the result containers, adding a second public runner, or changing the `Finder.run()` workflow. The target is the general residual anomaly-search grid already orchestrated by `SeasonGridRunner`.

## Mathematical design

For one `teff`, let `b(t - t0)` be the local box window. On a regular calculation grid, accumulate irregular observations into weighted sufficient-statistic arrays:

```text
N_j   = number of observations in bin j
W_j   = sum_i w_i
WY_j  = sum_i w_i y_i
WY2_j = sum_i w_i y_i^2.
```

The flux is first shifted by one global weighted mean. Every fitted model includes an intercept, so this leaves all local chi-square improvements unchanged while reducing cancellation in the constant-model sums.

For every translated center, box correlations give

```text
N    = N   star b
W    = W   star b
Y    = WY  star b
Y2   = WY2 star b.
```

The weighted constant-model chi-square is

```text
chi2_flat = Y2 - Y^2 / W.
```

For either template `x`, additionally compute

```text
X  = W  star (b x)
XX = W  star (b x^2)
XY = WY star (b x).
```

Then

```text
Sxx = XX - X^2 / W
Sxy = XY - X Y / W
delta_chi2 = Sxy^2 / Sxx.
```

The backend evaluates this expression for both `A0` and `A1`, applies the existing A1-on-tie rule, and retains the larger valid improvement at each `t0`.

## Irregular cadence

Observations do not need to be regularly sampled. For each `teff`, the calculation spacing is

```text
dt_fft = dt0_coeff * teff / fft_oversample.
```

The weighted sufficient statistics are accumulated on that grid. Empty bins have zero count and zero weight. Weighted squared-flux contributions are summed before the FFT, so the approximation is only that the translated window and template are constant inside one calculation bin.

## Exact cluster refinement

FFT values are used for the full, large candidate grid. After the existing cluster extractor chooses one representative per overlap group, those representatives are re-evaluated directly on the original irregular timestamps.

The direct refinement:

- fits the weighted constant model exactly,
- fits both `A0` and `A1` exactly and applies the existing tie rule,
- replaces the representative `dchi2`,
- computes `n_window`, `n_contrib`, `n_eff`, `peak_frac`, `rho1`, and `longest_run` with the same definitions as the established JAX and C++ implementations.

This keeps candidate selection, downstream seed use, automatic single-lens initialization, and quality criteria compatible while avoiding direct evaluation of every point in the complete grid.

## Configuration

Add:

- `grid_backend="fft"`
- `fft_oversample`, default `4`
- `fft_max_grid_points`, default `1_000_000`
- `fft_singular_rtol`, default `1e-12`

`grid_chunked` settings remain JAX-specific.

## Implementation

1. Add an internal `src/jacscanomaly/fft_grid.py` correlation engine; it is not exported as a second user workflow.
2. Extend `FinderConfig.grid_backend` and add FFT controls.
3. Route the existing `SeasonGridRunner` through the FFT evaluator when selected.
4. Run the unchanged cluster extractor on the approximate full grid.
5. Re-evaluate cluster representatives exactly and patch their rows in `grid_metrics`.
6. Document the backend and its relation to the separate global `PSPLFFTScanner` initializer.

## Tests

Add tests for:

1. FFT equality with direct weighted regression on aligned regular samples.
2. The weighted constant fit through the box correlation.
3. Correct selection and recovery for both `A0` and `A1`.
4. Irregular-cadence candidate recovery.
5. Stability under large constant flux offsets and strongly varying weights.
6. Exact representative metrics against the existing JAX definitions.
7. `SeasonGridRunner` integration with `grid_backend="fft"`.
8. Calculation-grid allocation guards.

## Non-goals

- Replacing the existing `SeasonGridRunner`, cluster extractor, or final single-lens/binary-lens fit.
- Applying FFTs to parallax or finite-source magnification calculations.
- Returning dense exact quality diagnostics for every non-candidate grid point; detailed diagnostics are materialized exactly for extracted representatives, where downstream selection uses them.
- Changing the default backend in this change.
