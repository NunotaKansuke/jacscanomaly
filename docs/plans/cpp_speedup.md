# Speeding up the scan + atoms robust fit

Goal: the same results, faster. Changes are either bit-identical, or change
only the floating-point rounding of a quantity that is then optimised
(stated per item).

## Baseline (f1d9053, 4 events 1111/1767/0042/1285, one thread)

| where | s | share |
|---|---|---|
| `_cpp_grid.run_grid` (396 calls, 66 `Finder.run`) | 105 | 52 % |
| atoms `_linear` (9694 calls: `lstsq` 13, `atom_basis` 5, stacking 3) | 33 | 16 % |
| PSPL FFT bank `search_tE` (12 calls) | 25 | 12 % |
| `extract_clusters` | 8 | 4 % |
| `ParallaxEvaluator.magnification` | 7 | 3 % |
| total | 203 | |

## Done

1. **run_grid** -- the two template shapes were re-evaluated (sqrt, divide)
   in each of eight passes over a window. Now they are evaluated once per
   window sample into a per-thread buffer and the passes are fused (both
   line fits share their sums; the improvement series is stored for the lag-1
   pass). Every sum keeps its order: outputs are bit-identical. 2.5x.
2. **atoms linear solve** (`_cpp_grid.atoms_linear`) -- the atom columns are
   non-zero only in their windows, so the normal equations cost O(n + sum of
   windows). Cholesky on the column-equilibrated Gram matrix plus two
   refinement steps on the true residual (corrected semi-normal equations);
   when the scaled pivot falls below 1e-10 it returns None and `_linear` uses
   the SVD solution as before. Agrees with `lstsq` to ~1e-12 in the
   residual; 11x per call. `least_squares` then lands on the same optimum to
   its tolerance (J differences <= 0.3, mostly 1e-3, on 30 events; same
   selected model and atom counts on all 30).
3. **PSPL FFT bank** -- the PSPL and parallax base fits ran the identical
   search on identical data. The last search is reused when its inputs (data
   hash, grids, settings) match. Bit-identical.
4. **extract_clusters** -- the 170k priorities were sorted with an indirect
   comparator; now a stable LSD radix sort on order-preserving keys (same
   order, ties by index). Bit-identical. 1.75x.
5. **ParallaxEvaluator.magnification** recomputed the ephemeris geometry per
   call; it now uses the per-sample geometry cached at construction, as
   `residual`/`jacobian` already did. Same arithmetic, bit-identical.

6. **run_grid AVX2 clone** -- the grid loop runs in chunks of 256 points
   through a `target_clones("avx2", "default")` function, so the basis loop
   is 4-wide where the CPU allows and the dispatch is paid once per chunk.
   No FMA, sums in the same order: bit-identical. ~1.15x.

After 1-6 the 4 events take 116 s (203 s at baseline; the machine is loaded,
so timings are +-15 %).

## Tried and dropped

* AVX2 clone of the basis loop alone: the per-window dispatch cost more than
  it saved.
* `omp simd` reductions in run_grid: another 7 %, but the sums change order
  (1e-13 differences). Not worth giving up bit-identity.
* The PSPL FFT batch is now mostly pocketfft itself (nfft = 2^18 per row);
  the template and post-processing are ~1/3 of it.

## Next

* Analytic variable-projection Jacobian for the atoms host fit (only the
  magnification column depends on the host parameters), with dA/dp from
  the PSPL closed form, the VBM gradient (FSPL) and the parallax evaluator.
  Replaces the 2-point finite differences (n_p + 1 solves per Jacobian). This
  changes the optimiser path, so it needs the 30-event comparison again.
