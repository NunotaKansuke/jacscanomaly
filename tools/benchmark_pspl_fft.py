"""Compare the legacy scalar PSPL FFT bank with the batched tE bank."""

from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np
import scipy

from jacscanomaly import PSPLFFTScanner, pspl_excess_magnification


def _median_runtime(function, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-points", type=int, default=32_768)
    parser.add_argument("--observations", type=int, default=4_000)
    parser.add_argument("--u0-count", type=int, default=8)
    parser.add_argument("--scale-count", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--fft-workers", type=int, default=-1)
    args = parser.parse_args()

    if args.grid_points < 2 or args.observations < 2:
        raise ValueError("grid-points and observations must be at least two.")
    if args.u0_count < 1 or args.scale_count < 1 or args.repeats < 1:
        raise ValueError("counts and repeats must be positive.")

    rng = np.random.default_rng(20260731)
    grid_dt = 0.02
    span = (args.grid_points - 1) * grid_dt
    observations = min(args.observations, args.grid_points)
    obs_time = np.sort(rng.uniform(0.0, span, observations))
    ferr = rng.uniform(0.02, 0.05, observations)
    flux = (
        1.4
        + 0.8 * pspl_excess_magnification(obs_time - 0.51 * span, 0.2, 1.2)
        + rng.normal(0.0, ferr)
    )
    u0_grid = np.geomspace(0.01, 1.0, args.u0_count)
    scales = np.geomspace(0.1, 30.0, args.scale_count)
    scanner = PSPLFFTScanner(
        grid_dt=grid_dt,
        max_grid_points=args.grid_points + 1,
        fft_workers=args.fft_workers,
    )

    legacy = lambda: scanner.search(
        obs_time,
        flux,
        ferr,
        u0_grid=u0_grid,
        teff_grid=scales,
        top_k=4,
    )
    batched = lambda: scanner.search_tE(
        obs_time,
        flux,
        ferr,
        u0_grid=u0_grid,
        tE_grid=scales,
        top_k=4,
    )

    # Warm FFT plans and allocation paths before recording medians.
    legacy()
    batched()
    legacy_seconds = _median_runtime(legacy, args.repeats)
    batched_seconds = _median_runtime(batched, args.repeats)

    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "grid_points": args.grid_points,
                "observations": observations,
                "u0_count": args.u0_count,
                "scale_count": args.scale_count,
                "templates_per_search": args.u0_count * args.scale_count,
                "repeats": args.repeats,
                "fft_workers": args.fft_workers,
                "legacy_seconds": legacy_seconds,
                "batched_tE_seconds": batched_seconds,
                "speedup": legacy_seconds / batched_seconds,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
