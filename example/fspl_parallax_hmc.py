"""Fit FSPL+annual parallax, then sample its posterior with NUTS."""

import numpy as np
import pandas as pd

from jacscanomaly import Finder, FinderConfig, sample_fspl_parallax_hmc


DATA_FILE = "pointlensdata/GrpProjData4.csv"
RA_DEG = 268.1715
DEC_DEG = -29.279525


def main() -> None:
    data = pd.read_csv(DATA_FILE)
    finder = Finder(FinderConfig(
        fitter_kind="fspl_parallax",
        ra_deg=RA_DEG,
        dec_deg=DEC_DEG,
    ))
    # A supplied optimization result can be passed here instead.  ``fit`` is
    # also initialized automatically if x0 is omitted.
    fit = finder.fit_single_lens(
        data["time"], data["magF146"], data["magF146err"], data_kind="mag"
    )
    posterior = sample_fspl_parallax_hmc(
        fit,
        rng_seed=42,
        num_warmup=1_000,
        num_samples=2_000,
        # Default peak_window_days="auto" follows the optimized rho * tE.
    )
    for name in ("t0", "tE", "u0", "rho", "piEN", "piEE", "fs", "fb"):
        print(f"{name:>4} = {posterior.median(name):.8g}")


if __name__ == "__main__":
    main()
