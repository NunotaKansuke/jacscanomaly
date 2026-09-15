"""Fit the canonical FSPL-parallax model for one CSV light curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jacscanomaly import Finder, FinderConfig


def _float_tuple(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--ra-deg", required=True, type=float)
    parser.add_argument("--dec-deg", required=True, type=float)
    parser.add_argument("--time-column", default="time")
    parser.add_argument("--value-column", required=True)
    parser.add_argument("--error-column", required=True)
    parser.add_argument("--data-kind", choices=("flux", "mag"), default="flux")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-piE", type=float, default=1.0)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument(
        "--x0",
        type=_float_tuple,
        default=None,
        help="optional t0,tE,u0,logrho,piEN,piEE; skips automatic seed search",
    )
    args = parser.parse_args()

    table = np.genfromtxt(
        args.csv, delimiter=",", names=True, dtype=None, encoding="utf-8"
    )
    try:
        time = np.asarray(table[args.time_column], dtype=float)
        value = np.asarray(table[args.value_column], dtype=float)
        error = np.asarray(table[args.error_column], dtype=float)
    except ValueError as exc:
        raise SystemExit(f"column not found in {args.csv}: {exc}") from exc

    finder = Finder(
        FinderConfig(
            fitter_kind="fspl_parallax",
            ra_deg=args.ra_deg,
            dec_deg=args.dec_deg,
            max_piE=args.max_piE,
            fitter_maxiter=args.maxiter,
        )
    )
    x0 = None if args.x0 is None else np.asarray(args.x0, dtype=float)
    if x0 is not None and x0.shape != (6,):
        raise SystemExit("--x0 requires exactly t0,tE,u0,logrho,piEN,piEE")
    fit = finder.fit_single_lens(
        time, value, error, x0=x0, data_kind=args.data_kind
    )

    result = {
        "data_file": str(args.csv),
        "backend": "scipy_lm_compiled_evaluator",
        "fitter_kind": "fspl_parallax",
        "n_data": int(time.size),
        "params": {
            name: float(value)
            for name, value in zip(fit.param_names, np.asarray(fit.params))
        },
        "raw_params": {
            name: float(value)
            for name, value in zip(
                ("t0", "log_tE", "u0", "logrho", "piEN", "piEE"),
                np.asarray(fit.raw_params),
            )
        },
        "chi2": float(fit.chi2),
        "chi2_dof": float(fit.chi2_dof),
        "fs": float(fit.fs),
        "fb": float(fit.fb),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
