"""Run canonical FSPL-parallax fits over a set of CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pattern", help="quoted CSV glob, e.g. 'pointlensdata/GrpProjData*.csv'"
    )
    parser.add_argument("--ra-deg", required=True, type=float)
    parser.add_argument("--dec-deg", required=True, type=float)
    parser.add_argument("--time-column", default="time")
    parser.add_argument("--value-column", required=True)
    parser.add_argument("--error-column", required=True)
    parser.add_argument("--data-kind", choices=("flux", "mag"), default="flux")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-piE", type=float, default=1.0)
    parser.add_argument("--maxiter", type=int, default=200)
    args = parser.parse_args()

    paths = sorted(Path().glob(args.pattern))
    if not paths:
        raise SystemExit(f"no files matched: {args.pattern}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for index, path in enumerate(paths, start=1):
        row: dict[str, object] = {"data_file": str(path), "status": "failed"}
        output = args.output_dir / f"{path.stem}_fit.json"
        try:
            if output.exists():
                result = json.loads(output.read_text())
                row.update(
                    {
                        "status": "existing",
                        "output": str(output),
                        **result["params"],
                        "chi2": result["chi2"],
                        "chi2_dof": result["chi2_dof"],
                    }
                )
                print(f"[{index}/{len(paths)}] {path.name}: existing")
                rows.append(row)
                continue

            # Isolate every event in a child process so one failed fit is
            # recorded as a failed row rather than terminating the batch.
            command = [
                sys.executable,
                str(Path(__file__).with_name("fit_fspl_parallax.py")),
                str(path),
                "--ra-deg",
                str(args.ra_deg),
                "--dec-deg",
                str(args.dec_deg),
                "--time-column",
                args.time_column,
                "--value-column",
                args.value_column,
                "--error-column",
                args.error_column,
                "--data-kind",
                args.data_kind,
                "--max-piE",
                str(args.max_piE),
                "--maxiter",
                str(args.maxiter),
                "--output",
                str(output),
            ]
            completed = subprocess.run(command, text=True, capture_output=True)
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout).strip().replace(
                    "\n", " "
                )
                raise RuntimeError(
                    f"child exit {completed.returncode}: {detail[-500:]}"
                )
            result = json.loads(output.read_text())
            row.update(
                {
                    "status": "ok",
                    "output": str(output),
                    **result["params"],
                    "chi2": result["chi2"],
                    "chi2_dof": result["chi2_dof"],
                }
            )
            print(
                f"[{index}/{len(paths)}] {path.name}: "
                f"chi2/dof={float(result['chi2_dof']):.6g}"
            )
        except Exception as exc:
            row["error"] = str(exc)
            print(f"[{index}/{len(paths)}] {path.name}: FAILED: {exc}")
        rows.append(row)

    columns = [
        "data_file",
        "status",
        "output",
        "t0",
        "tE",
        "u0",
        "rho",
        "piEN",
        "piEE",
        "chi2",
        "chi2_dof",
        "error",
    ]
    summary = args.output_dir / "fit_batch_summary.csv"
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(summary.resolve())


if __name__ == "__main__":
    main()
