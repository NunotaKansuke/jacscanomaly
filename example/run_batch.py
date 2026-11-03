"""Run the pipeline over many light curves and write one JSON.

Each file is a .npy array or a .csv table with the columns time, value, error
(names set with --columns). Example:

    python example/run_batch.py "data/*.csv" --columns BJD,flux,ferr \
        --ra-deg 268.4 --dec-deg -29.2 --time-offset 2450000 --output results.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from jacscanomaly import Event, FinderConfig, write_json


def load(path: Path, columns: list[str]):
    if path.suffix == ".npy":
        return np.load(path).T[:3]
    table = np.genfromtxt(path, delimiter=",", names=True)
    return [np.asarray(table[c], dtype=float) for c in columns]


def run(ev: Event, parallax: bool) -> None:
    ev.fit("pspl")
    pspl_scan = ev.scan()
    if not pspl_scan.detected:
        return
    ev.fit("fspl", start="pspl")
    scans = [ev.scan()]
    if parallax:
        ev.fit("parallax", start="pspl")
        scans.append(ev.scan())
    if all(s.detected for s in scans):
        for model in ev.fits:
            ev.fit_atoms(model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pattern", help="quoted glob, e.g. 'data/*.csv'")
    parser.add_argument("--columns", default="time,flux,ferr",
                        help="CSV columns for time, value, error")
    parser.add_argument("--kind", choices=("flux", "mag"), default="flux")
    parser.add_argument("--ra-deg", type=float)
    parser.add_argument("--dec-deg", type=float)
    parser.add_argument("--ephemeris", help="spacecraft ephemeris table")
    parser.add_argument("--time-offset", type=float, default=0.0,
                        help="added to time to get JD, e.g. 2450000")
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    args = parser.parse_args()

    paths = sorted(Path().glob(args.pattern))
    if not paths:
        raise SystemExit(f"no files matched: {args.pattern}")
    columns = args.columns.split(",")
    parallax = args.ra_deg is not None and args.dec_deg is not None
    config = FinderConfig(parallax_time_offset=args.time_offset)

    records = {}
    for i, path in enumerate(paths, start=1):
        time, value, error = load(path, columns)
        ev = Event(time, value, error, kind=args.kind, ra_deg=args.ra_deg,
                   dec_deg=args.dec_deg, ephemeris=args.ephemeris, config=config)
        records[path.stem] = record = {}
        try:
            run(ev, parallax)
        except Exception as exc:
            record["error"] = repr(exc)
        record.update(ev.to_dict())
        found = [m for m, a in ev.atoms.items() if a.features]
        print(f"[{i}/{len(paths)}] {path.name}: "
              f"{'anomaly (' + ', '.join(found) + ')' if found else 'no anomaly'}")

    write_json(args.output, records)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
