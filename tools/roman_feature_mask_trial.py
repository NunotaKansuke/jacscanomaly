#!/usr/bin/env python3
"""Run and publish a sparse Roman fit-mask comparison site."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
ROMAN_TOOL = WORKSPACE / "roman_simu" / "tool"
RESULT_ROOT = WORKSPACE / "roman_simu" / "anomaly_finder_result"
PORTAL_ROOT = WORKSPACE / "roman_simu" / "html_portal"
EVENT_CSV = WORKSPACE / "sample_rtmodel_v2.4" / "OMPLDG_croin_cassan.sample.csv"
SYNC_REQUEST = WORKSPACE / "html_portal" / "tool" / "request_sync.sh"
DEFAULT_EVENTS = (
    "0_680_836",
    "0_676_2084",
    "0_841_2794",
    "0_999_2677",
)


def run(command: list[str], *, env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=WORKSPACE, env=env, check=True)


def event_name(row: pd.Series) -> str:
    return f"{int(row['SubRun'])}_{int(row['Field'])}_{int(row['EventID'])}"


def resolve_events(values: list[str]) -> tuple[list[str], list[int]]:
    rows = pd.read_csv(EVENT_CSV)
    by_name = {
        event_name(row): int(index)
        for index, row in rows.iterrows()
    }
    missing = sorted(set(values) - set(by_name))
    if missing:
        raise ValueError(f"unknown Roman event names: {', '.join(missing)}")
    names = list(dict.fromkeys(values))
    return names, [by_name[name] for name in names]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events",
        default=",".join(DEFAULT_EVENTS),
        help="Comma-separated SubRun_Field_EventID names.",
    )
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--tag", default="feature_fit_mask_trial_v1")
    parser.add_argument(
        "--site-dir-name",
        default="anomaly_finder_model_result_feature_mask",
    )
    parser.add_argument(
        "--current-site-dir-name",
        default="anomaly_finder_model_result",
    )
    parser.add_argument(
        "--variant-name",
        default="fit mask trial",
    )
    parser.add_argument("--event-timeout", type=int, default=600)
    parser.add_argument(
        "--fit-bin-points",
        type=int,
        default=1,
        help="Bin only scan-stage PSPL baseline fits by this many observations.",
    )
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Reuse existing scan/final JSON and rebuild only the sparse site.",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Keep completed event JSON while filling missing trial outputs.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Request the normal portal synchronization after a successful build.",
    )
    args = parser.parse_args()

    requested = [
        value.strip() for value in str(args.events).split(",") if value.strip()
    ]
    names, indices = resolve_events(requested)
    index_arg = ",".join(str(index) for index in indices)
    event_arg = ",".join(names)
    tag = str(args.tag)

    scan_data = RESULT_ROOT / f"planet_signal_{tag}_scan_data"
    scan_figures = RESULT_ROOT / f"planet_signal_{tag}_scan_figures"
    physical_data = RESULT_ROOT / f"planet_signal_{tag}_physical_data"
    physical_figures = RESULT_ROOT / f"planet_signal_{tag}_physical_figures"
    post_data = RESULT_ROOT / f"planet_signal_{tag}_post_physical_data"
    post_figures = RESULT_ROOT / f"planet_signal_{tag}_post_physical_figures"
    final_data = RESULT_ROOT / f"planet_signal_{tag}_final_residual_data"
    final_figures = RESULT_ROOT / f"planet_signal_{tag}_final_residual_figures"
    route_result = RESULT_ROOT / f"planet_effect_route_{tag}.json"
    route_indices = route_result.with_suffix(".indices.txt")
    site_dir = (PORTAL_ROOT / str(args.site_dir_name)).resolve()
    if (
        site_dir.parent != PORTAL_ROOT.resolve()
        or site_dir.name == str(args.current_site_dir_name)
    ):
        parser.error("trial site must be a distinct direct child of the portal root")

    env = os.environ.copy()
    previous_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPOSITORY / "src") + (
        os.pathsep + previous_pythonpath if previous_pythonpath else ""
    )
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[variable] = "1"
    env.setdefault(
        "XLA_FLAGS",
        "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    )

    if not args.html_only:
        scan_command = [
            sys.executable,
            str(ROMAN_TOOL / "run_jacscanomaly_planet_signal.py"),
            "--stage",
            "scan",
            "--indices",
            index_arg,
            "--jobs",
            str(max(1, int(args.jobs))),
            "--result-tag",
            tag,
            "--event-timeout",
            str(max(1, int(args.event_timeout))),
            "--data-dir",
            str(scan_data),
            "--figure-dir",
            str(scan_figures),
            "--fit-bin-points",
            str(max(1, int(args.fit_bin_points))),
        ]
        if args.reuse:
            scan_command.append("--skip-existing")
        run(scan_command, env=env)

        run(
            [
                sys.executable,
                str(ROMAN_TOOL / "run_planet_effect_router.py"),
                "--jobs",
                str(max(1, int(args.jobs))),
                "--indices",
                index_arg,
                "--result",
                str(route_result),
                "--errors",
                str(route_result.with_suffix(".errors.log")),
                "--scan-data-dir",
                str(scan_data),
            ],
            env=env,
        )
        route_payload = json.loads(route_result.read_text(encoding="utf-8"))
        if int(route_payload.get("n_errors", 0)):
            parser.error(
                "physical-effect routing failed for "
                f"{int(route_payload['n_errors'])} event(s); see "
                f"{route_result.with_suffix('.errors.log')}"
            )

        if route_indices.exists() and route_indices.read_text().strip():
            physical_command = [
                sys.executable,
                str(ROMAN_TOOL / "run_jacscanomaly_planet_signal.py"),
                "--stage",
                "physical",
                "--indices-file",
                str(route_indices),
                "--jobs",
                str(max(1, int(args.jobs))),
                "--result-tag",
                tag,
                "--event-timeout",
                str(max(1, int(args.event_timeout))),
                "--effect-route-result",
                str(route_result),
                "--scan-data-dir",
                str(scan_data),
                "--data-dir",
                str(physical_data),
                "--figure-dir",
                str(physical_figures),
            ]
            if args.reuse:
                physical_command.append("--skip-existing")
            run(physical_command, env=env)

        post_command = [
            sys.executable,
            str(ROMAN_TOOL / "run_post_physical_planet_refinement.py"),
            "--physical-data-dir",
            str(physical_data),
            "--data-dir",
            str(post_data),
            "--figure-dir",
            str(post_figures),
            "--jobs",
            str(max(1, int(args.jobs))),
            "--max-refits",
            "3",
            "--event-timeout",
            str(max(1, int(args.event_timeout))),
        ]
        if args.reuse:
            post_command.append("--skip-existing")
        run(post_command, env=env)

        final_command = [
            sys.executable,
            str(ROMAN_TOOL / "run_final_residual_measurement.py"),
            "--scan-data-dir",
            str(scan_data),
            "--physical-data-dir",
            str(post_data),
            "--data-dir",
            str(final_data),
            "--figure-dir",
            str(final_figures),
            "--indices",
            index_arg,
            "--jobs",
            str(max(1, int(args.jobs))),
            "--event-timeout",
            str(max(1, int(args.event_timeout))),
        ]
        if args.reuse:
            final_command.append("--skip-existing")
        run(final_command, env=env)

    missing = [
        name
        for name, index in zip(names, indices)
        if not tuple(final_data.glob(f"planet_signal_result_{index}_*.json"))
    ]
    if missing:
        parser.error(
            "trial output is incomplete for: " + ", ".join(missing)
        )

    portal_env = {
        **env,
        "ROMAN_ADOPTED_MODEL_SITE": "1",
        "ROMAN_FINAL_PIPELINE_TAG": tag,
        "ROMAN_PS_DATA_DIR": str(final_data),
        "ROMAN_PHYSICAL_DATA_DIR": str(final_data),
        "ROMAN_HTML_DIR": str(site_dir),
        "ROMAN_EVENT_FILTER": event_arg,
        "ROMAN_SPARSE_SITE": "1",
        "ROMAN_SITE_VARIANT_NAME": str(args.variant_name),
        "ROMAN_SITE_VARIANT_PEER_DIR": str(args.current_site_dir_name),
        "ROMAN_SITE_VARIANT_PEER_LABEL": "Current mask",
        # This focused trial is driven entirely by the trial's adopted-model
        # JSON. Do not mix in global legacy scan statistics or figures.
        "ROMAN_HIDE_LEGACY_SECTIONS": "1",
        "ROMAN_HIDE_TEMPLATE_FREE_STATS": "1",
        "SKIP_PORTAL_SYNC": "1",
    }
    # The trial site is generated output. Recreate this validated, dedicated
    # directory so removed sections cannot leave stale multi-run assets behind.
    if site_dir.exists():
        shutil.rmtree(site_dir)
    run([sys.executable, str(ROMAN_TOOL / "make_html.py")], env=portal_env)

    if args.publish:
        run([str(SYNC_REQUEST)], env=env)
    print(f"trial site: {site_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
