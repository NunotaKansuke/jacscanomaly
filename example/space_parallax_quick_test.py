from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import jax.numpy as jnp
import VBMicrolensing

from jacscanomaly.singlelens_model import A_pspl_space_parallax_func
from jacscanomaly.trajectory import make_space_parallax_projector, u_space_parallax_tau_beta


SATELLITE_PATH = Path("/rogue1_8/nunota/sample_rtmodel_v2.4/satellitedir/satellite1.txt")
SATELLITE_DIR = SATELLITE_PATH.parent
COORDINATE_PATH = Path("/rogue1_8/nunota/sample_rtmodel_v2.4/event_2_675_639/Data/event.coordinates")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Save light-curve and trajectory plots.")
    parser.add_argument("--show", action="store_true", help="Show the plot window after saving.")
    parser.add_argument(
        "--output",
        default="example/space_parallax_quick_test.png",
        help="Path for the plot when --plot is used.",
    )
    args = parser.parse_args()

    if not SATELLITE_PATH.exists():
        raise SystemExit(f"Satellite table not found: {SATELLITE_PATH}")
    if not COORDINATE_PATH.exists():
        raise SystemExit(f"Coordinate file not found: {COORDINATE_PATH}")

    ra_deg = 267.623337808
    dec_deg = -29.1164180355
    tref = 9000.0

    space_projector = make_space_parallax_projector(
        ra_deg,
        dec_deg,
        tref,
        str(SATELLITE_PATH),
        use_HJD=True,
    )

    tE = 80.0
    u0 = 0.12
    piN = 0.50
    piE = 0.60
    time = jnp.asarray(np.linspace(tref - 120.0, tref + 120.0, 700))
    params_jac = jnp.asarray([tref, tE, u0, piN, piE])

    amp_jac = A_pspl_space_parallax_func(params_jac, time, space_projector)
    tau_jac, beta_jac = u_space_parallax_tau_beta(time, *params_jac, space_projector)

    vbm = VBMicrolensing.VBMicrolensing()
    sun_table = Path(VBMicrolensing.__file__).parent / "data" / "SunEphemeris.txt"
    vbm.LoadSunTable(str(sun_table))
    vbm.SetObjectCoordinates(str(COORDINATE_PATH), str(SATELLITE_DIR))
    vbm.parallaxsystem = 1
    vbm.t0_par_fixed = 1
    vbm.t0_par = tref
    vbm.t_in_HJD = 1
    vbm.satellite = 1

    params_vbm = [u0, float(np.log(tE)), tref, piN, piE]
    amp_vbm, y1_vbm, y2_vbm = vbm.PSPLLightCurveParallax(params_vbm, np.asarray(time).tolist())
    amp_vbm = np.asarray(amp_vbm, dtype=float)
    tau_vbm = -np.asarray(y1_vbm, dtype=float)
    beta_vbm = -np.asarray(y2_vbm, dtype=float)

    amp_jac_np = np.asarray(amp_jac)
    tau_jac_np = np.asarray(tau_jac)
    beta_jac_np = np.asarray(beta_jac)
    eps = np.finfo(float).eps
    mag_abs = np.abs(amp_jac_np - amp_vbm)
    u_vbm = np.sqrt(tau_vbm * tau_vbm + beta_vbm * beta_vbm)
    traj_abs = np.sqrt((tau_jac_np - tau_vbm) ** 2 + (beta_jac_np - beta_vbm) ** 2)
    traj_rel = traj_abs / np.maximum(u_vbm, eps)

    print("space parallax quick test: jacscanomaly vs VBMicrolensing")
    print("params:", dict(t0=float(tref), tE=tE, u0=u0, piEN=piN, piEE=piE, satellite=1, t_in_HJD=1))
    print("max |jac - vbm| in magnification:", float(np.max(mag_abs)))
    print("max |jac tau - vbm tau|:", float(np.max(np.abs(tau_jac_np - tau_vbm))))
    print("max |jac beta - vbm beta|:", float(np.max(np.abs(beta_jac_np - beta_vbm))))
    print("max normalized trajectory error:", float(np.max(traj_rel)))
    print("ok:", np.isfinite(amp_jac_np).all() and np.isfinite(amp_vbm).all())

    if args.plot:
        import matplotlib.pyplot as plt

        fig, (ax_lc, ax_traj, ax_dev) = plt.subplots(
            1, 3, figsize=(14.0, 4.2), constrained_layout=True
        )
        ax_lc.plot(np.asarray(time), amp_vbm, lw=2, label="VBMicrolensing")
        ax_lc.plot(np.asarray(time), amp_jac_np, "--", lw=2, label="jacscanomaly")
        ax_lc.set_xlabel("time")
        ax_lc.set_ylabel("magnification")
        ax_lc.set_title("PSPL space-parallax magnification")
        ax_lc.legend()
        ax_lc.minorticks_on()

        ax_traj.plot(tau_vbm, beta_vbm, lw=2, label="VBMicrolensing")
        ax_traj.plot(tau_jac_np, beta_jac_np, "--", lw=2, label="jacscanomaly")
        ax_traj.scatter([0.0], [0.0], marker="+", s=80, c="k", label="lens")
        ax_traj.set_xlabel("tau")
        ax_traj.set_ylabel("beta")
        ax_traj.set_title("source trajectory")
        ax_traj.set_aspect("equal", adjustable="box")
        ax_traj.legend()
        ax_traj.minorticks_on()

        ax_dev.plot(np.asarray(time), np.maximum(mag_abs, eps), lw=2, label="|delta A|")
        ax_dev.plot(np.asarray(time), np.maximum(traj_rel, eps), lw=2, label="|delta trajectory| / u_VB")
        ax_dev.set_xlabel("time")
        ax_dev.set_ylabel("error")
        ax_dev.set_yscale("log")
        ax_dev.set_title("deltas")
        ax_dev.legend()
        ax_dev.minorticks_on()

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160)
        print(f"plot saved: {output}")
        if args.show:
            plt.show()
        else:
            plt.close(fig)


if __name__ == "__main__":
    main()
