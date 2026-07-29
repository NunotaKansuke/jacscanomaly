"""Canonical native parallax contract and SciPy orchestration.

The module deliberately contains no JAX imports.  JAX remains available to the
legacy grid/planet code, but a native parallax evaluator can be imported and
used in a process where JAX is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np

try:  # optional at import time; the error is raised when a parallax fit is requested
    from . import _parallax_cpp
except ImportError:  # pragma: no cover - source-only installations
    _parallax_cpp = None

try:
    from scipy.optimize import least_squares
except ImportError:  # pragma: no cover - optional dependency
    least_squares = None


@dataclass(frozen=True)
class TimeSpec:
    """Explicit time scale and offset used by a native parallax fit."""

    scale: Literal["jd", "hjd"] = "jd"
    offset: float = 0.0

    def __post_init__(self) -> None:
        if self.scale not in {"jd", "hjd"}:
            raise ValueError("TimeSpec.scale must be 'jd' or 'hjd'.")
        if not np.isfinite(float(self.offset)):
            raise ValueError("TimeSpec.offset must be finite.")

    def normalize(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1 or not np.all(np.isfinite(array)):
            raise ValueError("time must be a finite one-dimensional array.")
        return array + float(self.offset)


@dataclass(frozen=True)
class NativeParallaxDiagnostics:
    optimizer_success: bool
    optimizer_status: str
    nfev: int
    njev: int
    chi2: float
    rank: int
    jacobian_condition: float
    parameter_at_bound: bool
    ephemeris_extrapolated: bool
    nonfinite_evaluations: int
    observer_convention: str
    backend: str = "native_cpp_vbm_magnification_scipy_trf"


@dataclass(frozen=True)
class Ephemeris:
    """Validated Cartesian ephemeris in ICRF/J2000, AU and AU/day."""

    time: np.ndarray
    position_au: np.ndarray
    velocity_au_per_day: Optional[np.ndarray] = None
    frame: Literal["icrf_j2000"] = "icrf_j2000"
    origin: Literal[
        "solar_system_barycenter", "sun", "earth", "explicit_reference"
    ] = "explicit_reference"
    time_spec: TimeSpec = TimeSpec()
    extrapolation: Literal["reject", "linear"] = "reject"

    def __post_init__(self) -> None:
        time = np.asarray(self.time, dtype=np.float64)
        position = np.asarray(self.position_au, dtype=np.float64)
        velocity = None if self.velocity_au_per_day is None else np.asarray(self.velocity_au_per_day, dtype=np.float64)
        if self.frame != "icrf_j2000":
            raise ValueError("Only frame='icrf_j2000' is supported by the native backend.")
        if self.origin not in {"solar_system_barycenter", "sun", "earth", "explicit_reference"}:
            raise ValueError("Ephemeris.origin is not supported.")
        if self.extrapolation not in {"reject", "linear"}:
            raise ValueError("extrapolation must be 'reject' or 'linear'.")
        if time.ndim != 1 or time.size < 2:
            raise ValueError("ephemeris time must contain at least two rows.")
        if position.shape != (time.size, 3):
            raise ValueError("position_au must have shape (n, 3).")
        if velocity is not None and velocity.shape != (time.size, 3):
            raise ValueError("velocity_au_per_day must have shape (n, 3).")
        if not np.all(np.isfinite(time)) or not np.all(np.isfinite(position)) or (velocity is not None and not np.all(np.isfinite(velocity))):
            raise ValueError("ephemeris arrays must be finite.")
        if np.any(np.diff(time) <= 0.0):
            raise ValueError("ephemeris time must be strictly increasing; duplicate rows are not allowed.")
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "position_au", position)
        object.__setattr__(self, "velocity_au_per_day", velocity)

    def cpp_tuple(self) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        return self.time, self.position_au, self.velocity_au_per_day

    def contains(self, values: Sequence[float] | np.ndarray) -> bool:
        values = np.asarray(values, dtype=float)
        return bool(np.all((values >= self.time[0]) & (values <= self.time[-1])))

    def velocity_at(self, value: float) -> np.ndarray:
        """Return an explicitly local derivative at ``value``."""
        t = float(value)
        if self.velocity_au_per_day is not None:
            return np.column_stack([
                np.interp(t, self.time, self.velocity_au_per_day[:, axis])
                for axis in range(3)
            ])[0]
        if t <= self.time[0]:
            i, j = 0, 1
        elif t >= self.time[-1]:
            i, j = self.time.size - 2, self.time.size - 1
        else:
            j = int(np.searchsorted(self.time, t, side="right"))
            i = max(0, j - 1)
            if i > 0 and j < self.time.size - 1:
                return (self.position_au[j + 1] - self.position_au[i - 1]) / (self.time[j + 1] - self.time[i - 1])
        return (self.position_au[j] - self.position_au[i]) / (self.time[j] - self.time[i])


def _cartesian_ephemeris(
    time: Sequence[float], position: Sequence[Sequence[float]], *, velocity=None,
    origin: str = "explicit_reference", time_spec: TimeSpec = TimeSpec(),
) -> Ephemeris:
    order = np.argsort(np.asarray(time, dtype=float))
    t = np.asarray(time, dtype=float)[order]
    p = np.asarray(position, dtype=float)[order]
    v = None if velocity is None else np.asarray(velocity, dtype=float)[order]
    return Ephemeris(t, p, v, origin=origin, time_spec=time_spec)


def parse_vbm_satellite_table(table: Sequence[Sequence[float]], *, time_spec: TimeSpec = TimeSpec()) -> Ephemeris:
    """Convert ``JD RA_deg Dec_deg distance_AU`` to an Earth-relative ephemeris."""
    table = np.asarray(table, dtype=float)
    if table.ndim != 2 or table.shape[1] < 4:
        raise ValueError("satellite table must have [JD, RA_deg, Dec_deg, distance_AU].")
    order = np.argsort(table[:, 0])
    table = table[order]
    ra, dec, distance = np.deg2rad(table[:, 1]), np.deg2rad(table[:, 2]), table[:, 3]
    position = np.column_stack([
        distance * np.cos(dec) * np.cos(ra),
        distance * np.cos(dec) * np.sin(ra),
        distance * np.sin(dec),
    ])
    return Ephemeris(table[:, 0], position, origin="earth", time_spec=time_spec)


def load_vbm_satellite_ephemeris(
    path: str | Path,
    *,
    time_spec: TimeSpec = TimeSpec(),
    extrapolation: Literal["reject", "linear"] = "reject",
) -> Ephemeris:
    """Read an RTModel/VBM ``JD RA Dec distance`` satellite table."""
    rows: list[list[float]] = []
    in_horizons_block = False
    saw_marker = False
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if text.startswith("$$SOE"):
                saw_marker = True
                in_horizons_block = True
                continue
            if text.startswith("$$EOE"):
                break
            if saw_marker and not in_horizons_block:
                continue
            parts = text.replace(",", " ").split()
            if len(parts) < 4 or text.startswith("#"):
                continue
            try:
                rows.append([float(parts[index]) for index in range(4)])
            except ValueError:
                continue
    if len(rows) < 2:
        raise ValueError(f"No VBM satellite ephemeris rows found in {path}.")
    ephemeris = parse_vbm_satellite_table(rows, time_spec=time_spec)
    return Ephemeris(
        ephemeris.time,
        ephemeris.position_au,
        ephemeris.velocity_au_per_day,
        origin="earth",
        time_spec=time_spec,
        extrapolation=extrapolation,
    )


def load_cartesian_ephemeris(path: str | Path, *, origin: str, time_spec: TimeSpec = TimeSpec()) -> Ephemeris:
    """Read a plain ``time x y z [vx vy vz]`` Cartesian file."""
    rows: list[list[float]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.replace(",", " ").split()
            if not parts or parts[0].startswith("#"):
                continue
            try:
                values = [float(value) for value in parts]
            except ValueError:
                continue
            if len(values) >= 4:
                rows.append(values[:7])
    if len(rows) < 2:
        raise ValueError(f"No Cartesian ephemeris rows found in {path}.")
    rows_np = np.asarray(rows, dtype=float)
    order = np.argsort(rows_np[:, 0])
    rows_np = rows_np[order]
    velocity = rows_np[:, 4:7] if rows_np.shape[1] >= 7 else None
    return Ephemeris(rows_np[:, 0], rows_np[:, 1:4], velocity, origin=origin, time_spec=time_spec)


def _require_native() -> object:
    if _parallax_cpp is None:
        raise ImportError(
            "Native parallax backend is unavailable. Install jacscanomaly with the "
            "VBMicrolensing development sources (jacscanomaly[vbm]) and rebuild."
        )
    return _parallax_cpp


def _validate_origins(mode: str, earth: Optional[Ephemeris], observer: Optional[Ephemeris], reference: Optional[Ephemeris]) -> None:
    if mode == "earth_geocentric_offset":
        if earth is None or earth.origin not in {"sun", "solar_system_barycenter"}:
            raise ValueError("earth_geocentric_offset requires an Earth ephemeris with origin='sun' or 'solar_system_barycenter'.")
        if observer is not None and observer.origin != "earth":
            raise ValueError("satellite_or_observer_ephemeris must have origin='earth' in earth_geocentric_offset mode.")
    else:
        if observer is None or reference is None:
            raise ValueError(f"{mode} requires both observer and reference ephemerides; observer=self is not implicit.")
        if observer.origin not in {"sun", "solar_system_barycenter", "explicit_reference"} or reference.origin not in {"sun", "solar_system_barycenter", "explicit_reference"}:
            raise ValueError(f"{mode} requires complete heliocentric/barycentric observer and reference ephemerides.")


class ParallaxEvaluator:
    """Python-owned native evaluator with validated immutable ephemerides."""

    def __init__(
        self, time, flux, ferr, dataset_id=None, *, ra_deg: float, dec_deg: float, tref: float,
        time_spec: Optional[TimeSpec] = None, observer_convention: str = "earth_geocentric_offset",
        earth_ephemeris: Optional[Ephemeris] = None,
        satellite_or_observer_ephemeris: Optional[Ephemeris] = None,
        reference_ephemeris: Optional[Ephemeris] = None,
        finite_source: bool = False, espl_table_path: Optional[str] = None,
        vbm_tol: float = 1.0e-4, vbm_reltol: float = 1.0e-4,
    ) -> None:
        if time_spec is None:
            raise ValueError("ParallaxEvaluator requires an explicit TimeSpec; time origin is not inferred.")
        mode = {"vbm": "earth_geocentric_offset"}.get(observer_convention, observer_convention)
        if mode not in {"earth_geocentric_offset", "heliocentric_observer", "gulls"}:
            raise ValueError("observer_convention is invalid.")
        _validate_origins(mode, earth_ephemeris, satellite_or_observer_ephemeris, reference_ephemeris)
        for label, eph in (("earth", earth_ephemeris), ("observer", satellite_or_observer_ephemeris), ("reference", reference_ephemeris)):
            if eph is not None and eph.time_spec != time_spec:
                raise ValueError(f"{label} ephemeris TimeSpec must match the evaluator TimeSpec.")
        normalized_time = time_spec.normalize(time)
        normalized_tref = float(tref) + float(time_spec.offset)
        for label, eph in (("earth", earth_ephemeris), ("observer", satellite_or_observer_ephemeris), ("reference", reference_ephemeris)):
            if eph is not None and not eph.contains(normalized_time) and eph.extrapolation == "reject":
                raise ValueError(f"{label} ephemeris does not cover the requested data time range.")
            if eph is not None and not eph.contains([normalized_tref]) and eph.extrapolation == "reject":
                raise ValueError(f"{label} ephemeris does not cover tref.")
        dataset = np.zeros(len(normalized_time), dtype=np.int64) if dataset_id is None else np.asarray(dataset_id, dtype=np.int64)
        if dataset.shape != normalized_time.shape:
            raise ValueError("dataset_id must match time shape.")
        cpp = _require_native()
        if finite_source and espl_table_path is None:
            espl_table_path = default_espl_table_path()
        self.time_spec = time_spec
        self.observer_convention = mode
        self.ra_deg = float(ra_deg)
        self.dec_deg = float(dec_deg)
        self.tref = float(tref)
        self.finite_source = bool(finite_source)
        self.espl_table_path = espl_table_path
        self.vbm_tol = float(vbm_tol)
        self.vbm_reltol = float(vbm_reltol)
        self.earth_ephemeris = earth_ephemeris
        self.satellite_or_observer_ephemeris = satellite_or_observer_ephemeris
        self.reference_ephemeris = reference_ephemeris
        self._native = cpp.ParallaxEvaluator(
            normalized_time, np.asarray(flux, dtype=float), np.asarray(ferr, dtype=float), dataset,
            float(ra_deg), float(dec_deg), normalized_tref, time_spec.scale, mode,
            earth_ephemeris.cpp_tuple() if earth_ephemeris is not None else None,
            satellite_or_observer_ephemeris.cpp_tuple() if satellite_or_observer_ephemeris is not None else None,
            reference_ephemeris.cpp_tuple() if reference_ephemeris is not None else None,
            bool(finite_source), espl_table_path, float(vbm_tol), float(vbm_reltol),
            bool(any(eph is not None and eph.extrapolation == "linear" for eph in (earth_ephemeris, satellite_or_observer_ephemeris, reference_ephemeris))),
        )

    def trajectory(self, raw_params, *, components: bool = False):
        return self._native.trajectory(np.asarray(raw_params, dtype=float), components=components)

    def magnification(self, raw_params):
        return np.asarray(self._native.magnification(np.asarray(raw_params, dtype=float)))

    def magnification_at(self, time, raw_params):
        """Evaluate the same native trajectory/model at arbitrary user times."""
        values = np.asarray(time, dtype=float).reshape(-1)
        clone = ParallaxEvaluator(
            values,
            np.ones(values.size, dtype=float),
            np.ones(values.size, dtype=float),
            ra_deg=self.ra_deg,
            dec_deg=self.dec_deg,
            tref=self.tref,
            time_spec=self.time_spec,
            observer_convention=self.observer_convention,
            earth_ephemeris=self.earth_ephemeris,
            satellite_or_observer_ephemeris=self.satellite_or_observer_ephemeris,
            reference_ephemeris=self.reference_ephemeris,
            finite_source=self.finite_source,
            espl_table_path=self.espl_table_path,
            vbm_tol=self.vbm_tol,
            vbm_reltol=self.vbm_reltol,
        )
        return clone.magnification(raw_params)

    def evaluate(self, raw_params, active_mask=None):
        return np.asarray(self._native.evaluate(np.asarray(raw_params, dtype=float), active_mask=active_mask))

    def residual(self, raw_params, active_mask=None):
        return np.asarray(self._native.residual(np.asarray(raw_params, dtype=float), active_mask=active_mask))

    def jacobian(self, raw_params, active_mask=None, fd_step=1.0e-5):
        return np.asarray(self._native.jacobian(np.asarray(raw_params, dtype=float), active_mask=active_mask, fd_step=float(fd_step)))

    def residual_and_jacobian(self, raw_params, active_mask=None, fd_step=1.0e-5):
        residual, jacobian = self._native.residual_and_jacobian(np.asarray(raw_params, dtype=float), active_mask=active_mask, fd_step=float(fd_step))
        return np.asarray(residual), np.asarray(jacobian)


class NativeParallaxFitter:
    """Common bounded SciPy TRF fitter for PSPL and FSPL parallax models."""

    def __init__(self, *, ra_deg, dec_deg, tref, finite_source=False, time_spec=TimeSpec(), observer_convention="earth_geocentric_offset", earth_ephemeris=None, satellite_or_observer_ephemeris=None, reference_ephemeris=None, maxiter=1000, tol=1e-6, max_piE=1.0, espl_table_path=None, vbm_tol=1e-4, vbm_reltol=1e-4):
        self.ra_deg = float(ra_deg); self.dec_deg = float(dec_deg); self.tref = float(tref)
        self.finite_source = bool(finite_source); self.time_spec = time_spec; self.observer_convention = observer_convention
        self.earth_ephemeris = earth_ephemeris if earth_ephemeris is not None else default_earth_ephemeris(time_spec=time_spec)
        self.satellite_or_observer_ephemeris = satellite_or_observer_ephemeris
        self.reference_ephemeris = reference_ephemeris
        self.maxiter = int(maxiter); self.tol = float(tol); self.max_piE = float(max_piE)
        self.espl_table_path = espl_table_path; self.vbm_tol = float(vbm_tol); self.vbm_reltol = float(vbm_reltol)
        self._last_fit = None

    @property
    def parameter_dimension(self) -> int:
        return 6 if self.finite_source else 5

    def _raw_seed(self, value: np.ndarray) -> np.ndarray:
        """Translate the fallback seed contract to the evaluator contract.

        Fallback/public seeds always use physical ``tE`` and, for finite
        sources, ``log_rho``.  Optimizer continuation must therefore go
        through ``seed_from_fit`` instead of passing ``raw_params`` back into
        this method.  Keeping one unambiguous boundary avoids accidentally
        applying log(log(tE)) during robust alternating refits.
        """
        value = np.asarray(value, dtype=float).reshape(-1)
        if value.size != self.parameter_dimension:
            if value.size == 3:
                value = np.r_[value, np.log(0.01), 0.0, 0.0] if self.finite_source else np.r_[value, 0.0, 0.0]
            elif value.size == 4 and not self.finite_source:
                value = np.r_[value[:3], 0.0, 0.0]
            elif value.size == 5 and self.finite_source:
                value = np.r_[value[:3], np.log(0.01), value[3:]]
            else:
                raise ValueError(f"expected a {self.parameter_dimension}-parameter seed")
        raw = value.copy()
        raw[0] += float(self.time_spec.offset)
        if not np.isfinite(raw[1]) or raw[1] <= 0.0:
            raise ValueError("native parallax seeds require physical tE > 0")
        raw[1] = np.log(raw[1])
        if self.finite_source and not np.isfinite(raw[3]):
            raise ValueError("native FSPL parallax seeds require finite log_rho")
        return raw

    @staticmethod
    def seed_from_fit(fit) -> np.ndarray:
        """Return the canonical continuation seed from a completed fit."""
        params = np.asarray(fit.params, dtype=float).reshape(-1).copy()
        names = tuple(getattr(fit, "param_names", ()))
        if "rho" in names:
            rho_index = names.index("rho")
            params[rho_index] = np.log(max(abs(float(params[rho_index])), 1.0e-12))
        return params

    def _bounds(self, time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full(self.parameter_dimension, -np.inf, dtype=float)
        upper = np.full(self.parameter_dimension, np.inf, dtype=float)
        margin = max(np.ptp(time) * 0.25, 1.0)
        lower[0] = float(np.min(time)) - margin + float(self.time_spec.offset)
        upper[0] = float(np.max(time)) + margin + float(self.time_spec.offset)
        lower[1] = np.log(max(np.ptp(time) / 1000.0, 1e-6)); upper[1] = np.log(max(np.ptp(time) * 1000.0, 1.0))
        lower[2] = -10.0; upper[2] = 10.0
        if self.finite_source:
            lower[3] = np.log(1e-8); upper[3] = np.log(1e3); pi_index = 4
        else: pi_index = 3
        lower[pi_index:pi_index + 2] = -abs(self.max_piE); upper[pi_index:pi_index + 2] = abs(self.max_piE)
        return lower, upper

    def fit(self, time, flux, ferr, q0):
        if least_squares is None:
            raise ImportError("Native parallax fitting requires scipy.optimize.least_squares.")
        time = np.asarray(time, dtype=float); flux = np.asarray(flux, dtype=float); ferr = np.asarray(ferr, dtype=float)
        if time.ndim != 1 or not (time.shape == flux.shape == ferr.shape) or np.any(ferr <= 0) or not np.all(np.isfinite(time)):
            raise ValueError("time, flux, and ferr must be finite one-dimensional arrays with positive ferr.")
        evaluator = ParallaxEvaluator(
            time, flux, ferr, ra_deg=self.ra_deg, dec_deg=self.dec_deg, tref=self.tref,
            time_spec=self.time_spec, observer_convention=self.observer_convention,
            earth_ephemeris=self.earth_ephemeris, satellite_or_observer_ephemeris=self.satellite_or_observer_ephemeris,
            reference_ephemeris=self.reference_ephemeris, finite_source=self.finite_source,
            espl_table_path=self.espl_table_path, vbm_tol=self.vbm_tol, vbm_reltol=self.vbm_reltol,
        )
        raw0 = self._raw_seed(np.asarray(q0, dtype=float))
        lower, upper = self._bounds(time)
        raw0 = np.minimum(np.maximum(raw0, lower + 1e-10), upper - 1e-10)
        result = least_squares(evaluator.residual, raw0, jac=evaluator.jacobian, method="trf", bounds=(lower, upper), x_scale="jac", loss="linear", max_nfev=self.maxiter, xtol=self.tol, ftol=self.tol, gtol=self.tol)
        raw = np.asarray(result.x, dtype=float)
        mags = evaluator.magnification(raw)
        model = evaluator.evaluate(raw)
        residual = flux - model
        fs, fb = _profile_fluxes(mags, flux, ferr)
        params = np.r_[raw[0] - float(self.time_spec.offset), np.exp(raw[1]), raw[2]]
        names = ["t0", "tE", "u0"]
        if self.finite_source:
            params = np.r_[params, np.exp(raw[3])]; names.append("rho"); pi_index = 4
        else: pi_index = 3
        params = np.r_[params, raw[pi_index:pi_index + 2]]; names.extend(["piEN", "piEE"])
        from .singlelens_fit import SingleLensFitResult
        final_residual = evaluator.residual(raw)
        chi2 = float(np.sum(final_residual * final_residual))
        try:
            jacobian = evaluator.jacobian(raw)
            singular = np.linalg.svd(jacobian, compute_uv=False)
            rank = int(np.linalg.matrix_rank(jacobian))
            condition = float(singular[0] / singular[-1]) if singular.size and singular[-1] > 0 else float("inf")
        except Exception:
            rank, condition = 0, float("inf")
        at_bound = bool(np.any(np.isclose(raw, lower, rtol=0.0, atol=1e-7)) or np.any(np.isclose(raw, upper, rtol=0.0, atol=1e-7)))
        diagnostics = NativeParallaxDiagnostics(
            optimizer_success=bool(result.success), optimizer_status=str(result.message),
            nfev=int(getattr(result, "nfev", 0)), njev=int(getattr(result, "njev", 0)), chi2=chi2,
            rank=rank, jacobian_condition=condition, parameter_at_bound=at_bound,
            ephemeris_extrapolated=any(eph is not None and not eph.contains(time + self.time_spec.offset) for eph in (self.earth_ephemeris, self.satellite_or_observer_ephemeris, self.reference_ephemeris)),
            nonfinite_evaluations=0, observer_convention=self.observer_convention,
        )
        fit = SingleLensFitResult(time=time, flux=flux, ferr=ferr, params=params, param_names=tuple(names), chi2=chi2, chi2_dof=chi2 / max(time.size - self.parameter_dimension, 1), fs=fs, fb=fb, model_flux=model, residual=residual, raw_params=raw, parallax_projector=evaluator, optimizer_success=bool(result.success), optimizer_status=f"native_cpp_scipy_trf:{result.message}", diagnostics=diagnostics)
        self._last_fit = fit
        return fit

    def evaluate_fixed(self, time, flux, ferr, q0):
        """Evaluate a native parallax model without optimizing its parameters."""
        time = np.asarray(time, dtype=float)
        flux = np.asarray(flux, dtype=float)
        ferr = np.asarray(ferr, dtype=float)
        if time.ndim != 1 or not (time.shape == flux.shape == ferr.shape):
            raise ValueError("time, flux, and ferr must be one-dimensional arrays with equal length")
        if np.any(~np.isfinite(time)) or np.any(~np.isfinite(flux)) or np.any(~np.isfinite(ferr)):
            raise ValueError("time, flux, and ferr must be finite")
        if np.any(ferr <= 0.0):
            raise ValueError("ferr must be positive")
        evaluator = ParallaxEvaluator(
            time, flux, ferr, ra_deg=self.ra_deg, dec_deg=self.dec_deg, tref=self.tref,
            time_spec=self.time_spec, observer_convention=self.observer_convention,
            earth_ephemeris=self.earth_ephemeris,
            satellite_or_observer_ephemeris=self.satellite_or_observer_ephemeris,
            reference_ephemeris=self.reference_ephemeris, finite_source=self.finite_source,
            espl_table_path=self.espl_table_path, vbm_tol=self.vbm_tol, vbm_reltol=self.vbm_reltol,
        )
        raw = self._raw_seed(np.asarray(q0, dtype=float))
        magnification = evaluator.magnification(raw)
        fs, fb = _profile_fluxes(magnification, flux, ferr)
        model = fs * magnification + fb
        residual = flux - model
        normalized = residual / ferr
        chi2 = float(np.dot(normalized, normalized))
        params = np.r_[raw[0] - float(self.time_spec.offset), np.exp(raw[1]), raw[2]]
        names = ["t0", "tE", "u0"]
        if self.finite_source:
            params = np.r_[params, np.exp(raw[3])]
            names.append("rho")
            pi_index = 4
        else:
            pi_index = 3
        params = np.r_[params, raw[pi_index:pi_index + 2]]
        names.extend(["piEN", "piEE"])
        from .singlelens_fit import SingleLensFitResult
        return SingleLensFitResult(
            time=time, flux=flux, ferr=ferr, params=params, param_names=tuple(names),
            chi2=chi2, chi2_dof=chi2 / max(time.size - self.parameter_dimension, 1),
            fs=fs, fb=fb, model_flux=model, residual=residual, raw_params=raw,
            parallax_projector=evaluator, optimizer_success=True,
            optimizer_status="native_cpp_fixed_evaluation",
        )


def native_parallax_effect_score(
    fit,
    *,
    exclude_mask: Optional[np.ndarray] = None,
) -> float:
    """Score parallax structure remaining in a native-fit residual.

    The native evaluator Jacobian already includes the profiled flux solve.
    We project the two parallax columns against the local non-parallax
    parameter columns, then report the linearized chi-square improvement.
    This is the same acceptance concept as the detector score but does not
    re-enter the JAX trajectory implementation.
    """
    evaluator = getattr(fit, "parallax_projector", None)
    raw = getattr(fit, "raw_params", None)
    if not isinstance(evaluator, ParallaxEvaluator) or raw is None:
        raise ValueError("native parallax effect scoring requires a native fit")
    raw = np.asarray(raw, dtype=float).reshape(-1)
    residual = np.asarray(evaluator.residual(raw), dtype=float).reshape(-1)
    jacobian = np.asarray(evaluator.jacobian(raw), dtype=float)
    if jacobian.shape != (residual.size, raw.size) or raw.size < 5:
        raise ValueError("native parallax residual/Jacobian shapes are invalid")
    keep = np.isfinite(residual) & np.all(np.isfinite(jacobian), axis=1)
    if exclude_mask is not None:
        excluded = np.asarray(exclude_mask, dtype=bool).reshape(-1)
        if excluded.size != residual.size:
            raise ValueError("exclude_mask must match native fit length")
        keep &= ~excluded
    if np.count_nonzero(keep) <= raw.size + 2:
        return 0.0
    z = residual[keep]
    j = jacobian[keep]
    effect = j[:, -2:]
    nuisance = j[:, :-2]
    if nuisance.size:
        q, _ = np.linalg.qr(nuisance, mode="reduced")
        z = z - q @ (q.T @ z)
        effect = effect - q @ (q.T @ effect)
    coefficients, *_ = np.linalg.lstsq(effect, -z, rcond=None)
    improved = z + effect @ coefficients
    score = float(np.dot(z, z) - np.dot(improved, improved))
    return max(score, 0.0) if np.isfinite(score) else 0.0


class NativePSPLAnnualParallaxFitter(NativeParallaxFitter):
    def __init__(self, ra_deg, dec_deg, tref, **kwargs):
        super().__init__(ra_deg=ra_deg, dec_deg=dec_deg, tref=tref, finite_source=False, **kwargs)


class NativeFSPLAnnualParallaxFitter(NativeParallaxFitter):
    def __init__(self, ra_deg, dec_deg, tref, **kwargs):
        super().__init__(ra_deg=ra_deg, dec_deg=dec_deg, tref=tref, finite_source=True, **kwargs)


class NativePSPLSpaceParallaxFitter(NativeParallaxFitter):
    def __init__(self, ra_deg, dec_deg, tref, satellite_or_observer_ephemeris, *, convention="earth_geocentric_offset", **kwargs):
        if convention == "earth_geocentric_offset":
            kwargs.setdefault("earth_ephemeris", default_earth_ephemeris(time_spec=kwargs.get("time_spec", TimeSpec())))
        super().__init__(ra_deg=ra_deg, dec_deg=dec_deg, tref=tref, finite_source=False, satellite_or_observer_ephemeris=satellite_or_observer_ephemeris, observer_convention=convention, **kwargs)


class NativeFSPLSpaceParallaxFitter(NativeParallaxFitter):
    def __init__(self, ra_deg, dec_deg, tref, satellite_or_observer_ephemeris, *, convention="earth_geocentric_offset", **kwargs):
        if convention == "earth_geocentric_offset":
            kwargs.setdefault("earth_ephemeris", default_earth_ephemeris(time_spec=kwargs.get("time_spec", TimeSpec())))
        super().__init__(ra_deg=ra_deg, dec_deg=dec_deg, tref=tref, finite_source=True, satellite_or_observer_ephemeris=satellite_or_observer_ephemeris, observer_convention=convention, **kwargs)


def _profile_fluxes(magnification: np.ndarray, flux: np.ndarray, ferr: np.ndarray) -> tuple[float, float]:
    w = 1.0 / np.maximum(np.asarray(ferr, dtype=float), 1e-12) ** 2
    xm = np.sum(w * magnification) / np.sum(w); ym = np.sum(w * flux) / np.sum(w)
    denom = np.sum(w * (magnification - xm) ** 2)
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError("profiled flux solve is singular")
    fs = float(np.sum(w * (magnification - xm) * (flux - ym)) / denom)
    return fs, float(ym - fs * xm)


def default_earth_ephemeris(*, time_spec: TimeSpec = TimeSpec()) -> Ephemeris:
    from importlib import resources
    from .parallax import load_horizons_vectors_file
    path = resources.files("jacscanomaly.data").joinpath("earth_orbital_parallax_table.txt")
    table = load_horizons_vectors_file(str(path))
    return Ephemeris(table[:, 0], table[:, 1:4], table[:, 4:7], origin="sun", time_spec=time_spec)


def default_espl_table_path() -> Optional[str]:
    try:
        import VBMicrolensing  # type: ignore
    except ImportError:
        return None
    path = Path(VBMicrolensing.__file__).resolve().parent / "data" / "ESPL.tbl"
    return str(path) if path.is_file() else None


__all__ = [
    "TimeSpec", "Ephemeris", "NativeParallaxDiagnostics", "ParallaxEvaluator", "NativePSPLAnnualParallaxFitter",
    "NativeFSPLAnnualParallaxFitter", "NativePSPLSpaceParallaxFitter", "NativeFSPLSpaceParallaxFitter",
    "parse_vbm_satellite_table", "load_vbm_satellite_ephemeris", "load_cartesian_ephemeris", "default_earth_ephemeris", "default_espl_table_path",
    "native_parallax_effect_score",
]
