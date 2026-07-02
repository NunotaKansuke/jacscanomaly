from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..planet_signal import PlanetSignalComponentClassification


@dataclass(frozen=True)
class PlanetClassConfig:
    """
    Configuration for residual-atom morphology fitting.
    """

    polynomial_order_short: int = 0
    polynomial_order_default: int = 1
    polynomial_order_wide: int = 2
    short_duration_points: int = 8
    wide_duration_tE_fraction: float = 0.2
    min_points_per_segment: int = 5
    min_delta_chi2_for_seed: float = 20.0
    keep_top_atom_fits: int = 6
    keep_top_seeds_per_segment: int = 120
    q_floor: float = 1e-7
    q_ceil: float = 1.0
    q_width_factors: Tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
    s_central_grid: Tuple[float, ...] = (
        0.55,
        0.65,
        0.75,
        0.85,
        0.92,
        1.08,
        1.18,
        1.35,
        1.6,
        2.0,
    )
    alpha_grid_size_central: int = 8
    cusp_tail_powers: Tuple[float, ...] = (1.0, 2.0 / 3.0)
    central_window_factor: float = 3.0
    optimizer_maxiter: int = 300
    optimizer_ftol: float = 1e-8
    enable_positive_bump: bool = True
    enable_negative_dip: bool = True
    enable_central_perturbation: bool = True
    enable_fold_caustic: bool = True
    enable_curved_fold_caustic: bool = True
    enable_cusp_tail: bool = True
    enable_second_pspl: bool = True
    enable_pspl_misfit: bool = True


@dataclass(frozen=True)
class PSPLParams:
    """
    Baseline point-lens parameters in the trajectory frame used by seed rules.
    """

    t0: float
    tE: float
    u0: float
    Fs: float
    Fb: float


@dataclass(frozen=True)
class SegmentData:
    """
    Data slice for one connected anomaly component.
    """

    component: PlanetSignalComponentClassification
    time: np.ndarray
    flux: np.ndarray
    ferr: np.ndarray
    residual: np.ndarray
    model_flux: np.ndarray
    full_indices: np.ndarray
    pspl: PSPLParams


@dataclass(frozen=True)
class AtomFitResult:
    """
    Fit result for one residual-template atom.
    """

    atom_name: str
    class_label: str
    params: dict[str, float]
    param_errors: Optional[dict[str, float]]
    chi2: float
    chi2_baseline: float
    delta_chi2: float
    bic: float
    aic: float
    score: float
    n_data: int
    n_params: int
    success: bool
    warnings: Tuple[str, ...]


@dataclass(frozen=True)
class SeedCandidate:
    """
    Initial-value candidate for a downstream physical model.
    """

    model_type: str
    class_label: str
    params: dict[str, float]
    score: float
    source_atom: str
    degeneracy_tag: Optional[str]
    warnings: Tuple[str, ...]


@dataclass(frozen=True)
class SegmentModelResult:
    """
    Ranked atom fits and derived seeds for one anomaly component.
    """

    component: PlanetSignalComponentClassification
    features: dict[str, float]
    atom_fits: Tuple[AtomFitResult, ...]
    best_fit: Optional[AtomFitResult]
    class_probabilities: dict[str, float]
    seeds: Tuple[SeedCandidate, ...]
    warnings: Tuple[str, ...]


@dataclass(frozen=True)
class PlanetAnomalyFitResult:
    """
    Event-level morphology-classification result.
    """

    pspl: PSPLParams
    segment_results: Tuple[SegmentModelResult, ...]
    event_seeds: Tuple[SeedCandidate, ...]
    best_label: str
    best_atom: Optional[AtomFitResult]
    class_probabilities: dict[str, float]
    warnings: Tuple[str, ...]
