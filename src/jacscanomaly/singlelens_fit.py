"""Shared result container for the canonical single-lens fitters.

The concrete fitters live in :mod:`jacscanomaly.fitters`. Keeping the result
type in this small module avoids coupling downstream pipeline code to a
particular numerical backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import jax.numpy as jnp


@dataclass(frozen=True)
class SingleLensFitResult:
    """Result of a PSPL, FSPL, or parallax single-lens fit."""

    time: np.ndarray
    flux: np.ndarray
    ferr: np.ndarray

    params: jnp.ndarray
    param_names: Tuple[str, ...]
    chi2: jnp.ndarray
    chi2_dof: jnp.ndarray
    fs: jnp.ndarray
    fb: jnp.ndarray
    model_flux: jnp.ndarray
    residual: jnp.ndarray

    # Optional optimizer coordinates, e.g. log(tE) and log(rho).
    raw_params: Optional[jnp.ndarray] = None
    # For parallax fits this is the C++ trajectory/magnification evaluator.
    parallax_projector: Optional[Any] = None
    # Optional evaluator used for plotting/re-evaluating non-parallax models.
    model_evaluator: Optional[Any] = None
    # Canonical model family, e.g. ``"pspl"`` or ``"fspl_parallax"``.
    model_kind: Optional[str] = None
    optimizer_success: Optional[bool] = None
    optimizer_status: Optional[str] = None
    diagnostics: Optional[Any] = None


__all__ = ["SingleLensFitResult"]
