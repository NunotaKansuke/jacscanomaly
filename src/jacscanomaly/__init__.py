# scanomaly/__init__.py
from __future__ import annotations

from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

from .config import FinderConfig
from .finder import Finder
from .plot import AnomalyPlotter
from .singlelens import PSPLFitter, PSPLFitResult

__all__ = [
    "FinderConfig",
    "Finder",
    "AnomalyPlotter",
    "PSPLFitter",
    "PSPLFitResult",
]

__version__ = "0.1.1"
