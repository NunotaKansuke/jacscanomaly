# scanomaly/__init__.py
from __future__ import annotations

from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

from .config import FinderConfig
from .finder import Finder
from .plot import AnomalyPlotter
from .singlelens import PSPLFitter, FSPLFitter, PSPLParallaxFitter, PSPLFitResult

__all__ = [
    "FinderConfig",
    "Finder",
    "AnomalyPlotter",
    "PSPLFitter",
    "FSPLFitter",
    "PSPLFitResult",
    "PSPLParallaxFitter"
]

__version__ = "0.1.1"
