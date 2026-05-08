# scanomaly/__init__.py
from __future__ import annotations

from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

from .config import FinderConfig
from .finder import Finder
from .models import AnomalyResult, BestCandidate, CandidateQuality, SeasonSummary
from .plot import AnomalyPlotter
from .singlelens_fit import (
    SingleLensFitResult,
    PSPLFitter,
    FSPLFitter,
    PSPLParallaxFitter,
    FSPLParallaxFitter,
)
from .template_free import (
    TemplateFreeCandidate,
    TemplateFreeScanner,
    TemplateFreeSearchConfig,
    TemplateFreeSearchResult,
)

__all__ = [
    "FinderConfig",
    "Finder",
    "AnomalyResult",
    "BestCandidate",
    "CandidateQuality",
    "SeasonSummary",
    "AnomalyPlotter",
    "PSPLFitter",
    "FSPLFitter",
    "PSPLParallaxFitter",
    "FSPLParallaxFitter",
    "SingleLensFitResult",
    "TemplateFreeCandidate",
    "TemplateFreeScanner",
    "TemplateFreeSearchConfig",
    "TemplateFreeSearchResult",
]

__version__ = "0.1.1"
