# scanomaly/__init__.py
from __future__ import annotations

from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

from .config import FinderConfig
from .criteria import CandidateCriteria
from .finder import Finder
from .models import AnomalyResult, BestCandidate, CandidateQuality, SeasonSummary
from .plot import AnomalyPlotter
from .singlelens_fit import (
    SingleLensFitResult,
    PSPLFitter,
    CPPPSPLFitter,
    FSPLFitter,
    VBMFiniteDiffFSPLFitter,
    PSPLParallaxFitter,
    FSPLParallaxFitter,
    PSPLSpaceParallaxFitter,
    FSPLSpaceParallaxFitter,
    VBMFiniteDiffGullsFSPLSpaceParallaxFitter,
    BICSingleLensFitter,
    CVFitter,
)
from .template_free import (
    TemplateFreeCandidate,
    TemplateFreeScanner,
    TemplateFreeSearchConfig,
    TemplateFreeSearchResult,
)
from .planet_signal import (
    FlatBaselineDiagnostic,
    PlanetSignalClassification,
    PlanetSignalClassificationConfig,
    PlanetSignalClassifier,
    PlanetSignalComponentClassification,
    PlanetSignalCandidate,
    PlanetSignalConfig,
    PlanetSignalExtractor,
    PlanetSignalIteration,
    PlanetSignalPeak,
    PlanetSignalResult,
)
from .planet_class import (
    AtomFitResult,
    LocalPhysicalFitResult,
    PlanetAnomalyClassifier,
    PlanetAnomalyFitResult,
    PlanetClassConfig,
    PSPLParams,
    SegmentData,
    SegmentModelResult,
)

__all__ = [
    "FinderConfig",
    "CandidateCriteria",
    "Finder",
    "AnomalyResult",
    "BestCandidate",
    "CandidateQuality",
    "SeasonSummary",
    "AnomalyPlotter",
    "PSPLFitter",
    "CPPPSPLFitter",
    "FSPLFitter",
    "VBMFiniteDiffFSPLFitter",
    "PSPLParallaxFitter",
    "FSPLParallaxFitter",
    "PSPLSpaceParallaxFitter",
    "FSPLSpaceParallaxFitter",
    "VBMFiniteDiffGullsFSPLSpaceParallaxFitter",
    "BICSingleLensFitter",
    "CVFitter",
    "SingleLensFitResult",
    "TemplateFreeCandidate",
    "TemplateFreeScanner",
    "TemplateFreeSearchConfig",
    "TemplateFreeSearchResult",
    "FlatBaselineDiagnostic",
    "PlanetSignalClassification",
    "PlanetSignalClassificationConfig",
    "PlanetSignalClassifier",
    "PlanetSignalComponentClassification",
    "PlanetSignalCandidate",
    "PlanetSignalConfig",
    "PlanetSignalExtractor",
    "PlanetSignalIteration",
    "PlanetSignalPeak",
    "PlanetSignalResult",
    "AtomFitResult",
    "LocalPhysicalFitResult",
    "PlanetAnomalyClassifier",
    "PlanetAnomalyFitResult",
    "PlanetClassConfig",
    "PSPLParams",
    "SegmentData",
    "SegmentModelResult",
]

__version__ = "0.3.2"
