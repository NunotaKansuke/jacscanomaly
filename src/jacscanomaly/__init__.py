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
    CPPVBMFSPLParallaxFitter,
    PSPLParallaxFitter,
    FSPLParallaxFitter,
    PSPLSpaceParallaxFitter,
    FSPLSpaceParallaxFitter,
    VBMFiniteDiffGullsFSPLSpaceParallaxFitter,
    BICSingleLensFitter,
    CVFitter,
)
from .hmc import (
    FSPLHMCResult,
    FSPLParallaxHMCResult,
    PSPLHMCResult,
    sample_fspl_hmc,
    sample_fspl_parallax_hmc,
    sample_pspl_hmc,
)
from .template_free import (
    TemplateFreeCandidate,
    TemplateFreeScanner,
    TemplateFreeSearchConfig,
    TemplateFreeSearchResult,
)
from .planet_signal import (
    FlatBaselineDiagnostic,
    PlanetFeature,
    PlanetFeatureConfig,
    PlanetFeatureResult,
    PlanetSignalCandidate,
    PlanetSignalConfig,
    PlanetSignalExtractor,
    PlanetSignalIteration,
    PlanetSignalResult,
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
    "CPPVBMFSPLParallaxFitter",
    "PSPLParallaxFitter",
    "FSPLParallaxFitter",
    "PSPLSpaceParallaxFitter",
    "FSPLSpaceParallaxFitter",
    "VBMFiniteDiffGullsFSPLSpaceParallaxFitter",
    "FSPLParallaxHMCResult",
    "sample_fspl_parallax_hmc",
    "FSPLHMCResult",
    "sample_fspl_hmc",
    "PSPLHMCResult",
    "sample_pspl_hmc",
    "BICSingleLensFitter",
    "CVFitter",
    "SingleLensFitResult",
    "TemplateFreeCandidate",
    "TemplateFreeScanner",
    "TemplateFreeSearchConfig",
    "TemplateFreeSearchResult",
    "FlatBaselineDiagnostic",
    "PlanetFeature",
    "PlanetFeatureConfig",
    "PlanetFeatureResult",
    "PlanetSignalCandidate",
    "PlanetSignalConfig",
    "PlanetSignalExtractor",
    "PlanetSignalIteration",
    "PlanetSignalResult",
]

__version__ = "0.4.0"
