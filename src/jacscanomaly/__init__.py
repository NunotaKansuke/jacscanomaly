# scanomaly/__init__.py
from __future__ import annotations

from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

from .config import FinderConfig
from .criteria import CandidateCriteria
from .finder import Finder
from .models import (
    AnomalyResult,
    BestCandidate,
    CandidateQuality,
    ScoredCandidate,
    SeasonSummary,
)
from .plot import AnomalyPlotter
from .singlelens_fit import SingleLensFitResult
from .fitters import (
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
from .candidate import Candidate, candidates_from_scan
from .signal_mask import SignalMask, signal_mask
from .features import (
    CausticDetail,
    Feature,
    FeatureConfig,
    FeatureResult,
    measure_features,
)
from .signal_scale import (
    ObservedSignalScale,
    ResolvedSignalScale,
    measure_observed_magnification_fwhm,
    measure_observed_signal_scale,
    resolve_event_signal_scale,
)
from .pspl_fft import (
    PSPLFFTCandidate,
    PSPLFFTProfile,
    PSPLFFTScanner,
    PSPLFFTSearchResult,
    pspl_excess_magnification,
)
from .effect_detection import (
    EffectCandidate,
    EffectMorphologyConfig,
    EffectMorphologyScores,
    ParallaxSupport,
    ProjectionDiagnostics,
    build_fspl_template_bank,
    compute_effect_morphology_scores,
    detect_fspl_from_pspl_fit,
    detect_parallax_from_pspl_fit,
    detect_physical_effects,
    find_compact_blocks,
    measure_parallax_support,
    parallax_score_test,
    project_out_nuisance,
)
from .fspl_initialization import fspl_template_initial_guesses
from .effect_routing import (
    RoutingThresholds,
    route_candidate,
    route_candidates,
    routing_pareto_curve,
    select_fallback_candidates,
)
from .parallax_backend import (
    TimeSpec,
    Ephemeris,
    ParallaxDiagnostics,
    ParallaxEvaluator,
    parallax_effect_score,
    parallax_fitter_from_config,
)
from .exact_probe import (
    ExactProbeEvaluation,
    ExactProbeResult,
    run_exact_probe,
    run_fspl_exact_probe,
    run_parallax_exact_probe,
)

__all__ = [
    "FinderConfig",
    "CandidateCriteria",
    "Finder",
    "AnomalyResult",
    "BestCandidate",
    "ScoredCandidate",
    "CandidateQuality",
    "SeasonSummary",
    "AnomalyPlotter",
    "PSPLFitter",
    "FSPLFitter",
    "PSPLParallaxFitter",
    "FSPLParallaxFitter",
    "SingleLensFitResult",
    "Candidate",
    "candidates_from_scan",
    "SignalMask",
    "signal_mask",
    "CausticDetail",
    "Feature",
    "FeatureConfig",
    "FeatureResult",
    "measure_features",
    "TemplateFreeCandidate",
    "TemplateFreeScanner",
    "TemplateFreeSearchConfig",
    "TemplateFreeSearchResult",
    "ObservedSignalScale",
    "ResolvedSignalScale",
    "measure_observed_signal_scale",
    "resolve_event_signal_scale",
    "PSPLFFTCandidate",
    "PSPLFFTProfile",
    "PSPLFFTScanner",
    "PSPLFFTSearchResult",
    "pspl_excess_magnification",
    "EffectCandidate",
    "EffectMorphologyConfig",
    "EffectMorphologyScores",
    "ParallaxSupport",
    "ProjectionDiagnostics",
    "build_fspl_template_bank",
    "compute_effect_morphology_scores",
    "detect_fspl_from_pspl_fit",
    "detect_parallax_from_pspl_fit",
    "detect_physical_effects",
    "find_compact_blocks",
    "measure_parallax_support",
    "parallax_score_test",
    "project_out_nuisance",
    "fspl_template_initial_guesses",
    "RoutingThresholds",
    "route_candidate",
    "route_candidates",
    "routing_pareto_curve",
    "select_fallback_candidates",
    "TimeSpec",
    "Ephemeris",
    "ParallaxDiagnostics",
    "ParallaxEvaluator",
    "parallax_effect_score",
    "parallax_fitter_from_config",
    "ExactProbeEvaluation",
    "ExactProbeResult",
    "run_exact_probe",
    "run_fspl_exact_probe",
    "run_parallax_exact_probe",
]

__version__ = "0.5.3"
