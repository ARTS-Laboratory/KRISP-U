"""Small user-facing workflows for KRISP-U field reconstruction."""

from __future__ import annotations

from numpy.typing import ArrayLike

from krispu.config import GPRConfig
from krispu.domains import CandidateDomain
from krispu.observations import ObservationSet
from krispu.recommender import KrispURecommender
from krispu.results import RecommendationResult, UncertaintyDiagnostics
from krispu.surrogates import GPRSurrogate


def fit_reconstruction(
    domain: CandidateDomain,
    observations: ObservationSet,
    *,
    gpr_config: GPRConfig | None = None,
) -> GPRSurrogate:
    """Fit and return a normalized-coordinate Gaussian-process reconstruction."""

    if not isinstance(observations, ObservationSet):
        raise TypeError("observations must be an ObservationSet.")
    domain.validate_points(observations.X, "observations.X")
    return GPRSurrogate(gpr_config or GPRConfig()).fit(
        domain.normalize(observations.X),
        observations.y,
        observations.observation_variances,
    )


def evaluate_uncertainty(
    domain: CandidateDomain,
    observations: ObservationSet,
    reference_points: ArrayLike,
    *,
    uncertainty: str = "support_adjusted_krispu",
    gpr_config: GPRConfig | None = None,
) -> UncertaintyDiagnostics:
    """Evaluate reconstruction uncertainty at supplied reference locations."""

    recommender = KrispURecommender(
        domain,
        uncertainty=uncertainty,
        gpr_config=gpr_config,
    )
    return recommender.evaluate_uncertainty(observations, reference_points)


def recommend_next_point(
    domain: CandidateDomain,
    observations: ObservationSet,
    *,
    candidates: ArrayLike | None = None,
    uncertainty: str = "support_adjusted_krispu",
    gpr_config: GPRConfig | None = None,
) -> RecommendationResult:
    """Rank candidate measurements and return the best available point."""

    recommender = KrispURecommender(
        domain,
        uncertainty=uncertainty,
        gpr_config=gpr_config,
    )
    return recommender.recommend(observations, candidates=candidates)


__all__ = ["evaluate_uncertainty", "fit_reconstruction", "recommend_next_point"]
