"""Public single-point KRISP-U recommendation workflow."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from krispu.acquisition.loo_uncertainty import loo_uncertainty_scores
from krispu.acquisition.posterior_std import posterior_std_scores
from krispu.candidates import generate_candidates, nearest_normalized_distance, valid_candidate_mask
from krispu.config import GPRConfig
from krispu.domains import CandidateDomain
from krispu.observations import ObservationSet
from krispu.results import Recommendation, RecommendationResult, UncertaintyDiagnostics
from krispu.surrogates.gpr import GPRSurrogate
from krispu.uncertainty.jackknife import (
    combine_uncertainties,
    jackknife_std,
    loo_calibration_factor,
)
from krispu.uncertainty.loo_bruteforce import compute_bruteforce_loo


class KrispURecommender:
    """Recommend measurements where the reconstructed field is most uncertain.

    The default score is candidate-level LOO field uncertainty. It is not an
    objective optimizer and has no expected-improvement path.
    """

    def __init__(
        self,
        domain: CandidateDomain,
        uncertainty: str = "krispu_loo",
        gpr_config: GPRConfig | None = None,
        random_state: int | np.random.Generator | None = None,
        n_candidates: int = 2048,
        candidate_method: str = "lhs",
        min_normalized_distance: float = 0.05,
        excluded_regions: (
            Iterable[Callable[[NDArray[np.float64]], NDArray[np.bool_]]]
            | Callable[[NDArray[np.float64]], NDArray[np.bool_]]
            | None
        ) = None,
    ) -> None:
        if uncertainty == "loo_uncertainty":
            uncertainty = "krispu_loo"
        if uncertainty not in {"krispu_loo", "posterior_std"}:
            raise ValueError("uncertainty must be 'krispu_loo' or explicit 'posterior_std'.")
        if n_candidates <= 0:
            raise ValueError("n_candidates must be positive.")
        if min_normalized_distance < 0:
            raise ValueError("min_normalized_distance must be non-negative.")
        self.domain = domain
        self.uncertainty = uncertainty
        self.gpr_config = gpr_config or GPRConfig()
        self.random_state = random_state
        self.n_candidates = int(n_candidates)
        self.candidate_method = candidate_method
        self.min_normalized_distance = float(min_normalized_distance)
        self.excluded_regions = excluded_regions
        self.surrogate_: GPRSurrogate | None = None

    def evaluate_uncertainty(
        self,
        observations: ObservationSet,
        reference_points: ArrayLike,
    ) -> UncertaintyDiagnostics:
        """Compute all full-fit and candidate-level LOO quantities."""

        self._validate_observations(observations)
        reference = self.domain.validate_points(reference_points, "reference_points")
        X_normalized = self.domain.normalize(observations.X)
        reference_normalized = self.domain.normalize(reference)
        surrogate = GPRSurrogate(self.gpr_config).fit(
            X_normalized,
            observations.y,
            observations.observation_variances,
        )
        self.surrogate_ = surrogate
        predicted_mean, posterior_std = surrogate.predict(reference_normalized)
        loo = compute_bruteforce_loo(
            surrogate,
            observations,
            reference_normalized,
            X_normalized=X_normalized,
            epsilon=self.gpr_config.response_epsilon,
        )
        loo_mean, loo_field_uncertainty = jackknife_std(loo.field_means)
        dominant_columns = np.argmax(
            (loo.field_means - loo_mean[:, None]) ** 2,
            axis=1,
        )
        dominant_indices = loo.loo_eligible_indices[dominant_columns]
        dominant_coordinates = observations.X[dominant_indices].copy()
        calibration = loo_calibration_factor(loo.standardized_residuals)
        calibrated, combined = combine_uncertainties(
            loo_field_uncertainty, posterior_std, calibration
        )
        if not np.all(np.isfinite(predicted_mean)):
            raise FloatingPointError("predicted means are non-finite.")
        for name, value in (
            ("posterior standard deviations", posterior_std),
            ("LOO field uncertainties", loo_field_uncertainty),
            ("calibrated posterior uncertainties", calibrated),
            ("combined uncertainties", combined),
        ):
            if not np.all(np.isfinite(value)) or np.any(value < 0):
                raise FloatingPointError(f"{name} are non-finite or negative.")
        if not np.all(np.isfinite(loo_mean)):
            raise FloatingPointError("LOO means are non-finite.")
        if not np.isfinite(calibration) or calibration < 0:
            raise FloatingPointError("LOO calibration factor is non-finite or negative.")
        return UncertaintyDiagnostics(
            reference_points=reference.copy(),
            predicted_mean=predicted_mean,
            posterior_std=posterior_std,
            loo_mean=loo_mean,
            loo_field_uncertainty=loo_field_uncertainty,
            loo_calibration_factor=calibration,
            calibrated_posterior_std=calibrated,
            combined_std=combined,
            loo_field_means=loo.field_means,
            loo_field_stds=loo.field_stds,
            loo_residuals=loo.residuals,
            loo_standardized_residuals=loo.standardized_residuals,
            loo_eligible_indices=loo.loo_eligible_indices,
            dominant_loo_observation_indices=dominant_indices,
            dominant_loo_observation_coordinates=dominant_coordinates,
            heldout_predicted_mean=loo.heldout_means,
            heldout_predicted_std=loo.heldout_stds,
        )

    def recommend(
        self,
        observations: ObservationSet,
        n_recommendations: int = 1,
        candidates: ArrayLike | None = None,
        reference_points: ArrayLike | None = None,
    ) -> RecommendationResult:
        """Return the highest-scoring valid candidate(s).

        For this first pass, requests larger than one use an explicitly named
        independent ranking.  Conditional batch fantasies are deferred until
        the single-point scientific core is stable.
        """

        if n_recommendations <= 0:
            raise ValueError("n_recommendations must be positive.")
        self._validate_observations(observations)
        pool = (
            self.domain.validate_points(candidates, "candidates")
            if candidates is not None
            else generate_candidates(
                self.domain, self.n_candidates, self.candidate_method, self.random_state
            )
        )
        mask = valid_candidate_mask(
            self.domain,
            pool,
            observations.X,
            minimum_normalized_distance=self.min_normalized_distance,
            excluded_regions=self.excluded_regions,
        )
        pool = pool[mask]
        if len(pool) == 0:
            raise ValueError("No valid candidates remain after domain and exclusion filtering.")
        references = (
            pool
            if reference_points is None
            else self.domain.validate_points(reference_points, "reference_points")
        )
        diagnostics = self.evaluate_uncertainty(observations, references)
        scores = (
            loo_uncertainty_scores(diagnostics)
            if self.uncertainty == "krispu_loo"
            else posterior_std_scores(diagnostics)
        )
        # Scores must correspond to candidate rows. This explicit check avoids
        # silently ranking a differently ordered reference field.
        if reference_points is not None and (
            len(references) != len(pool) or not np.allclose(references, pool)
        ):
            raise ValueError(
                "reference_points must equal candidates when recommending from a supplied pool."
            )
        n_selected = min(int(n_recommendations), len(pool))
        order = np.argsort(-scores, kind="stable")[:n_selected]
        distances = nearest_normalized_distance(self.domain, pool, observations.X)
        recommendations = [
            Recommendation(
                rank=rank,
                x=pool[index].copy(),
                acquisition_score=float(scores[index]),
                predicted_mean=float(diagnostics.predicted_mean[index]),
                posterior_std=float(diagnostics.posterior_std[index]),
                loo_field_uncertainty=float(diagnostics.loo_field_uncertainty[index]),
                calibrated_posterior_std=float(diagnostics.calibrated_posterior_std[index]),
                combined_std=float(diagnostics.combined_std[index]),
                distance_to_nearest_observation=float(distances[index]),
            )
            for rank, index in enumerate(order, start=1)
        ]
        return RecommendationResult(
            recommendations=recommendations,
            diagnostics=diagnostics,
            feature_names=tuple(
                self.domain.names or tuple(f"x{i + 1}" for i in range(self.domain.dimension))
            ),
            observed_X=observations.X.copy(),
            observed_y=observations.y.copy(),
            uncertainty=self.uncertainty,
        )

    def _validate_observations(self, observations: ObservationSet) -> None:
        if not isinstance(observations, ObservationSet):
            raise TypeError("observations must be an ObservationSet.")
        self.domain.validate_points(observations.X, "observations.X")
