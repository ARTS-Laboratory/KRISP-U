"""Public single-point KRISP-U recommendation workflow."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from krispu.acquisition.krispu_uncertainty import krispu_uncertainty_scores
from krispu.acquisition.posterior_std import posterior_std_scores
from krispu.acquisition.raw_jackknife_sensitivity import raw_jackknife_sensitivity_scores
from krispu.candidates import generate_candidates, nearest_normalized_distance, valid_candidate_mask
from krispu.config import GPRConfig
from krispu.domains import CandidateDomain
from krispu.jackknife import (
    BufferedJackknifePlan,
    build_buffered_jackknife_plan,
    jackknife_calibration_factor,
    jackknife_field_sensitivity,
)
from krispu.observations import ObservationSet
from krispu.results import Recommendation, RecommendationResult, UncertaintyDiagnostics
from krispu.surrogates.gpr import GPRSurrogate
from krispu.uncertainty.buffered_jackknife import compute_buffered_jackknife
from krispu.uncertainty.support import kernel_support_deficit


class KrispURecommender:
    """Recommend measurements where the reconstructed field is most uncertain.

    The default score is support-adjusted KRISP-U uncertainty. It is not an
    objective optimizer and has no expected-improvement path.
    """

    def __init__(
        self,
        domain: CandidateDomain,
        uncertainty: str = "support_adjusted_krispu",
        gpr_config: GPRConfig | None = None,
        random_state: int | np.random.Generator | None = None,
        n_candidates: int = 2048,
        candidate_method: str = "lhs",
        min_normalized_distance: float = 1.0e-4,
        excluded_regions: (
            Iterable[Callable[[NDArray[np.float64]], NDArray[np.bool_]]]
            | Callable[[NDArray[np.float64]], NDArray[np.bool_]]
            | None
        ) = None,
    ) -> None:
        if uncertainty not in {
            "support_adjusted_krispu",
            "raw_jackknife_sensitivity",
            "posterior_std",
        }:
            raise ValueError(
                "uncertainty must be 'support_adjusted_krispu', "
                "'raw_jackknife_sensitivity', or 'posterior_std'."
            )
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
        buffered_jackknife_plan: BufferedJackknifePlan | None = None,
    ) -> UncertaintyDiagnostics:
        """Compute the complete fit and all buffered-jackknife quantities."""

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
        plan_config = self.gpr_config.jackknife
        plan = buffered_jackknife_plan or build_buffered_jackknife_plan(
            X_normalized,
            observations.jackknife_eligible,
            multiplier=plan_config.multiplier,
            minimum_radius=plan_config.minimum_radius,
            maximum_radius=plan_config.maximum_radius,
            minimum_training_points=plan_config.minimum_training_points,
        )
        jackknife = compute_buffered_jackknife(
            surrogate,
            observations,
            reference_normalized,
            plan,
            X_normalized=X_normalized,
            epsilon=self.gpr_config.response_epsilon,
        )
        jackknife_mean, sensitivity = jackknife_field_sensitivity(jackknife.field_means)
        support_deficit, maximum_kernel_correlation = kernel_support_deficit(
            surrogate,
            X_normalized,
            reference_normalized,
            epsilon=self.gpr_config.response_epsilon,
        )
        krispu_uncertainty = sensitivity * np.sqrt(support_deficit)
        dominant_columns = np.argmax(
            (jackknife.field_means - jackknife_mean[:, None]) ** 2,
            axis=1,
        )
        dominant_indices = jackknife.anchor_indices[dominant_columns]
        dominant_coordinates = observations.X[dominant_indices].copy()
        calibration = jackknife_calibration_factor(jackknife.standardized_residuals)
        calibrated = calibration * np.maximum(posterior_std, 0.0)
        combined = krispu_uncertainty.copy()
        if not np.all(np.isfinite(predicted_mean)):
            raise FloatingPointError("predicted means are non-finite.")
        for name, value in (
            ("posterior standard deviations", posterior_std),
            ("jackknife field sensitivities", sensitivity),
            ("kernel support deficits", support_deficit),
            ("KRISP-U uncertainties", krispu_uncertainty),
            ("maximum kernel correlations", maximum_kernel_correlation),
            ("calibrated posterior uncertainties", calibrated),
            ("combined uncertainties", combined),
        ):
            if not np.all(np.isfinite(value)) or np.any(value < 0):
                raise FloatingPointError(f"{name} are non-finite or negative.")
        if not np.all(np.isfinite(jackknife_mean)):
            raise FloatingPointError("jackknife means are non-finite.")
        if not np.isfinite(calibration) or calibration < 0:
            raise FloatingPointError("jackknife calibration factor is non-finite or negative.")
        return UncertaintyDiagnostics(
            reference_points=reference.copy(),
            predicted_mean=predicted_mean,
            posterior_std=posterior_std,
            jackknife_mean=jackknife_mean,
            jackknife_field_sensitivity=sensitivity,
            kernel_support_deficit=support_deficit,
            krispu_uncertainty=krispu_uncertainty,
            maximum_kernel_correlation_to_observations=maximum_kernel_correlation,
            jackknife_calibration_factor=calibration,
            calibrated_posterior_std=calibrated,
            combined_std=combined,
            jackknife_field_means=jackknife.field_means,
            jackknife_field_stds=jackknife.field_stds,
            jackknife_residuals=jackknife.residuals,
            jackknife_standardized_residuals=jackknife.standardized_residuals,
            jackknife_eligible_indices=jackknife.anchor_indices,
            dominant_jackknife_observation_indices=dominant_indices,
            dominant_jackknife_observation_coordinates=dominant_coordinates,
            heldout_predicted_mean=jackknife.heldout_means,
            heldout_predicted_std=jackknife.heldout_stds,
            buffered_jackknife_plan=plan,
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
            krispu_uncertainty_scores(diagnostics)
            if self.uncertainty == "support_adjusted_krispu"
            else (
                raw_jackknife_sensitivity_scores(diagnostics)
                if self.uncertainty == "raw_jackknife_sensitivity"
                else posterior_std_scores(diagnostics)
            )
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
                jackknife_field_sensitivity=float(diagnostics.jackknife_field_sensitivity[index]),
                kernel_support_deficit=float(diagnostics.kernel_support_deficit[index]),
                krispu_uncertainty=float(diagnostics.krispu_uncertainty[index]),
                nearest_normalized_distance=float(distances[index]),
                maximum_kernel_correlation_to_observations=float(
                    diagnostics.maximum_kernel_correlation_to_observations[index]
                ),
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
