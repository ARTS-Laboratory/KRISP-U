"""Inspectable result objects and JSON/record serialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class UncertaintyDiagnostics:
    """All uncertainty quantities at the supplied reference locations."""

    reference_points: NDArray[np.float64]
    predicted_mean: NDArray[np.float64]
    posterior_std: NDArray[np.float64]
    loo_mean: NDArray[np.float64]
    loo_field_sensitivity: NDArray[np.float64]
    kernel_support_deficit: NDArray[np.float64]
    krispu_uncertainty: NDArray[np.float64]
    maximum_kernel_correlation_to_observations: NDArray[np.float64]
    loo_calibration_factor: float
    calibrated_posterior_std: NDArray[np.float64]
    combined_std: NDArray[np.float64]
    loo_field_means: NDArray[np.float64]
    loo_field_stds: NDArray[np.float64]
    loo_residuals: NDArray[np.float64]
    loo_standardized_residuals: NDArray[np.float64]
    loo_eligible_indices: NDArray[np.int_]
    dominant_loo_observation_indices: NDArray[np.int_]
    dominant_loo_observation_coordinates: NDArray[np.float64]
    heldout_predicted_mean: NDArray[np.float64]
    heldout_predicted_std: NDArray[np.float64]
    # Deprecated storage fields retained for dataclasses.replace() and old
    # clients.  New serialization and calculations use the fields above.
    loo_field_uncertainty: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        n = len(self.reference_points)
        for name in (
            "predicted_mean",
            "posterior_std",
            "loo_mean",
            "loo_field_sensitivity",
            "kernel_support_deficit",
            "krispu_uncertainty",
            "maximum_kernel_correlation_to_observations",
            "calibrated_posterior_std",
            "combined_std",
        ):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{name} must have one value per reference point.")
        if self.loo_field_means.shape != (n, len(self.loo_eligible_indices)):
            raise ValueError("loo_field_means has an invalid shape.")
        if self.dominant_loo_observation_indices.shape != (n,):
            raise ValueError("dominant_loo_observation_indices has an invalid shape.")
        if self.dominant_loo_observation_coordinates.shape[0] != n:
            raise ValueError("dominant_loo_observation_coordinates has an invalid shape.")
        if self.loo_field_uncertainty is None:
            object.__setattr__(self, "loo_field_uncertainty", self.loo_field_sensitivity)
        elif self.loo_field_uncertainty.shape != (n,):
            raise ValueError("loo_field_uncertainty must have one value per reference point.")

    def to_dict(self) -> dict[str, Any]:
        """Return all diagnostics as JSON-friendly lists and scalars."""

        return {
            "reference_points": self.reference_points.tolist(),
            "predicted_mean": self.predicted_mean.tolist(),
            "posterior_std": self.posterior_std.tolist(),
            "loo_mean": self.loo_mean.tolist(),
            "loo_field_sensitivity": self.loo_field_sensitivity.tolist(),
            "kernel_support_deficit": self.kernel_support_deficit.tolist(),
            "krispu_uncertainty": self.krispu_uncertainty.tolist(),
            "maximum_kernel_correlation_to_observations": (
                self.maximum_kernel_correlation_to_observations.tolist()
            ),
            "loo_field_means": self.loo_field_means.tolist(),
            "loo_field_stds": self.loo_field_stds.tolist(),
            "loo_residuals": self.loo_residuals.tolist(),
            "loo_standardized_residuals": self.loo_standardized_residuals.tolist(),
            "loo_eligible_indices": self.loo_eligible_indices.tolist(),
            "dominant_loo_observation_indices": self.dominant_loo_observation_indices.tolist(),
            "dominant_loo_observation_coordinates": self.dominant_loo_observation_coordinates.tolist(),
            "heldout_predicted_mean": self.heldout_predicted_mean.tolist(),
            "heldout_predicted_std": self.heldout_predicted_std.tolist(),
        }

    @property
    def jackknife_std(self) -> NDArray[np.float64]:
        """Compatibility alias for LOO field sensitivity."""

        return self.loo_field_sensitivity

    @property
    def loo_field_means_mean(self) -> NDArray[np.float64]:
        """Compatibility alias for the mean of the LOO fields."""

        return self.loo_mean


@dataclass(frozen=True)
class Recommendation:
    """One field-reconstruction measurement recommendation."""

    rank: int
    x: NDArray[np.float64]
    acquisition_score: float
    predicted_mean: float
    posterior_std: float
    loo_field_sensitivity: float
    kernel_support_deficit: float
    krispu_uncertainty: float
    nearest_normalized_distance: float
    maximum_kernel_correlation_to_observations: float

    @property
    def jackknife_std(self) -> float:
        """Compatibility alias for LOO field sensitivity."""

        return self.loo_field_sensitivity

    @property
    def loo_field_uncertainty(self) -> float:
        """Deprecated compatibility alias for LOO field sensitivity."""

        return self.loo_field_sensitivity

    @property
    def calibrated_posterior_std(self) -> float:
        """Deprecated compatibility value; posterior std is not canonical."""

        return self.posterior_std

    @property
    def combined_std(self) -> float:
        """Deprecated compatibility alias for KRISP-U uncertainty."""

        return self.krispu_uncertainty

    @property
    def distance_to_nearest_observation(self) -> float:
        """Deprecated compatibility alias for normalized distance."""

        return self.nearest_normalized_distance


@dataclass(frozen=True)
class RecommendationResult:
    """Recommendations plus the exact diagnostics used to rank them."""

    recommendations: list[Recommendation]
    diagnostics: UncertaintyDiagnostics
    feature_names: tuple[str, ...]
    observed_X: NDArray[np.float64]
    observed_y: NDArray[np.float64]
    uncertainty: str

    def as_array(self) -> NDArray[np.float64]:
        if not self.recommendations:
            return np.empty((0, len(self.feature_names)))
        return np.vstack([item.x for item in self.recommendations])

    def to_records(self) -> list[dict[str, float | int]]:
        records: list[dict[str, float | int]] = []
        for item in self.recommendations:
            record: dict[str, float | int] = {
                "rank": item.rank,
                "acquisition_score": item.acquisition_score,
                "predicted_mean": item.predicted_mean,
                "posterior_std": item.posterior_std,
                "loo_field_sensitivity_at_selection": item.loo_field_sensitivity,
                "kernel_support_deficit_at_selection": item.kernel_support_deficit,
                "krispu_uncertainty_at_selection": item.krispu_uncertainty,
                "nearest_normalized_distance": item.nearest_normalized_distance,
                "maximum_kernel_correlation_to_observations": (
                    item.maximum_kernel_correlation_to_observations
                ),
            }
            for name, value in zip(self.feature_names, item.x, strict=True):
                record[name] = float(value)
            records.append(record)
        return records
