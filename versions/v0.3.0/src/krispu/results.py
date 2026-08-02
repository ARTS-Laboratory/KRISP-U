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
    loo_field_uncertainty: NDArray[np.float64]
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

    def __post_init__(self) -> None:
        n = len(self.reference_points)
        for name in (
            "predicted_mean",
            "posterior_std",
            "loo_mean",
            "loo_field_uncertainty",
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

    def to_dict(self) -> dict[str, Any]:
        """Return all diagnostics as JSON-friendly lists and scalars."""

        return {
            "reference_points": self.reference_points.tolist(),
            "predicted_mean": self.predicted_mean.tolist(),
            "posterior_std": self.posterior_std.tolist(),
            "loo_mean": self.loo_mean.tolist(),
            "loo_field_uncertainty": self.loo_field_uncertainty.tolist(),
            "loo_calibration_factor": self.loo_calibration_factor,
            "calibrated_posterior_std": self.calibrated_posterior_std.tolist(),
            "combined_std": self.combined_std.tolist(),
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
        """Compatibility alias for the candidate-level LOO field spread."""

        return self.loo_field_uncertainty

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
    loo_field_uncertainty: float
    calibrated_posterior_std: float
    combined_std: float
    distance_to_nearest_observation: float

    @property
    def jackknife_std(self) -> float:
        """Compatibility alias for the LOO field uncertainty."""

        return self.loo_field_uncertainty


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
                "loo_field_uncertainty": item.loo_field_uncertainty,
                "calibrated_posterior_std": item.calibrated_posterior_std,
                "combined_std": item.combined_std,
                "distance_to_nearest_observation": item.distance_to_nearest_observation,
            }
            for name, value in zip(self.feature_names, item.x, strict=True):
                record[name] = float(value)
            records.append(record)
        return records
