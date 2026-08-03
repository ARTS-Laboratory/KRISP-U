"""Inspectable result objects and JSON/record serialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class UncertaintyDiagnostics:
    reference_points: NDArray[np.float64]
    predicted_mean: NDArray[np.float64]
    posterior_std: NDArray[np.float64]
    jackknife_mean: NDArray[np.float64]
    jackknife_field_sensitivity: NDArray[np.float64]
    kernel_support_deficit: NDArray[np.float64]
    krispu_uncertainty: NDArray[np.float64]
    maximum_kernel_correlation_to_observations: NDArray[np.float64]
    jackknife_calibration_factor: float
    calibrated_posterior_std: NDArray[np.float64]
    combined_std: NDArray[np.float64]
    jackknife_field_means: NDArray[np.float64]
    jackknife_field_stds: NDArray[np.float64]
    jackknife_residuals: NDArray[np.float64]
    jackknife_standardized_residuals: NDArray[np.float64]
    jackknife_eligible_indices: NDArray[np.int_]
    dominant_jackknife_observation_indices: NDArray[np.int_]
    dominant_jackknife_observation_coordinates: NDArray[np.float64]
    heldout_predicted_mean: NDArray[np.float64]
    heldout_predicted_std: NDArray[np.float64]
    buffered_jackknife_plan: Any | None = None

    def __post_init__(self) -> None:
        n = len(self.reference_points)
        for name in (
            "predicted_mean", "posterior_std", "jackknife_mean", "jackknife_field_sensitivity",
            "kernel_support_deficit", "krispu_uncertainty", "maximum_kernel_correlation_to_observations",
            "calibrated_posterior_std", "combined_std",
        ):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{name} must have one value per reference point.")
        if self.jackknife_field_means.shape != (n, len(self.jackknife_eligible_indices)):
            raise ValueError("jackknife_field_means has an invalid shape.")
        if self.dominant_jackknife_observation_indices.shape != (n,):
            raise ValueError("dominant_jackknife_observation_indices has an invalid shape.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_points": self.reference_points.tolist(),
            "predicted_mean": self.predicted_mean.tolist(),
            "posterior_std": self.posterior_std.tolist(),
            "jackknife_mean": self.jackknife_mean.tolist(),
            "jackknife_field_sensitivity": self.jackknife_field_sensitivity.tolist(),
            "kernel_support_deficit": self.kernel_support_deficit.tolist(),
            "krispu_uncertainty": self.krispu_uncertainty.tolist(),
            "maximum_kernel_correlation_to_observations": self.maximum_kernel_correlation_to_observations.tolist(),
            "jackknife_field_means": self.jackknife_field_means.tolist(),
            "jackknife_field_stds": self.jackknife_field_stds.tolist(),
            "jackknife_residuals": self.jackknife_residuals.tolist(),
            "jackknife_standardized_residuals": self.jackknife_standardized_residuals.tolist(),
            "jackknife_eligible_indices": self.jackknife_eligible_indices.tolist(),
            "dominant_jackknife_observation_indices": self.dominant_jackknife_observation_indices.tolist(),
            "dominant_jackknife_observation_coordinates": self.dominant_jackknife_observation_coordinates.tolist(),
            "heldout_predicted_mean": self.heldout_predicted_mean.tolist(),
            "heldout_predicted_std": self.heldout_predicted_std.tolist(),
        }


@dataclass(frozen=True)
class Recommendation:
    rank: int
    x: NDArray[np.float64]
    acquisition_score: float
    predicted_mean: float
    posterior_std: float
    jackknife_field_sensitivity: float
    kernel_support_deficit: float
    krispu_uncertainty: float
    nearest_normalized_distance: float
    maximum_kernel_correlation_to_observations: float


@dataclass(frozen=True)
class RecommendationResult:
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
                "jackknife_field_sensitivity_at_selection": item.jackknife_field_sensitivity,
                "kernel_support_deficit_at_selection": item.kernel_support_deficit,
                "krispu_uncertainty_at_selection": item.krispu_uncertainty,
                "nearest_normalized_distance": item.nearest_normalized_distance,
                "maximum_kernel_correlation_to_observations": item.maximum_kernel_correlation_to_observations,
            }
            for name, value in zip(self.feature_names, item.x, strict=True):
                record[name] = float(value)
            records.append(record)
        return records
