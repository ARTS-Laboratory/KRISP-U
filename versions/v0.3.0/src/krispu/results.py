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
    jackknife_std: NDArray[np.float64]
    loo_calibration_factor: float
    calibrated_posterior_std: NDArray[np.float64]
    combined_std: NDArray[np.float64]
    loo_field_means: NDArray[np.float64]
    loo_field_stds: NDArray[np.float64]
    loo_residuals: NDArray[np.float64]
    loo_standardized_residuals: NDArray[np.float64]
    loo_eligible_indices: NDArray[np.int_]
    heldout_predicted_mean: NDArray[np.float64]
    heldout_predicted_std: NDArray[np.float64]

    def __post_init__(self) -> None:
        n = len(self.reference_points)
        for name in (
            "predicted_mean",
            "posterior_std",
            "loo_mean",
            "jackknife_std",
            "calibrated_posterior_std",
            "combined_std",
        ):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{name} must have one value per reference point.")
        if self.loo_field_means.shape != (n, len(self.loo_eligible_indices)):
            raise ValueError("loo_field_means has an invalid shape.")

    def to_dict(self) -> dict[str, Any]:
        """Return all diagnostics as JSON-friendly lists and scalars."""

        return {
            "reference_points": self.reference_points.tolist(),
            "predicted_mean": self.predicted_mean.tolist(),
            "posterior_std": self.posterior_std.tolist(),
            "loo_mean": self.loo_mean.tolist(),
            "jackknife_std": self.jackknife_std.tolist(),
            "loo_calibration_factor": self.loo_calibration_factor,
            "calibrated_posterior_std": self.calibrated_posterior_std.tolist(),
            "combined_std": self.combined_std.tolist(),
            "loo_field_means": self.loo_field_means.tolist(),
            "loo_field_stds": self.loo_field_stds.tolist(),
            "loo_residuals": self.loo_residuals.tolist(),
            "loo_standardized_residuals": self.loo_standardized_residuals.tolist(),
            "loo_eligible_indices": self.loo_eligible_indices.tolist(),
            "heldout_predicted_mean": self.heldout_predicted_mean.tolist(),
            "heldout_predicted_std": self.heldout_predicted_std.tolist(),
        }


@dataclass(frozen=True)
class Recommendation:
    """One field-reconstruction measurement recommendation."""

    rank: int
    x: NDArray[np.float64]
    acquisition_score: float
    predicted_mean: float
    posterior_std: float
    jackknife_std: float
    calibrated_posterior_std: float
    combined_std: float
    distance_to_nearest_observation: float


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
                "jackknife_std": item.jackknife_std,
                "calibrated_posterior_std": item.calibrated_posterior_std,
                "combined_std": item.combined_std,
                "distance_to_nearest_observation": item.distance_to_nearest_observation,
            }
            for name, value in zip(self.feature_names, item.x, strict=True):
                record[name] = float(value)
            records.append(record)
        return records
