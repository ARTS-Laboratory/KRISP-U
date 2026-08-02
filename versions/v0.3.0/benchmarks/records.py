"""Scalar and spatial benchmark record serialization."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

RECORD_FIELDS = [
    "field",
    "trial",
    "method",
    "sample_count",
    "rmse",
    "nrmse",
    "mae",
    "nmae",
    "r2",
    "p95_absolute_error",
    "max_absolute_error",
    "loo_calibration_factor",
    "mean_posterior_std",
    "mean_loo_field_uncertainty",
    "mean_calibrated_posterior_std",
    "max_posterior_std",
    "max_loo_field_uncertainty",
    "max_calibrated_posterior_std",
    "recommended_x",
    "recommended_y",
    "distance_to_nearest_observation",
    "distance_to_domain_boundary",
    "near_domain_boundary",
    "on_current_sample_hull",
    "dominant_loo_observation_index",
    "dominant_loo_observation_x",
    "dominant_loo_observation_y",
    "dominant_observation_is_anchor",
    "dominant_observation_near_boundary",
    "wall_time_seconds",
]


def write_records(path: Path, records: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def save_spatial_state(path: Path, state: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "true_field": state.true_field,
        "predicted_field": state.predicted_field,
        "absolute_error": state.metrics.absolute_error,
        "squared_error": state.metrics.squared_error,
        "observed_X": state.observed_X,
        "evaluation_points": state.evaluation_points,
    }
    for name in (
        "posterior_std",
        "loo_field_uncertainty",
        "loo_field_means",
        "loo_residuals",
        "loo_standardized_residuals",
        "calibrated_posterior_std",
    ):
        value = getattr(state, name)
        arrays[name] = np.asarray(value) if value is not None else np.array([], dtype=float)
    np.savez_compressed(path, **arrays)
