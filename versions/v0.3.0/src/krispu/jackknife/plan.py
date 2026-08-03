"""Deterministic buffered-jackknife fold plans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class BufferedJackknifePlan:
    anchor_indices: NDArray[np.int_]
    removed_indices_by_fold: tuple[NDArray[np.int_], ...]
    effective_radius_by_fold: NDArray[np.float64]
    removed_count_by_fold: NDArray[np.int_]
    training_count_by_fold: NDArray[np.int_]
    global_buffer_radius: float

    def __post_init__(self) -> None:
        anchors = np.asarray(self.anchor_indices, dtype=int).reshape(-1)
        radii = np.asarray(self.effective_radius_by_fold, dtype=float).reshape(-1)
        removed = np.asarray(self.removed_count_by_fold, dtype=int).reshape(-1)
        training = np.asarray(self.training_count_by_fold, dtype=int).reshape(-1)
        count = len(anchors)
        if any(len(values) != count for values in (radii, removed, training)):
            raise ValueError("BufferedJackknifePlan fold arrays must have matching lengths.")
        if len(self.removed_indices_by_fold) != count:
            raise ValueError("BufferedJackknifePlan must contain one removal set per anchor.")
        if not np.isfinite(self.global_buffer_radius) or self.global_buffer_radius < 0:
            raise ValueError("global_buffer_radius must be finite and non-negative.")
        if np.any(radii < 0) or np.any(radii > self.global_buffer_radius + 1e-15):
            raise ValueError("effective fold radii must lie within the global radius.")
        object.__setattr__(self, "anchor_indices", anchors.copy())
        object.__setattr__(self, "effective_radius_by_fold", radii.copy())
        object.__setattr__(self, "removed_count_by_fold", removed.copy())
        object.__setattr__(self, "training_count_by_fold", training.copy())
        object.__setattr__(
            self,
            "removed_indices_by_fold",
            tuple(np.asarray(values, dtype=int).reshape(-1).copy() for values in self.removed_indices_by_fold),
        )

    @classmethod
    def from_normalized_coordinates(
        cls,
        coordinates: ArrayLike,
        eligible: ArrayLike | None = None,
        *,
        multiplier: float = 1.0,
        minimum_radius: float = 0.025,
        maximum_radius: float = 0.20,
        minimum_training_points: int = 3,
    ) -> BufferedJackknifePlan:
        points = np.asarray(coordinates, dtype=float)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 1:
            raise ValueError("coordinates must contain at least two observations.")
        if not np.all(np.isfinite(points)):
            raise ValueError("coordinates must be finite normalized coordinates.")
        if multiplier < 0 or not np.isfinite(multiplier):
            raise ValueError("multiplier must be finite and non-negative.")
        if not (0 < minimum_radius <= maximum_radius):
            raise ValueError("radius bounds must be positive and increasing.")
        if minimum_training_points < 1 or minimum_training_points >= len(points):
            raise ValueError("minimum_training_points must be less than the observation count.")
        mask = np.ones(len(points), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
        if mask.shape != (len(points),) or not np.any(mask):
            raise ValueError("eligible must be a non-empty Boolean mask matching coordinates.")
        differences = points[:, None, :] - points[None, :, :]
        distances = np.linalg.norm(differences, axis=2)
        nonzero = distances[distances > 0]
        nearest = np.min(np.where(distances > 0, distances, np.inf), axis=1)
        if len(nonzero) == 0:
            median_nearest = 0.0
        else:
            median_nearest = float(np.median(nearest[np.isfinite(nearest)]))
        global_radius = float(np.clip(multiplier * median_nearest, minimum_radius, maximum_radius))
        anchors = np.flatnonzero(mask)
        removed_sets: list[NDArray[np.int_]] = []
        effective: list[float] = []
        for anchor in anchors:
            row = distances[anchor]
            initial = np.flatnonzero(row <= global_radius + 1e-15)
            if len(initial) <= len(points) - minimum_training_points:
                selected = initial
                radius = global_radius
            else:
                allowed = len(points) - minimum_training_points
                ordered = np.sort(np.unique(row[row > 0]))
                candidates = np.concatenate(([0.0], ordered[ordered <= global_radius + 1e-15]))
                radius = 0.0
                selected = np.array([anchor], dtype=int)
                for candidate in candidates:
                    candidate_indices = np.flatnonzero(row <= candidate + 1e-15)
                    if len(candidate_indices) <= allowed:
                        radius = float(candidate)
                        selected = candidate_indices
                    else:
                        break
            if anchor not in selected:
                selected = np.sort(np.concatenate((selected, [anchor])))
            removed_sets.append(np.sort(selected))
            effective.append(radius)
        removed_counts = np.asarray([len(values) for values in removed_sets], dtype=int)
        training_counts = len(points) - removed_counts
        return cls(
            anchor_indices=anchors,
            removed_indices_by_fold=tuple(removed_sets),
            effective_radius_by_fold=np.asarray(effective, dtype=float),
            removed_count_by_fold=removed_counts,
            training_count_by_fold=training_counts,
            global_buffer_radius=global_radius,
        )


def build_buffered_jackknife_plan(
    coordinates: ArrayLike,
    eligible: ArrayLike | None = None,
    *,
    multiplier: float = 1.0,
    minimum_radius: float = 0.025,
    maximum_radius: float = 0.20,
    minimum_training_points: int = 3,
) -> BufferedJackknifePlan:
    return BufferedJackknifePlan.from_normalized_coordinates(
        coordinates,
        eligible,
        multiplier=multiplier,
        minimum_radius=minimum_radius,
        maximum_radius=maximum_radius,
        minimum_training_points=minimum_training_points,
    )
