"""Locked deterministic robustness transformations for benchmark fields."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

from evaluation.fields.canonical.smooth import AuditField


def transformed_field(
    field: AuditField,
    name: str,
    *,
    translation: tuple[float, float] = (0.0, 0.0),
    axis_scale: tuple[float, float] = (1.0, 1.0),
    rotation_degrees: float = 0.0,
    output_scale: float = 1.0,
    noise_scale: float = 0.0,
    seed: int = 0,
) -> AuditField:
    """Apply a deterministic coordinate/output transformation with locked noise."""

    if any(value <= 0 for value in axis_scale) or output_scale <= 0:
        raise ValueError("axis_scale and output_scale must be positive")
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=65536)
    angle = np.deg2rad(rotation_degrees)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    def evaluate(values: ArrayLike) -> NDArray[np.float64]:
        points = np.asarray(values, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        normalized = field.domain.normalize(points)
        centered = (normalized - 0.5 - np.asarray(translation)) / np.asarray(axis_scale)
        transformed = np.clip(centered @ rotation.T + 0.5, 0.0, 1.0)
        source = field.domain.denormalize(transformed)
        result = np.asarray(field.evaluate(source), dtype=float) * output_scale
        if noise_scale:
            indices = np.arange(len(points)) % len(noise)
            result = result + noise_scale * noise[indices]
        return result

    metadata = {
        **field.metadata,
        "true_kernel": None,
        "robustness_transformation": {
            "translation": list(translation),
            "axis_scale": list(axis_scale),
            "rotation_degrees": rotation_degrees,
            "output_scale": output_scale,
            "noise_scale": noise_scale,
            "seed": seed,
        },
    }
    return replace(field, name=name, evaluate=evaluate, metadata=metadata)


def smooth_translation_field(seed: int = 0) -> AuditField:
    from evaluation.fields.canonical.smooth import smooth_field

    return transformed_field(
        smooth_field(), "smooth_translation", translation=(0.1, -0.08), seed=seed
    )


def smooth_axis_rescaled_field(seed: int = 0) -> AuditField:
    from evaluation.fields.canonical.smooth import smooth_field

    return transformed_field(
        smooth_field(), "smooth_axis_rescaled", axis_scale=(0.65, 1.25), seed=seed
    )


def smooth_rotation_field(seed: int = 0) -> AuditField:
    from evaluation.fields.canonical.smooth import smooth_field

    return transformed_field(smooth_field(), "smooth_rotation", rotation_degrees=30.0, seed=seed)


def smooth_output_scaled_field(seed: int = 0) -> AuditField:
    from evaluation.fields.canonical.smooth import smooth_field

    return transformed_field(smooth_field(), "smooth_output_scaled", output_scale=3.0, seed=seed)


def smooth_clean_field(seed: int = 0) -> AuditField:
    from evaluation.fields.canonical.smooth import smooth_field

    return transformed_field(smooth_field(), "smooth_clean", seed=seed)


def smooth_noisy_field(seed: int = 0) -> AuditField:
    from evaluation.fields.canonical.smooth import smooth_field

    return transformed_field(smooth_field(), "smooth_noisy", noise_scale=0.15, seed=seed)


__all__ = [
    "smooth_axis_rescaled_field",
    "smooth_clean_field",
    "smooth_noisy_field",
    "smooth_output_scaled_field",
    "smooth_rotation_field",
    "smooth_translation_field",
    "transformed_field",
]
