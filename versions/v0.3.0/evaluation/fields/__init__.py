"""Field registry for development, recovery, and reconstruction studies."""

from dataclasses import replace

import numpy as np

from evaluation.fields.benchmarks import (
    borehole_field,
    branin_hoo_field,
    franke_field,
    goldstein_price_field,
    hartmann_3d_field,
    hartmann_6d_field,
    otl_circuit_field,
    piston_simulation_field,
    six_hump_camel_field,
)
from evaluation.fields.canonical.anisotropic import anisotropic_field
from evaluation.fields.canonical.localized import localized_field
from evaluation.fields.canonical.noisy import noisy_field
from evaluation.fields.canonical.smooth import smooth_field
from evaluation.fields.synthetic_gp.kernel_fields import (
    periodic_field,
    rough_correlated_field,
    rough_multiscale_field,
    trend_plus_local_field,
)
from evaluation.fields.synthetic_gp.sampled import (
    axis_rescaled_anisotropic_gp_field,
    exponential_gp_field,
    gaussian_gp_field,
    matern32_gp_field,
    matern52_gp_field,
    rational_quadratic_gp_field,
    rotated_anisotropic_gp_field,
    spherical_gp_field,
    wendland_gp_field,
)
from evaluation.fields.transforms import (
    smooth_axis_rescaled_field,
    smooth_clean_field,
    smooth_noisy_field,
    smooth_output_scaled_field,
    smooth_rotation_field,
    smooth_translation_field,
)


def _rename(factory: object, name: str, **metadata: object) -> object:
    """Return a compact development-field factory with explicit metadata."""

    def build(seed: int = 0) -> object:
        del seed
        field = factory()  # type: ignore[operator]
        return replace(
            field,
            name=name,
            metadata={
                **field.metadata,  # type: ignore[attr-defined]
                "field_category": "development",
                "true_kernel": None,
                **metadata,
            },
        )

    return build


def clustered_observations_field(seed: int = 0) -> object:
    del seed
    return replace(
        localized_field(),
        name="clustered_observations",
        metadata={
            "description": "Localized feature used with nearly duplicate observations",
            "field_category": "development",
            "true_kernel": None,
            "observation_design": "clustered",
        },
    )


def white_noise_field(seed: int = 0) -> object:
    rng = np.random.default_rng(seed)
    base = smooth_field()
    values = rng.normal(0.0, 0.15, 4096)
    lookup = values.reshape(64, 64)

    def evaluate(points: object) -> np.ndarray:
        coordinates = np.asarray(points, dtype=float)
        if coordinates.ndim == 1:
            coordinates = coordinates.reshape(1, -1)
        indices = np.clip(((coordinates + 1.0) * 31.5).round().astype(int), 0, 63)
        return lookup[indices[:, 0], indices[:, 1]]

    base = replace(
        base,
        name="white_noise",
        evaluate=evaluate,
        metadata={
            "description": "Locked-seed white-noise response",
            "field_category": "development",
            "true_kernel": None,
            "seed": seed,
        },
    )
    return base


def baseline_drift_field(seed: int = 0) -> object:
    field = smooth_field()
    original = field.evaluate

    def evaluate(points: object) -> np.ndarray:
        coordinates = np.asarray(points, dtype=float)
        if coordinates.ndim == 1:
            coordinates = coordinates.reshape(1, -1)
        return np.asarray(original(coordinates), dtype=float) + 0.7 * coordinates[:, 0]

    return replace(
        field,
        name="baseline_drift",
        evaluate=evaluate,
        metadata={
            "description": "Deterministic broad baseline drift",
            "field_category": "development",
            "true_kernel": None,
        },
    )


def baseline_plus_noise_field(seed: int = 0) -> object:
    field = baseline_drift_field(seed)
    noisy = white_noise_field(seed + 11)
    original = field.evaluate

    def evaluate(points: object) -> np.ndarray:
        return np.asarray(original(points), dtype=float) + 0.2 * np.asarray(
            noisy.evaluate(points), dtype=float
        )

    return replace(
        field,
        name="baseline_plus_noise",
        evaluate=evaluate,
        metadata={
            "description": "Baseline drift with locked noise",
            "field_category": "development",
            "true_kernel": None,
        },
    )


def heteroscedastic_noise_field(seed: int = 0) -> object:
    field = smooth_field()
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=4096).reshape(64, 64)

    def evaluate(points: object) -> np.ndarray:
        coordinates = np.asarray(points, dtype=float)
        if coordinates.ndim == 1:
            coordinates = coordinates.reshape(1, -1)
        indices = np.clip(((coordinates + 1.0) * 31.5).round().astype(int), 0, 63)
        base = np.asarray(field.evaluate(coordinates), dtype=float)
        level = 0.05 + 0.25 * ((coordinates[:, 0] + 1.0) / 2.0)
        return base + level * noise[indices[:, 0], indices[:, 1]]

    return replace(
        field,
        name="heteroscedastic_noise",
        evaluate=evaluate,
        metadata={
            "description": "Locked-seed spatially varying noise",
            "field_category": "development",
            "true_kernel": None,
        },
    )


FIELD_FACTORIES = {
    "smooth": smooth_field,
    "localized": localized_field,
    "anisotropic": anisotropic_field,
    "noisy_baseline": noisy_field,
    "noisy": noisy_field,
    "rough_correlated": rough_correlated_field,
    "rough_multiscale": rough_multiscale_field,
    "periodic": periodic_field,
    "trend_plus_local": trend_plus_local_field,
    "smooth_single_scale": _rename(smooth_field, "smooth_single_scale"),
    "localized_feature": _rename(localized_field, "localized_feature"),
    "anisotropic_ridge": _rename(anisotropic_field, "anisotropic_ridge"),
    "multiscale_response": rough_multiscale_field,
    "boundary_feature": _rename(localized_field, "boundary_feature", feature_center=[0.8, 0.8]),
    "baseline_drift": baseline_drift_field,
    "white_noise": white_noise_field,
    "baseline_plus_noise": baseline_plus_noise_field,
    "heteroscedastic_noise": heteroscedastic_noise_field,
    "clustered_observations": clustered_observations_field,
    "smooth_translation": smooth_translation_field,
    "smooth_axis_rescaled": smooth_axis_rescaled_field,
    "smooth_rotation": smooth_rotation_field,
    "smooth_output_scaled": smooth_output_scaled_field,
    "smooth_clean": smooth_clean_field,
    "smooth_noisy": smooth_noisy_field,
    "Franke": franke_field,
    "Branin-Hoo": branin_hoo_field,
    "Goldstein-Price": goldstein_price_field,
    "Six-hump camel": six_hump_camel_field,
    "Hartmann 3D": hartmann_3d_field,
    "Hartmann 6D": hartmann_6d_field,
    "Borehole": borehole_field,
    "OTL circuit": otl_circuit_field,
    "Piston simulation": piston_simulation_field,
    "franke": franke_field,
    "branin_hoo": branin_hoo_field,
    "goldstein_price": goldstein_price_field,
    "six_hump_camel": six_hump_camel_field,
    "hartmann_3d": hartmann_3d_field,
    "hartmann_6d": hartmann_6d_field,
    "borehole": borehole_field,
    "otl_circuit": otl_circuit_field,
    "piston_simulation": piston_simulation_field,
    "gp_rotated_anisotropic": rotated_anisotropic_gp_field,
    "gp_axis_rescaled_anisotropic": axis_rescaled_anisotropic_gp_field,
    "gp_gaussian_ard": gaussian_gp_field,
    "gp_exponential_ard": exponential_gp_field,
    "gp_spherical_ard": spherical_gp_field,
    "gp_matern_32_ard": matern32_gp_field,
    "gp_matern_52_ard": matern52_gp_field,
    "gp_rational_quadratic_ard": rational_quadratic_gp_field,
    "gp_wendland_c2_ard": wendland_gp_field,
}

__all__ = [
    "FIELD_FACTORIES",
    "anisotropic_field",
    "localized_field",
    "noisy_field",
    "periodic_field",
    "rough_correlated_field",
    "rough_multiscale_field",
    "smooth_field",
    "trend_plus_local_field",
]
