"""Controlled domain-knowledge profiles for hybrid selection."""

from __future__ import annotations

from dataclasses import dataclass

from krispu.kernels.registry import registered_kernel_ids


@dataclass(frozen=True)
class KernelProfile:
    name: str
    allowed_kernel_ids: tuple[str, ...]
    default_kernel_id: str
    description: str


_ALL = registered_kernel_ids()
PROFILE_REGISTRY: dict[str, KernelProfile] = {
    "smooth_global": KernelProfile(
        "smooth_global",
        ("rbf_ard", "matern_52_ard", "matern_32_ard"),
        "matern_52_ard",
        "Globally smooth fields with no explicit multiscale or periodic structure.",
    ),
    "rough_single_scale": KernelProfile(
        "rough_single_scale",
        ("matern_32_ard", "matern_12_ard", "rational_quadratic"),
        "matern_32_ard",
        "Rough fields dominated by one spatial scale.",
    ),
    "rough_multiscale": KernelProfile(
        "rough_multiscale",
        (
            "matern_52_long_plus_matern_12_short",
            "matern_32_long_plus_matern_12_short",
            "matern_52_long_plus_matern_32_short",
            "rational_quadratic",
        ),
        "matern_32_long_plus_matern_12_short",
        "A broad component plus a distinct short correlated component.",
    ),
    "trend_plus_local": KernelProfile(
        "trend_plus_local",
        ("linear_plus_matern_32", "linear_plus_matern_52", "matern_52_long_plus_matern_12_short"),
        "linear_plus_matern_32",
        "A global trend with local residual structure.",
    ),
    "periodic": KernelProfile(
        "periodic",
        ("periodic_times_matern_32", "periodic_plus_matern_32", "matern_32_ard"),
        "periodic_times_matern_32",
        "Approximately periodic fields with local deviations.",
    ),
    "unrestricted_standard": KernelProfile(
        "unrestricted_standard",
        _ALL,
        "matern_32_ard",
        "All registered initial candidates.",
    ),
    "broad_standard": KernelProfile(
        "broad_standard",
        _ALL,
        "matern_32_ard",
        "A broad but finite standard candidate profile.",
    ),
}


def get_profile(name: str) -> KernelProfile:
    try:
        return PROFILE_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown kernel profile: {name}") from exc
