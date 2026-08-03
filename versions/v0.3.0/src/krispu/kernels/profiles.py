"""Finite profile subsets over the single-family standard registry."""

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
PROFILE_REGISTRY = {
    "smooth_global": KernelProfile(
        "smooth_global", ("gaussian_ard", "matern_52_ard", "matern_32_ard"), "matern_52_ard",
        "Globally smooth single-family fields.",
    ),
    "rough_single_scale": KernelProfile(
        "rough_single_scale", ("exponential_ard", "matern_32_ard", "matern_52_ard"), "matern_32_ard",
        "Rough fields represented by one global scale vector.",
    ),
    "compact_support": KernelProfile(
        "compact_support", ("spherical_ard", "wendland_c2_ard"), "wendland_c2_ard",
        "Compactly supported single-family fields in dimensions one through three.",
    ),
    "unrestricted_standard": KernelProfile("unrestricted_standard", _ALL, "matern_32_ard", "All standard families."),
    "broad_standard": KernelProfile("broad_standard", _ALL, "matern_32_ard", "All standard families."),
}


def get_profile(name: str) -> KernelProfile:
    try:
        return PROFILE_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown kernel profile: {name}") from exc
