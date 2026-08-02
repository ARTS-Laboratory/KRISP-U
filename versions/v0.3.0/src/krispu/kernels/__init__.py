"""Explicit kernel construction, registry, scoring, and selection."""

from krispu.kernels.builders import build_kernel_from_spec
from krispu.kernels.profiles import PROFILE_REGISTRY, get_profile
from krispu.kernels.registry import (
    KERNEL_REGISTRY,
    KernelDefinition,
    candidate_ids,
    get_kernel_definition,
    registered_kernel_ids,
)
from krispu.kernels.scoring import (
    CandidateScore,
    score_candidate_set,
    spatial_block_folds,
)
from krispu.kernels.selection import (
    KernelSelectionResult,
    KernelSelector,
    select_kernel,
)
from krispu.kernels.specification import (
    KernelSelectionConfig,
    parse_kernel_configuration,
)

__all__ = [
    "KERNEL_REGISTRY",
    "PROFILE_REGISTRY",
    "CandidateScore",
    "KernelDefinition",
    "KernelSelectionConfig",
    "KernelSelectionResult",
    "KernelSelector",
    "build_kernel_from_spec",
    "candidate_ids",
    "get_kernel_definition",
    "get_profile",
    "parse_kernel_configuration",
    "registered_kernel_ids",
    "score_candidate_set",
    "select_kernel",
    "spatial_block_folds",
]
