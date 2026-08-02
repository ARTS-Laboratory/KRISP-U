"""Validated, non-evaluated configuration for kernel selection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KernelSelectionConfig:
    """Configuration shared by manual, automatic, and hybrid selection."""

    mode: str = "automatic"
    specification: Mapping[str, Any] | None = None
    candidate_set: str = "standard"
    profile: str = "unrestricted_standard"
    selection_metric: str = "spatial_cv_composite"
    optimizer_restarts: int = 0
    reevaluate_every: int = 3
    minimum_score_improvement: float = 0.05
    minimum_points_before_selection: int = 6
    nlpd_weight: float = 0.5
    nrmse_weight: float = 0.4
    calibration_weight: float = 0.1
    random_state: int = 0
    spatial_folds: int = 4
    optimize_hyperparameters: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"manual", "automatic", "hybrid"}:
            raise ValueError("kernel mode must be manual, automatic, or hybrid.")
        if self.selection_metric not in {
            "loo_predictive",
            "spatial_block_cv",
            "spatial_cv_composite",
        }:
            raise ValueError("Unknown kernel selection metric.")
        if self.optimizer_restarts < 0:
            raise ValueError("optimizer_restarts must be non-negative.")
        if self.reevaluate_every <= 0 or self.minimum_points_before_selection < 2:
            raise ValueError("selection intervals and minimum points must be positive.")
        if self.minimum_score_improvement < 0:
            raise ValueError("minimum_score_improvement must be non-negative.")
        weights = (self.nlpd_weight, self.nrmse_weight, self.calibration_weight)
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("composite score weights must be non-negative and non-zero.")
        if self.spatial_folds < 2:
            raise ValueError("spatial_folds must be at least two.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> KernelSelectionConfig:
        """Parse either a complete config or the value of a ``kernel`` key."""

        raw = {} if value is None else dict(value)
        if "kernel" in raw and isinstance(raw["kernel"], Mapping):
            raw = dict(raw["kernel"])
        known = {
            "mode",
            "specification",
            "candidate_set",
            "profile",
            "selection_metric",
            "optimizer_restarts",
            "reevaluate_every",
            "minimum_score_improvement",
            "minimum_points_before_selection",
            "nlpd_weight",
            "nrmse_weight",
            "calibration_weight",
            "random_state",
            "spatial_folds",
            "optimize_hyperparameters",
        }
        unknown = set(raw).difference(known)
        if unknown:
            raise ValueError(f"Unknown kernel configuration keys: {sorted(unknown)}")
        return cls(**{key: raw[key] for key in known if key in raw})


def parse_kernel_configuration(
    config: Mapping[str, Any] | KernelSelectionConfig | None,
) -> KernelSelectionConfig:
    """Return a validated kernel-selection configuration."""

    if isinstance(config, KernelSelectionConfig):
        return config
    return KernelSelectionConfig.from_mapping(config)


def as_float_pair(value: Any, name: str) -> tuple[float, float]:
    """Validate a finite positive lower/upper bound pair."""

    try:
        pair = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain two finite numbers.") from exc
    if len(pair) != 2 or not all(item > 0 for item in pair) or pair[0] >= pair[1]:
        raise ValueError(f"{name} must be an increasing positive pair.")
    return pair


def as_length_scale(value: Any, dimension: int, name: str) -> float | list[float]:
    """Validate a scalar or ARD length-scale initial value."""

    if isinstance(value, (list, tuple)):
        result = [float(item) for item in value]
        if len(result) != dimension or not all(item > 0 for item in result):
            raise ValueError(f"{name} must contain one positive value per dimension.")
        return result
    scalar = float(value)
    if scalar <= 0:
        raise ValueError(f"{name} must be positive.")
    return scalar
