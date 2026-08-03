"""Configuration contracts for always-ARD kernel optimization and reselection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class KernelOptimizationConfig:
    every_step: bool = True
    restarts: int = 2


@dataclass(frozen=True)
class KernelReselectionConfig:
    minimum_points: int = 6
    maximum_interval: int = 5
    score_degradation_fraction: float = 0.10
    bound_proximity_fraction: float = 0.05
    bound_contact_steps: int = 2
    minimum_switch_improvement: float = 0.05


@dataclass(frozen=True)
class KernelSelectionConfig:
    mode: str = "automatic"
    candidate_set: str = "standard"
    profile: str | None = None
    specification: Mapping[str, Any] | None = None
    optimization: KernelOptimizationConfig = KernelOptimizationConfig()
    reselection: KernelReselectionConfig = KernelReselectionConfig()
    random_state: int = 0
    optimize_hyperparameters: bool = True

    @property
    def optimizer_restarts(self) -> int:
        return self.optimization.restarts

    @property
    def minimum_points_before_selection(self) -> int:
        return self.reselection.minimum_points

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> KernelSelectionConfig:
        raw = {} if value is None else dict(value)
        if "kernel" in raw and isinstance(raw["kernel"], Mapping):
            raw = dict(raw["kernel"])
        known = {
            "enabled", "mode", "candidate_set", "profile", "specification", "optimization", "reselection",
            "random_state", "optimize_hyperparameters", "optimizer_restarts",
            "minimum_points_before_selection", "minimum_score_improvement", "reevaluate_every",
            "selection_metric", "spatial_folds", "nlpd_weight", "nrmse_weight", "calibration_weight",
        }
        unknown = set(raw).difference(known)
        if unknown:
            raise ValueError(f"Unknown kernel configuration keys: {sorted(unknown)}")
        optimization_raw = dict(raw.get("optimization", {}))
        if "optimizer_restarts" in raw:
            optimization_raw["restarts"] = raw["optimizer_restarts"]
        reselection_raw = dict(raw.get("reselection", {}))
        legacy_map = {
            "minimum_points_before_selection": "minimum_points",
            "reevaluate_every": "maximum_interval",
            "minimum_score_improvement": "minimum_switch_improvement",
        }
        for old, new in legacy_map.items():
            if old in raw:
                reselection_raw[new] = raw[old]
        optimization = KernelOptimizationConfig(**optimization_raw)
        reselection = KernelReselectionConfig(**reselection_raw)
        result = cls(
            mode=str(raw.get("mode", "automatic")),
            candidate_set=str(raw.get("candidate_set", "standard")),
            profile=raw.get("profile"),
            specification=raw.get("specification"),
            optimization=optimization,
            reselection=reselection,
            random_state=int(raw.get("random_state", 0)),
            optimize_hyperparameters=bool(raw.get("optimize_hyperparameters", True)),
        )
        result._validate()
        return result

    def _validate(self) -> None:
        if self.mode not in {"automatic", "hybrid", "manual"}:
            raise ValueError("kernel mode must be automatic, hybrid, or manual.")
        if self.optimization.restarts < 0:
            raise ValueError("kernel.optimization.restarts must be non-negative.")
        r = self.reselection
        if r.minimum_points < 2 or r.maximum_interval <= 0:
            raise ValueError("kernel reselection point and interval limits must be positive.")
        if any(value < 0 for value in (r.score_degradation_fraction, r.bound_proximity_fraction, r.minimum_switch_improvement)):
            raise ValueError("kernel reselection fractions must be non-negative.")
        if r.bound_contact_steps < 1:
            raise ValueError("kernel.reselection.bound_contact_steps must be positive.")


def parse_kernel_configuration(config: Mapping[str, Any] | KernelSelectionConfig | None) -> KernelSelectionConfig:
    if isinstance(config, KernelSelectionConfig):
        config._validate()
        return config
    return KernelSelectionConfig.from_mapping(config)


def as_float_pair(value: Any, name: str) -> tuple[float, float]:
    pair = tuple(float(item) for item in value)
    if len(pair) != 2 or not np.all(np.isfinite(pair)) or not (0 < pair[0] < pair[1]):
        raise ValueError(f"{name} must be an increasing positive pair.")
    return pair


def as_length_scale(value: Any, dimension: int, name: str) -> list[float]:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        result = np.full(dimension, float(array), dtype=float)
    elif array.shape == (dimension,):
        result = array
    else:
        raise ValueError(f"{name} must be scalar or contain one value per dimension.")
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError(f"{name} must contain finite positive values.")
    return result.tolist()
