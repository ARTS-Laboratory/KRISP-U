"""Explicit per-step kernel optimization and reselection records."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KernelOptimizationEvent:
    sample_count: int
    kernel_id: str
    hyperparameters_optimized: bool
    current_length_scales: tuple[float, ...]
    length_scale_minimums: tuple[float, ...]
    length_scale_maximums: tuple[float, ...]
    current_validation_score: float
    optimization_runtime: float


@dataclass(frozen=True)
class KernelReselectionEvent:
    sample_count: int
    reselection_triggered: bool
    reselection_reasons: tuple[str, ...]
    candidates_evaluated: tuple[str, ...]
    previous_kernel_id: str | None
    selected_kernel_id: str
    current_validation_score: float
    challenger_validation_score: float | None
    score_improvement: float
    reselection_runtime: float
    best_challenger_kernel_id: str | None = None


@dataclass(frozen=True)
class KernelSwitchEvent:
    sample_count: int
    previous_kernel_id: str | None
    selected_kernel_id: str
    switch_accepted: bool
    current_validation_score: float
    challenger_validation_score: float | None
    score_improvement: float
