"""Explicit candidate-failure and degeneracy diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from krispu.kernels.registry import KernelDefinition


@dataclass(frozen=True)
class DegeneracyDiagnostics:
    penalty: float
    reasons: tuple[str, ...]
    valid: bool
    condition_number: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "degeneracy_penalty": self.penalty,
            "penalty_reasons": list(self.reasons),
            "valid": self.valid,
            "condition_number": self.condition_number,
        }


def inspect_fitted_model(model: Any, definition: KernelDefinition) -> DegeneracyDiagnostics:
    """Check fitted-kernel pathologies that can make predictive scores misleading."""

    reasons: list[str] = []
    penalty = 0.0
    condition_number: float | None = None
    try:
        if model is None or not np.all(np.isfinite(model.kernel_.theta)):
            reasons.append("nonfinite fit")
            penalty += 100.0
        if model is not None and hasattr(model, "L_"):
            condition_number = float(np.linalg.cond(model.L_) ** 2)
            if not np.isfinite(condition_number):
                reasons.append("failed covariance factorization")
                penalty += 100.0
            elif condition_number > 1e12:
                reasons.append("extreme covariance condition number")
                penalty += 10.0
    except (FloatingPointError, np.linalg.LinAlgError, AttributeError, TypeError):
        reasons.append("failed covariance factorization")
        penalty += 100.0

    if model is None:
        return DegeneracyDiagnostics(penalty, tuple(reasons), False, condition_number)

    try:
        values: dict[str, np.ndarray] = {}
        theta_offset = 0
        for hyperparameter in model.kernel_.hyperparameters:
            count = int(hyperparameter.n_elements)
            if hyperparameter.fixed:
                theta_offset += count
                continue
            values[hyperparameter.name] = _hyperparameter_values(model, theta_offset, count)
            theta_offset += count
            if "length_scale" in hyperparameter.name:
                _bound_reason(
                    values[hyperparameter.name], hyperparameter.bounds, reasons, "length scale"
                )
            if "constant_value" in hyperparameter.name:
                _bound_reason(
                    values[hyperparameter.name],
                    hyperparameter.bounds,
                    reasons,
                    "component amplitude",
                )
            if "noise_level" in hyperparameter.name:
                _bound_reason(
                    values[hyperparameter.name],
                    hyperparameter.bounds,
                    reasons,
                    "white-noise variance",
                )
        penalty += float(len(reasons))
        noise = [value for name, value in values.items() if "noise_level" in name]
        amplitudes = [value for name, value in values.items() if "constant_value" in name]
        if noise and amplitudes:
            noise_value = float(np.sum(noise[0]))
            response_scale = noise_value + sum(float(np.sum(value)) for value in amplitudes)
            if response_scale > 0 and noise_value / response_scale > 0.95:
                reasons.append("white-noise variance absorbing nearly all response variance")
                penalty += 5.0
    except (FloatingPointError, ValueError, TypeError):
        reasons.append("nonfinite fit")
        penalty += 100.0

    unique_reasons = tuple(dict.fromkeys(reasons))
    valid = not any(
        reason in {"nonfinite fit", "failed covariance factorization", "nonfinite CV predictions"}
        for reason in unique_reasons
    )
    return DegeneracyDiagnostics(float(penalty), unique_reasons, valid, condition_number)


def _hyperparameter_values(model: Any, offset: int, count: int) -> np.ndarray:
    values = np.exp(np.asarray(model.kernel_.theta[offset : offset + count], dtype=float))
    return values.reshape(-1)


def _bound_reason(
    values: np.ndarray,
    bounds: np.ndarray,
    reasons: list[str],
    label: str,
) -> None:
    lower = np.asarray(bounds[:, 0], dtype=float).reshape(-1)
    upper = np.asarray(bounds[:, 1], dtype=float).reshape(-1)
    values = values.reshape(-1)
    tolerance = 1e-6
    if np.any(values <= lower * (1.0 + tolerance)):
        reasons.append(f"{label} at lower bound")
    if np.any(values >= upper * (1.0 - tolerance)):
        reasons.append(f"{label} at upper bound")


def _length_scale_values(values: dict[str, np.ndarray]) -> list[np.ndarray]:
    return [value for name, value in sorted(values.items()) if "length_scale" in name]


def fitted_hyperparameters(model: Any) -> dict[str, list[float]]:
    """Return JSON/CSV-friendly fitted hyperparameters."""

    result: dict[str, list[float]] = {}
    if model is None:
        return result
    theta_offset = 0
    for hyperparameter in model.kernel_.hyperparameters:
        count = int(hyperparameter.n_elements)
        result[hyperparameter.name] = _hyperparameter_values(model, theta_offset, count).tolist()
        theta_offset += count
    return result
