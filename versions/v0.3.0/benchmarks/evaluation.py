"""Common field reconstruction metrics."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ReconstructionMetrics:
    rmse: float
    nrmse: float
    mae: float
    nmae: float
    r2: float
    p95_absolute_error: float
    max_absolute_error: float
    absolute_error: NDArray[np.float64]
    squared_error: NDArray[np.float64]


def nrmse_auc(sample_counts: ArrayLike, nrmse_values: ArrayLike) -> float:
    """Integrate NRMSE using the benchmark's right-endpoint convention."""

    counts = np.asarray(sample_counts, dtype=float).reshape(-1)
    values = np.asarray(nrmse_values, dtype=float).reshape(-1)
    if counts.shape != values.shape or len(counts) < 2:
        raise ValueError("sample_counts and nrmse_values need at least two matching values.")
    if not np.all(np.isfinite(counts)) or not np.all(np.isfinite(values)):
        raise ValueError("sample_counts and nrmse_values must be finite.")
    if np.any(np.diff(counts) <= 0):
        raise ValueError("sample_counts must be strictly increasing.")
    return float(np.sum(values[1:] * np.diff(counts)))


def paired_difference(krispu: ArrayLike, baseline: ArrayLike) -> NDArray[np.float64]:
    """Return paired ``KRISP-U - baseline`` differences."""

    left = np.asarray(krispu, dtype=float)
    right = np.asarray(baseline, dtype=float)
    if left.shape != right.shape:
        raise ValueError("paired metric arrays must have matching shapes.")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("paired metric arrays must be finite.")
    return left - right


def reconstruction_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> ReconstructionMetrics:
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    prediction = np.asarray(y_pred, dtype=float).reshape(-1)
    if truth.shape != prediction.shape or truth.size == 0:
        raise ValueError("y_true and y_pred must have the same non-zero shape.")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
        raise ValueError("y_true and y_pred must be finite.")
    absolute = np.abs(truth - prediction)
    squared = absolute**2
    value_range = float(np.max(truth) - np.min(truth))
    if value_range <= np.finfo(float).eps:
        raise ValueError("NRMSE and NMAE are undefined for a numerically constant true field.")
    sst = float(np.sum((truth - np.mean(truth)) ** 2))
    if sst <= np.finfo(float).eps:
        raise ValueError("R2 is undefined for a numerically constant true field.")
    rmse = float(np.sqrt(np.mean(squared)))
    mae = float(np.mean(absolute))
    return ReconstructionMetrics(
        rmse=rmse,
        nrmse=rmse / value_range,
        mae=mae,
        nmae=mae / value_range,
        r2=1.0 - float(np.sum(squared)) / sst,
        p95_absolute_error=float(np.percentile(absolute, 95)),
        max_absolute_error=float(np.max(absolute)),
        absolute_error=absolute,
        squared_error=squared,
    )
