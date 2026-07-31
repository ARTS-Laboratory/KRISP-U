"""Metric utilities used by KRISP-U validation and benchmarks."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_1d_float(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _paired_arrays(
    y_true: ArrayLike, y_predicted: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    true = _as_1d_float(y_true, "y_true")
    predicted = _as_1d_float(y_predicted, "y_predicted")
    if true.shape != predicted.shape:
        raise ValueError("Input arrays must have the same shape.")
    return true, predicted


def mse(y_true: ArrayLike, y_predicted: ArrayLike) -> float:
    """Return mean squared error."""

    true, predicted = _paired_arrays(y_true, y_predicted)
    return float(np.mean((true - predicted) ** 2))


def rmse(y_true: ArrayLike, y_predicted: ArrayLike) -> float:
    """Return root mean squared error."""

    return float(np.sqrt(mse(y_true, y_predicted)))


def mae(y_true: ArrayLike, y_predicted: ArrayLike) -> float:
    """Return mean absolute error."""

    true, predicted = _paired_arrays(y_true, y_predicted)
    return float(np.mean(np.abs(true - predicted)))


def kld(y_true: ArrayLike, y_predicted: ArrayLike, epsilon: float = 1e-12) -> float:
    """Return Kullback-Leibler divergence between normalized magnitudes."""

    true, predicted = _paired_arrays(y_true, y_predicted)
    p = np.abs(true) + epsilon
    q = np.abs(predicted) + epsilon
    p = p / np.sum(p)
    q = q / np.sum(q)
    value = np.sum(p * np.log(p / q))
    if not np.isfinite(value):
        return 0.0
    return float(value)


def jsd(y_true: ArrayLike, y_predicted: ArrayLike, epsilon: float = 1e-12) -> float:
    """Return Jensen-Shannon divergence between normalized magnitudes."""

    true, predicted = _paired_arrays(y_true, y_predicted)
    middle = 0.5 * (np.abs(true) + np.abs(predicted))
    return float(
        0.5 * kld(true, middle, epsilon) + 0.5 * kld(predicted, middle, epsilon)
    )


def best_so_far(values: ArrayLike, objective: str = "minimize") -> NDArray[np.float64]:
    """Return cumulative best values for a minimization or maximization trace."""

    array = _as_1d_float(values, "values")
    if objective == "minimize":
        return np.minimum.accumulate(array)
    if objective == "maximize":
        return np.maximum.accumulate(array)
    raise ValueError("objective must be either 'minimize' or 'maximize'.")


def simple_regret(
    best_values: ArrayLike, optimum: float, objective: str = "minimize"
) -> NDArray[np.float64]:
    """Return non-negative simple regret to a known optimum."""

    best = _as_1d_float(best_values, "best_values")
    if objective == "minimize":
        regret = best - float(optimum)
    elif objective == "maximize":
        regret = float(optimum) - best
    else:
        raise ValueError("objective must be either 'minimize' or 'maximize'.")
    return np.maximum(regret, 0.0)


KLD = kld
MSE = mse
JSD = jsd
