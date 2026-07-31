"""Acquisition functions for KRISP-U field sampling."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_solve
from scipy.stats import norm

from krispu.space import as_2d_float_array, validate_objective


def normalize_acquisition_name(name: str) -> str:
    """Normalize acquisition-function aliases."""

    normalized = name.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ei": "expected_improvement",
        "pi": "probability_of_improvement",
        "poi": "probability_of_improvement",
        "max_uncertainty": "uncertainty",
        "predictive_uncertainty": "uncertainty",
        "lcb": "confidence_bound",
        "ucb": "confidence_bound",
        "lower_confidence_bound": "confidence_bound",
        "upper_confidence_bound": "confidence_bound",
        "weighted_centroid": "thresholded_weighted_centroid",
        "kl": "kld",
        "kld_score": "kld",
        "kl_divergence": "kld",
        "kullback_leibler": "kld",
        "information_gain": "kld",
        "field_information_gain": "kld",
        "expected_information_gain": "kld",
    }
    return aliases.get(normalized, normalized)


def incumbent(y_observed: ArrayLike, objective: str) -> float:
    """Return the current best observed objective value."""

    objective = validate_objective(objective)
    values = np.asarray(y_observed, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("y_observed must contain at least one value.")
    if not np.all(np.isfinite(values)):
        raise ValueError("y_observed must contain only finite values.")
    if objective == "minimize":
        return float(np.min(values))
    return float(np.max(values))


def expected_improvement(
    mean: ArrayLike,
    std: ArrayLike,
    y_observed: ArrayLike,
    objective: str = "minimize",
    xi: float = 0.0,
) -> NDArray[np.float64]:
    """Return expected-improvement scores."""

    objective = validate_objective(objective)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    std = np.asarray(std, dtype=float).reshape(-1)
    best = incumbent(y_observed, objective)
    safe_std = np.maximum(std, 1e-15)
    if objective == "minimize":
        improvement = best - mean - xi
    else:
        improvement = mean - best - xi
    z_value = improvement / safe_std
    scores = improvement * norm.cdf(z_value) + safe_std * norm.pdf(z_value)
    scores[std <= 0] = 0.0
    return np.maximum(scores, 0.0)


def probability_of_improvement(
    mean: ArrayLike,
    std: ArrayLike,
    y_observed: ArrayLike,
    objective: str = "minimize",
    xi: float = 0.0,
) -> NDArray[np.float64]:
    """Return probability-of-improvement scores."""

    objective = validate_objective(objective)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    std = np.asarray(std, dtype=float).reshape(-1)
    best = incumbent(y_observed, objective)
    safe_std = np.maximum(std, 1e-15)
    if objective == "minimize":
        improvement = best - mean - xi
    else:
        improvement = mean - best - xi
    scores = norm.cdf(improvement / safe_std)
    scores[std <= 0] = 0.0
    return scores


def confidence_bound(
    mean: ArrayLike,
    std: ArrayLike,
    objective: str = "minimize",
    kappa: float = 2.0,
) -> NDArray[np.float64]:
    """Return confidence-bound scores where larger is always better."""

    objective = validate_objective(objective)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    std = np.asarray(std, dtype=float).reshape(-1)
    if objective == "minimize":
        return -mean + kappa * std
    return mean + kappa * std


def acquisition_scores(
    method: str,
    mean: ArrayLike,
    std: ArrayLike,
    y_observed: ArrayLike,
    objective: str = "minimize",
    xi: float = 0.0,
    kappa: float = 2.0,
) -> NDArray[np.float64]:
    """Compute acquisition scores for candidate predictions."""

    method = normalize_acquisition_name(method)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    std = np.asarray(std, dtype=float).reshape(-1)
    if mean.shape != std.shape:
        raise ValueError("mean and std must have the same shape.")
    if method == "uncertainty":
        return np.maximum(std, 0.0)
    if method in {"expected_improvement", "thresholded_weighted_centroid"}:
        return expected_improvement(mean, std, y_observed, objective, xi)
    if method == "probability_of_improvement":
        return probability_of_improvement(mean, std, y_observed, objective, xi)
    if method == "confidence_bound":
        return confidence_bound(mean, std, objective, kappa)
    if method == "kld":
        raise ValueError(
            "KLD acquisition requires a fitted Gaussian-process model. "
            "Use KrispUOptimizer.ask(acquisition='kld')."
        )
    raise ValueError(f"Unknown acquisition method: {method}")


def field_information_gain_scores(
    model: object,
    candidates: ArrayLike,
    reference_points: ArrayLike | None = None,
    chunk_size: int = 512,
    epsilon: float = 1e-12,
) -> NDArray[np.float64]:
    """Score candidates by expected KL information gain over a reference field.

    The score is the average Gaussian entropy reduction induced across
    ``reference_points`` by measuring a candidate point. This is the expected
    KL divergence between the current field posterior and the posterior after
    a hypothetical measurement at the candidate, so larger means the point is
    expected to reduce more field uncertainty.
    """

    candidates = as_2d_float_array(candidates, "candidates")
    if reference_points is None:
        reference = candidates
    else:
        reference = as_2d_float_array(reference_points, "reference_points")
    if len(candidates) == 0:
        raise ValueError("At least one candidate is required.")
    if len(reference) == 0:
        raise ValueError("At least one reference point is required.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if not hasattr(model, "kernel_") or not hasattr(model, "X_train_"):
        raise ValueError("KLD acquisition requires a fitted Gaussian-process model.")

    _, reference_std = model.predict(reference, return_std=True)
    _, candidate_std = model.predict(candidates, return_std=True)
    reference_var = np.maximum(np.asarray(reference_std, dtype=float) ** 2, epsilon)
    candidate_var = np.maximum(np.asarray(candidate_std, dtype=float) ** 2, epsilon)

    train = np.asarray(model.X_train_, dtype=float)
    train_reference_cov = model.kernel_(train, reference)
    solved_train_reference_cov = cho_solve(
        (model.L_, True), train_reference_cov, check_finite=False
    )

    scores = np.empty(len(candidates), dtype=float)
    for start in range(0, len(candidates), chunk_size):
        stop = min(start + chunk_size, len(candidates))
        chunk = candidates[start:stop]
        prior_cross_cov = model.kernel_(chunk, reference)
        chunk_train_cov = model.kernel_(chunk, train)
        posterior_cross_cov = (
            prior_cross_cov - chunk_train_cov @ solved_train_reference_cov
        )
        ratio = (posterior_cross_cov**2) / (
            candidate_var[start:stop, None] * reference_var[None, :]
        )
        ratio = np.clip(ratio, 0.0, 1.0 - epsilon)
        scores[start:stop] = np.mean(-0.5 * np.log1p(-ratio), axis=1)

    return np.maximum(scores, 0.0)


def thresholded_weighted_candidate_index(
    candidates: ArrayLike,
    scores: ArrayLike,
    threshold: float = 0.8,
) -> int:
    """Return the candidate nearest the weighted centroid of high-score candidates."""

    candidates = as_2d_float_array(candidates, "candidates")
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(candidates) != len(scores):
        raise ValueError("candidates and scores must have the same length.")
    if scores.size == 0:
        raise ValueError("At least one candidate score is required.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must contain only finite values.")
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")

    max_score = float(np.max(scores))
    cutoff = threshold * max_score if 0 <= threshold <= 1 else threshold
    mask = scores >= cutoff
    if not np.any(mask):
        return int(np.argmax(scores))

    selected = candidates[mask]
    selected_scores = scores[mask]
    weights = selected_scores - float(np.min(selected_scores))
    if np.sum(weights) <= 0:
        weights = np.ones_like(selected_scores)
    centroid = np.average(selected, axis=0, weights=weights)
    distances = np.linalg.norm(candidates - centroid, axis=1)
    return int(np.argmin(distances))
