"""A smooth global audit field."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from krispu.domains import ContinuousDomain


@dataclass(frozen=True)
class AuditField:
    name: str
    domain: ContinuousDomain
    evaluate: object
    recommended_plot_limits: tuple[float, float]
    metadata: dict[str, object]


DOMAIN = ContinuousDomain([[-1.0, 1.0], [-1.0, 1.0]], names=("x", "y"))


def _evaluate(X: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(X, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    x, y = values[:, 0], values[:, 1]
    return (
        0.6 * np.sin(2.5 * x)
        - 0.4 * np.cos(2.0 * y)
        - 0.25 * x * y
        - 0.5 * np.exp(-3.0 * ((x - 0.35) ** 2 + (y + 0.25) ** 2))
    )


def smooth_field() -> AuditField:
    return AuditField(
        name="smooth",
        domain=DOMAIN,
        evaluate=_evaluate,
        recommended_plot_limits=(-1.7, 1.2),
        metadata={"description": "Smooth multi-feature global field", "true_kernel": None, "field_category": "development"},
    )
