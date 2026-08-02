"""A field with substantially faster variation in x than y."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from benchmarks.fields.smooth import DOMAIN, AuditField


def _evaluate(X: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(X, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    x, y = values[:, 0], values[:, 1]
    return np.sin(7.0 * x + 1.5 * y) - 0.25 * np.cos(1.2 * y) - 0.15 * x


def anisotropic_field() -> AuditField:
    return AuditField(
        name="anisotropic",
        domain=DOMAIN,
        evaluate=_evaluate,
        recommended_plot_limits=(-1.5, 1.5),
        metadata={"description": "Fast x-direction variation with slow y-direction variation"},
    )
