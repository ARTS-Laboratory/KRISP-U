"""A smooth background with a narrow localized feature."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from evaluation.fields.canonical.smooth import DOMAIN, AuditField


def _evaluate(X: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(X, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    x, y = values[:, 0], values[:, 1]
    return (
        0.25 * x
        + 0.2 * y
        - 0.3 * np.sin(2.0 * x)
        - 1.5 * np.exp(-((x - 0.35) ** 2 / 0.025) - ((y + 0.2) ** 2 / 0.04))
    )


def localized_field() -> AuditField:
    return AuditField(
        name="localized",
        domain=DOMAIN,
        evaluate=_evaluate,
        recommended_plot_limits=(-1.7, 0.9),
        metadata={
            "description": "Smooth background with a narrow negative localized feature",
            "feature_center": [0.35, -0.2],
            "true_kernel": None,
            "field_category": "development",
        },
    )
