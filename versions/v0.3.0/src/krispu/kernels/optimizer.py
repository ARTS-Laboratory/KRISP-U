"""Small optimizer entry point for one global ARD kernel family."""

from __future__ import annotations

from typing import Any

from numpy.typing import ArrayLike

from krispu.config import GPRConfig
from krispu.surrogates.gpr import GPRSurrogate


def optimize_kernel(
    kernel: Any,
    X_normalized: ArrayLike,
    y: ArrayLike,
    *,
    gpr_config: GPRConfig | None = None,
    restarts: int = 0,
) -> GPRSurrogate:
    """Optimize amplitude, global ARD scales, and configured nugget once."""

    from dataclasses import replace

    base = gpr_config or GPRConfig()
    return GPRSurrogate(
        replace(base, kernel=kernel, optimize_hyperparameters=True, n_restarts_optimizer=restarts)
    ).fit(X_normalized, y)
