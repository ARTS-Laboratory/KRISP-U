"""Small surrogate protocol used to keep fitting separate from acquisition."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Surrogate(Protocol):
    def fit(
        self, X: ArrayLike, y: ArrayLike, observation_variances: ArrayLike | None = None
    ) -> Surrogate: ...

    def predict(self, X: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
