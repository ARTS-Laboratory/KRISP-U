"""Direct full-model posterior uncertainty, kept separate from KRISP-U."""

from __future__ import annotations

from numpy.typing import ArrayLike, NDArray

from krispu.surrogates.gpr import GPRSurrogate


def posterior_predictions(surrogate: GPRSurrogate, points: ArrayLike) -> tuple[NDArray, NDArray]:
    return surrogate.predict(points)
