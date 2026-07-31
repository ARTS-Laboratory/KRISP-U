"""Initial-design helpers for KRISP-U workflows."""

from __future__ import annotations

from itertools import product

import numpy as np
from numpy.typing import ArrayLike, NDArray

from krispu.space import make_rng, validate_bounds


def corner_plus_interior_design(
    bounds: ArrayLike,
    random_state: int | np.random.Generator | None = None,
    interior_fraction: tuple[float, float] = (0.25, 0.75),
    max_corner_points: int = 64,
) -> NDArray[np.float64]:
    """Return domain corners plus one random interior point.

    For a 2D rectangular domain this returns five rows: the four corners that
    define the convex hull plus one interior point. For n dimensions it returns
    the 2**n hyper-rectangle vertices plus one interior point.
    """

    bounds_array = validate_bounds(bounds)
    dimension = bounds_array.shape[0]
    n_corners = 2**dimension
    if n_corners > max_corner_points:
        raise ValueError(
            "Corner design would create too many points; increase max_corner_points "
            "explicitly if that is intended."
        )
    low_fraction, high_fraction = interior_fraction
    if not 0.0 < low_fraction < high_fraction < 1.0:
        raise ValueError("interior_fraction must satisfy 0 < low < high < 1.")

    axes = [(low, high) for low, high in bounds_array]
    corners = np.asarray(list(product(*axes)), dtype=float)
    rng = make_rng(random_state)
    unit = rng.uniform(low_fraction, high_fraction, size=dimension)
    interior = bounds_array[:, 0] + unit * (bounds_array[:, 1] - bounds_array[:, 0])
    return np.vstack((corners, interior.reshape(1, -1)))
