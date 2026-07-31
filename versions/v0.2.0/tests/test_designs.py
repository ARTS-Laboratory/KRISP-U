from __future__ import annotations

import numpy as np
import pytest

from krispu.designs import corner_plus_interior_design


def test_corner_plus_interior_design_returns_2d_hull_plus_one_point() -> None:
    bounds = np.asarray([[-1.0, 1.0], [10.0, 20.0]])
    design = corner_plus_interior_design(bounds, random_state=4)

    assert design.shape == (5, 2)
    expected_corners = {
        (-1.0, 10.0),
        (-1.0, 20.0),
        (1.0, 10.0),
        (1.0, 20.0),
    }
    assert expected_corners == {tuple(row) for row in design[:4]}
    assert -1.0 < design[-1, 0] < 1.0
    assert 10.0 < design[-1, 1] < 20.0


def test_corner_plus_interior_design_is_reproducible() -> None:
    bounds = [[0.0, 1.0], [0.0, 1.0]]

    first = corner_plus_interior_design(bounds, random_state=9)
    second = corner_plus_interior_design(bounds, random_state=9)

    np.testing.assert_allclose(first, second)


def test_corner_plus_interior_design_rejects_too_many_corners() -> None:
    bounds = np.asarray([[0.0, 1.0]] * 7)

    with pytest.raises(ValueError):
        corner_plus_interior_design(bounds)
