"""Canonical two-dimensional and scalar DOE benchmark fields.

The deterministic fields in this module deliberately carry no generating-kernel
claim.  They are response functions used to compare reconstruction methods.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from evaluation.fields.canonical.smooth import AuditField
from krispu.domains import ContinuousDomain


def _points(values: ArrayLike, dimension: int) -> NDArray[np.float64]:
    points = np.asarray(values, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2 or points.shape[1] != dimension:
        raise ValueError(f"expected a two-dimensional array with {dimension} columns")
    return points


def _normalized_field(
    name: str,
    bounds: list[list[float]],
    function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    description: str,
) -> AuditField:
    domain = ContinuousDomain(bounds, names=tuple(f"x{index + 1}" for index in range(len(bounds))))
    probe = np.random.default_rng(1901 + len(name)).random((4096, len(bounds)))
    probe = domain.denormalize(probe)
    response = np.asarray(function(probe), dtype=float)
    center = float(np.mean(response))
    scale = max(float(np.ptp(response)), np.finfo(float).eps)

    def evaluate(values: ArrayLike) -> NDArray[np.float64]:
        return (np.asarray(function(_points(values, len(bounds))), dtype=float) - center) / scale

    return AuditField(
        name=name,
        domain=domain,
        evaluate=evaluate,
        recommended_plot_limits=(-0.6, 0.6),
        metadata={
            "description": description,
            "field_category": "canonical_doe" if len(bounds) > 2 else "canonical_2d",
            "deterministic": True,
            "true_kernel": None,
            "domain": bounds,
            "normalization": {"center": center, "scale": scale},
        },
    )


def franke_field() -> AuditField:
    def function(values: NDArray[np.float64]) -> NDArray[np.float64]:
        x, y = values[:, 0], values[:, 1]
        return (
            0.75 * np.exp(-((9 * x - 2) ** 2 + (9 * y - 2) ** 2) / 4)
            + 0.75 * np.exp(-((9 * x + 1) ** 2) / 49 - (9 * y + 1) / 10)
            + 0.5 * np.exp(-((9 * x - 7) ** 2 + (9 * y - 3) ** 2) / 4)
            - 0.2 * np.exp(-((9 * x - 4) ** 2 + (9 * y - 7) ** 2))
        )

    return _normalized_field("Franke", [[0, 1], [0, 1]], function, "Franke response surface")


def branin_hoo_field() -> AuditField:
    def function(values: NDArray[np.float64]) -> NDArray[np.float64]:
        x, y = values[:, 0], values[:, 1]
        a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5 / np.pi
        return a * (y - b * x**2 + c * x - 6) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x) + 10

    return _normalized_field(
        "Branin-Hoo", [[-5, 10], [0, 15]], function, "Branin-Hoo response surface"
    )


def goldstein_price_field() -> AuditField:
    def function(values: NDArray[np.float64]) -> NDArray[np.float64]:
        x, y = values[:, 0], values[:, 1]
        first = 1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)
        second = 30 + (2 * x - 3 * y) ** 2 * (
            18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2
        )
        return first * second

    return _normalized_field(
        "Goldstein-Price", [[-2, 2], [-2, 2]], function, "Goldstein-Price response surface"
    )


def six_hump_camel_field() -> AuditField:
    def function(values: NDArray[np.float64]) -> NDArray[np.float64]:
        x, y = values[:, 0], values[:, 1]
        return (4 - 2.1 * x**2 + x**4 / 3) * x**2 + x * y + (-4 + 4 * y**2) * y**2

    return _normalized_field(
        "Six-hump camel", [[-3, 3], [-2, 2]], function, "Six-hump camel response surface"
    )


def _hartmann3(values: NDArray[np.float64]) -> NDArray[np.float64]:
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    alpha_a = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
    p = 1e-4 * np.array(
        [[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]]
    )
    return np.sum(alpha * np.exp(-np.sum(alpha_a * (values[:, None, :] - p) ** 2, axis=2)), axis=1)


def _hartmann6(values: NDArray[np.float64]) -> NDArray[np.float64]:
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    alpha_a = np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
    )
    p = 1e-4 * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )
    return np.sum(alpha * np.exp(-np.sum(alpha_a * (values[:, None, :] - p) ** 2, axis=2)), axis=1)


def _borehole(values: NDArray[np.float64]) -> NDArray[np.float64]:
    rw, r, tu, hu, tl, hl, l, kw = values.T
    return (
        2
        * np.pi
        * tu
        * (hu - hl)
        / (np.log(r / rw) * (1 + 2 * l * tu / (np.log(r / rw) * rw**2 * kw)) + tu / tl)
    )


def _otl(values: NDArray[np.float64]) -> NDArray[np.float64]:
    rb, rf, ru, tu, tl, au = values.T
    return ((ru * (rf + 0.1) + 2 * au * tl) / (rb * (rf + 0.1) + 2 * au * (tl + tu))) * (tu + tl)


def _piston(values: NDArray[np.float64]) -> NDArray[np.float64]:
    m, s, v0, k, p0, ta, t0 = values.T
    return s * np.sqrt(m / (k + p0 * v0 / (t0 * ta)))


def hartmann_3d_field() -> AuditField:
    return _normalized_field(
        "Hartmann 3D", [[0, 1]] * 3, _hartmann3, "Hartmann 3D scalar benchmark"
    )


def hartmann_6d_field() -> AuditField:
    return _normalized_field(
        "Hartmann 6D", [[0, 1]] * 6, _hartmann6, "Hartmann 6D scalar benchmark"
    )


def borehole_field() -> AuditField:
    bounds = [
        [0.05, 0.15],
        [100, 50000],
        [63070, 115600],
        [990, 1110],
        [63.1, 116],
        [700, 820],
        [1120, 1680],
        [9855, 12045],
    ]
    return _normalized_field("Borehole", bounds, _borehole, "Borehole flow-rate benchmark")


def otl_circuit_field() -> AuditField:
    bounds = [[50, 150], [25, 70], [0.5, 3], [0.5, 2], [0.05, 0.15], [200, 500]]
    return _normalized_field("OTL circuit", bounds, _otl, "OTL circuit scalar benchmark")


def piston_simulation_field() -> AuditField:
    bounds = [
        [30, 60],
        [0.005, 0.02],
        [0.002, 0.01],
        [1000, 5000],
        [90000, 110000],
        [290, 296],
        [340, 360],
    ]
    return _normalized_field(
        "Piston simulation", bounds, _piston, "Piston simulation scalar benchmark"
    )


__all__ = [
    "borehole_field",
    "branin_hoo_field",
    "franke_field",
    "goldstein_price_field",
    "hartmann_3d_field",
    "hartmann_6d_field",
    "otl_circuit_field",
    "piston_simulation_field",
    "six_hump_camel_field",
]
