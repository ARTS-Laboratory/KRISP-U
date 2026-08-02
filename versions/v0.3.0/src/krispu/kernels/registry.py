"""Finite, documented registry used by automatic and hybrid selection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)

Builder = Callable[[int, bool], Any]


@dataclass(frozen=True)
class KernelDefinition:
    kernel_id: str
    display_name: str
    builder: Builder
    number_of_hyperparameters: int
    supported_dimensions: tuple[int, ...] | str
    profile_tags: tuple[str, ...]
    default_bounds: dict[str, tuple[float, float]]
    supports_measurement_noise: bool


def _bounds(value: tuple[float, float], optimize: bool) -> str | tuple[float, float]:
    return value if optimize else "fixed"


def _amplitude_base(base: Any, amplitude: float, optimize: bool) -> Any:
    return ConstantKernel(amplitude, constant_value_bounds=_bounds((1e-3, 5.0), optimize)) * base


def _matern(dimension: int, optimize: bool, nu: float, initial: float = 0.25) -> Any:
    return _amplitude_base(
        Matern(
            length_scale=np.full(dimension, initial),
            length_scale_bounds=_bounds((0.02, 2.0), optimize),
            nu=nu,
        ),
        1.0,
        optimize,
    )


def _rbf(dimension: int, optimize: bool) -> Any:
    return _amplitude_base(
        RBF(
            length_scale=np.full(dimension, 0.25),
            length_scale_bounds=_bounds((0.02, 2.0), optimize),
        ),
        1.0,
        optimize,
    )


def _rational_quadratic(dimension: int, optimize: bool) -> Any:
    return _amplitude_base(
        RationalQuadratic(
            length_scale=0.25,
            alpha=1.0,
            length_scale_bounds=_bounds((0.02, 2.0), optimize),
            alpha_bounds=_bounds((1e-3, 100.0), optimize),
        ),
        1.0,
        optimize,
    )


def _multiscale(
    dimension: int,
    optimize: bool,
    long_nu: float,
    short_nu: float,
    long_amplitude: float = 1.0,
    short_amplitude: float = 0.4,
) -> Any:
    long = _amplitude_base(
        Matern(
            length_scale=np.full(dimension, 0.6),
            length_scale_bounds=_bounds((0.15, 2.0), optimize),
            nu=long_nu,
        ),
        long_amplitude,
        optimize,
    )
    short = _amplitude_base(
        Matern(
            length_scale=np.full(dimension, 0.08),
            length_scale_bounds=_bounds((0.01, 0.20), optimize),
            nu=short_nu,
        ),
        short_amplitude,
        optimize,
    )
    return long + short


def _rbf_multiscale(dimension: int, optimize: bool) -> Any:
    long = _amplitude_base(
        RBF(
            length_scale=np.full(dimension, 0.6),
            length_scale_bounds=_bounds((0.15, 2.0), optimize),
        ),
        1.0,
        optimize,
    )
    short = _amplitude_base(
        Matern(
            length_scale=np.full(dimension, 0.08),
            length_scale_bounds=_bounds((0.01, 0.20), optimize),
            nu=0.5,
        ),
        0.4,
        optimize,
    )
    return long + short


def _linear_plus_matern(dimension: int, optimize: bool, nu: float) -> Any:
    return DotProduct(
        sigma_0=1.0,
        sigma_0_bounds=_bounds((1e-5, 10.0), optimize),
    ) + _matern(dimension, optimize, nu)


def _periodic_times_matern(dimension: int, optimize: bool) -> Any:
    periodic = ExpSineSquared(
        length_scale=0.35,
        periodicity=0.9,
        length_scale_bounds=_bounds((0.02, 2.0), optimize),
        periodicity_bounds=_bounds((0.2, 4.0), optimize),
    )
    return periodic * _matern(dimension, optimize, 1.5) + WhiteKernel(
        noise_level=1.0,
        noise_level_bounds=_bounds((1e-6, 10.0), optimize),
    )


def _periodic_plus_matern(dimension: int, optimize: bool) -> Any:
    periodic = ExpSineSquared(
        length_scale=0.35,
        periodicity=0.9,
        length_scale_bounds=_bounds((0.02, 2.0), optimize),
        periodicity_bounds=_bounds((0.2, 4.0), optimize),
    )
    return (
        periodic
        + _matern(dimension, optimize, 1.5)
        + WhiteKernel(
            noise_level=1.0,
            noise_level_bounds=_bounds((1e-6, 10.0), optimize),
        )
    )


def _matern_52(dimension: int, optimize: bool) -> Any:
    return _matern(dimension, optimize, 2.5)


def _matern_32(dimension: int, optimize: bool) -> Any:
    return _matern(dimension, optimize, 1.5)


def _matern_12(dimension: int, optimize: bool) -> Any:
    return _matern(dimension, optimize, 0.5)


KERNEL_REGISTRY: dict[str, KernelDefinition] = {
    "matern_52_ard": KernelDefinition(
        "matern_52_ard",
        "Matérn-5/2 ARD",
        _matern_52,
        3,
        "continuous",
        ("smooth",),
        {"length_scale": (0.02, 2.0), "amplitude": (1e-3, 5.0)},
        True,
    ),
    "matern_32_ard": KernelDefinition(
        "matern_32_ard",
        "Matérn-3/2 ARD",
        _matern_32,
        3,
        "continuous",
        ("rough",),
        {"length_scale": (0.02, 2.0), "amplitude": (1e-3, 5.0)},
        True,
    ),
    "matern_12_ard": KernelDefinition(
        "matern_12_ard",
        "Matérn-1/2 ARD",
        _matern_12,
        3,
        "continuous",
        ("rough",),
        {"length_scale": (0.02, 2.0), "amplitude": (1e-3, 5.0)},
        True,
    ),
    "rbf_ard": KernelDefinition(
        "rbf_ard",
        "RBF ARD",
        _rbf,
        3,
        "continuous",
        ("smooth",),
        {"length_scale": (0.02, 2.0), "amplitude": (1e-3, 5.0)},
        True,
    ),
    "rational_quadratic": KernelDefinition(
        "rational_quadratic",
        "Rational Quadratic",
        _rational_quadratic,
        4,
        "continuous",
        ("rough", "multiscale"),
        {"length_scale": (0.02, 2.0), "alpha": (1e-3, 100.0)},
        True,
    ),
    "matern_52_long_plus_matern_12_short": KernelDefinition(
        "matern_52_long_plus_matern_12_short",
        "Matérn-5/2 long + Matérn-1/2 short",
        lambda dimension, optimize: _multiscale(dimension, optimize, 2.5, 0.5),
        6,
        "continuous",
        ("multiscale",),
        {"long_scale": (0.15, 2.0), "short_scale": (0.01, 0.20)},
        True,
    ),
    "matern_32_long_plus_matern_12_short": KernelDefinition(
        "matern_32_long_plus_matern_12_short",
        "Matérn-3/2 long + Matérn-1/2 short",
        lambda dimension, optimize: _multiscale(dimension, optimize, 1.5, 0.5),
        6,
        "continuous",
        ("multiscale",),
        {"long_scale": (0.15, 2.0), "short_scale": (0.01, 0.20)},
        True,
    ),
    "matern_52_long_plus_matern_32_short": KernelDefinition(
        "matern_52_long_plus_matern_32_short",
        "Matérn-5/2 long + Matérn-3/2 short",
        lambda dimension, optimize: _multiscale(dimension, optimize, 2.5, 1.5),
        6,
        "continuous",
        ("multiscale",),
        {"long_scale": (0.15, 2.0), "short_scale": (0.01, 0.20)},
        True,
    ),
    "rbf_long_plus_matern_12_short": KernelDefinition(
        "rbf_long_plus_matern_12_short",
        "RBF long + Matérn-1/2 short",
        _rbf_multiscale,
        6,
        "continuous",
        ("multiscale",),
        {"long_scale": (0.15, 2.0), "short_scale": (0.01, 0.20)},
        True,
    ),
    "linear_plus_matern_32": KernelDefinition(
        "linear_plus_matern_32",
        "Linear + Matérn-3/2",
        lambda d, o: _linear_plus_matern(d, o, 1.5),
        4,
        "continuous",
        ("trend",),
        {"length_scale": (0.02, 2.0)},
        True,
    ),
    "linear_plus_matern_52": KernelDefinition(
        "linear_plus_matern_52",
        "Linear + Matérn-5/2",
        lambda d, o: _linear_plus_matern(d, o, 2.5),
        4,
        "continuous",
        ("trend",),
        {"length_scale": (0.02, 2.0)},
        True,
    ),
    "periodic_times_matern_32": KernelDefinition(
        "periodic_times_matern_32",
        "Periodic × Matérn-3/2",
        _periodic_times_matern,
        6,
        "continuous",
        ("periodic",),
        {"periodicity": (0.2, 4.0), "length_scale": (0.02, 2.0)},
        True,
    ),
    "periodic_plus_matern_32": KernelDefinition(
        "periodic_plus_matern_32",
        "Periodic + Matérn-3/2",
        _periodic_plus_matern,
        6,
        "continuous",
        ("periodic",),
        {"periodicity": (0.2, 4.0), "length_scale": (0.02, 2.0)},
        True,
    ),
}

CANDIDATE_SETS: dict[str, tuple[str, ...]] = {
    "standard": tuple(KERNEL_REGISTRY),
}


def registered_kernel_ids() -> tuple[str, ...]:
    return tuple(KERNEL_REGISTRY)


def get_kernel_definition(kernel_id: str) -> KernelDefinition:
    try:
        return KERNEL_REGISTRY[kernel_id]
    except KeyError as exc:
        raise ValueError(f"Unknown registered kernel: {kernel_id}") from exc


def candidate_ids(candidate_set: str = "standard") -> tuple[str, ...]:
    try:
        return CANDIDATE_SETS[candidate_set]
    except KeyError as exc:
        raise ValueError(f"Unknown registered candidate set: {candidate_set}") from exc
