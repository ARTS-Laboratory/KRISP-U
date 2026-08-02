"""Small explicit builders for the supported scikit-learn kernel families."""

from __future__ import annotations

from collections.abc import Mapping
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

from krispu.kernels.specification import as_float_pair, as_length_scale


def build_kernel_from_spec(
    specification: Mapping[str, Any],
    dimension: int,
    optimize_hyperparameters: bool = True,
) -> Any:
    """Build a validated manual kernel without evaluating Python expressions."""

    if dimension < 1:
        raise ValueError("dimension must be positive.")
    spec = dict(specification)
    kernel = _build_node(spec, dimension, optimize_hyperparameters)
    noise = spec.get("observation_noise")
    if noise is not None:
        if not isinstance(noise, Mapping):
            raise ValueError("observation_noise must be a mapping.")
        if bool(noise.get("enabled", False)):
            initial = float(noise.get("initial", noise.get("variance_initial", 1e-6)))
            bounds = as_float_pair(
                noise.get("bounds", noise.get("variance_bounds", [1e-10, 1.0])),
                "observation_noise.bounds",
            )
            kernel = kernel + WhiteKernel(
                noise_level=initial,
                noise_level_bounds=_bounds(bounds, optimize_hyperparameters),
            )
    return kernel


def build_named_kernel(
    kernel_id: str,
    dimension: int,
    optimize_hyperparameters: bool = True,
) -> Any:
    """Build one registry candidate by explicit kernel id."""

    from krispu.kernels.registry import get_kernel_definition

    definition = get_kernel_definition(kernel_id)
    return definition.builder(dimension, optimize_hyperparameters)


def _build_node(specification: Mapping[str, Any], dimension: int, optimize: bool) -> Any:
    kind = str(specification.get("type", "")).lower().replace("-", "_")
    if kind in {"additive", "sum", "multiplicative", "product"}:
        components = specification.get("components")
        if not isinstance(components, (list, tuple)) or not components:
            raise ValueError(f"{kind} kernels require a non-empty components list.")
        built = [_build_node(component, dimension, optimize) for component in components]
        result = built[0]
        for component in built[1:]:
            result = result + component if kind in {"additive", "sum"} else result * component
        return result
    if kind in {"matern", "matern_ard"}:
        nu = float(specification.get("nu", 1.5))
        if nu not in {0.5, 1.5, 2.5}:
            raise ValueError("manual Matérn nu must be 0.5, 1.5, or 2.5.")
        base = Matern(
            length_scale=_initial_length_scale(specification, dimension),
            length_scale_bounds=_length_bounds(specification, optimize),
            nu=nu,
        )
    elif kind in {"rbf", "rbf_ard"}:
        base = RBF(
            length_scale=_initial_length_scale(specification, dimension),
            length_scale_bounds=_length_bounds(specification, optimize),
        )
    elif kind in {"rational_quadratic", "rq"}:
        alpha_initial = float(specification.get("alpha_initial", 1.0))
        alpha_bounds = as_float_pair(specification.get("alpha_bounds", [1e-3, 1e3]), "alpha_bounds")
        base = RationalQuadratic(
            length_scale=float(_initial_length_scale(specification, dimension)),
            alpha=alpha_initial,
            length_scale_bounds=_length_bounds(specification, optimize),
            alpha_bounds=_bounds(alpha_bounds, optimize),
        )
    elif kind in {"dotproduct", "dot_product", "linear"}:
        sigma_initial = float(specification.get("sigma_0_initial", 1.0))
        sigma_bounds = as_float_pair(
            specification.get("sigma_0_bounds", [1e-5, 1e3]), "sigma_0_bounds"
        )
        base = DotProduct(
            sigma_0=sigma_initial,
            sigma_0_bounds=_bounds(sigma_bounds, optimize),
        )
    elif kind in {"expsinesquared", "exp_sine_squared", "periodic"}:
        periodicity = float(specification.get("periodicity_initial", 1.0))
        periodicity_bounds = as_float_pair(
            specification.get("periodicity_bounds", [0.2, 4.0]), "periodicity_bounds"
        )
        base = ExpSineSquared(
            length_scale=float(_initial_length_scale(specification, dimension)),
            periodicity=periodicity,
            length_scale_bounds=_length_bounds(specification, optimize),
            periodicity_bounds=_bounds(periodicity_bounds, optimize),
        )
    elif kind in {"white", "white_kernel"}:
        base = WhiteKernel(
            noise_level=float(specification.get("noise_initial", 1e-6)),
            noise_level_bounds=_bounds(
                as_float_pair(specification.get("noise_bounds", [1e-10, 1.0]), "noise_bounds"),
                optimize,
            ),
        )
    else:
        raise ValueError(f"Unsupported explicit kernel type: {specification.get('type')!r}")

    if kind in {"white", "white_kernel"}:
        return base
    amplitude = float(specification.get("amplitude_initial", 1.0))
    amplitude_bounds = as_float_pair(
        specification.get("amplitude_bounds", [1e-3, 1e3]), "amplitude_bounds"
    )
    return (
        ConstantKernel(
            constant_value=amplitude,
            constant_value_bounds=_bounds(amplitude_bounds, optimize),
        )
        * base
    )


def _initial_length_scale(specification: Mapping[str, Any], dimension: int) -> float | np.ndarray:
    value = specification.get("length_scale_initial", 0.25)
    checked = as_length_scale(value, dimension, "length_scale_initial")
    return np.asarray(checked, dtype=float) if isinstance(checked, list) else checked


def _length_bounds(specification: Mapping[str, Any], optimize: bool) -> str | tuple[float, float]:
    bounds = as_float_pair(
        specification.get("length_scale_bounds", [0.02, 2.0]), "length_scale_bounds"
    )
    return _bounds(bounds, optimize)


def _bounds(bounds: tuple[float, float], optimize: bool) -> str | tuple[float, float]:
    return bounds if optimize else "fixed"
