"""Toy response-field datasets and benchmark functions for KRISP-U."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from krispu.space import (
    ContinuousSpace,
    DiscreteCandidateSpace,
    as_2d_float_array,
    make_rng,
    validate_bounds,
    validate_objective,
)

ObjectiveFunction = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class ToyDataset:
    """Metadata and callable response field for a deterministic toy problem."""

    name: str
    function: ObjectiveFunction
    bounds: NDArray[np.float64] | ArrayLike
    objective: str = "minimize"
    optimum_x: NDArray[np.float64] | None = None
    optimum_y: float | None = None
    labels: Sequence[str] | None = None
    description: str = ""
    candidates: NDArray[np.float64] | None = None
    recommended_initial_n: int | None = None

    def __post_init__(self) -> None:
        bounds = validate_bounds(self.bounds)
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "objective", validate_objective(self.objective))
        if self.labels is not None and len(self.labels) != bounds.shape[0]:
            raise ValueError("labels must match the number of dimensions.")
        if self.optimum_x is not None:
            optimum_x = np.asarray(self.optimum_x, dtype=float).reshape(-1)
            if optimum_x.shape[0] != bounds.shape[0]:
                raise ValueError("optimum_x must match the number of dimensions.")
            object.__setattr__(self, "optimum_x", optimum_x)
        if self.candidates is not None:
            candidates = as_2d_float_array(self.candidates, "candidates")
            if candidates.shape[1] != bounds.shape[0]:
                raise ValueError("candidates must match the number of dimensions.")
            object.__setattr__(self, "candidates", candidates)

    @property
    def dimension(self) -> int:
        return int(self.bounds.shape[0])

    def evaluate(self, X: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the response field at one or more rows."""

        points = as_2d_float_array(X, "X")
        if points.shape[1] != self.dimension:
            raise ValueError(f"X must have {self.dimension} columns.")
        values = np.asarray(self.function(points), dtype=float).reshape(-1)
        if len(values) != len(points):
            raise ValueError("Dataset function returned the wrong number of values.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Dataset function returned non-finite values.")
        return values

    def space(self) -> ContinuousSpace | DiscreteCandidateSpace:
        """Return the natural candidate space for this dataset."""

        if self.candidates is not None:
            return DiscreteCandidateSpace(self.candidates, names=self.labels)
        return ContinuousSpace(self.bounds, names=self.labels)

    def initial_design(
        self,
        n: int | None = None,
        method: str = "lhs",
        random_state: int | np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """Return a recommended initial design."""

        n = n or self.recommended_initial_n or max(4, 2 * self.dimension + 1)
        return self.space().sample(n, method=method, random_state=random_state)


def _check_dimension(X: NDArray[np.float64], dimension: int) -> None:
    if X.shape[1] != dimension:
        raise ValueError(f"Expected {dimension} dimensions, got {X.shape[1]}.")


def forrester(X: NDArray[np.float64]) -> NDArray[np.float64]:
    _check_dimension(X, 1)
    x = X[:, 0]
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)


def sinusoid_trend(X: NDArray[np.float64]) -> NDArray[np.float64]:
    _check_dimension(X, 1)
    x = X[:, 0]
    return np.sin(10 * x) + 0.5 * x


def branin(X: NDArray[np.float64]) -> NDArray[np.float64]:
    _check_dimension(X, 2)
    x1 = X[:, 0]
    x2 = X[:, 1]
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


def himmelblau(X: NDArray[np.float64]) -> NDArray[np.float64]:
    _check_dimension(X, 2)
    x = X[:, 0]
    y = X[:, 1]
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


def six_hump_camel(X: NDArray[np.float64]) -> NDArray[np.float64]:
    _check_dimension(X, 2)
    x = X[:, 0]
    y = X[:, 1]
    return (4 - 2.1 * x**2 + x**4 / 3) * x**2 + x * y + (-4 + 4 * y**2) * y**2


def ackley_2d(X: NDArray[np.float64]) -> NDArray[np.float64]:
    _check_dimension(X, 2)
    x = X[:, 0]
    y = X[:, 1]
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    return term1 + term2 + np.e + 20.0


def goldstein_price(X: NDArray[np.float64]) -> NDArray[np.float64]:
    _check_dimension(X, 2)
    x = X[:, 0]
    y = X[:, 1]
    term1 = 1 + (x + y + 1) ** 2 * (
        19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2
    )
    term2 = 30 + (2 * x - 3 * y) ** 2 * (
        18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2
    )
    return term1 * term2


def rosenbrock_2d(X: NDArray[np.float64]) -> NDArray[np.float64]:
    _check_dimension(X, 2)
    x = X[:, 0]
    y = X[:, 1]
    return 100.0 * (y - x**2) ** 2 + (1 - x) ** 2


def anisotropic_ridge(X: NDArray[np.float64]) -> NDArray[np.float64]:
    _check_dimension(X, 2)
    x = X[:, 0]
    y = X[:, 1]
    curved_center = -0.35 + 0.55 * np.sin(2.5 * x)
    return 0.2 * (x - 0.1) ** 2 + 40.0 * (y - curved_center) ** 2 + 0.1 * x


def gaussian_mixture_2d(X: NDArray[np.float64]) -> NDArray[np.float64]:
    _check_dimension(X, 2)
    x = X[:, 0]
    y = X[:, 1]
    peak_1 = 1.25 * np.exp(-((x - 0.55) ** 2 / 0.015 + (y + 0.35) ** 2 / 0.03))
    peak_2 = 0.70 * np.exp(-((x + 0.45) ** 2 / 0.05 + (y - 0.40) ** 2 / 0.06))
    peak_3 = 0.35 * np.exp(-((x - 0.10) ** 2 / 0.20 + (y - 0.10) ** 2 / 0.20))
    return -(peak_1 + peak_2 + peak_3) + 0.04 * (x**2 + y**2)


def quadratic_bowl_2d(X: NDArray[np.float64]) -> NDArray[np.float64]:
    _check_dimension(X, 2)
    x = X[:, 0]
    y = X[:, 1]
    return (x - 0.25) ** 2 + 1.5 * (y + 0.40) ** 2


def noisy_smooth_surface_2d(X: NDArray[np.float64]) -> NDArray[np.float64]:
    _check_dimension(X, 2)
    x = X[:, 0]
    y = X[:, 1]
    centers = np.asarray(
        [[-0.75, -0.10], [-0.25, 0.70], [0.35, -0.45], [0.80, 0.35]],
        dtype=float,
    )
    weights = np.asarray([0.35, -0.55, 0.70, -0.45], dtype=float)
    values = 0.25 * np.sin(4 * x) + 0.20 * np.cos(3 * y) + 0.15 * np.sin(5 * (x + y))
    for center, weight in zip(centers, weights, strict=True):
        values += weight * np.exp(-6 * ((x - center[0]) ** 2 + (y - center[1]) ** 2))
    return values


def hartmann_3d(X: NDArray[np.float64]) -> NDArray[np.float64]:
    _check_dimension(X, 3)
    alpha = np.asarray([1.0, 1.2, 3.0, 3.2])
    a = np.asarray([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
    p = 1e-4 * np.asarray(
        [[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]]
    )
    outer = np.zeros(X.shape[0])
    for i in range(4):
        outer += alpha[i] * np.exp(-np.sum(a[i] * (X - p[i]) ** 2, axis=1))
    return -outer


def hartmann_6d(X: NDArray[np.float64]) -> NDArray[np.float64]:
    _check_dimension(X, 6)
    alpha = np.asarray([1.0, 1.2, 3.0, 3.2])
    a = np.asarray(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ],
        dtype=float,
    )
    p = 1e-4 * np.asarray(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ],
        dtype=float,
    )
    outer = np.zeros(X.shape[0])
    for i in range(4):
        outer += alpha[i] * np.exp(-np.sum(a[i] * (X - p[i]) ** 2, axis=1))
    return -outer


def additive_5d(X: NDArray[np.float64]) -> NDArray[np.float64]:
    _check_dimension(X, 5)
    return (
        np.sin(4 * X[:, 0])
        + 0.5 * (X[:, 1] - 0.2) ** 2
        + np.cos(3 * X[:, 2])
        + 0.25 * X[:, 3]
        + (X[:, 4] + 0.4) ** 2
    )


def _continuous_datasets() -> dict[str, ToyDataset]:
    return {
        "forrester": ToyDataset(
            name="forrester",
            function=forrester,
            bounds=np.asarray([[0.0, 1.0]]),
            optimum_x=np.asarray([0.75725]),
            optimum_y=-6.02074,
            labels=("x",),
            description="One-dimensional multimodal Forrester function.",
            recommended_initial_n=4,
        ),
        "sinusoid_trend": ToyDataset(
            name="sinusoid_trend",
            function=sinusoid_trend,
            bounds=np.asarray([[0.0, 1.0]]),
            labels=("x",),
            description="One-dimensional sinusoid with a linear trend.",
            recommended_initial_n=4,
        ),
        "branin": ToyDataset(
            name="branin",
            function=branin,
            bounds=np.asarray([[-5.0, 10.0], [0.0, 15.0]]),
            optimum_x=np.asarray([-np.pi, 12.275]),
            optimum_y=0.397887,
            labels=("x1", "x2"),
            description="Classic 2D Branin-Hoo minimization problem.",
        ),
        "himmelblau": ToyDataset(
            name="himmelblau",
            function=himmelblau,
            bounds=np.asarray([[-6.0, 6.0], [-6.0, 6.0]]),
            optimum_x=np.asarray([3.0, 2.0]),
            optimum_y=0.0,
            labels=("x", "y"),
            description="Multimodal 2D Himmelblau minimization problem.",
        ),
        "six_hump_camel": ToyDataset(
            name="six_hump_camel",
            function=six_hump_camel,
            bounds=np.asarray([[-3.0, 3.0], [-2.0, 2.0]]),
            optimum_x=np.asarray([0.0898, -0.7126]),
            optimum_y=-1.0316,
            labels=("x", "y"),
            description="Six-hump camel minimization problem.",
        ),
        "ackley_2d": ToyDataset(
            name="ackley_2d",
            function=ackley_2d,
            bounds=np.asarray([[-5.0, 5.0], [-5.0, 5.0]]),
            optimum_x=np.asarray([0.0, 0.0]),
            optimum_y=0.0,
            labels=("x", "y"),
            description="2D Ackley function with many local minima.",
        ),
        "goldstein_price": ToyDataset(
            name="goldstein_price",
            function=goldstein_price,
            bounds=np.asarray([[-2.0, 2.0], [-2.0, 2.0]]),
            optimum_x=np.asarray([0.0, -1.0]),
            optimum_y=3.0,
            labels=("x", "y"),
            description="Goldstein-Price 2D minimization problem.",
        ),
        "rosenbrock_2d": ToyDataset(
            name="rosenbrock_2d",
            function=rosenbrock_2d,
            bounds=np.asarray([[-2.0, 2.0], [-1.0, 3.0]]),
            optimum_x=np.asarray([1.0, 1.0]),
            optimum_y=0.0,
            labels=("x", "y"),
            description="2D Rosenbrock banana valley.",
        ),
        "anisotropic_ridge": ToyDataset(
            name="anisotropic_ridge",
            function=anisotropic_ridge,
            bounds=np.asarray([[-1.0, 1.0], [-1.0, 1.0]]),
            labels=("x", "y"),
            description="Anisotropic curved ridge/valley surface.",
        ),
        "gaussian_mixture_2d": ToyDataset(
            name="gaussian_mixture_2d",
            function=gaussian_mixture_2d,
            bounds=np.asarray([[-1.0, 1.0], [-1.0, 1.0]]),
            optimum_x=np.asarray([0.55, -0.35]),
            optimum_y=-1.225,
            labels=("x", "y"),
            description="Localized Gaussian mixture with a narrow feature.",
        ),
        "quadratic_bowl_2d": ToyDataset(
            name="quadratic_bowl_2d",
            function=quadratic_bowl_2d,
            bounds=np.asarray([[-2.0, 2.0], [-2.0, 2.0]]),
            optimum_x=np.asarray([0.25, -0.40]),
            optimum_y=0.0,
            labels=("x", "y"),
            description="Convex quadratic bowl sanity-check problem.",
        ),
        "noisy_smooth_surface_2d": ToyDataset(
            name="noisy_smooth_surface_2d",
            function=noisy_smooth_surface_2d,
            bounds=np.asarray([[-1.0, 1.0], [-1.0, 1.0]]),
            labels=("x", "y"),
            description="Deterministic rough smooth surface with fixed features.",
        ),
        "hartmann_3d": ToyDataset(
            name="hartmann_3d",
            function=hartmann_3d,
            bounds=np.asarray([[0.0, 1.0]] * 3),
            objective="minimize",
            optimum_y=-3.86278,
            labels=("x1", "x2", "x3"),
            description="Hartmann 3D benchmark, expressed as minimization.",
        ),
        "hartmann_6d": ToyDataset(
            name="hartmann_6d",
            function=hartmann_6d,
            bounds=np.asarray([[0.0, 1.0]] * 6),
            objective="minimize",
            optimum_y=-3.32237,
            labels=("x1", "x2", "x3", "x4", "x5", "x6"),
            description="Hartmann 6D benchmark, expressed as minimization.",
            recommended_initial_n=13,
        ),
        "additive_5d": ToyDataset(
            name="additive_5d",
            function=additive_5d,
            bounds=np.asarray([[-1.0, 1.0]] * 5),
            objective="minimize",
            labels=("x1", "x2", "x3", "x4", "x5"),
            description="Synthetic additive 5D response surface.",
            recommended_initial_n=11,
        ),
    }


def _candidate_subset(
    dataset: ToyDataset,
    n_candidates: int,
    random_state: int,
    method: str = "lhs",
) -> NDArray[np.float64]:
    rng = make_rng(random_state)
    candidates = ContinuousSpace(dataset.bounds).sample(
        n_candidates, method=method, random_state=rng
    )
    if dataset.optimum_x is not None:
        candidates = np.vstack((dataset.optimum_x.reshape(1, -1), candidates))
    return np.unique(np.round(candidates, decimals=12), axis=0)


def _discrete_from(
    dataset: ToyDataset,
    name: str,
    n_candidates: int,
    random_state: int,
    description: str,
    method: str = "lhs",
) -> ToyDataset:
    candidates = _candidate_subset(dataset, n_candidates, random_state, method)
    values = dataset.evaluate(candidates)
    if dataset.objective == "minimize":
        index = int(np.argmin(values))
    else:
        index = int(np.argmax(values))
    return ToyDataset(
        name=name,
        function=dataset.function,
        bounds=dataset.bounds,
        objective=dataset.objective,
        optimum_x=candidates[index],
        optimum_y=float(values[index]),
        labels=dataset.labels,
        description=description,
        candidates=candidates,
        recommended_initial_n=min(5, max(2, len(candidates) // 10)),
    )


def _build_datasets() -> dict[str, ToyDataset]:
    datasets = _continuous_datasets()
    datasets["branin_irregular_candidates"] = _discrete_from(
        datasets["branin"],
        name="branin_irregular_candidates",
        n_candidates=160,
        random_state=21,
        description="Branin evaluated only on an irregular preset candidate pool.",
    )
    datasets["gaussian_mixture_sparse_candidates"] = _discrete_from(
        datasets["gaussian_mixture_2d"],
        name="gaussian_mixture_sparse_candidates",
        n_candidates=90,
        random_state=22,
        description="Narrow Gaussian mixture sampled on a sparse preset candidate pool.",
    )
    datasets["anisotropic_ridge_coarse_candidates"] = _discrete_from(
        datasets["anisotropic_ridge"],
        name="anisotropic_ridge_coarse_candidates",
        n_candidates=120,
        random_state=23,
        description="Anisotropic ridge sampled on a coarse preset candidate pool.",
    )
    return datasets


_DATASETS = _build_datasets()


def list_datasets() -> list[str]:
    """Return available toy-dataset names."""

    return sorted(_DATASETS)


def get_dataset(name: str) -> ToyDataset:
    """Return a toy dataset by name."""

    try:
        return _DATASETS[name]
    except KeyError as exc:
        available = ", ".join(list_datasets())
        raise KeyError(
            f"Unknown dataset '{name}'. Available datasets: {available}"
        ) from exc
