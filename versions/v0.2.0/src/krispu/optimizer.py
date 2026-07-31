"""GPR-based KRISP-U sequential field sampler."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor

from krispu.acquisition import (
    acquisition_scores,
    field_information_gain_scores,
    normalize_acquisition_name,
    thresholded_weighted_candidate_index,
)
from krispu.metrics import best_so_far
from krispu.models import (
    GprConfig,
    KernelPriorConfig,
    KernelPriorResult,
    clone_gpr,
    fit_prior_optimized_gpr,
    make_default_gpr,
)
from krispu.space import (
    ContinuousSpace,
    DiscreteCandidateSpace,
    HybridCandidateSpace,
    ensure_unique_rows,
    make_rng,
    validate_objective,
)

CandidateSpace = ContinuousSpace | DiscreteCandidateSpace | HybridCandidateSpace


@dataclass
class AcquisitionResult:
    """Result returned by a single KRISP-U field-sampling step."""

    x_next: NDArray[np.float64]
    acquisition: str
    score: float
    mean: float
    std: float
    candidate_index: int | None = None
    candidates: NDArray[np.float64] | None = None
    scores: NDArray[np.float64] | None = None
    predicted_mean: NDArray[np.float64] | None = None
    predicted_std: NDArray[np.float64] | None = None
    random_state: int | None = None


@dataclass
class OptimizationResult:
    """Structured history for a sequential KRISP-U run."""

    X: NDArray[np.float64]
    y: NDArray[np.float64]
    objective: str
    acquisitions: list[AcquisitionResult] = field(default_factory=list)
    random_state: int | None = None

    @property
    def best_y_history(self) -> NDArray[np.float64]:
        return best_so_far(self.y, self.objective)

    @property
    def best_y(self) -> float:
        return float(self.best_y_history[-1])

    @property
    def best_x(self) -> NDArray[np.float64]:
        if self.objective == "minimize":
            index = int(np.argmin(self.y))
        else:
            index = int(np.argmax(self.y))
        return self.X[index].copy()


class KrispUOptimizer:
    """Gaussian-process KRISP-U sampler for continuous and preset spaces."""

    def __init__(
        self,
        space: CandidateSpace,
        objective: str = "minimize",
        acquisition: str = "uncertainty",
        model: GaussianProcessRegressor | None = None,
        gpr_config: GprConfig | None = None,
        n_candidates: int = 4096,
        candidate_method: str = "lhs",
        random_state: int | np.random.Generator | None = None,
        xi: float = 0.0,
        kappa: float = 2.0,
        weighted_centroid_threshold: float = 0.8,
        chunk_size: int = 20_000,
        optimize_continuous_acquisition: bool = True,
        n_restarts: int = 8,
        exclude_observed: bool = True,
        kernel_prior_config: KernelPriorConfig | None = None,
    ) -> None:
        self.space = space
        self.objective = validate_objective(objective)
        self.acquisition = normalize_acquisition_name(acquisition)
        self.gpr_config = gpr_config or GprConfig()
        self.model_template = model or make_default_gpr(self.gpr_config)
        self.fixed_model_supplied = model is not None
        self.kernel_prior_config = (
            kernel_prior_config
            or self.gpr_config.kernel_prior_config
            or KernelPriorConfig(random_state=self.gpr_config.random_state)
        )
        self.n_candidates = int(n_candidates)
        self.candidate_method = candidate_method
        self.random_state = random_state if isinstance(random_state, int) else None
        self.rng = make_rng(random_state)
        self.xi = float(xi)
        self.kappa = float(kappa)
        self.weighted_centroid_threshold = float(weighted_centroid_threshold)
        self.chunk_size = int(chunk_size)
        self.optimize_continuous_acquisition = bool(optimize_continuous_acquisition)
        self.n_restarts = int(n_restarts)
        self.exclude_observed = bool(exclude_observed)
        self.model_: GaussianProcessRegressor | None = None
        self.X_train_: NDArray[np.float64] | None = None
        self.y_train_: NDArray[np.float64] | None = None
        self.kernel_prior_result_: KernelPriorResult | None = None
        self.model_selection_history_: list[KernelPriorResult] = []

        if self.n_candidates <= 0:
            raise ValueError("n_candidates must be positive.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if self.n_restarts <= 0:
            raise ValueError("n_restarts must be positive.")

    def fit(self, X: ArrayLike, y: ArrayLike) -> KrispUOptimizer:
        """Fit the GPR surrogate to observed data."""

        points = self.space.validate_points(X, "X")
        values = np.asarray(y, dtype=float).reshape(-1)
        if len(points) != len(values):
            raise ValueError("X and y must contain the same number of observations.")
        if len(points) < 2:
            raise ValueError("At least two observations are required to fit KRISP-U.")
        if not np.all(np.isfinite(values)):
            raise ValueError("y must contain only finite values.")
        ensure_unique_rows(points, "X")

        if self._uses_adaptive_kernel:
            model, prior_result = fit_prior_optimized_gpr(
                points,
                values,
                gpr_config=self.gpr_config,
                kernel_prior_config=self.kernel_prior_config,
                bounds=getattr(self.space, "bounds", None),
            )
            self.kernel_prior_result_ = prior_result
            self.model_selection_history_.append(prior_result)
        else:
            model = clone_gpr(self.model_template)
            model.fit(points, values)
            self.kernel_prior_result_ = None
        self.model_ = model
        self.X_train_ = points.copy()
        self.y_train_ = values.copy()
        return self

    @property
    def _uses_adaptive_kernel(self) -> bool:
        return (
            not self.fixed_model_supplied
            and self.gpr_config.kernel is None
            and self.gpr_config.adaptive_kernel
            and self.kernel_prior_config.enabled
        )

    def predict(self, X: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predict mean and standard deviation for candidate points."""

        if self.model_ is None:
            raise ValueError("Call fit() before predict().")
        points = self.space.validate_points(X, "X")
        mean, std = self.model_.predict(points, return_std=True)
        return np.asarray(mean, dtype=float).reshape(-1), np.asarray(
            std, dtype=float
        ).reshape(-1)

    def ask(
        self,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        acquisition: str | None = None,
        candidates: ArrayLike | None = None,
        n_candidates: int | None = None,
        candidate_method: str | None = None,
        store_candidates: bool = False,
    ) -> AcquisitionResult:
        """Fit if data are supplied, then propose the next candidate."""

        if X is not None or y is not None:
            if X is None or y is None:
                raise ValueError("X and y must be supplied together.")
            self.fit(X, y)
        if self.model_ is None or self.X_train_ is None or self.y_train_ is None:
            raise ValueError("Call fit() before ask(), or pass X and y into ask().")

        method = normalize_acquisition_name(acquisition or self.acquisition)
        seed = int(self.rng.integers(0, np.iinfo(np.int32).max))
        pool = self._candidate_pool(candidates, n_candidates, candidate_method, seed)
        if self.exclude_observed:
            pool = self._remove_observed(pool)
        if len(pool) == 0:
            raise ValueError("No valid unevaluated candidates remain.")

        mean, std, scores = self._score_candidates(pool, method)
        candidate_index = self._select_candidate_index(pool, scores, method)
        x_next = pool[candidate_index].copy()
        score = float(scores[candidate_index])
        selected_mean = float(mean[candidate_index])
        selected_std = float(std[candidate_index])

        if self._can_optimize_continuous(method, candidates):
            optimized = self._optimize_continuous(pool, scores, method)
            if optimized is not None and optimized[1] > score:
                x_next, score, selected_mean, selected_std = optimized
                candidate_index = None

        return AcquisitionResult(
            x_next=x_next,
            acquisition=method,
            score=score,
            mean=selected_mean,
            std=selected_std,
            candidate_index=candidate_index,
            candidates=pool.copy() if store_candidates else None,
            scores=scores.copy() if store_candidates else None,
            predicted_mean=mean.copy() if store_candidates else None,
            predicted_std=std.copy() if store_candidates else None,
            random_state=seed,
        )

    def tell(self, x_new: ArrayLike, y_new: float) -> KrispUOptimizer:
        """Add one observation and refit the surrogate."""

        if self.X_train_ is None or self.y_train_ is None:
            raise ValueError("Call fit() before tell().")
        point = self.space.validate_points(x_new, "x_new")
        if point.shape[0] != 1:
            raise ValueError("tell() accepts exactly one new point.")
        X = np.vstack((self.X_train_, point))
        y = np.append(self.y_train_, float(y_new))
        return self.fit(X, y)

    def run(
        self,
        objective_fn: Callable[[NDArray[np.float64]], ArrayLike],
        initial_X: ArrayLike,
        initial_y: ArrayLike | None = None,
        n_iterations: int = 10,
        acquisition: str | None = None,
        store_candidates: bool = False,
    ) -> OptimizationResult:
        """Run a sequential KRISP-U field-sampling loop."""

        if n_iterations < 0:
            raise ValueError("n_iterations must be non-negative.")
        X = self.space.validate_points(initial_X, "initial_X")
        if initial_y is None:
            y = self._evaluate_objective(objective_fn, X)
        else:
            y = np.asarray(initial_y, dtype=float).reshape(-1)
        if len(X) != len(y):
            raise ValueError("initial_X and initial_y must have the same length.")

        acquisitions: list[AcquisitionResult] = []
        for _ in range(n_iterations):
            self.fit(X, y)
            acquisition_result = self.ask(
                acquisition=acquisition,
                store_candidates=store_candidates,
            )
            y_next = self._evaluate_objective(
                objective_fn, acquisition_result.x_next.reshape(1, -1)
            )[0]
            X = np.vstack((X, acquisition_result.x_next.reshape(1, -1)))
            y = np.append(y, y_next)
            acquisitions.append(acquisition_result)

        self.fit(X, y)
        return OptimizationResult(
            X=X,
            y=y,
            objective=self.objective,
            acquisitions=acquisitions,
            random_state=self.random_state,
        )

    def _candidate_pool(
        self,
        candidates: ArrayLike | None,
        n_candidates: int | None,
        candidate_method: str | None,
        seed: int,
    ) -> NDArray[np.float64]:
        if candidates is not None:
            return self.space.validate_points(candidates, "candidates")
        n = int(n_candidates or self.n_candidates)
        method = candidate_method or self.candidate_method
        if isinstance(self.space, DiscreteCandidateSpace):
            return self.space.candidates.copy()
        if method.lower() in {"grid", "mesh"}:
            points_per_dimension = max(2, int(np.ceil(n ** (1 / self.space.dimension))))
            return self.space.dense_grid(points_per_dimension=points_per_dimension)[:n]
        return self.space.sample(n, method=method, random_state=seed)

    def _remove_observed(self, candidates: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.X_train_ is None:
            return candidates
        keep = np.ones(len(candidates), dtype=bool)
        for observed in self.X_train_:
            keep &= ~np.all(np.isclose(candidates, observed, atol=1e-10), axis=1)
        return candidates[keep]

    def _score_candidates(
        self, candidates: NDArray[np.float64], method: str
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        if self.model_ is None or self.y_train_ is None:
            raise ValueError("Call fit() before scoring candidates.")
        if method == "kld":
            mean, std = self.model_.predict(candidates, return_std=True)
            scores = field_information_gain_scores(
                self.model_,
                candidates,
                reference_points=candidates,
                chunk_size=self.chunk_size,
            )
            return (
                np.asarray(mean, dtype=float).reshape(-1),
                np.asarray(std, dtype=float).reshape(-1),
                scores,
            )

        mean_parts = []
        std_parts = []
        score_parts = []
        for start in range(0, len(candidates), self.chunk_size):
            chunk = candidates[start : start + self.chunk_size]
            mean, std = self.model_.predict(chunk, return_std=True)
            scores = acquisition_scores(
                method,
                mean,
                std,
                self.y_train_,
                objective=self.objective,
                xi=self.xi,
                kappa=self.kappa,
            )
            mean_parts.append(np.asarray(mean, dtype=float).reshape(-1))
            std_parts.append(np.asarray(std, dtype=float).reshape(-1))
            score_parts.append(scores)
        return (
            np.concatenate(mean_parts),
            np.concatenate(std_parts),
            np.concatenate(score_parts),
        )

    def _select_candidate_index(
        self, candidates: NDArray[np.float64], scores: NDArray[np.float64], method: str
    ) -> int:
        if method == "thresholded_weighted_centroid":
            return thresholded_weighted_candidate_index(
                candidates, scores, threshold=self.weighted_centroid_threshold
            )
        return int(np.argmax(scores))

    def _can_optimize_continuous(
        self, method: str, user_candidates: ArrayLike | None
    ) -> bool:
        return (
            self.optimize_continuous_acquisition
            and user_candidates is None
            and isinstance(self.space, ContinuousSpace)
            and method not in {"thresholded_weighted_centroid", "kld"}
        )

    def _optimize_continuous(
        self,
        pool: NDArray[np.float64],
        scores: NDArray[np.float64],
        method: str,
    ) -> tuple[NDArray[np.float64], float, float, float] | None:
        if self.X_train_ is None:
            return None
        start_indices = np.argsort(scores)[-self.n_restarts :]
        starts = pool[start_indices]
        bounds = [tuple(pair) for pair in self.space.bounds]
        best: tuple[NDArray[np.float64], float, float, float] | None = None

        def objective(x_value: NDArray[np.float64]) -> float:
            point = np.asarray(x_value, dtype=float).reshape(1, -1)
            mean, std, acq = self._score_candidates(point, method)
            return -float(acq[0])

        for start in starts:
            result = minimize(objective, start, method="L-BFGS-B", bounds=bounds)
            if not result.success:
                continue
            point = np.asarray(result.x, dtype=float).reshape(1, -1)
            if not np.all(self.space.contains(point)):
                continue
            if np.any(np.all(np.isclose(self.X_train_, point, atol=1e-8), axis=1)):
                continue
            mean, std, acq = self._score_candidates(point, method)
            candidate = (
                point.reshape(-1),
                float(acq[0]),
                float(mean[0]),
                float(std[0]),
            )
            if best is None or candidate[1] > best[1]:
                best = candidate
        return best

    @staticmethod
    def _evaluate_objective(
        objective_fn: Callable[[NDArray[np.float64]], ArrayLike],
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        values = np.asarray(objective_fn(X), dtype=float).reshape(-1)
        if len(values) != len(X):
            raise ValueError("objective_fn must return one value per input row.")
        if not np.all(np.isfinite(values)):
            raise ValueError("objective_fn returned non-finite values.")
        return values


KRISPU = KrispUOptimizer
