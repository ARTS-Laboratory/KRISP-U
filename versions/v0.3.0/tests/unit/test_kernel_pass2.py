import numpy as np
import pytest
from sklearn.gaussian_process.kernels import Product, Sum

from krispu.kernels.builders import build_kernel_from_spec, build_named_kernel
from krispu.kernels.registry import registered_kernel_ids
from krispu.kernels.scoring import score_candidate_set
from krispu.kernels.selection import KernelSelector


def _data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.4, 0.6], [0.2, 0.8]])
    return X, np.sin(X[:, 0]) + X[:, 1]


def test_manual_additive_and_multiplicative_specs_are_rejected() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        build_kernel_from_spec({"type": "additive", "components": [{"type": "gaussian_ard"}, {"type": "exponential_ard"}]}, 2)


def test_registered_kernels_are_single_global_ard_families() -> None:
    assert "rbf_ard" not in registered_kernel_ids()
    assert "exponential_ard" in registered_kernel_ids()
    for kernel_id in registered_kernel_ids():
        kernel = build_named_kernel(kernel_id, 2)
        assert np.asarray(kernel.length_scale).shape == (2,)
        assert not isinstance(kernel, (Sum, Product))


def test_kernel_gradients_and_covariance_matrices_are_finite() -> None:
    X, _ = _data()
    for kernel_id in registered_kernel_ids():
        kernel = build_named_kernel(kernel_id, 2)
        covariance, gradient = kernel(X, eval_gradient=True)
        assert covariance.shape == (len(X), len(X))
        assert gradient.shape[-1] == len(kernel.theta)
        assert np.all(np.isfinite(covariance))
        assert np.all(np.linalg.eigvalsh(covariance) > -1e-8)


def test_every_candidate_receives_the_same_fold_plan() -> None:
    X, y = _data()
    scores = score_candidate_set(X, y, registered_kernel_ids(), optimizer_restarts=0)
    assert len(scores) == len(registered_kernel_ids())
    assert all(score.valid for score in scores)
    assert all(score.fold_plan is scores[0].fold_plan for score in scores)
    assert all("buffered_predictive_log_score" in score.as_record(len(X), None, score.candidate_kernel_id, True) for score in scores)


def test_current_family_is_optimized_each_step_and_reselection_is_triggered_at_interval() -> None:
    X, y = _data()
    selector = KernelSelector({"mode": "automatic", "optimization": {"restarts": 0}, "reselection": {"minimum_points": 6, "maximum_interval": 2}})
    first = selector.select(X, y)
    second = selector.select(np.vstack((X, [[0.8, 0.2]])), np.r_[y, 0.7])
    assert first.optimization_event.hyperparameters_optimized
    assert second.optimization_event.hyperparameters_optimized
    assert second.reselection_event.reselection_triggered
    assert second.reselection_event.candidates_evaluated == registered_kernel_ids()


def test_reselection_can_retain_the_current_family() -> None:
    X, y = _data()
    selector = KernelSelector({"mode": "automatic", "optimization": {"restarts": 0}, "reselection": {"minimum_points": 6, "maximum_interval": 1, "minimum_switch_improvement": 1e6}})
    first = selector.select(X, y)
    second = selector.select(np.vstack((X, [[0.8, 0.2]])), np.r_[y, 0.7])
    assert second.reselection_event.reselection_triggered
    assert second.selected_kernel_id == first.selected_kernel_id
    assert not second.switch_accepted


def test_fitted_axis_scales_change_when_an_anisotropic_field_is_rotated() -> None:
    grid = np.linspace(0.1, 0.9, 7)
    X = np.array([(x, y) for x in grid for y in grid])
    def field(first: float, second: float) -> np.ndarray:
        return np.exp(-((X[:, 0] - 0.5) / first) ** 2 - ((X[:, 1] - 0.5) / second) ** 2)
    from krispu.config import GPRConfig
    from krispu.surrogates.gpr import GPRSurrogate
    first = GPRSurrogate(GPRConfig(kernel=build_named_kernel("gaussian_ard", 2))).fit(X, field(0.15, 0.45))
    second = GPRSurrogate(GPRConfig(kernel=build_named_kernel("gaussian_ard", 2))).fit(X, field(0.45, 0.15))
    assert np.allclose(first.model_.kernel_.length_scale, second.model_.kernel_.length_scale[::-1], atol=0.05)


def test_spherical_and_wendland_reject_unsupported_dimensions() -> None:
    with pytest.raises(ValueError):
        build_named_kernel("spherical_ard", 4)
    with pytest.raises(ValueError):
        build_named_kernel("wendland_c2_ard", 4)
