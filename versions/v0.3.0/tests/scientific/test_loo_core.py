import numpy as np

from krispu import ContinuousDomain, GPRConfig, KrispURecommender, ObservationSet


def _problem() -> tuple[ContinuousDomain, ObservationSet, np.ndarray]:
    domain = ContinuousDomain([[-1, 1], [-1, 1]], names=["x", "y"])
    X = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1], [0, 0]], dtype=float)
    y = X[:, 0] ** 2 + 0.3 * X[:, 1]
    references = np.array([[0, 0], [0.5, 0.2], [-0.7, 0.4], [0.9, -0.8]], dtype=float)
    return domain, ObservationSet(X, y, loo_eligible=[True, True, False, True, True]), references


def test_bruteforce_loo_returns_one_field_per_eligible_observation_and_protects_anchor() -> None:
    domain, observations, references = _problem()
    diagnostics = KrispURecommender(
        domain, gpr_config=GPRConfig(n_restarts_optimizer=0)
    ).evaluate_uncertainty(observations, references)
    assert diagnostics.loo_field_means.shape == (len(references), 4)
    assert np.array_equal(diagnostics.loo_eligible_indices, [0, 1, 3, 4])
    assert np.all(np.isfinite(diagnostics.loo_field_means))
    assert np.all(np.isfinite(diagnostics.combined_std))
    assert np.all(diagnostics.combined_std >= 0)


def test_reordering_observations_does_not_change_field_uncertainty() -> None:
    domain, observations, references = _problem()
    order = [4, 2, 0, 3, 1]
    shuffled = ObservationSet(
        observations.X[order], observations.y[order], observations.loo_eligible[order]
    )
    first = KrispURecommender(domain).evaluate_uncertainty(observations, references)
    second = KrispURecommender(domain).evaluate_uncertainty(shuffled, references)
    assert np.allclose(first.predicted_mean, second.predicted_mean)
    assert np.allclose(first.combined_std, second.combined_std)


def test_response_shift_and_positive_scale_preserve_recommendation_coordinates() -> None:
    domain, observations, _ = _problem()
    candidates = domain.denormalize(np.array([[0.1, 0.2], [0.4, 0.4], [0.8, 0.1], [0.2, 0.9]]))
    first = KrispURecommender(domain).recommend(observations, candidates=candidates)
    shifted = ObservationSet(observations.X, observations.y + 23.0, observations.loo_eligible)
    scaled = ObservationSet(observations.X, observations.y * 4.0, observations.loo_eligible)
    second = KrispURecommender(domain).recommend(shifted, candidates=candidates)
    third = KrispURecommender(domain).recommend(scaled, candidates=candidates)
    assert np.allclose(first.as_array(), second.as_array())
    assert np.allclose(first.as_array(), third.as_array())


def test_physical_unit_scaling_preserves_normalized_recommendation() -> None:
    _, observations, _ = _problem()
    scaled_domain = ContinuousDomain([[-100, 100], [-0.01, 0.01]])
    scaled_X = np.column_stack((observations.X[:, 0] * 100, observations.X[:, 1] * 0.01))
    scaled_observations = ObservationSet(scaled_X, observations.y, observations.loo_eligible)
    candidates = np.array([[-0.8, -0.4], [0.2, 0.7], [0.9, 0.1]])
    scaled_candidates = np.column_stack((candidates[:, 0] * 100, candidates[:, 1] * 0.01))
    first = KrispURecommender(ContinuousDomain([[-1, 1], [-1, 1]])).recommend(
        observations, candidates=candidates
    )
    second = KrispURecommender(scaled_domain).recommend(
        scaled_observations, candidates=scaled_candidates
    )
    assert np.allclose(first.as_array(), second.as_array() / [100, 0.01])


def test_recommendations_are_valid_and_not_observed() -> None:
    domain, observations, _ = _problem()
    candidates = np.vstack((observations.X, [[0.2, -0.2], [0.5, 0.6]]))
    result = KrispURecommender(domain).recommend(observations, candidates=candidates)
    assert len(result.recommendations) == 1
    assert not np.any(np.all(np.isclose(observations.X, result.as_array()[0]), axis=1))
    assert bool(domain.contains(result.as_array())[0])


def test_constant_field_has_finite_scores() -> None:
    domain = ContinuousDomain([[0, 1], [0, 1]])
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    result = KrispURecommender(domain).recommend(ObservationSet(X, np.ones(4)), n_recommendations=1)
    assert np.isfinite(result.recommendations[0].combined_std)
