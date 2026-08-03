import numpy as np

from krispu.domains import ContinuousDomain
from krispu.observations import ObservationSet
from krispu.recommender import KrispURecommender


def test_seeded_candidate_recommendation_is_reproducible() -> None:
    domain = ContinuousDomain([[0, 1], [0, 1]])
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    observations = ObservationSet(X, np.array([0.0, 1.0, 1.0, 0.0]))
    first = KrispURecommender(domain, random_state=19, n_candidates=64).recommend(observations)
    second = KrispURecommender(domain, random_state=19, n_candidates=64).recommend(observations)
    assert np.allclose(first.as_array(), second.as_array())
    assert np.isclose(first.recommendations[0].krispu_uncertainty, second.recommendations[0].krispu_uncertainty)
