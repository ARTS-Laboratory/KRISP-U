"""Minimal two-dimensional field reconstruction recommendation."""

import numpy as np

from krispu import ContinuousDomain, KrispURecommender, ObservationSet

domain = ContinuousDomain([[-1, 1], [-1, 1]], names=["x", "y"])
X = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=float)
y = X[:, 0] ** 2 + np.sin(3 * X[:, 1])
result = KrispURecommender(domain, random_state=7).recommend(ObservationSet(X, y))
print(result.to_records()[0])
