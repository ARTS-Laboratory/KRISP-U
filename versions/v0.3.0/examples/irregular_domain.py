"""Recommendation inside a polygon with a central excluded hole."""

import numpy as np

from krispu import KrispURecommender, ObservationSet, PolygonDomain

domain = PolygonDomain(
    [[0, 0], [2, 0], [2, 2], [0, 2]],
    holes=[[[0.8, 0.8], [1.2, 0.8], [1.2, 1.2], [0.8, 1.2]]],
    names=["x", "y"],
)
X = np.array([[0.2, 0.2], [1.8, 0.2], [0.2, 1.8], [1.8, 1.8]])
y = X[:, 0] + X[:, 1]
result = KrispURecommender(domain, random_state=11, n_candidates=512).recommend(
    ObservationSet(X, y)
)
print(result.to_records()[0])
