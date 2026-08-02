# KRISP-U v0.3.0

This version implements KRISP-U as a field-reconstruction method.  It fits a
response-standardized Matérn-3/2 ARD Gaussian process, removes each explicitly
eligible measurement using the same fitted kernel hyperparameters, predicts
the complete candidate/reference field for every fold, and ranks candidates by
the combined candidate-level uncertainty.

The first pass intentionally stops at the auditable brute-force LOO core.
Analytic LOO, conditional batch selection, and the paired random-field
benchmark suite are deferred until this reference implementation has been
validated.

Install from this directory:

```bash
python -m pip install -e ".[dev,plot]"
```

Minimal use:

```python
import numpy as np
from krispu import ContinuousDomain, KrispURecommender, ObservationSet

domain = ContinuousDomain([[-1, 1], [-1, 1]], names=["x", "y"])
observations = ObservationSet(
    X=np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]]),
    y=np.array([0.0, 1.0, 1.0, 0.0]),
)
result = KrispURecommender(domain, random_state=7).recommend(observations)
print(result.recommendations[0].x)
```

See `docs/algorithm.md` and `docs/uncertainty.md` for the exact method.
