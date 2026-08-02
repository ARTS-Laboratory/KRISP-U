# KRISP-U v0.3.0

This version implements KRISP-U as a field-reconstruction method.  It fits a
response-standardized Matérn-3/2 ARD Gaussian process, removes each explicitly
eligible measurement using the same fitted kernel hyperparameters, predicts
the complete candidate/reference field for every fold, and ranks candidates by
support-adjusted KRISP-U uncertainty.

The first pass intentionally uses the auditable brute-force LOO core. Analytic
LOO, conditional batch selection, and broad random-field generation remain
deferred. The deterministic benchmark audit below is diagnostic rather than
final performance evidence.

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

Benchmark audit
---------------

From this directory, install the plotting and development dependencies and run
the paired smoke audit:

```bash
python -m pip install -e ".[dev,plot]"
python -m benchmarks.runner --config benchmarks/configs/paired_smoke.yaml
```

The larger visual audit is run with:

```bash
python -m benchmarks.runner --config benchmarks/configs/visual_audit.yaml
```

Outputs are written only to `benchmark_outputs/<timestamp>_<experiment>/`.
The audit uses smooth, localized, anisotropic, rough-correlated, and
rough-multiscale fields; the methods `raw_loo_sensitivity`,
`support_adjusted_krispu`, `posterior_std`, `random`, `lhs`, and `maximin`; and the common
five-point interior maximin initial design.
Metrics are RMSE, NRMSE, MAE, NMAE, R², p95 absolute error, and maximum
absolute error. Every paired method shares the field, initial responses,
candidate pool, evaluation grid, and budget, with separate recorded seeds.
