# v0.3.0 benchmark protocol

Run the audit from `versions/v0.3.0` with:

```bash
python -m benchmarks.runner --config benchmarks/configs/paired_smoke.yaml
python -m benchmarks.runner --config benchmarks/configs/visual_audit.yaml
```

Each run writes `benchmark_outputs/<timestamp>_<experiment_name>/`, including
the YAML configuration, JSON manifest, scalar CSV records, compressed spatial
arrays, and PNG/PDF figures. The manifest records package and Python versions,
the available Git commit, GPR settings, field metadata, grid and candidate
settings, and all field, initial-design, candidate, and method seeds.

The audit fields are the smooth multi-feature field, the localized-feature
field, and the anisotropic field, all on `[-1, 1]^2`. The exact methods are
`krispu_loo`, `posterior_std`, `random`, `lhs`, and
`maximin`. All methods in a paired trial share the hidden field, domain, five
initial interior-maximin points, initial responses, candidate
pool, evaluation grid, and final measurement budget.

Field metrics are RMSE, NRMSE = RMSE / (max(true) - min(true)), MAE,
NMAE = MAE / (max(true) - min(true)), R², p95 absolute error, and maximum
absolute error. A numerically constant true field raises an explicit error.
Large arrays are stored in NPZ files rather than CSV cells. Missing LOO values
for baseline methods are left missing; they are never replaced by zero.

The figures include six-panel field audits, uncertainty components, learning
curves, sampling paths, uncertainty-versus-error scatters, error-concentration
curves, component evolution, and paired final-performance differences. The
LOO field uncertainty is plotted alongside GP posterior standard deviation;
there is no combined-uncertainty acquisition or plot panel.

The primary comparison is field reconstruction, not objective regret. The
first benchmark is diagnostic: it is intended to reveal weaknesses in the
current brute-force LOO and canonical uncertainty equation, not to establish
performance superiority. Analytic LOO, batch recommendation, and broad random
domain generation are intentionally deferred.
