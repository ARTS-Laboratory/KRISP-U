# v0.3.0 benchmark protocol

Run the audit from `versions/v0.3.0` with:

```bash
python -m evaluation.runners.suite --config configs/suites/paired_smoke.yaml
python -m evaluation.runners.suite --config configs/suites/visual_audit.yaml
```

Each run writes `outputs/<experiment_name>/` and overwrites that exact suite
directory. Summary mode includes the resolved YAML configuration, YAML
manifest, scalar CSV records, and the official figures. Diagnostic and debug
profiles add compressed spatial arrays and optional frames. The manifest records package and Python versions,
the available Git commit, GPR settings, field metadata, grid and candidate
settings, and all field, initial-design, candidate, and method seeds.

The audit fields are `smooth`, `localized`, `anisotropic`, `rough_correlated`,
and `rough_multiscale`, all on `[-1, 1]^2`. The exact methods are
`raw_jackknife_sensitivity`, `support_adjusted_krispu`, `posterior_std`, `random`,
`lhs`, and `maximin`. All methods in a paired trial share the hidden field, domain, five
initial interior-maximin points, initial responses, candidate
pool, evaluation grid, and final measurement budget.

Field metrics are RMSE, NRMSE = RMSE / (max(true) - min(true)), MAE,
NMAE = MAE / (max(true) - min(true)), R², p95 absolute error, and maximum
absolute error. A numerically constant true field raises an explicit error.
Large arrays are stored in NPZ files rather than CSV cells. Missing jackknife values
for baseline methods are left missing; they are never replaced by zero.

The primary 2×2 panels and GIFs show true field, current reconstruction,
KRISP-U uncertainty, and absolute reconstruction error. A separate diagnostic
figure shows buffered-jackknife field sensitivity, kernel support deficit, final KRISP-U
uncertainty, and absolute error with observations and the proposed point
overlaid. Reports include final NRMSE, NRMSE AUC, median nearest-neighbor
distance, the fraction of selections within `0.05` normalized distance, and
the fraction with kernel correlation above `0.95`.

The primary comparison is field reconstruction, not objective regret. The
first benchmark is diagnostic: it is intended to reveal weaknesses in the
current buffered-jackknife and canonical uncertainty equation, not to establish
performance superiority. Batch recommendation and broad random
domain generation are intentionally deferred.
## Pass 3 study separation

`kernel_recovery.yaml` is the only covariance-recovery study. Its fields are
finite draws from the registered Gaussian-process families and retain family,
amplitude, ARD scales, nugget, transformation, and seed metadata. Recovery
tables report family frequency, scale error, bound contact, reselection, and
switch behavior.

`development.yaml`, `canonical_2d.yaml`, `canonical_doe.yaml`, and
`noise_robustness.yaml` are reconstruction-performance studies. Their
deterministic response functions deliberately have no generating-kernel claim.
Two-dimensional fields receive the four compact field figures; higher-
dimensional DOE problems contribute scalar learning curves and tables only.

The summary output is overwrite-only and contains exactly four global figures:
`aggregate_learning_curve.png`, `performance_profile.png`,
`kernel_ablation.png`, and `robustness_matrix.png`. Adaptive runs serialize one
event row per sampling step in `kernel/events.csv`; candidate rows are written
only for steps at which a full reselection was evaluated.
