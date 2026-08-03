# Global anisotropic kernel selection

At every sample count there is exactly one selected spatial covariance family
and one global ARD vector. The active standard registry is:

```text
gaussian_ard
exponential_ard
spherical_ard
matern_32_ard
matern_52_ard
rational_quadratic_ard
wendland_c2_ard
```

Gaussian is the displayed replacement for RBF, and exponential is the
displayed replacement for Matérn-1/2. They are not scored as duplicate aliases.
The rational-quadratic family uses a scalar shape parameter plus a genuine
per-axis ARD scale vector. Spherical and Wendland C2 are restricted to the
dimensions in which their covariance formulations are positive definite.

An observation-noise nugget may be added as a separate diagonal noise
parameter. It is not a second spatial covariance component. Additive and
multiplicative manual kernel specifications are rejected.

The removed composite families are:

```text
matern_52_long_plus_matern_12_short
matern_32_long_plus_matern_12_short
matern_52_long_plus_matern_32_short
rbf_long_plus_matern_12_short
linear_plus_matern_32
linear_plus_matern_52
periodic_times_matern_32
periodic_plus_matern_32
```

## Optimization and reselection

`kernel.optimization.every_step: true` optimizes the current family at every
step after the minimum fit size. `kernel.optimization.restarts` controls
optimizer restarts. Reselection is triggered by the small configured set of
fit, validation, conditioning, bound-contact, degradation, and interval
conditions. A triggered check scores every permitted family using one shared
buffered-jackknife plan.

The primary score is buffered predictive log score. Upper-tail normalized
absolute error is the tie-breaker. The current family may be retained when a
challenger does not exceed `minimum_switch_improvement`.

## YAML

```yaml
jackknife:
  buffer:
    mode: median_nearest_neighbor
    multiplier: 1.0
    minimum_radius: 0.025
    maximum_radius: 0.20
    minimum_training_points: 3

kernel:
  optimization:
    every_step: true
    restarts: 2
  reselection:
    minimum_points: 6
    maximum_interval: 5
    score_degradation_fraction: 0.10
    bound_proximity_fraction: 0.05
    bound_contact_steps: 2
    minimum_switch_improvement: 0.05
```

Every event records whether hyperparameters were optimized, whether
reselection was triggered, its reasons, candidates evaluated, previous and
selected family IDs, current and challenger scores, score improvement,
per-axis current/minimum/maximum scales, and runtimes. The explicit records
are `KernelOptimizationEvent`, `KernelReselectionEvent`, and
`KernelSwitchEvent`.

The public diagnostic renames are `jackknife_eligible`,
`jackknife_field_sensitivity`, `jackknife_field_means`,
`jackknife_residuals`, `jackknife_standardized_residuals`, and
`BufferedJackknifeResult`. The former pointwise-deletion names are not active
v0.3.0 API fields.
