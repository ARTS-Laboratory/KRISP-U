# KRISP-U uncertainty fields

The universal uncertainty backend is buffered jackknife. For each eligible
anchor (i), the fold removes the anchor and all observations within its
normalized buffer radius. Nearby noneligible observations are removed too.
The complete-fit kernel hyperparameters remain fixed in every fold.

If (F_i(x)) is the reference-field prediction from fold (i), the field
ensemble mean and sensitivity are

```text
J(x) = mean_i F_i(x)
S_J(x) = sqrt((m - 1)/m * sum_i (F_i(x) - J(x))^2)
```

The universal buffer radius is

```text
clip(multiplier * median(nonzero nearest-neighbor distance),
     minimum_radius, maximum_radius)
```

All folds store their removed indices, effective radius, removed count, and
training count in `BufferedJackknifePlan`. This makes fold construction
reproducible and prevents pointwise deletion from hiding unsupported regions
inside dense observation clusters.

The fitted latent-process kernel defines support deficit (C(x)). The
canonical KRISP-U field is

```text
U_KRISPU(x) = S_J(x) * sqrt(C(x))
```

Posterior standard deviation, calibrated posterior standard deviation, and
held-out standardized residuals remain diagnostics. None is introduced into
the canonical product.
