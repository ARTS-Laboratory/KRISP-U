# v0.3.0 algorithm

Let `X` and `R` be measured and reference coordinates. Continuous coordinates
are normalized with the declared domain bounds:

\[
x'_j = (x_j-l_j)/(u_j-l_j).
\]

The same transform is used for observations, candidates, references, and all
kernel distances. Discrete coordinates are validated against their finite
option sets; candidate generation never optimizes through an invalid option.

Responses use the affine transform `y'=(y-y_mean)/s_y`, where `s_y=1` for a
constant response. Predictions and standard deviations are returned in
response units.

The surrogate is `ConstantKernel × Matérn-3/2 ARD`, with one length scale per
coordinate. Length scales are bounded to `[0.02, 2.0]` in normalized
coordinates by default. A small fixed `alpha` is used in deterministic mode;
no freely fitted white-noise term is added in that mode.

The complete fit optimizes hyperparameters once. Every eligible LOO fold uses a
clone of the fitted kernel with `optimizer=None`, so differences between folds
represent the influence of removing a measurement rather than a new
kernel-selection decision. Protected observations remain in every fold.

The brute-force implementation stores the complete matrix
`F_LOO.shape == (n_reference, n_loo)`. It never attaches one scalar to each
removed point or interpolates those scalars into an acquisition field.

The default recommendation is the valid candidate maximizing the combined
uncertainty described in `uncertainty.md`.
