# Stage K7: universal kriging design

Stage K7 is a planned extension and is intentionally not implemented in
v0.3.0. The current benchmark suite uses one global anisotropic residual
kernel at every fit.

The planned model is:

```text
global mean/trend component
+
one global anisotropic residual covariance kernel
+
optional observation nugget
```

Planned trend modes are `constant`, `linear`, and `quadratic`. Trend
coefficients should eventually be estimated with a universal-kriging or
generalized-least-squares formulation, using the same global residual kernel
for all observations. This is intended to address baseline drift and broad
global structure without introducing multiple spatial kernels or local scale
maps.

Implementation is deferred until the single-global-kernel adaptive system and
the benchmark suite pass. No pointwise kernels, local length-scale maps, or
multikernel model are part of this stage.
