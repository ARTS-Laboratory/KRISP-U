# Researcher CSV workflow

Install from `versions/v0.3.0/`, then run:

```bash
krispu-recommend measurements.csv \
  --target response \
  --features x y \
  --bounds x:-1:1 y:-1:1 \
  --n-recommendations 3 \
  --output recommendation_outputs/recommendations.csv
```

Rows with a finite target are measured observations. Rows with a blank target
are optional user-supplied candidates. If no blank-target rows are present,
the command generates a Latin-hypercube candidate pool inside the declared
domain. The output contains coordinates, predicted mean, posterior standard
deviation, jackknife uncertainty, calibrated posterior uncertainty, combined
score, rank, and distance to the nearest measured point.
