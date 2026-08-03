# KRISP-U benchmark report

The benchmark separates sampled-GP covariance recovery from deterministic reconstruction performance.

## Field summaries

| Field | Best final NRMSE | Best NRMSE AUC | Kernel sequence | Reselection n | Switch n | Final ARD scales | Near-neighbor rate |
|---|---|---|---|---|---|---|---|
| gp_axis_rescaled_anisotropic | krispu_adaptive | krispu_adaptive | matern_32_ard -> exponential_ard -> matern_32_ard -> exponential_ard -> matern_32_ard -> exponential_ard -> matern_32_ard -> exponential_ard -> gaussian_ard -> matern_32_ard -> exponential_ard -> gaussian_ard -> matern_32_ard -> exponential_ard -> gaussian_ard -> matern_32_ard -> exponential_ard -> gaussian_ard -> matern_32_ard -> exponential_ard -> gaussian_ard -> matern_32_ard -> exponential_ard -> gaussian_ard -> matern_32_ard -> exponential_ard | 10, 11, 13, 16, 17, 18, 6, 7, 9 | 13, 9 | 0.25706108781077214;0.5207340334645849 | 0.1538 |
| gp_rotated_anisotropic | krispu_adaptive | krispu_adaptive | matern_32_ard -> spherical_ard -> matern_32_ard -> spherical_ard -> matern_32_ard -> spherical_ard -> matern_32_ard -> spherical_ard -> matern_32_ard -> spherical_ard -> matern_32_ard -> wendland_c2_ard -> matern_32_ard -> gaussian_ard -> matern_32_ard -> gaussian_ard -> matern_32_ard -> gaussian_ard -> matern_32_ard -> gaussian_ard -> matern_32_ard | 10, 11, 12, 14, 15, 16, 18, 6, 7, 8, 9 | 14, 15, 9 | 0.07455857775773006;0.04429999285333965 | 0.2051 |

## Kernel-event narrative

### gp_axis_rescaled_anisotropic
n=6: reselection triggered by bound-contact trigger; family retained.
n=7: reselection triggered by score-degradation trigger; family retained.
n=9: reselection triggered by score-degradation trigger; family retained.
n=9: reselection triggered by score-degradation trigger; accepted switch.
n=10: reselection triggered by score-degradation trigger; family retained.
n=11: reselection triggered by score-degradation trigger; family retained.
n=13: reselection triggered by score-degradation trigger; accepted switch.
n=13: reselection triggered by score-degradation trigger; family retained.
n=16: reselection triggered by maximum-interval trigger; family retained.
n=17: reselection triggered by score-degradation trigger; family retained.
n=18: reselection triggered by maximum-interval trigger; family retained.

### gp_rotated_anisotropic
n=6: reselection triggered by bound-contact trigger; family retained.
n=6: reselection triggered by bound-contact trigger;score-degradation trigger; family retained.
n=6: reselection triggered by score-degradation trigger; family retained.
n=7: reselection triggered by bound-contact trigger; family retained.
n=8: reselection triggered by bound-contact trigger;score-degradation trigger; family retained.
n=8: reselection triggered by score-degradation trigger; family retained.
n=9: reselection triggered by bound-contact trigger;score-degradation trigger; accepted switch.
n=10: reselection triggered by score-degradation trigger; family retained.
n=10: reselection triggered by score-degradation trigger; family retained.
n=10: reselection triggered by bound-contact trigger; family retained.
n=11: reselection triggered by score-degradation trigger; family retained.
n=11: reselection triggered by bound-contact trigger; family retained.
n=12: reselection triggered by bound-contact trigger; family retained.
n=14: reselection triggered by score-degradation trigger; accepted switch.
n=15: reselection triggered by score-degradation trigger; accepted switch.
n=15: reselection triggered by maximum-interval trigger; accepted switch.
n=15: reselection triggered by score-degradation trigger; family retained.
n=16: reselection triggered by score-degradation trigger; family retained.
n=18: reselection triggered by score-degradation trigger; family retained.

## Scientific cautions

Deterministic development and canonical fields are response functions; this report does not assign them a true kernel. Synthetic recovery claims are restricted to fields whose metadata records an actual GP draw.

Major failure observations should be added from diagnostic runs; summary mode intentionally stores no per-step spatial arrays.
