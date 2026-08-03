# Output schema

Every evaluation suite writes to `outputs/<suite_name>/`. The target directory
is safely replaced before execution, so reruns overwrite the same suite and do
not create timestamped directories.

Summary mode uses this exact file layout:

```text
config_resolved.yaml
manifest.yaml
metrics/per_step.csv
metrics/final.csv
metrics/aggregate.csv
kernel/events.csv
kernel/candidate_scores.csv
figures/fields/<field>/process.gif
figures/fields/<field>/checkpoints.png
figures/fields/<field>/learning_curve.png
figures/fields/<field>/kernel_history.png
figures/global/aggregate_learning_curve.png
figures/global/performance_profile.png
figures/global/kernel_ablation.png
figures/global/robustness_matrix.png
report.md
```

`summary` contains official figures and scalar tables only. `diagnostic` adds
checkpoint arrays and complete candidate-score tables. `debug` retains all
intermediate arrays and optional animation frames. Temporary animation frames
are deleted after GIF construction unless the mode is `debug`.

The manifest records resolved settings, software versions, field definitions,
and deterministic seeds. `config_resolved.yaml` is the authoritative record of
the profile after defaults and aliases have been resolved.
