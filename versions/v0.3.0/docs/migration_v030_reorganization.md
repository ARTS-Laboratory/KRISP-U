# v0.3.0 reorganization note

The former `benchmarks` namespace was active evaluation code, so its reusable
fields, metrics, methods, plotting, records, and runners were promoted into
`evaluation`. The old namespace and timestamp-oriented output conventions are
not compatibility surfaces.

Generated directories from earlier local audits are disposable artifacts, not
scientific inputs. They are intentionally not migrated into the new layout;
new runs use `outputs/<suite_name>/` only. The obsolete sequential algorithm
module was removed from `src/krispu`; callers should use the evaluation runner
when reproducing a benchmark and `krispu.api` for user-facing reconstruction.
