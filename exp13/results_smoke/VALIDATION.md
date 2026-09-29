# Exp13 smoke validation

The requested smoke contract remains 64 training examples, 64 test examples,
batch and microbatch size 16, one epoch, four optimizer steps, five strategy-fit
steps, and two replay draws. Smoke artifacts were regenerated after replacing
the separate legacy readout variants with one fixed IME abs readout.

## Training

All five methods produced finite summaries. The IME runs stayed below the
`1e10` loss diagnostic threshold.

| Method | Test loss | Test accuracy |
|---|---:|---:|
| `nonprivate_adam` | 6.169782 | 0.046875 |
| `iid_adam` | 2.426233 | 0.140625 |
| `bandmf_single_m` | 2.339809 | 0.171875 |
| `iid_ime` | 2.446308 | 0.140625 |
| `bandmf_ime_sep` | 2.477137 | 0.125000 |

## Raw moment-state and workload checks

The reported state MSEs are before bias correction. The theoretical and
empirical columns use the same EMA workload and the installed
`banded.per_query_error` objective.

| Method | Channel | Theoretical | Empirical | Empirical / theoretical |
|---|---|---:|---:|---:|
| `iid_ime` | first | 3.124835e-4 | 3.123975e-4 | 0.999725 |
| `iid_ime` | second | 1.414965e-7 | 1.414968e-7 | 1.000002 |
| `bandmf_ime_sep` | first | 2.274146e-4 | 2.273827e-4 | 0.999860 |
| `bandmf_ime_sep` | second | 9.731780e-8 | 9.732988e-8 | 1.000124 |

| Method | Negative fraction | Denominator min | p001 | p01 | Direction p99 | p999 | Max | Direction RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `iid_ime` | 0.500096 | 1.000000e-8 | 0.013925 | 0.044018 | 1.881453 | 5.958228 | 2.381106e6 | 358.934245 |
| `bandmf_ime_sep` | 0.500046 | 6.969159e-5 | 0.012668 | 0.039980 | 1.824142 | 5.770193 | 1206.038574 | 1.051529 |

The IID replay contains a rare denominator at `adam_eps` in step 4, producing a
large direction maximum despite ordinary pooled p99/p999 values. The per-step
table records this directly; abs readout removes the half-zero-denominator
failure mode but does not eliminate all near-zero risk.

## Strategy cross-evaluation

| Workload | Strategy | Objective |
|---|---|---:|
| first moment | `C_m` | 0.0150531661 |
| first moment | `C_v` | 0.0151106473 |
| second moment | `C_m` | 1.7225136162e-6 |
| second moment | `C_v` | 1.7160047037e-6 |

`second_moment_specialization_gain` is 0.0037787292. Full per-step denominator
and direction diagnostics are in `replay/per_step_diagnostics.{csv,json}`.

## Configuration and tests

Formal seeds are `[0, 1, 2]`, giving 15 jobs. The ordered partition assigns five
jobs each to GPUs 1, 2, and 3. `GPUS=(1,2,3)`, the config GPU IDs, launcher
bindings, worker GPU handling, and three-worker partition are unchanged.

`conda run -n curve pytest exp13/tests`: **29 passed**. The five warnings come
from the installed `jax_privacy` one-band strategy initializer and do not fail
the workload, fit, replay, training-summary, or privacy-calibration checks.
