# Development validation

The installed curve environment reports jax_privacy 2.3.0.dev0. Actual API
signatures are recorded in `logs/installed_api.txt`. No datasets or checkpoint
were downloaded. Only development smoke workloads were launched.

Commands executed from the repository root, with PYTHONDONTWRITEBYTECODE=1:

- GPU 1: `conda run --no-capture-output -n curve python -B exp13/fit_strategies.py --smoke`
- GPU 3: `conda run --no-capture-output -n curve python -B exp13/replay.py --smoke`
- GPU 2: `conda run --no-capture-output -n curve python -B exp13/full_training.py --smoke`
- CPU: `conda run --no-capture-output -n curve pytest exp13/tests`
- `bash -n exp13/launch_full.sh`

## Cross-evaluation (four-step smoke)

| Workload | C_m | C_v |
|---|---:|---:|
| H_beta1 | 0.0150531661 | 0.0151106473 |
| H_beta2 | 1.722513616e-6 | 1.716004704e-6 |

Second-moment specialization gain is 0.0037787292 (0.377873%). These are small
smoke fits, not five-epoch/ten-seed experiment estimates.

Replay covers all 5,501,002 parameter coordinates, four frozen clean-gradient
steps and two independent paired draws. Both IME methods have approximately
50% negative v_hat_raw coordinates. Direction RMSE is about 6.28e6 (IID) and
6.32e6 (separate BandMF). See `replay/metrics.json` for all required moment MSEs.

The specified projection/readout produces a denominator of 1e-8 for negative
v_hat_raw. Consequently IME can make extremely large updates. Training smoke
completion establishes execution, not stability or useful utility. No clamp on
the private second query, recurrence projection, or epsilon adjustment was added.

The fitting library emits a divide-by-zero warning in its default initializer
before replacing its first coefficient; fitted strategies and validation outputs
are finite. The local CIFAR loader emits a NumPy pickle deprecation warning.

## Actual training smoke results

| Method | Test accuracy | Test loss |
|---|---:|---:|
| bandmf_ime_sep | 0.109375 | 2.31e+11 |
| bandmf_single_m | 0.171875 | 2.33972 |
| iid_adam | 0.140625 | 2.42646 |
| iid_ime | 0.125000 | 1.40428e+11 |
| nonprivate_adam | 0.046875 | 6.16225 |

Final full test run: **19 passed, 0 skipped**, three upstream initializer warnings,
14.16 seconds. All five actual training smoke runs completed with finite loss.
The final suite also verifies paired initialization/schedules and all-pair
aggregation, including rejection of missing run summaries.

No files outside `exp13/` were edited by this implementation. The pre-existing
`exp12/results/` working-tree changes were left untouched. The formal ten-seed
experiment was not launched.
