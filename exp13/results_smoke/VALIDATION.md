# IME relu versus abs readout diagnostic

Only `exp13/` files were changed. GPU settings remain GPUS=(1,2,3), with
unchanged launcher bindings and worker partition logic. No full experiment was
launched. The 64-example, four-step smoke configuration is unchanged.

`abs` is DP post-processing of the same private moment state. Both readouts keep
v_raw = beta2*v_raw + (1-beta2)*private_q and its bias correction unchanged.
Only the Adam denominator reads either max(v_hat_raw,0) or abs(v_hat_raw).
Neither is written back into raw state. Query sensitivity, GDP split, noise
scales, learning rate, adam_eps, BandMF workload and fitting are unchanged.
Existing C_m/C_v artifacts were reused without refitting; SHA256 checks are in
`logs/readout_invariants.sha256`. Non-IME behavior remains unchanged.

Commands run with PYTHONDONTWRITEBYTECODE=1:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n curve python -B exp13/replay.py --smoke
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n curve python -B exp13/full_training.py --smoke
conda run -n curve pytest exp13/tests
```

## Frozen replay

All 5,501,002 coordinates, four steps, two paired draws. Raw second-moment MSE,
first-moment MSE and negative fractions are exactly equal for corresponding
relu/abs methods. Errors refer to bias-corrected moment readouts. Direction RMSE
compares with the clean Adam direction; direction magnitude statistics describe
the private direction itself. Quantiles are exact pooled coordinate quantiles.

| Metric | iid_ime | iid_ime_abs | bandmf_ime_sep | bandmf_ime_sep_abs |
|---|---:|---:|---:|---:|
| negative_fraction | 0.50009589 | 0.50009589 | 0.50004558 | 0.50004558 |
| raw_second_moment_mse | 0.029536351 | 0.029536351 | 0.031176162 | 0.031176162 |
| readout_second_moment_mse | 0.014767067 | 0.029536342 | 0.015586616 | 0.031176155 |
| first_moment_mse | 0.00789338 | 0.00789338 | 0.0079890115 | 0.0079890115 |
| adam_direction_rmse | 6282911.6 | 358.93424 | 6320710.9 | 1.0515289 |
| absolute_direction_mean | 3421631.1 | 0.34081712 | 3232324.8 | 0.27435534 |
| absolute_direction_median | 1972.9178 | 0.17692338 | 752.40973 | 0.16658121 |
| absolute_direction_p90 | 10828580 | 0.58199918 | 9963373 | 0.56086329 |
| absolute_direction_p99 | 22420671 | 1.8814538 | 25157987 | 1.8241382 |
| denominator_le_eps_1p01_fraction | 0.50009591 | 2.2723133e-08 | 0.50004558 | 0 |
| denominator_lt_1e6_fraction | 0.50009591 | 2.2723133e-08 | 0.50004558 | 0 |
| denominator_lt_1e4_fraction | 0.50009594 | 9.0892532e-08 | 0.50004558 | 2.2723133e-08 |

Denominator thresholds above are respectively <= 1.01e-8, < 1e-6, < 1e-4
(the fields `lt_1e6` and `lt_1e4` denote these negative-exponent thresholds).

| Mechanism | Draw | Direction RMSE abs/relu | Readout MSE abs/relu |
|---|---:|---:|---:|
| iid_ime | 0 | 8.0802731e-05 | 2.0003117 |
| iid_ime | 1 | 1.7072179e-07 | 1.9999874 |
| bandmf_ime_sep | 0 | 1.6447083e-07 | 2.0003226 |
| bandmf_ime_sep | 1 | 1.6823191e-07 | 2.0000525 |

Abs drastically reduces typical direction magnitude and near-epsilon denominator
frequency, even though its second-moment readout MSE is about twice relu's. The
IID abs replay still contains a rare near-zero denominator and has pooled RMSE
358.9 despite p99 magnitude 1.88; abs is not a guarantee against every tiny
positive denominator. No floor or additional hyperparameter was introduced.

## Actual paired training smoke

All seven methods completed the actual local pretrained ViT training smoke.

| Method | Train loss | Test loss | Test accuracy | Loss >= 1e10 |
|---|---:|---:|---:|---|
| iid_ime | 1.64480639e+11 | 1.40428012e+11 | 0.125000 | True |
| iid_ime_abs | 2.39447194 | 2.44603097 | 0.140625 | False |
| bandmf_ime_sep | 2.44251611e+11 | 2.31000465e+11 | 0.109375 | True |
| bandmf_ime_sep_abs | 2.29154831 | 2.47749859 | 0.125000 | False |

Both relu controls reproduce losses of order 1e11; both abs versions recover
losses around 2.3–2.5. Combined with identical frozen raw states/noise and the
replay denominator/direction diagnostics, this supports the smoke diagnosis:
negative v_hat_raw -> relu maps it to zero -> denominator becomes adam_eps
(not literally zero) -> very large Adam direction -> training instability.
The relu-to-zero denominator readout is supported as the primary explosion
source in this paired smoke. This is empirical diagnostic evidence, not a
theoretical proof, a full-experiment result, or a universal stability guarantee.

Logs: `logs/replay_readout.log`, `logs/training_readout.log`,
`logs/pytest_readout.log`. Detailed per-draw ratios and pooled metrics are in
`replay/`; training comparisons are `readout_training_comparison.{csv,json}`.

## Final verification

`conda run -n curve pytest exp13/tests`: **28 passed, 0 skipped**, 15.80 seconds.
Three warnings come from the existing jax_privacy default strategy initializer.
Tests cover abs/relu readout identities, non-feedback into recurrence, exact
paired raw states/noise keys, identical calibration, replay ratios/thresholds,
all seven real training artifacts and paired training configuration metadata.

`config.yaml`, `workloads.py`, `fit_strategies.py` and both saved strategies passed
before/after SHA256 checks. `launch_full.sh` and `worker.py` retain their exact
original hashes; common.GPUS and jobs() are unchanged. No file outside `exp13/`
was changed, and no formal full experiment was run.
