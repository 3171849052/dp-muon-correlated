# Experiment 2: full AdamW correlated-noise cancellation

Exp2 freezes one clean CIFAR-10 AdamW trajectory at `g_t`, after the existing
clipping query and before the AdamW update.  `start_step=0` and a fixed learning
rate are required.  The collector stores the AdamW public parameters (`beta1`,
`beta2`, `eps`, `weight_decay`, and `learning_rate`) alongside `g`.

The two covariance designs are deliberately different:

* `decayed-prefix` uses the existing Toeplitz workload
  `decayed_prefix_sum_workload_coef`, i.e. `A_0=P_rho` with
  `rho=1-learning_rate*weight_decay`.
* `adam-m-aware` uses the existing `adam_first_moment_workload_matrix`, i.e.
  `A_m=learning_rate*P_rho*H_beta1^m`.  It is fitted through the general causal
  BandInvMF path and is stored with a dense workload matrix.

Neither design changes clipping, privacy accounting, the noise sampler, or the
training optimizer.  Replay runs clean and noisy gradients through the complete
nonlinear AdamW moments and bias corrections, then applies
`delta_theta_t=rho*delta_theta_(t-1)-learning_rate*Delta q_t`.  It reports
`J_k`, cumulative `D_k`, `R_k`, and aggregate `R=sum_k J_k/sum_k D_k`.
`delta_R=R_adam-m-aware-R_decayed-prefix`; negative means improved
cancellation.  An optional `R_linear` for `adam-m-aware` is the same recurrence
with the first-moment linear direction, and is only a reference.

## Run

```bash
python exp2/collect_trajectory.py --config config/cifar10_dpadamw.yaml \
  --steps 64 --output exp2/trajectory.npz
python exp2/run.py --config exp2/config.yaml
```

`run.py` uses 1,000 samples, seed 0, and targets `[0.01, 0.1, 1.0]` by
default.  A target receives one deterministic global scalar for each strategy;
there is no sample- or step-dependent rescaling.  The same latent Gaussian
draws are passed to both strategies in paired batches.  Outputs are
`results/results.csv`, `results/prefix_results.csv`, and `results/summary.json`.
