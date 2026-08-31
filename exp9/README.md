# Experiment 9: Muon nonlinear cancellation decomposition

Experiment 9 runs one real correlated DP-Muon trajectory per seed. The
clipped-clean gradient is retained in the train state and drives independent
shadow diagnostics on Muon-labelled rank-two leaves only.

The primary float32 paths are:

* `P0`: linear classic-Nesterov noise response;
* `P1`: JVP of the production-equivalent smooth Muon Q map;
* `P2`: antithetic odd response;
* `P3`: raw private-clean response minus an independent antithetic output-bias estimate.

`G_C` and `G_J` are computed from exact raw-step endpoints. Bias and raw
private-clean gaps are reported separately and are not cancellation metrics.
The IID branch matches the correlated branch's raw gradient-noise marginal at
each step and is a mechanism control, not a formal same-DP baseline.

Smoke check:

```bash
conda run -n curve python exp9/run.py --smoke --output-dir /tmp/exp9-smoke --seeds 0
```

Formal run:

```bash
conda run -n curve python exp9/run.py \
  --config config/cifar10_bandinv_dpmuon_naive.yaml \
  --output-dir exp9/results \
  --seeds 0 1 2
```
