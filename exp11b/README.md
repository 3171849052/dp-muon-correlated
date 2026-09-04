# Exp11b

Exp11b extends Exp11 to three same-type Muon matrices:

- `blocks/0/attention/query/kernel`
- `blocks/5/attention/query/kernel`
- `blocks/11/attention/query/kernel`

It runs one real clipped clean trajectory and two real IID DP-Muon
trajectories, using the same pretrained parameters, fixed-cycle schedule,
clipping, optimizer settings, `delta=1e-5`, and data order. The DP trajectories
share the IID standard-normal realization and differ only in the calibrated
scale for `epsilon=3` versus `epsilon=8`. The pre-Q hook runs after Muon
Nesterov and before Newton--Schulz/Q, retaining only exact singular values.

Run the checks and smoke experiment with:

```bash
conda run -n curve pytest exp11b/tests -q
conda run -n curve python exp11b/run.py --smoke --output-dir /tmp/exp11b-smoke
```

Run the formal experiment with:

```bash
conda run -n curve python exp11b/run.py \
  --config config/cifar10_dpmuon.yaml \
  --output-dir exp11b/results
```

`spectra.npz` is the source artifact for `spectra.csv` and both plots. Its
arrays are `epsilons [2]`, `steps [3]`, `layers [3]`, and clean/DP spectra with
shape `[epsilon, step, layer, singular-index]`. The CSV header is:
`epsilon,step,layer,index,clean,dp`.
