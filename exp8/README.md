# Experiment 8: layer-by-layer BandInvMF cancellation

Exp8 runs one real correlated DP-AdamW baseline trajectory per seed. At each
step it evaluates the clipped gradient once, lets the real update consume its
own BandInvMF noise stream, and maintains shadow-only correlated and
matched-marginal IID branches from the same standard-normal innovation.

The diagnostic paths are:

* P0: momentum noise response;
* P1: clean Adam preconditioner;
* P2: deterministic mean-square bias in the second moment;
* P3: full private noisy preconditioner.

The resulting `G0 -> Gc -> Gphi -> Gp` comparison identifies the layer where
cross-iteration cancellation is lost. The matched-IID branch is a conditional
mechanism control for this comparison; it is not a new DP algorithm and does
not claim the same formal privacy guarantee as BandInvMF.

Run only the requested small check with `--smoke`. The formal run is:

```bash
conda run -n curve python exp8/run.py \
  --config config/cifar10_bandinv_dpadamw_naive.yaml \
  --output-dir exp8/results \
  --seeds 0 1 2
```

