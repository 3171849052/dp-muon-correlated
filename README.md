# dp-muon-correlated

Research scaffolding for a non-amplified, linear BandInvMF baseline that will
later support DP-Muon work. This stage contains no training loop, clipping,
privacy calibration, streaming noise, or Muon implementation.

The project intentionally uses the `jax_privacy` already installed in the
`curve` conda environment. It imports that package as a dependency and does
not fork or modify its source.

## Setup

Run commands from this directory with the existing environment:

```bash
conda run -n curve python scripts/check_jax_privacy.py
conda run -n curve pytest -q
```

The integration check reports the installed package metadata, source path, and
the actual signatures of the Toeplitz APIs it uses. It also runs a tiny
BandInvMF smoke test.

## Create a strategy artifact

The only supported public workload is the prefix-sum workload:

```bash
conda run -n curve python scripts/fit_bandinvmf.py \
  --horizon 16 --bandwidth 4 --min-sep 1 --max-optimizer-steps 20
```

This writes an `.npz` artifact to `artifacts/strategies/`. The artifact records
the workload, BandInvMF noising and strategy coefficients, sensitivity, and
objective. `noising_coef` always denotes coefficients of `C^{-1}` and
`strategy_coef` coefficients of `C`.
