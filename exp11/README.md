# Exp11: real Muon pre-Q spectra

Exp11 runs two strictly paired CIFAR-10 fine-tuning trajectories using the
existing IID DP-Muon trainer:

- `Clean` uses the same per-example global clipping and the same Muon/AdamW
  update, but skips Gaussian noise sampling.
- `IID DP` uses the same clipped query and the existing IID Gaussian mechanism.

The trajectories load one pretrained snapshot, consume one shared fixed-cycle
batch schedule, and use the same optimizer settings. The only recorded
parameter is `blocks/0/attention/query/kernel`. Its spectrum is captured from
the optimizer's post-Nesterov, pre-Newton--Schulz value at steps 32, 244, and
480. Only singular values are written to `results/spectra.npz`.

Run the small end-to-end check with:

```bash
conda run -n curve python exp11/run.py --smoke --output-dir /tmp/exp11-smoke
```

Run the full experiment with:

```bash
conda run -n curve python exp11/run.py \
  --config config/cifar10_dpmuon.yaml \
  --output-dir exp11/results
```

The formal output contains `spectra.npz` and `singular_spectra.png`. The plot
has three shared-y log-scale panels and exactly two curves per panel.
