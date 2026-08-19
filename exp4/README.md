# Experiment 4

4A reads the post-update `nu` field from the real Optax AdamW state and computes
`p_t = 1 / (sqrt(nu / (1-beta2**t)) + eps)` after every optimizer step. Only
summary statistics are written; the previous flattened vector is retained in
memory solely for `relative_change`.

4B fits one BandInvMF strategy per contiguous block. At a boundary only the FIR
latent-noise ring buffer is cleared; model parameters, AdamW `count`, `mu`, and
`nu` continue unchanged. The blocks are one block-diagonal Gaussian transcript,
so calibration uses `sum_i sensitivity_squared_i` before the single GDP
calibration to the requested `(epsilon, delta)`. No per-block privacy budget or
sampling amplification is introduced.

Validation command:

```bash
conda activate curve
python exp4/run.py --config config/cifar10_bandinv_dpadamw_naive.yaml --output-dir exp4/results --seeds 0 1 2 --smoke
```
