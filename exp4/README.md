# Experiment 4

4A reads the post-update `nu` field from the real Optax AdamW state and computes
`p_t = 1 / (sqrt(nu / (1-beta2**t)) + eps)` after every optimizer step. Only
summary statistics are written; the previous flattened vector is retained in
memory solely for `relative_change`.

4B fits the same decayed-prefix workload as the continuous baseline separately
for every contiguous block. At a boundary only the FIR latent-noise ring buffer
and its block-local step are cleared; model parameters, the global training
step, and AdamW `count`, `mu`, and `nu` continue unchanged.

The blocks form one block-diagonal Gaussian transcript. Its sensitivity is
computed exactly under the original full-horizon `min_sep` and
`max_participations` contract. Since every Exp4 block is no longer than
`min_sep`, a record can occur at most once in a block. Dynamic programming over
blocks retains the last global participation position and participation count,
tries every legal strategy-matrix column, and maximizes the accumulated squared
L2 energy. The resulting global maximum is passed once to the existing GDP
calibration. Blocks do not receive separate privacy budgets and no sampling
amplification is used.

Validation command:

```bash
conda activate curve
python exp4/run.py --config config/cifar10_bandinv_dpadamw_naive.yaml --output-dir exp4/results --seeds 0 1 2 --smoke
```
