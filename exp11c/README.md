# Exp11c

Exp11c inherits Exp11b's three strictly paired closed-loop trajectories:

- `clean`: clipped query, without Gaussian noise;
- `eps3`: IID DP-Muon with the calibrated epsilon=3 scale;
- `eps8`: IID DP-Muon with the calibrated epsilon=8 scale.

All three use the same pretrained initialization, fixed-cycle schedule, data
order, clipping, `delta=1e-5`, and Muon/AdamW settings. The two DP states start
from the same RNG key, so they use the same standard Gaussian realization and
only differ in calibrated scale.

At steps 32, 244, and 480, a matrix-only optimizer hook reads the actual
post-Nesterov/pre-Newton--Schulz target matrices. It does not reconstruct a
gradient after the fact. The matrices are copied to host only at requested
steps, immediately reduced with float64 SVD,

```python
u, _, vh = np.linalg.svd(x.astype(np.float64), full_matrices=False)
q = u @ vh
```

Only matrix norms and pairwise ideal-Q distances/cosines are written. The full
pre-Q and ideal-Q matrices are never placed in an output artifact.

Run the checks and smoke experiment with:

```bash
conda run -n curve pytest exp11c/tests -q
conda run -n curve python exp11c/run.py --smoke --output-dir /tmp/exp11c-smoke
```

The output contains `scale_blindness.npz`, `scale_blindness.csv`, and
`scale_blindness.png`. For two matrices differing only by a positive scalar,
ideal Muon predicts zero pairwise Q distance and cosine similarity one.
