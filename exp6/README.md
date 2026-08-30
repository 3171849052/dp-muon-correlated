# Experiment 6: local BandInvMF cancellation after AdamW preconditioning

Experiment 6 keeps the continuous CIFAR-10 setup from Experiment 3: one
clipped gradient, one continuous `decayed-prefix` BandInvMF noise stream, and
one real DP-AdamW update at every step.  The default seeds are `0 1 2`.

The added code consumes the moments already maintained by
`exp3.online_shadow`; it does not call the model, optimizer, privacy
mechanism, or RNG a second time.  Four device-side shadow paths are updated:

* `C_momentum`: `x_t = -eta r_t^lin`;
* `C_frozen_p`: `x_t = -eta (p_s * r_t^lin)`, with `p_s` fixed at the first
  step of the 16-step window;
* `C_dynamic_clean_p`: `x_t = -eta (p_t^clean * r_t^lin)`;
* `C_real_adamw`: `x_t = -eta (q_t^DP - q_t^clean)`.

All four use `y_t = (1 - eta lambda)y_{t-1} + x_t`, so the final local score
is exactly the requested exponentially weighted squared norm ratio.  Windows
are one-based and non-overlapping: `[1,16], [17,32], ...`; a short final
window is retained.  `delta_p_cancellation` is
`C_dynamic_clean_p - C_frozen_p`, and `extra_real_effect` is
`C_real_adamw - C_dynamic_clean_p`.  Since the first training step has no
defined `p_0`, its `delta_p` contribution is recorded as zero, matching the
first-step convention used by Experiment 4.

Run the complete experiment with:

```bash
source /home/longt29/miniconda3/etc/profile.d/conda.sh
conda activate curve
python exp6/run.py \
  --config config/cifar10_bandinv_dpadamw_naive.yaml \
  --output-dir exp6/results \
  --seeds 0 1 2
```

Do not pass `--smoke` for the complete experiment.  A small path-only smoke
run is:

```bash
python exp6/run.py --smoke --output-dir /tmp/exp6-smoke --seeds 0
```

Outputs include `window_diagnostics_seed*.csv`, `window_summary.csv`,
`summary.json`, `diagnostics_over_steps.png`, and
`p_change_vs_cancellation.png`.  Stage means use the actual step overlap of a
window with steps 1--97 and 98--488, so a boundary-crossing window is split
by its covered step counts rather than assigned wholesale to one stage.
