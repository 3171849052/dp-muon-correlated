#!/usr/bin/env python3
"""Aggregate all 60 completed runs and paired seed differences."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import json
import numpy as np
from scipy.stats import t
from exp_adam_workload.core import HERE, NAMES, SEEDS, write_table
from exp_adam_workload.plotting import bars

METRICS = ('final_train_loss', 'final_test_loss', 'final_test_accuracy',
           'best_test_loss', 'best_test_accuracy', 'calibrated_iid_noise_std',
           'sensitivity_squared', 'mean_marginal_correlated_noise_variance')


def stats(values):
    x = np.asarray(values)
    mean, std = float(x.mean()), float(x.std(ddof=1))
    se = std / np.sqrt(len(x))
    half = float(t.ppf(.975, len(x)-1) * se)
    return dict(n=len(x), mean=mean, std=std, se=float(se),
                ci95_low=mean-half, ci95_high=mean+half)


def aggregate(output):
    records = {}
    for name in NAMES:
        for seed in SEEDS:
            record = json.loads((output / 'runs' / name / f'seed{seed}' / 'result.json').read_text())
            assert record['strategy'] == name and record['seed'] == seed
            records[name,seed] = record
    write_table(output, 'utility_runs', [dict(strategy=n, seed=s,
        **{m: records[n,s][m] for m in METRICS}) for n in NAMES for s in SEEDS])
    rows = [dict(strategy=n, metric=m, **stats([records[n,s][m] for s in SEEDS]))
            for n in NAMES for m in METRICS]
    write_table(output, 'utility_summary', rows)
    paired = [dict(strategy_a=a, strategy_b=b, metric=m,
        **stats([records[a,s][m]-records[b,s][m] for s in SEEDS]))
        for i,a in enumerate(NAMES) for b in NAMES[i+1:] for m in METRICS]
    write_table(output, 'utility_paired_differences', paired)
    for metric in METRICS[:5]:
        subset = [r for r in rows if r['metric']==metric]
        bars([r['mean'] for r in subset], [r['ci95_high']-r['mean'] for r in subset],
             output / f'utility_{metric}.png', f'{metric}: mean and 95% t CI (10 paired seeds)')


if __name__ == '__main__':
    aggregate(HERE / 'results')
