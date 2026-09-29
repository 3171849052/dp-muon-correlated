import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'results_smoke/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def replay_plot(dest,rows):
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,metric in zip(axes,('raw_second_moment_mse','adam_direction_rmse','negative_fraction')):
        ax.bar([r['method'] for r in rows],[r[metric] for r in rows])
        ax.set_title(metric); ax.tick_params(axis='x',rotation=35)
    fig.tight_layout(); fig.savefig(dest/'metrics.png',dpi=160); plt.close(fig)

def utility_plot(dest,rows):
    fig,ax=plt.subplots(figsize=(8,4))
    selected=[r for r in rows if r['metric']=='final_test_accuracy']
    ax.bar([r['method'] for r in selected],[r['mean'] for r in selected],
        yerr=[r['ci95_high']-r['mean'] for r in selected],capsize=4)
    ax.set_ylabel('Final test accuracy (Student-t 95% CI)')
    fig.tight_layout(); fig.savefig(dest/'accuracy.png',dpi=160); plt.close(fig)
