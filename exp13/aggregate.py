import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'src')]
import json
import numpy as np
from scipy.stats import t
from exp13.common import METHODS, output, table
from exp13.plotting import utility_plot
METRICS=('final_test_accuracy','best_test_accuracy','final_test_loss','best_test_loss','accuracy_auc')
from itertools import combinations
PAIRS = tuple((b,a) for a,b in combinations(METHODS,2))

def statistics(x):
    x=np.asarray(x); mean=float(x.mean()); std=float(x.std(ddof=1)); se=std/np.sqrt(len(x)); ci=float(t.ppf(.975,len(x)-1)*se)
    return dict(n=len(x),mean=mean,sample_std=std,se=float(se),ci95_low=mean-ci,ci95_high=mean+ci)

def aggregate():
    root=output(); records={(m,s):json.loads((root/'training'/f'{m}_seed{s}'/'summary.json').read_text()) for m in METHODS for s in range(10)}
    table(root/'runs',list(records.values()))
    rows=[dict(method=m,metric=k,**statistics([records[m,s][k] for s in range(10)])) for m in METHODS for k in METRICS]
    table(root/'aggregate',rows)
    paired=[dict(comparison=f'{a} - {b}',metric=k,**statistics([records[a,s][k]-records[b,s][k] for s in range(10)])) for a,b in PAIRS for k in METRICS]
    table(root/'paired_seed_differences',[dict(comparison=f'{a} - {b}', seed=s, metric=k, difference=records[a,s][k]-records[b,s][k]) for a,b in PAIRS for s in range(10) for k in METRICS])
    table(root/'paired_differences',paired); utility_plot(root,rows)
if __name__=='__main__': aggregate()
