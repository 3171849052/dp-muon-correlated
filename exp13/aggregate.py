import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'src')]
import json
import numpy as np
from scipy.stats import t
from exp13.common import METHODS, configuration, output, table
from exp13.plotting import utility_plot
METRICS=('final_test_accuracy','best_test_accuracy','final_test_loss','best_test_loss','accuracy_auc')
PAIRS = (('bandmf_single_m','iid_adam'),
         ('bandmf_ime_sep','iid_ime'),
         ('bandmf_ime_sep','bandmf_single_m'))

def statistics(x):
    x=np.asarray(x); mean=float(x.mean()); std=float(x.std(ddof=1)); se=std/np.sqrt(len(x)); ci=float(t.ppf(.975,len(x)-1)*se)
    return dict(n=len(x),mean=mean,sample_std=std,se=float(se),ci95_low=mean-ci,ci95_high=mean+ci)

def aggregate():
    root=output(); seeds=configuration().seeds
    expected=[(m,s) for m in METHODS for s in seeds]
    records={(m,s):json.loads((root/'training'/f'{m}_seed{s}'/'summary.json').read_text()) for m,s in expected}
    if len(records)!=len(METHODS)*len(seeds):
        raise ValueError(f'expected {len(METHODS)*len(seeds)} complete jobs; found {len(records)}')
    for method,seed in expected:
        record=records[method,seed]
        if record.get('method')!=method or record.get('seed')!=seed:
            raise ValueError(f'mismatched summary for {method}, seed {seed}')
    table(root/'runs',list(records.values()))
    rows=[dict(method=m,metric=k,**statistics([records[m,s][k] for s in seeds])) for m in METHODS for k in METRICS]
    table(root/'aggregate',rows)
    paired=[dict(comparison=f'{a} - {b}',metric=k,**statistics([records[a,s][k]-records[b,s][k] for s in seeds])) for a,b in PAIRS for k in METRICS]
    table(root/'paired_seed_differences',[dict(comparison=f'{a} - {b}', seed=s, metric=k, difference=records[a,s][k]-records[b,s][k]) for a,b in PAIRS for s in seeds for k in METRICS])
    table(root/'paired_differences',paired); utility_plot(root,rows)
if __name__=='__main__': aggregate()
