from pathlib import Path
from types import SimpleNamespace
import csv
import json
import os
import numpy as np
import yaml
from dp_muon.data import load_cifar10
from dp_muon.training.cifar10_experiment import derive_fixed_cycle_participation
from dp_muon.training.cifar10_driver import build_fixed_cycle_logical_schedule

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'exp13'
METHODS = ('nonprivate_adam', 'iid_adam', 'bandmf_single_m', 'iid_ime', 'bandmf_ime_sep_raw', 'bandmf_ime_sep_vbc', 'bandmf_ime_sep_bc')
GPUS = (1, 2, 3)
SMOKE_EXAMPLES = 64

def configuration(smoke=False):
    config_path = Path(os.environ.get('EXP13_CONFIG', HERE / 'config.yaml'))
    c = yaml.safe_load(config_path.read_text())
    if smoke:
        c.update(epochs=1, batch_size=16, microbatch_size=16, fit_steps=5, replay_draws=2)
        smoke_contract = derive_fixed_cycle_participation(
            SMOKE_EXAMPLES, c['epochs'], c['batch_size'])
        c['bandwidth'] = min(c['bandwidth'], smoke_contract.min_sep)
    return SimpleNamespace(**c)

def dataset(c, smoke=False, train=True):
    x, y = load_cifar10(ROOT / c.data_dir, train=train, download=False)
    if smoke:
        x, y = x[:SMOKE_EXAMPLES], y[:SMOKE_EXAMPLES]
    return x, y

def contract(c, smoke=False):
    x, _ = dataset(c, smoke)
    return derive_fixed_cycle_participation(len(x), c.epochs, c.batch_size)

def schedule(c, p, n, seed):
    return build_fixed_cycle_logical_schedule(num_examples=n, batch_size=c.batch_size,
        horizon=p.horizon, min_sep=p.min_sep, max_participations=p.max_participations, seed=seed)

def output(smoke=False, stage="stage1_numerical"):
    path = HERE / ('results_smoke' if smoke else 'results')
    path = path / stage
    path.mkdir(parents=True, exist_ok=True)
    return path

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def table(path, rows):
    write_json(path.with_suffix('.json'), rows)
    with path.with_suffix('.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)

def jobs(gpu, gpus=GPUS):
    rank = gpus.index(gpu)
    seeds = configuration().seeds
    ordered = [(method, seed) for method in METHODS for seed in seeds]
    return [(method, seed) for i, (method, seed) in enumerate(ordered) if i % len(gpus) == rank]


def config_hash(c, p):
    import hashlib
    payload = dict(config=vars(c), participation=vars(p))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
