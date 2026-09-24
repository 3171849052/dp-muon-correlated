"""Six public workloads, exact mechanism controls, and shared experiment I/O."""
from dataclasses import asdict, replace
import csv
import json
from pathlib import Path

import jax
import numpy as np
from dp_muon.optim import (adam_first_moment_workload_matrix,
    decayed_prefix_sum_workload_coef, momentum_trajectory_workload_matrix)
from exp2.common import derive_contract, load_config_and_contract
from exp2.full_training import calibration_metadata
from exp2.run import _lower_toeplitz
from dp_muon.training.cifar10_bandinv_dpadamw_experiment import load_cifar10_bandinv_dpadamw_config

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'exp_adam_workload'
CONFIG = ROOT / 'config/cifar10_bandinv_dpadamw_naive.yaml'
NAMES = ('sgd', 'sgd_wd', 'momentum', 'momentum_wd', 'adam_momentum', 'adam_momentum_wd')
SEEDS = tuple(range(10))
ADAM_FIELDS = ('learning_rate', 'beta1', 'beta2', 'eps', 'weight_decay')


def configuration(smoke=False):
    if smoke:
        config = replace(load_cifar10_bandinv_dpadamw_config(CONFIG),
                         epochs=2, batch_size=4, max_optimizer_steps=5)
        return config, derive_contract(config, num_examples=20)
    return load_config_and_contract(CONFIG)


def workload(name, horizon, config):
    wd = config.weight_decay if name.endswith('_wd') else 0.0
    if name.startswith('sgd'):
        return _lower_toeplitz(np.asarray(decayed_prefix_sum_workload_coef(
            horizon, config.learning_rate, wd)), horizon)
    if name.startswith('adam_'):
        # The shared helper includes eta; all six definitions here omit eta.
        with jax.default_matmul_precision('highest'):
            return np.asarray(adam_first_moment_workload_matrix(
                horizon, config.beta1, config.learning_rate, wd)) / config.learning_rate
    if name.startswith('momentum'):
        return np.asarray(momentum_trajectory_workload_matrix(
            horizon, config.beta1, config.learning_rate, wd))
    raise ValueError(name)


def operator(strategy):
    return _lower_toeplitz(np.asarray(strategy.noising_coef), strategy.horizon)


def mechanism(strategy, config):
    calibration = calibration_metadata(config, strategy)
    D = operator(strategy) * calibration['calibrated_noise_stddev']
    return D, np.sum(D * D, axis=1), calibration


def linear_energy(A, D):
    """Exact per-coordinate output energies for correlated and matched IID noise."""
    marginal = np.sum(D * D, axis=1)
    return np.sum((A @ D)**2, axis=1), (A * A) @ marginal


def energy_rows(correlated, iid, split):
    horizon = len(correlated)
    for window, start, stop in [('early', 0, split), ('late', split, horizon), ('full', 0, horizon)]:
        for metric in ('trajectory', 'endpoint'):
            c = correlated[start:stop].sum() if metric == 'trajectory' else correlated[stop-1]
            d = iid[start:stop].sum() if metric == 'trajectory' else iid[stop-1]
            yield dict(window=window, metric=metric, start_step=start+1, end_step=stop,
                       correlated_energy=float(c), matched_iid_energy=float(d),
                       ratio=float(c/d), gain=float(1-c/d))


def jobs(worker_index, num_workers):
    if not 0 <= worker_index < num_workers:
        raise ValueError('worker-index must lie in [0, num-workers)')
    return [(name, seed) for i, (name, seed) in enumerate(
        (name, seed) for name in NAMES for seed in SEEDS) if i % num_workers == worker_index]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_table(directory, stem, rows):
    write_json(directory / f'{stem}.json', rows)
    with (directory / f'{stem}.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()} for row in rows)


def provenance(config, contract, smoke):
    return dict(smoke=smoke, config=asdict(config), contract=asdict(contract),
                workload_scale='P_rho H; no overall learning-rate factor')
