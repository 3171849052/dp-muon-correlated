#!/usr/bin/env python3
"""Single GPU worker for fixed-order round-robin AdamW training jobs."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import json
from dp_muon.bandinvmf import load_bandinv_strategy
from dp_muon.data import load_cifar10
from dp_muon.training.cifar10_driver import build_logical_schedule
from exp2.full_training import _train_one_strategy
from exp_adam_workload.core import HERE, configuration, jobs, mechanism, write_json, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-index', type=int, required=True)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()
    assigned = jobs(args.worker_index, args.num_workers)
    config, contract = configuration()
    output = HERE / 'results'
    manifest = json.loads((output / 'experiment.json').read_text())
    assert manifest == provenance(config, contract, False), 'Prepare artifacts with the current config first'
    train_images, train_labels = load_cifar10(ROOT / 'data', train=True)
    test_images, test_labels = load_cifar10(ROOT / 'data', train=False)
    for name, seed in assigned:
        run_dir = output / 'runs' / name / f'seed{seed}'
        run_dir.mkdir(parents=True, exist_ok=True)
        strategy_path = output / 'strategies' / f'{name}.npz'
        strategy = load_bandinv_strategy(strategy_path)
        schedule = build_logical_schedule(num_examples=contract.num_examples,
            batch_size=contract.batch_size, strategy=strategy, seed=seed)
        result = _train_one_strategy(config=config, contract=contract,
            strategy_name=name, strategy_path=strategy_path, strategy=strategy, seed=seed,
            train_images=train_images, train_labels=train_labels,
            test_images=test_images, test_labels=test_labels, schedule=schedule, output_dir=run_dir)
        _, marginal, calibration = mechanism(strategy, config)
        result.update(calibrated_iid_noise_std=calibration['calibrated_noise_stddev'],
            sensitivity_squared=float(strategy.sensitivity_squared),
            mean_marginal_correlated_noise_variance=float(marginal.mean()))
        write_json(run_dir / 'result.json', result)
        print(f'Completed {name} seed={seed}', flush=True)


if __name__ == '__main__':
    main()
