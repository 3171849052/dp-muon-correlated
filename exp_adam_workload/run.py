#!/usr/bin/env python3
"""Prepare strategies, exact cross-evaluation, and paired nonlinear replay."""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
from scipy.signal import lfilter
from dp_muon.bandinvmf import fit_bandinv_strategy, save_bandinv_strategy
from exp2.collect_trajectory import collect_trajectory
from exp2.run import adamw_perturbations, load_trajectory
from exp_adam_workload.core import (NAMES, ADAM_FIELDS, HERE, CONFIG, configuration,
    workload, mechanism, operator, linear_energy, energy_rows, write_json,
    write_table, provenance)
from exp_adam_workload.plotting import heatmap, coefficients, bars


def prepare(output, config, contract):
    strategy_dir = output / 'strategies'
    strategy_dir.mkdir(parents=True, exist_ok=True)
    strategies, summaries = {}, []
    for name in NAMES:
        print(f'Fitting {name}', flush=True)
        A = workload(name, contract.horizon, config)
        strategy = fit_bandinv_strategy(contract.horizon, config.bandwidth,
            contract.min_sep, max_participations=contract.max_participations,
            workload_matrix=A, reduction=config.reduction,
            max_optimizer_steps=config.max_optimizer_steps)
        wd = config.weight_decay if name.endswith('_wd') else 0.0
        save_bandinv_strategy(strategy_dir / f'{name}.npz', strategy,
            workload_type=name, momentum=config.beta1, learning_rate=config.learning_rate,
            weight_decay=wd, reduction=config.reduction,
            max_optimizer_steps=config.max_optimizer_steps)
        strategies[name] = strategy
        _, marginal, calibration = mechanism(strategy, config)
        summary = dict(strategy=name, workload_type=name.removesuffix('_wd'),
            includes_weight_decay=name.endswith('_wd'), beta1=config.beta1,
            learning_rate=config.learning_rate, weight_decay=wd,
            actual_adamw_weight_decay=config.weight_decay,
            **asdict(contract), bandwidth=config.bandwidth, reduction=config.reduction,
            max_optimizer_steps=config.max_optimizer_steps,
            sensitivity_squared=float(strategy.sensitivity_squared),
            objective=float(strategy.objective),
            noising_coefficients=np.asarray(strategy.noising_coef).tolist(),
            C_coefficients=np.asarray(strategy.strategy_coef).tolist(),
            calibrated_iid_noise_std=calibration['calibrated_noise_stddev'],
            mean_marginal_correlated_noise_variance=float(marginal.mean()),
            privacy_calibration=calibration)
        summaries.append(summary)
        write_json(strategy_dir / f'{name}.json', summary)
    write_table(output, 'strategy_summary', summaries)
    coefficients(strategies, output / 'strategy_noising_coefficients.png')
    operators = [operator(strategies[n]) for n in NAMES]
    # Matrix L2 here means vectorized Euclidean (Frobenius) norm, not spectral norm.
    distances = np.array([[np.linalg.norm(a-b)/np.linalg.norm(a) for b in operators]
                          for a in operators])
    write_table(output, 'strategy_pairwise_distance', [dict(strategy=n,
        **{m: float(distances[i,j]) for j,m in enumerate(NAMES)}) for i,n in enumerate(NAMES)])
    heatmap(distances, output / 'strategy_pairwise_distance.png',
            'Relative Frobenius distance ||D_i-D_j|| / ||D_i||', 'reference strategy i')
    rows = []
    for evaluation in NAMES:
        A = workload(evaluation, contract.horizon, config)
        for design in NAMES:
            D, _, _ = mechanism(strategies[design], config)
            c, d = linear_energy(A, D)
            rows.extend(dict(evaluation_workload=evaluation, design_strategy=design, **r)
                        for r in energy_rows(c, d, contract.min_sep))
    write_table(output, 'linear_cross_eval', rows)
    for metric, suffix in [('trajectory', 'full'), ('endpoint', 'endpoint')]:
        matrix = np.array([r['ratio'] for r in rows
                           if r['window']=='full' and r['metric']==metric]).reshape(6,6)
        heatmap(matrix, output / f'linear_G_{suffix}_heatmap.png',
                f'Correlated / matched-IID energy: {metric}')
    return strategies


def replay(output, config, contract, strategies, trajectory, samples, seed):
    g = trajectory['g']
    assert g.shape[0] == contract.horizon
    for field in ADAM_FIELDS:
        assert trajectory[field] == getattr(config, field), field
    totals = {n: np.zeros((2, contract.horizon)) for n in NAMES}
    sample_ratios = {n: [] for n in NAMES}
    rng = np.random.default_rng(seed)
    mechanisms = {n: mechanism(s, config) for n,s in strategies.items()}
    # One sample at a time bounds memory even for a complete ViT parameter leaf.
    for sample in range(samples):
        z = rng.standard_normal((1, *g.shape))
        for name, strategy in strategies.items():
            _, marginal, calibration = mechanisms[name]
            correlated = lfilter(np.asarray(strategy.noising_coef), [1.0], z, axis=1)
            correlated *= calibration['calibrated_noise_stddev']
            iid = z * np.sqrt(marginal)[None, :, None, None]
            energies = []
            for noise in (correlated, iid):
                _, delta = adamw_perturbations(g, noise, **{f: trajectory[f] for f in ADAM_FIELDS})
                energies.append(np.sum(delta[0]**2, axis=(1,2)))
            totals[name] += np.asarray(energies)
            sample_ratios[name].append([float(x.sum()) for x in energies])
        if (sample+1) % 10 == 0:
            print(f'Replay {sample+1}/{samples}', flush=True)
    rows = []
    for name in NAMES:
        c, d = totals[name] / samples
        rows.extend(dict(strategy=name, samples=samples, **r)
                    for r in energy_rows(c, d, contract.min_sep))
    write_table(output, 'adam_replay', rows)
    # Paired bootstrap resamples the SAME latent sample indices across designs.
    indices = np.random.default_rng(seed+1).integers(samples, size=(2000, samples))
    boot = {}
    for name in NAMES:
        sums = np.asarray(sample_ratios[name])[indices].sum(axis=1)
        boot[name] = sums[:,0] / sums[:,1]
    paired = []
    for i,a in enumerate(NAMES):
        for b in NAMES[i+1:]:
            delta = boot[a]-boot[b]
            ca, da = totals[a]; cb, db = totals[b]
            paired.append(dict(strategy_a=a, strategy_b=b,
                ratio_difference=float(ca.sum()/da.sum()-cb.sum()/db.sum()),
                ci95_low=float(np.quantile(delta,.025)), ci95_high=float(np.quantile(delta,.975))))
    write_table(output, 'adam_replay_paired', paired)
    write_json(output / 'adam_replay_metadata.json', dict(samples=samples, seed=seed,
        parameter_name=trajectory['parameter_name'], paired_latents=True,
        noise_scale='independent formal privacy calibration per strategy',
        energy='squared parameter displacement from clean AdamW, summed over selected leaf',
        **{f: trajectory[f] for f in ADAM_FIELDS}))
    bars([r['ratio'] for r in rows if r['window']=='full' and r['metric']=='trajectory'],
         None, output / 'adam_replay_summary.png', 'Full AdamW replay: correlated / matched-IID energy')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--parameter-name', default='blocks/0/attention/query/kernel')
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    config, contract = configuration(args.smoke)
    output = HERE / ('results_smoke' if args.smoke else 'results')
    output.mkdir(parents=True, exist_ok=True)
    strategies = prepare(output, config, contract)
    write_json(output / 'experiment.json', provenance(config, contract, args.smoke))
    if args.prepare_only:
        return
    if args.smoke:
        trajectory = dict(g=np.random.default_rng(10).normal(size=(contract.horizon,3,2))*.05,
            parameter_name='synthetic-smoke', **{f: getattr(config,f) for f in ADAM_FIELDS})
    else:
        path = collect_trajectory(config_path=CONFIG, parameter_name=args.parameter_name,
                                  output=output / 'trajectory.npz')
        trajectory = load_trajectory(path)
    replay(output, config, contract, strategies, trajectory, 3 if args.smoke else args.samples, args.seed)
    if args.smoke:
        from exp_adam_workload.smoke import train_smoke
        train_smoke(output, config, contract, strategies)
    print(f'Wrote {output}', flush=True)


if __name__ == '__main__':
    main()
