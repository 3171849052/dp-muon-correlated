"""Sampled DP-query replay along a frozen clean Adam parameter trajectory."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'src')]
import argparse
import jax
import jax.numpy as jnp
import numpy as np
from jax_privacy.matrix_factorization import banded
from exp13.common import configuration, output, contract, table, write_json, config_hash
import time
import json
from exp13.adam import init, moments, channels, private_inputs, IME_METHODS
from exp13.privacy import calibration
from exp13.strategies import load, strategy_names, validate
from exp13.workloads import ema


THRESHOLDS = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4)
THRESHOLD_FIELDS = tuple(f'denominator_lt_1e_minus_{-int(np.log10(x))}_fraction'
                         for x in THRESHOLDS)
STEP_SUMS = ('negative_count', 'first_state_sq_sum', 'second_state_sq_sum',
             'first_hat_sq_sum', 'second_hat_raw_sq_sum', 'second_hat_abs_sq_sum',
             'adam_direction_sq_sum', 'absolute_direction_sum',
             *(field.removesuffix('_fraction') + '_count' for field in THRESHOLD_FIELDS))


def squared_error_sum(actual, reference):
    """Sum squared coordinate errors, including uncorrected moment states."""
    return jnp.sum(jnp.square(actual-reference))


def theoretical_state_mse(strategy, beta, sigma, corrected=False):
    """Exact linear raw or bias-corrected moment MSE from the workload."""
    row_norms_squared = banded.per_query_error(strategy, A=ema(beta, corrected))
    return float(sigma**2 * jnp.mean(row_norms_squared))


def make_runner(c,p,method,root):
    @jax.jit
    def run(g,key,selected):
        priv=channels(c,p,method,root,key)
        def step(state,x):
            clean,private,noise_state=state
            clean,clean_read=moments(clean,x,x*x,c,method='nonprivate_adam')
            first,second,noise_state=private_inputs(x,noise_state,priv,method)
            private,private_read=moments(private,first,second,c,method=method)
            mh,vh,used,direction=private_read
            clean_mh,clean_vh,_,clean_direction=clean_read
            denominator=jnp.sqrt(used)+c.adam_eps
            magnitude=jnp.abs(direction)
            values=jnp.stack([
                jnp.sum(vh<0),
                squared_error_sum(private[1],clean[1]),
                squared_error_sum(private[2],clean[2]),
                squared_error_sum(mh,clean_mh),
                squared_error_sum(vh,clean_vh),
                squared_error_sum(used,clean_vh),
                squared_error_sum(direction,clean_direction),
                jnp.sum(magnitude),
                *(jnp.sum(denominator<threshold) for threshold in THRESHOLDS),
            ])
            return (clean,private,noise_state),(values,jnp.min(denominator),jnp.max(magnitude),magnitude[selected],denominator[selected])
        state=(init(g[0]),init(g[0]),tuple(x.init(g[0]) for x in priv))
        _,statistics=jax.lax.scan(step,state,g)
        return statistics

    return run


def replay(smoke=False):
    started=time.monotonic()
    c,root=configuration(smoke),output(smoke)
    p=contract(c,smoke)
    validate(root,c,p)
    trajectory=root/'trajectory/gradients_sampled.npy'
    if not trajectory.exists():
        raise FileNotFoundError('Run Stage 1 collect step before replay: missing sampled trajectory')
    metadata=json.loads((trajectory.parent/'metadata.json').read_text())
    if metadata['config_hash'] != config_hash(c,p):
        raise ValueError('Stage 1 trajectory/config mismatch')
    gradients=np.load(trajectory,mmap_mode='r')
    if gradients.shape != (p.horizon,c.replay_num_coordinates):
        raise ValueError('Sampled trajectory shape/config mismatch')
    (root/'replay').mkdir(exist_ok=True)
    quantile_indices=np.sort(np.random.default_rng(c.replay_coordinate_seed+1).choice(
        gradients.shape[1],c.replay_quantile_coordinates,replace=False))
    np.save(root/'replay/quantile_indices.npy',quantile_indices)

    quantile_width=max(np.count_nonzero((quantile_indices>=start)&(quantile_indices<start+c.replay_chunk))
        for start in range(0,gradients.shape[1],c.replay_chunk))

    rows=[]
    per_step_rows=[]
    empirical={}
    for method in IME_METHODS:
        sums_by_step=np.zeros((p.horizon,len(STEP_SUMS)),dtype=np.float64)
        denominator_min_by_step=np.full(p.horizon,np.inf,dtype=np.float64)
        direction_max_by_step=np.zeros(p.horizon,dtype=np.float64)
        temporary_shape=(c.replay_draws,p.horizon,c.replay_quantile_coordinates)
        magnitudes=np.empty(temporary_shape,dtype=np.float32)
        denominators=np.empty(temporary_shape,dtype=np.float32)

        run=make_runner(c,p,method,root)

        for draw in range(c.replay_draws):
            for start in range(0,gradients.shape[1],c.replay_chunk):
                end=min(start+c.replay_chunk,gradients.shape[1])
                g=jnp.asarray(gradients[:,start:end])
                key=jax.random.fold_in(jax.random.key(c.replay_seed+draw),start)
                positions=np.flatnonzero((quantile_indices>=start)&(quantile_indices<end))
                selected=jnp.asarray(np.pad(quantile_indices[positions]-start,(0,quantile_width-len(positions)),mode="edge"))
                values,den_min,mag_max,magnitude,denominator=run(g,key,selected)
                sums_by_step+=np.asarray(values,dtype=np.float64)
                denominator_min_by_step=np.minimum(denominator_min_by_step,np.asarray(den_min))
                direction_max_by_step=np.maximum(direction_max_by_step,np.asarray(mag_max))
                magnitudes[draw][:,positions]=np.asarray(magnitude)[:,:len(positions)]
                denominators[draw][:,positions]=np.asarray(denominator)[:,:len(positions)]

        step_coordinate_count=c.replay_draws*gradients.shape[1]
        coordinate_count=c.replay_draws*gradients.size
        pooled=sums_by_step.sum(axis=0)/coordinate_count
        row=dict(method=method,optimizer_second_moment='abs(v_hat_raw)',
            negative_fraction=float(pooled[0]),
            first_state_mse=float(pooled[1]),second_state_mse=float(pooled[2]),
            first_hat_mse=float(pooled[3]),second_hat_raw_mse=float(pooled[4]),
            second_hat_abs_mse=float(pooled[5]),adam_direction_rmse=float(np.sqrt(pooled[6])),
            absolute_direction_mean=float(pooled[7]))
        row['denominator_min']=float(denominator_min_by_step.min())
        for field,value in zip(THRESHOLD_FIELDS,pooled[8:]):
            row[field]=float(value)
        row['absolute_direction_max']=float(direction_max_by_step.max())
        empirical[method]=row
        rows.append(row)

        for step_index in range(p.horizon):
            per_step_rows.append(dict(method=method,step=step_index+1,
                negative_fraction=float(sums_by_step[step_index,0]/step_coordinate_count),
                first_state_mse=float(sums_by_step[step_index,1]/step_coordinate_count),
                second_state_mse=float(sums_by_step[step_index,2]/step_coordinate_count),
                first_hat_mse=float(sums_by_step[step_index,3]/step_coordinate_count),
                second_hat_raw_mse=float(sums_by_step[step_index,4]/step_coordinate_count),
                denominator_min=float(denominator_min_by_step[step_index]),
                denominator_p001=float(np.quantile(denominators[:,step_index,:],.001)),
                absolute_direction_p99=float(np.quantile(magnitudes[:,step_index,:],.99)),
                absolute_direction_max=float(direction_max_by_step[step_index])))
        denominator_q=np.quantile(denominators,[.0001,.001,.01,.1,.5])
        row.update(zip(('denominator_p0001','denominator_p001','denominator_p01',
                        'denominator_p1','denominator_median'),denominator_q.tolist()))
        direction_q=np.quantile(magnitudes,[.5,.9,.99,.999])
        row.update(zip(('absolute_direction_median','absolute_direction_p90',
                        'absolute_direction_p99','absolute_direction_p999'),direction_q.tolist()))
        print(row,flush=True)

    table(root/'replay/metrics',rows)
    table(root/'replay/per_step_diagnostics',per_step_rows)
    from exp13.plotting import replay_plot
    replay_plot(root/'replay',rows)

    identity=banded.ColumnNormalizedBanded.default(p.horizon,1)
    workload_rows=[]
    for method in IME_METHODS:
        method_cal=calibration(c,p,method)
        channels_for_method=(('first',c.beta1,method_cal['sigma1'],strategy_names(method)[0]),
                             ('second',c.beta2,method_cal['sigma2'],strategy_names(method)[1]))
        for channel,beta,sigma,bandmf_name in channels_for_method:
            strategy_name='identity' if method=='iid_ime' else bandmf_name
            strategy=identity if method=='iid_ime' else load(root,bandmf_name)
            theory=theoretical_state_mse(strategy,beta,sigma)
            theory_bc=theoretical_state_mse(strategy,beta,sigma,True)
            measured=empirical[method]['first_state_mse' if channel=='first' else 'second_state_mse']
            measured_bc=empirical[method]['first_hat_mse' if channel=='first' else 'second_hat_raw_mse']
            workload_rows.append(dict(method=method,channel=channel,beta=beta,
                strategy=strategy_name,sigma=sigma,theoretical_raw_state_mse=theory,
                empirical_raw_state_mse=measured,raw_empirical_over_theoretical=measured/theory,
                theoretical_bias_corrected_mse=theory_bc,empirical_bias_corrected_mse=measured_bc,
                bc_empirical_over_theoretical=measured_bc/theory_bc))
    table(root/'replay/workload_validation',workload_rows)

    write_json(root/'replay/provenance.json',dict(draws=c.replay_draws,seed=c.replay_seed,
        coordinates=gradients.shape[1],horizon=p.horizon,
        normalization='uncorrected state errors averaged over draws, steps, and coordinates',
        reference='clipped DP-query gradients evaluated along a clean Adam parameter trajectory',
        fitting_objective='sigma^2 * mean(banded.per_query_error(strategy, A=ema(beta, corrected)))',
        iid_strategy='identity represented as a one-band column-normalized strategy',
        moment_readout='IME denominator uses abs(v_hat_raw); abs is excluded from raw recurrence',
        quantile_coordinates=c.replay_quantile_coordinates,
        runtime_seconds=time.monotonic()-started,
        quantiles='quantiles are computed from a deterministic coordinate subsample.'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true')
    replay(parser.parse_args().smoke)
