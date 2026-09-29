"""Frozen clean-gradient replay with raw-state and denominator diagnostics."""
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
from exp13.common import configuration, output, contract, table, write_json
from exp13.collect_trajectory import collect
from exp13.adam import init, moments, channels, private_inputs
from exp13.privacy import calibration
from exp13.strategies import load
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


def theoretical_state_mse(strategy, beta, sigma):
    """Per-coordinate uncorrected EMA-state MSE from the BandMF workload."""
    row_norms_squared = banded.per_query_error(strategy, A=ema(beta))
    return float(sigma**2 * jnp.mean(row_norms_squared))


def replay(smoke=False):
    collect(smoke)
    c,root=configuration(smoke),output(smoke)
    p=contract(c,smoke)
    gradients=np.load(root/'replay/gradients.npy',mmap_mode='r')
    if gradients.shape[0] != p.horizon:
        raise ValueError(f'gradient trajectory has {gradients.shape[0]} steps; expected {p.horizon}')

    rows=[]
    per_step_rows=[]
    empirical={}
    for method in ('iid_ime','bandmf_ime_sep'):
        c1=calibration(c,p,method)
        sums_by_step=np.zeros((p.horizon,len(STEP_SUMS)),dtype=np.float64)
        denominator_min_by_step=np.full(p.horizon,np.inf,dtype=np.float64)
        direction_max_by_step=np.zeros(p.horizon,dtype=np.float64)
        temporary_shape=(c.replay_draws,p.horizon,gradients.shape[1])
        magnitude_path=root/'replay/absolute_directions.npy'
        denominator_path=root/'replay/denominators.npy'
        magnitudes=np.lib.format.open_memmap(magnitude_path,mode='w+',dtype=np.float32,shape=temporary_shape)
        denominators=np.lib.format.open_memmap(denominator_path,mode='w+',dtype=np.float32,shape=temporary_shape)

        @jax.jit
        def run(g,key):
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
                return (clean,private,noise_state),(values,magnitude,denominator)
            state=(init(g[0]),init(g[0]),tuple(x.init(g[0]) for x in priv))
            _,(values,magnitude,denominator)=jax.lax.scan(step,state,g)
            return values,magnitude,denominator

        for draw in range(c.replay_draws):
            for start in range(0,gradients.shape[1],c.replay_chunk):
                end=min(start+c.replay_chunk,gradients.shape[1])
                g=jnp.asarray(gradients[:,start:end])
                key=jax.random.fold_in(jax.random.key(c.replay_seed+draw),start)
                values,magnitude,denominator=run(g,key)
                values=np.asarray(values,dtype=np.float64)
                magnitude=np.asarray(magnitude,dtype=np.float32)
                denominator=np.asarray(denominator,dtype=np.float32)
                sums_by_step+=values
                denominator_min_by_step=np.minimum(denominator_min_by_step,denominator.min(axis=1))
                direction_max_by_step=np.maximum(direction_max_by_step,magnitude.max(axis=1))
                magnitudes[draw,:,start:end]=magnitude
                denominators[draw,:,start:end]=denominator

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
        empirical[method]=(row['first_state_mse'],row['second_state_mse'])
        rows.append(row)

        for step_index in range(p.horizon):
            per_step_rows.append(dict(method=method,step=step_index+1,
                negative_fraction=float(sums_by_step[step_index,0]/step_coordinate_count),
                first_state_mse=float(sums_by_step[step_index,1]/step_coordinate_count),
                second_state_mse=float(sums_by_step[step_index,2]/step_coordinate_count),
                denominator_min=float(denominator_min_by_step[step_index]),
                denominator_p001=float(np.quantile(denominators[:,step_index,:],.001)),
                absolute_direction_p99=float(np.quantile(magnitudes[:,step_index,:],.99)),
                absolute_direction_max=float(direction_max_by_step[step_index])))
        denominator_q=np.quantile(denominators,[.0001,.001,.01,.1,.5],overwrite_input=True)
        row.update(zip(('denominator_p0001','denominator_p001','denominator_p01',
                        'denominator_p1','denominator_median'),denominator_q.tolist()))
        direction_q=np.quantile(magnitudes,[.5,.9,.99,.999],overwrite_input=True)
        row.update(zip(('absolute_direction_median','absolute_direction_p90',
                        'absolute_direction_p99','absolute_direction_p999'),direction_q.tolist()))
        magnitudes.flush(); denominators.flush()
        del magnitudes,denominators
        magnitude_path.unlink()
        denominator_path.unlink()
        print(row,flush=True)

    table(root/'replay/metrics',rows)
    table(root/'replay/per_step_diagnostics',per_step_rows)
    from exp13.plotting import replay_plot
    replay_plot(root/'replay',rows)

    identity=banded.ColumnNormalizedBanded.default(p.horizon,1)
    workload_rows=[]
    for method in ('iid_ime','bandmf_ime_sep'):
        method_cal=calibration(c,p,method)
        channels_for_method=(('first',c.beta1,method_cal['sigma1'],'C_m'),
                             ('second',c.beta2,method_cal['sigma2'],'C_v'))
        for channel,beta,sigma,bandmf_name in channels_for_method:
            strategy_name='identity' if method=='iid_ime' else bandmf_name
            strategy=identity if method=='iid_ime' else load(root,bandmf_name)
            theory=theoretical_state_mse(strategy,beta,sigma)
            measured=empirical[method][0 if channel=='first' else 1]
            workload_rows.append(dict(method=method,channel=channel,beta=beta,
                strategy=strategy_name,sigma=sigma,theoretical_state_mse=theory,
                empirical_state_mse=measured,empirical_over_theoretical=measured/theory))
    table(root/'replay/workload_validation',workload_rows)

    write_json(root/'replay/provenance.json',dict(draws=c.replay_draws,seed=c.replay_seed,
        coordinates=gradients.shape[1],horizon=p.horizon,
        normalization='uncorrected state errors averaged over draws, steps, and coordinates',
        reference='clean Adam frozen clipped batch mean trajectory',
        fitting_objective='sigma^2 * mean(banded.per_query_error(strategy, A=ema(beta)))',
        iid_strategy='identity represented as a one-band column-normalized strategy',
        moment_readout='IME denominator uses abs(v_hat_raw); abs is excluded from raw recurrence',
        quantiles='exact pooled coordinate quantiles over all replay draws and steps'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true')
    replay(parser.parse_args().smoke)
