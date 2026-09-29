"""Frozen clean-gradient replay, all coordinates, bounded coordinate-block memory."""
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
from exp13.common import configuration, output, contract, table, write_json
from exp13.collect_trajectory import collect
from exp13.adam import init, moments, channels, private_inputs, readout_for_method


def replay(smoke=False):
    collect(smoke)
    c,root=configuration(smoke),output(smoke)
    p=contract(c,smoke)
    gradients=np.load(root/'replay/gradients.npy',mmap_mode='r')
    metrics=('negative_fraction','raw_second_moment_mse','readout_second_moment_mse',
             'first_moment_mse','adam_direction_mse','absolute_direction_mean',
             'denominator_le_eps_1p01_fraction','denominator_lt_1e6_fraction',
             'denominator_lt_1e4_fraction')
    rows=[]; draw_rows=[]
    magnitude_path=root/'replay/absolute_directions.npy'
    for method in ('iid_ime','iid_ime_abs','bandmf_ime_sep','bandmf_ime_sep_abs'):
        totals=np.zeros(len(metrics),dtype=np.float64); count=0
        # Exact pooled quantiles, with one reusable disk-backed array per method.
        magnitudes=np.lib.format.open_memmap(magnitude_path,mode='w+',dtype=np.float32,
            shape=(c.replay_draws*gradients.size,))
        @jax.jit
        def run(g,key):
            priv=channels(c,p,method,root,key)
            def step(state,x):
                clean,noisy,noise_state=state
                clean,reference=moments(clean,x,x*x,c)
                first,second,noise_state=private_inputs(x,noise_state,priv,method)
                noisy,observed=moments(noisy,first,second,c,readout_for_method(method))
                mh,vh,used,direction=observed
                denominator=jnp.sqrt(used)+c.adam_eps
                magnitude=jnp.abs(direction)
                values=jnp.stack([jnp.sum(vh<0),jnp.sum((vh-reference[1])**2),
                    jnp.sum((used-reference[2])**2),jnp.sum((mh-reference[0])**2),
                    jnp.sum((direction-reference[3])**2),jnp.sum(magnitude),
                    jnp.sum(denominator<=1e-8*1.01),jnp.sum(denominator<1e-6),
                    jnp.sum(denominator<1e-4)])
                return (clean,noisy,noise_state),(values,magnitude)
            state=(init(g[0]),init(g[0]),tuple(x.init(g[0]) for x in priv))
            _,(values,magnitude)=jax.lax.scan(step,state,g)
            return values.sum(axis=0),magnitude
        for draw in range(c.replay_draws):
            draw_total=np.zeros(len(metrics),dtype=np.float64)
            for start in range(0,gradients.shape[1],c.replay_chunk):
                g=jnp.asarray(gradients[:,start:start+c.replay_chunk])
                key=jax.random.fold_in(jax.random.key(c.replay_seed+draw),start)
                values,magnitude=run(g,key)
                draw_total+=np.asarray(values,dtype=np.float64)
                magnitudes[count:count+g.size]=np.asarray(magnitude).ravel()
                count+=g.size
            draw_row=dict(method=method,draw=draw,**dict(zip(metrics,(draw_total/gradients.size).tolist())))
            draw_row['adam_direction_rmse']=float(np.sqrt(draw_row.pop('adam_direction_mse')))
            draw_rows.append(draw_row)
            totals+=draw_total
        row=dict(method=method,second_moment_readout=readout_for_method(method),
                 **dict(zip(metrics,(totals/count).tolist())))
        row['adam_direction_rmse']=float(np.sqrt(row.pop('adam_direction_mse')))
        quantiles=np.quantile(magnitudes,[.5,.9,.99],overwrite_input=True)
        row.update(zip(('absolute_direction_median','absolute_direction_p90','absolute_direction_p99'),
                       quantiles.tolist()))
        del magnitudes
        rows.append(row)
        print(row,flush=True)
    magnitude_path.unlink()
    table(root/'replay/metrics',rows)
    table(root/'replay/draw_metrics',draw_rows)
    paired=[]
    for base in ('iid_ime','bandmf_ime_sep'):
        for draw in range(c.replay_draws):
            relu=next(r for r in draw_rows if r['method']==base and r['draw']==draw)
            absolute=next(r for r in draw_rows if r['method']==base+'_abs' and r['draw']==draw)
            paired.append(dict(pair=base,draw=draw,
                direction_rmse_abs_over_relu=absolute['adam_direction_rmse']/relu['adam_direction_rmse'],
                readout_mse_abs_over_relu=absolute['readout_second_moment_mse']/relu['readout_second_moment_mse']))
    table(root/'replay/paired_readout_ratios',paired)
    print('paired readout ratios',paired,flush=True)
    from exp13.plotting import replay_plot
    replay_plot(root/'replay',rows)
    write_json(root/'replay/provenance.json',dict(draws=c.replay_draws,seed=c.replay_seed,
        coordinates=gradients.shape[1],horizon=p.horizon,normalization='mean over draws, steps, coordinates',
        reference='clean Adam frozen clipped batch mean trajectory', pairing='same latent keys across mechanisms and relu/abs; ratios computed within draw',
        quantiles='exact pooled absolute coordinate direction magnitudes over steps and draws',
        moment_readout='bias-corrected; raw means unprojected v_hat_raw'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true')
    replay(parser.parse_args().smoke)
