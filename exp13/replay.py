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
from exp13.adam import init, moments, channels, private_inputs


def replay(smoke=False):
    collect(smoke)
    c,root=configuration(smoke),output(smoke)
    p=contract(c,smoke)
    gradients=np.load(root/'replay/gradients.npy',mmap_mode='r')
    metrics=('negative_fraction','raw_second_moment_mse','projected_second_moment_mse',
             'first_moment_mse','adam_direction_mse')
    rows=[]
    for method in ('iid_ime','bandmf_ime_sep'):
        totals=np.zeros(5,dtype=np.float64); count=0
        @jax.jit
        def run(g,key):
            priv=channels(c,p,method,root,key)
            def step(state,x):
                clean,noisy,noise_state=state
                clean,reference=moments(clean,x,x*x,c)
                first,second,noise_state=private_inputs(x,noise_state,priv,method)
                noisy,observed=moments(noisy,first,second,c)
                mh,vh,proj,direction=observed
                values=jnp.stack([jnp.sum(vh<0),jnp.sum((vh-reference[1])**2),
                    jnp.sum((proj-reference[2])**2),jnp.sum((mh-reference[0])**2),
                    jnp.sum((direction-reference[3])**2)])
                return (clean,noisy,noise_state),values
            state=(init(g[0]),init(g[0]),tuple(x.init(g[0]) for x in priv))
            _,values=jax.lax.scan(step,state,g)
            return values.sum(axis=0)
        for draw in range(c.replay_draws):
            for start in range(0,gradients.shape[1],c.replay_chunk):
                g=jnp.asarray(gradients[:,start:start+c.replay_chunk])
                key=jax.random.fold_in(jax.random.key(c.replay_seed+draw),start)
                totals+=np.asarray(run(g,key),dtype=np.float64); count+=g.size
        row=dict(method=method,**dict(zip(metrics,(totals/count).tolist())))
        row['adam_direction_rmse']=float(np.sqrt(row.pop('adam_direction_mse')))
        rows.append(row)
        print(row,flush=True)
    table(root/'replay/metrics',rows)
    from exp13.plotting import replay_plot
    replay_plot(root/'replay',rows)
    write_json(root/'replay/provenance.json',dict(draws=c.replay_draws,seed=c.replay_seed,
        coordinates=gradients.shape[1],horizon=p.horizon,normalization='mean over draws, steps, coordinates',
        reference='clean Adam frozen clipped batch mean trajectory', pairing='same latent keys across mechanisms'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true')
    replay(parser.parse_args().smoke)
