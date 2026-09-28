import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'src')]
import argparse
import jax
import jax.numpy as jnp
import numpy as np
from exp12.adam import recurrence
from exp12.objectives import filter_noise
from exp12.strategies import coefficients
from exp12.common import METHODS, configuration, contract, output, calibration, table, write_json

def sufficient_statistics(g,noise,c):
    u,p=recurrence(g,c.beta1,c.beta2,c.adam_eps)
    un,pn=recurrence(g+noise,c.beta1,c.beta2,c.adam_eps)
    d=un-u; theta=-c.learning_rate*jnp.cumsum(d,axis=0)
    return jnp.stack([jnp.sum(a,axis=1) for a in (
        theta**2,d**2,(pn-p)**2,un*u,un**2,u**2,
        (jnp.sign(un)==jnp.sign(u)).astype(g.dtype),noise**2,g**2)])

def metrics(stats,coordinates):
    # stats: draws x nine sufficient statistics x time, summed over all parameters.
    ratio=np.sqrt(stats[:,7])/(np.sqrt(stats[:,8])+1e-12)
    return dict(parameter_trajectory_rmse=float(np.sqrt(stats[:,0].mean())),
        endpoint_rmse=float(np.sqrt(stats[:,0,-1].mean())),
        adam_direction_rmse=float(np.sqrt(stats[:,1].mean())),
        update_cosine_similarity=float((stats[:,3]/np.sqrt(np.maximum(stats[:,4]*stats[:,5],1e-30))).mean()),
        sign_agreement=float(stats[:,6].mean()/coordinates),
        preconditioner_rmse=float(np.sqrt(stats[:,2].mean())),
        noise_gradient_ratio_mean=float(ratio.mean()),noise_gradient_ratio_median=float(np.median(ratio)),
        noise_gradient_ratio_p10=float(np.quantile(ratio,.1)),noise_gradient_ratio_p90=float(np.quantile(ratio,.9)),
        global_energy_ratio=float(stats[:,7].sum()/stats[:,8].sum()))

def replay(smoke=False):
    c,root=configuration(smoke),output(smoke); p=contract(c,smoke)
    dest=root/'replay'; dest.mkdir(exist_ok=True)
    if smoke:
        from exp12.collect_trajectory import collect
        collect(True)
    g=np.load(dest/'gradients.npy',mmap_mode='r')
    compute=jax.jit(lambda g,n:sufficient_statistics(g,n,c))
    stats={m:np.zeros((c.replay_draws,9,len(g)),dtype=np.float64) for m in METHODS}
    coefs={m:coefficients(root,m).astype(jnp.float32) for m in METHODS}
    sigmas={m:calibration(c,p,m).iid_noise_std for m in METHODS}
    for start in range(0,g.shape[1],c.replay_chunk):
        chunk=jnp.asarray(np.array(g[:,start:start+c.replay_chunk]))
        for draw in range(c.replay_draws):
            key=jax.random.fold_in(jax.random.fold_in(jax.random.key(c.replay_seed),draw),start)
            z=jax.random.normal(key,chunk.shape)
            for m in METHODS:
                noise=sigmas[m]*filter_noise(coefs[m],z)
                stats[m][draw]+=np.asarray(compute(chunk,noise))
    rows=[dict(method=m,**metrics(stats[m],g.shape[1])) for m in METHODS]
    table(dest/'metrics',rows)
    np.savez(dest/'sufficient_statistics.npz',**stats)
    write_json(dest/'metadata.json',dict(draws=c.replay_draws,seed=c.replay_seed,
        coordinates=g.shape[1],horizon=len(g),paired_draws=True,global_energy_ratio='sum ||noise||^2 / sum ||gradient||^2'))
    from exp12.plotting import replay_plot
    replay_plot(dest,rows)
    print(rows,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); replay(p.parse_args().smoke)
