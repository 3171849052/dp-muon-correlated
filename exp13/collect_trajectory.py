import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'src')]
import argparse
import hashlib
from dataclasses import asdict
import jax
import jax.numpy as jnp
import numpy as np
import optax
from dp_muon.data import iter_logical_batches
from exp13.common import ROOT, configuration, output, write_json, config_hash
from exp13.full_training import setup
from exp13.adam import optimizer, batch_mean_loss

def sample_coordinates(total, count, seed):
    return np.random.default_rng(seed).choice(total, count, replace=False)

def collect(smoke=False):
    c,root=configuration(smoke),output(smoke)
    dest=root/'trajectory'; dest.mkdir(exist_ok=True)
    x,y,p,batches,model,params,key,query=setup(c,smoke,0)
    leaves,tree=jax.tree_util.tree_flatten(params)
    count=sum(a.size for a in leaves)
    indices=sample_coordinates(count,c.replay_num_coordinates,c.replay_coordinate_seed)
    np.save(dest/'coordinate_indices.npy',indices)
    device_indices=jnp.asarray(indices)
    gfile=np.lib.format.open_memmap(dest/'gradients_sampled.npy',mode='w+',dtype=np.float32,shape=(p.horizon,len(indices)))
    opt=optimizer(c); state=opt.init(params)
    @jax.jit
    def step(params,state,b):
        g=query(params,b)
        clean=jax.grad(lambda parameters: batch_mean_loss(parameters,b,model))(params)
        u,state=opt.update(clean,state,params)
        sampled=jnp.concatenate([a.ravel() for a in jax.tree.leaves(g)])[device_indices]
        return optax.apply_updates(params,u),state,sampled
    for i,b in enumerate(iter_logical_batches(x,y,batches)):
        params,state,g=step(params,state,jax.tree.map(jnp.asarray,b))
        gfile[i]=np.asarray(g)
    gfile.flush()
    write_json(dest/'metadata.json',dict(config_hash=config_hash(c,p),config=vars(c),contract=asdict(p),seed=0,
        schedule_sha256=hashlib.sha256(np.asarray(batches).tobytes()).hexdigest(),
        pretrained_sha256=hashlib.sha256((ROOT/c.pretrained).read_bytes()).hexdigest(),
        gradient='clipped DP-query gradients evaluated along a clean Adam parameter trajectory',
        total_coordinates=count,sampled_coordinates=len(indices),coordinate_seed=c.replay_coordinate_seed,horizon=p.horizon,
        shapes=[list(a.shape) for a in leaves],coordinates=count))
    print('collected',gfile.shape,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); collect(p.parse_args().smoke)
