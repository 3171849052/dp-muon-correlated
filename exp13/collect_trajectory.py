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
from exp13.common import ROOT, configuration, output, write_json
from exp13.full_training import setup
from exp13.adam import optimizer

def collect(smoke=False):
    c,root=configuration(smoke),output(smoke)
    dest=root/'replay'; dest.mkdir(exist_ok=True)
    x,y,p,batches,model,params,key,query=setup(c,smoke,0)
    leaves,tree=jax.tree_util.tree_flatten(params)
    count=sum(a.size for a in leaves)
    gfile=np.lib.format.open_memmap(dest/'gradients.npy',mode='w+',dtype=np.float32,shape=(p.horizon,count))
    opt=optimizer(c); state=opt.init(params)
    @jax.jit
    def step(params,state,b):
        g=query(params,b)
        u,state=opt.update(g,state,params)
        return optax.apply_updates(params,u),state,g
    for i,b in enumerate(iter_logical_batches(x,y,batches)):
        params,state,g=step(params,state,jax.tree.map(jnp.asarray,b))
        gfile[i]=np.concatenate([np.asarray(a).ravel() for a in jax.tree.leaves(g)])
    gfile.flush()
    write_json(dest/'trajectory.json',dict(config=vars(c),contract=asdict(p),seed=0,
        schedule_sha256=hashlib.sha256(np.asarray(batches).tobytes()).hexdigest(),
        pretrained_sha256=hashlib.sha256((ROOT/c.pretrained).read_bytes()).hexdigest(),
        gradient='all parameters, post global per-example clipping, mean, pre optimizer',
        shapes=[list(a.shape) for a in leaves],coordinates=count))
    print('collected',gfile.shape,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); collect(p.parse_args().smoke)
