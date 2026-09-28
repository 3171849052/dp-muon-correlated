import os
os.environ.setdefault('JAX_ENABLE_X64','true')
import sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'src')]
import argparse
import jax
import jax.numpy as jnp
from exp12.common import configuration, contract, output, calibration, table, write_json
from exp12.strategies import coefficients
from exp12.objectives import momentum_objective, exact, mean_field

def cross_eval(smoke=False):
    c,root=configuration(smoke),output(smoke); p=contract(c,smoke)
    sigma=calibration(c,p,'adam_exact').iid_noise_std
    z=jax.random.normal(jax.random.key(c.eval_seed),(p.horizon,c.K_eval),dtype=jnp.float64)
    rows=[]
    for method in ('bandinv_momentum','adam_exact','adam_mf'):
        s=coefficients(root,method)
        rows.append(dict(strategy=method,J_mom=float(momentum_objective(s,p.horizon,c)),
            J_exact=float(exact(s,z,sigma,c)),J_mf=float(mean_field(s,p.horizon,sigma,c))))
    gap=rows[2]['J_exact']/rows[1]['J_exact']-1
    table(root/'cross_eval',rows)
    write_json(root/'cross_eval.json',dict(rows=rows,mf_exact_gap=gap,K_eval=c.K_eval,eval_seed=c.eval_seed))
    print('cross evaluation',rows,'mf_exact_gap',gap,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); cross_eval(p.parse_args().smoke)
