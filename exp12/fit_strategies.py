import os
os.environ.setdefault('JAX_ENABLE_X64','true')
import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'src')]
import argparse
from dataclasses import asdict
import jax
import jax.numpy as jnp
import numpy as np
import optax
from dp_muon.bandinvmf import fit_bandinv_strategy
from exp12.common import configuration, contract, output, calibration, write_json
from exp12.objectives import momentum, momentum_objective, exact, mean_field
from exp12.strategies import normalize, save

def fit(smoke=False):
    c, root = configuration(smoke), output(smoke)
    p = contract(c,smoke)
    directory = root/'strategies'
    directory.mkdir(exist_ok=True)
    sigma = calibration(c,p,'adam_exact').iid_noise_std
    base = fit_bandinv_strategy(p.horizon,c.bandwidth,p.min_sep,
        max_participations=p.max_participations, workload_matrix=momentum(c,p.horizon),
        max_optimizer_steps=c.fit_steps,reduction=c.reduction)
    metadata = dict(config=vars(c),contract=asdict(p),sigma_corr=sigma,
        sensitivity_convention='jax_privacy non-increasing absolute coefficient majorant; use_matrix_upper_bound=False',
        initialization='bandinv_momentum', fit_seed=c.fit_seed, eval_seed=c.eval_seed)
    write_json(root/'contract.json',metadata)
    raw = base.noising_coef
    save(directory/'bandinv_momentum.npz',raw,p,momentum_objective(normalize(raw,p),p.horizon,c),metadata)
    z = jax.random.normal(jax.random.key(c.fit_seed),(p.horizon,c.K_fit),dtype=jnp.float64)
    eval_z = jax.random.normal(jax.random.key(c.eval_seed),(p.horizon,c.K_eval),dtype=jnp.float64)
    for name in ('adam_exact','adam_mf'):
        def loss(x):
            s = normalize(x,p)
            return exact(s,z,sigma,c) if name=='adam_exact' else mean_field(s,p.horizon,sigma,c)
        # Fix the diagonal to one: normalization removes this redundant scale.
        def reduced(x):
            return loss(jnp.concatenate((jnp.ones(1),x)))
        opt = optax.adam(c.fit_learning_rate)
        x = raw[1:]/raw[0]
        state = opt.init(x)
        @jax.jit
        def step(x,state):
            value,grad = jax.value_and_grad(reduced)(x)
            update,state = opt.update(grad,state,x)
            return optax.apply_updates(x,update),state,value
        best_x, best_value = x, float(reduced(x))
        trace=[]
        for i in range(c.fit_steps):
            new_x,state,value = step(x,state)
            value=float(value)
            if not np.isfinite(value):
                raise FloatingPointError(f'{name} objective nonfinite at {i}')
            if value < best_value:
                best_x,best_value=x,value
            x=new_x
            trace.append(value)
        last=float(reduced(x))
        if last < best_value:
            best_x,best_value=x,last
        fitted=jnp.concatenate((jnp.ones(1),best_x))
        s=normalize(fitted,p)
        independent=float(exact(s,eval_z,sigma,c))
        meta=dict(metadata,optimizer='optax.adam direct objective',steps=c.fit_steps,
            fitting_objective=best_value, independent_exact_objective=independent,loss_trace=trace)
        objective=independent if name=='adam_exact' else float(mean_field(s,p.horizon,sigma,c))
        save(directory/f'{name}.npz',fitted,p,objective,meta)
        print(name,objective,flush=True)
    from exp12.cross_eval import cross_eval
    cross_eval(smoke)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true')
    fit(parser.parse_args().smoke)
