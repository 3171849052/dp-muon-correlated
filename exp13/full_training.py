import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'src')]
import argparse
from dataclasses import asdict
import hashlib
import time
import jax
import jax.numpy as jnp
import numpy as np
from dp_muon.data import iter_logical_batches
from dp_muon.models import ViTTiny, load_pretrained_vit_tiny
from dp_muon.privacy import make_clipped_gradient_query
from dp_muon.training.cifar10_driver import cross_entropy_loss, evaluate_classifier_metrics
from dp_muon.training.cifar10_experiment import derive_fixed_cycle_participation
from exp13.common import ROOT, METHODS, configuration, dataset, output, schedule, table, write_json
from exp13.adam import init, moments, channels, private_inputs, uses_ime
from exp13.privacy import calibration, iid_calibration
from exp13.adam import clean_step
from exp13.strategies import validate, strategy_names
from dp_muon.training.nonamplified_dpadamw import make_nonamplified_dpadamw_train_step, init_nonamplified_dpadamw_state
from dp_muon.privacy.nonamplified import epsilon_spent_for_iid_prefix


def setup(c,smoke,seed, clipped=True):
    x,y=dataset(c,smoke)
    p=derive_fixed_cycle_participation(len(x),c.epochs,c.batch_size)
    batches=schedule(c,p,len(x),seed)
    model=ViTTiny()
    pk,nk=jax.random.split(jax.random.key(seed))
    params=load_pretrained_vit_tiny(ROOT/c.pretrained,key=pk)
    query=make_clipped_gradient_query(lambda params,b: cross_entropy_loss(params,b,model),
        clip_norm=c.clip_norm,normalize_by=float(c.batch_size),batch_argnums=1,
        keep_batch_dim=True,microbatch_size=c.microbatch_size) if clipped else None
    return x,y,p,batches,model,params,nk,query

def train(method='iid_adam',seed=0,smoke=False):
    c,root=configuration(smoke),output(smoke, "stage2_training")
    strategy_root=output(smoke)
    from exp13.common import contract
    validate(strategy_root,c,contract(c,smoke))
    dest=root/'training'/f'{method}_seed{seed}'; dest.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    x,y,p,batches,model,params,key,query=setup(c,smoke,seed, clipped=method not in ("nonprivate_adam", "iid_adam"))
    tx,ty=dataset(c,smoke,False)
    if method == 'nonprivate_adam':
        update, opt = clean_step(c,model)
        state = opt.init(params)
        cal = None
        def step(params,state,noise_state,batch):
            params,state = update(params,state,batch)
            return params,state,None
        noise_state = None
    elif method == 'iid_adam':
        canonical = iid_calibration(c,p)
        update,opt = make_nonamplified_dpadamw_train_step(
            lambda params,b: cross_entropy_loss(params,b,model), canonical,
            learning_rate=c.learning_rate,beta1=c.beta1,beta2=c.beta2,
            eps=c.adam_eps,weight_decay=0,microbatch_size=c.microbatch_size)
        update = jax.jit(update)
        state = init_nonamplified_dpadamw_state(params,key,opt)
        cal = asdict(canonical)
        def step(params,state,noise_state,batch):
            state = update(state,batch)
            return state.params,state,None
        noise_state = None
    else:
        state=init(params)
        privatizers=channels(c,p,method,strategy_root,key)
        noise_state=tuple(priv.init(params) for priv in privatizers)
        cal=calibration(c,p,method)
        @jax.jit
        def step(params,state,noise_state,batch):
            g=query(params,batch)
            first,second,noise_state=private_inputs(g,noise_state,privatizers,method)
            state,readout=moments(state,first,second,c,method=method)
            params=jax.tree.map(lambda x,d:x-c.learning_rate*d,params,readout[3])
            return params,state,noise_state
    names = strategy_names(method)
    strategy_meta=dict(method=method,first_strategy=names[0] or 'identity',second_strategy=names[1])
    metadata=dict(method=method,seed=seed,config=vars(c),contract=asdict(p),
        privacy_calibration=cal,privacy=cal,clipping=method != "nonprivate_adam",
        noise=method != "nonprivate_adam",optimizer="clean_adam" if method == "nonprivate_adam" else method,
        privacy_accounting=None if cal is None else "non_amplified_fixed_cycle_gdp",
        sampling_amplification=False,strategy=strategy_meta,
        optimizer_second_moment=('abs(v_hat_raw)' if uses_ime(method) else 'v_hat_raw'),
        schedule_sha256=hashlib.sha256(np.asarray(batches).tobytes()).hexdigest(),
        pretrained_sha256=hashlib.sha256((ROOT/c.pretrained).read_bytes()).hexdigest())
    np.save(dest/'schedule.npy',np.asarray(batches))
    initial_test=evaluate_classifier_metrics(params,model,tx,ty,batch_size=c.batch_size)
    metadata['initial_test_metrics']=initial_test
    write_json(dest/'metadata.json',metadata)
    rows=[]; next_epoch=1
    for t,b in enumerate(iter_logical_batches(x,y,batches),1):
        params,state,noise_state=step(params,state,noise_state,jax.tree.map(jnp.asarray,b))
        progress=t*c.batch_size/len(x)
        if progress>=next_epoch or t==p.horizon:
            train_metrics=evaluate_classifier_metrics(params,model,x,y,batch_size=c.batch_size)
            test=evaluate_classifier_metrics(params,model,tx,ty,batch_size=c.batch_size)
            row=dict(epoch=next_epoch,step=t,effective_epoch=progress,epsilon_full_mechanism=None if cal is None else c.epsilon,
                epsilon_spent=(epsilon_spent_for_iid_prefix(prefix_steps=t,horizon=p.horizon,min_sep=p.min_sep,
                    max_participations=p.max_participations,calibration=canonical) if method=='iid_adam' else None),
                train_loss=train_metrics['test_loss'],**test,elapsed_seconds=time.monotonic()-start)
            rows.append(row)
            row['best_test_accuracy']=max(r['test_accuracy'] for r in rows)
            row['best_test_loss']=min(r['test_loss'] for r in rows)
            table(dest/'metrics',rows)
            print(method,seed,row,flush=True); next_epoch+=1
    # AUC integrates initial and epoch-end test measurements over effective epochs.
    times=np.array([0.]+[r['effective_epoch'] for r in rows])
    accuracy=np.array([initial_test['test_accuracy']]+[r['test_accuracy'] for r in rows])
    summary=dict(method=method,seed=seed,final_test_accuracy=rows[-1]['test_accuracy'],
        best_test_accuracy=rows[-1]['best_test_accuracy'],final_test_loss=rows[-1]['test_loss'],
        best_test_loss=rows[-1]['best_test_loss'],accuracy_auc=float(np.trapezoid(accuracy,times)/times[-1]),
        elapsed_seconds=time.monotonic()-start)
    write_json(dest/'summary.json',summary)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true')
    p.add_argument('--method',choices=METHODS,default='iid_adam'); p.add_argument('--seed',type=int,default=0)
    a=p.parse_args()
    train(a.method,a.seed,a.smoke)
