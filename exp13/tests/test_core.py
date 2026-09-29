from types import SimpleNamespace
import itertools
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_privacy.matrix_factorization import banded
from exp13.workloads import ema
from exp13.common import configuration, jobs, METHODS
from exp13.strategies import fit
from exp13.privacy import calibration
from exp13.adam import init, moments, square_mean, channels, private_inputs

@pytest.mark.parametrize('beta',[.9,.999])
def test_workload(beta):
    t,s=np.indices((8,8))
    dense=np.where(t>=s,(1-beta)*beta**(t-s),0)
    np.testing.assert_allclose(ema(beta).materialize(8),dense,rtol=2e-6,atol=1e-8)

@pytest.mark.parametrize('beta',[.9,.999])
def test_strategy(beta):
    c=configuration(True); p=SimpleNamespace(horizon=8,min_sep=4,max_participations=2)
    strategy,meta=fit(beta,c,p)
    C=np.asarray(strategy.materialize())
    assert C.shape==(8,8)
    np.testing.assert_allclose(np.linalg.norm(C,axis=0),1,atol=2e-6)
    np.testing.assert_allclose(C@np.asarray(strategy.inverse_as_streaming_matrix().materialize(8)),np.eye(8),atol=2e-6)
    assert meta['sensitivity_squared']==2
    for i in range(4):
        assert np.sum((C[:,i]+C[:,i+4])**2)==pytest.approx(2,abs=2e-6)
    with pytest.raises(ValueError): banded.minsep_sensitivity_squared(strategy,3,2)
    H=np.asarray(ema(beta).materialize(8))
    assert meta['objective']==pytest.approx(np.mean(np.sum((H@np.linalg.inv(C))**2,axis=1)),rel=2e-5)

def test_calibration():
    c=configuration(); p=SimpleNamespace(max_participations=5)
    v=calibration(c,p,'iid_ime')
    assert v['a1']==1/512
    assert v['a2']==1023/512**2
    assert v['mu1']**2+v['mu2']**2==pytest.approx(v['mu']**2)
    assert v['sigma1']==pytest.approx(v['a1']*np.sqrt(10)/v['mu'])
    assert v['sigma2']==pytest.approx(v['a2']*np.sqrt(10)/v['mu'])
    assert calibration(c,p,'iid_adam')['sigma1']==pytest.approx(v['sigma1']/np.sqrt(2))

def test_square_after_mean():
    per_example=jnp.array([[1.,0.],[-1.,0.]])
    result=square_mean(per_example.mean(0))
    np.testing.assert_array_equal(result,[0,0])
    assert not np.array_equal(result,(per_example**2).mean(0))

def test_raw_linear_projection_no_feedback():
    c=configuration(True); x=jnp.array([1.,2.])
    state,read=moments(init(x),x,jnp.array([-2.,3.]),c)
    assert float(read[1][0])<0
    assert np.all(np.asarray(read[2])>=0)
    state2,read2=moments(state,x,jnp.zeros(2),c)
    np.testing.assert_allclose(state2[2],c.beta2*state[2],rtol=1e-6)
    assert float(state2[2][0])<0
    a,_=moments(init(x),x,x,c); b,_=moments(init(x),x,2*x,c)
    np.testing.assert_allclose(b[2],2*a[2])

@pytest.mark.parametrize('gpus',[(1,2,3),(0,1,2,3)])
def test_partition(gpus):
    assigned=[item for gpu in gpus for item in jobs(gpu,gpus)]
    expected=list(itertools.product(METHODS,range(10)))
    assert len(set(assigned))==len(assigned)==50
    assert set(assigned)==set(expected)
    for gpu in gpus:
        assert jobs(gpu,gpus)==expected[gpus.index(gpu)::len(gpus)]

def test_adam_matches_optax():
    import optax
    from exp13.adam import optimizer
    c=configuration(True); g=jnp.array([.1,-.2]); params=jnp.ones(2)
    opt=optimizer(c); state=opt.init(params); raw=init(params)
    for i in range(1,5):
        updates,state=opt.update(g*i,state)
        raw,read=moments(raw,g*i,(g*i)**2,c)
        np.testing.assert_allclose(-c.learning_rate*read[3],updates,rtol=2e-5)

@pytest.mark.parametrize('method',METHODS)
def test_methods_train_real_smoke(method):
    # Integration artifacts are created by the explicit GPU smoke command.
    import json
    from exp13.common import HERE
    path=HERE/'results_smoke/training'/f'{method}_seed0'
    summary=json.loads((path/'summary.json').read_text())
    rows=json.loads((path/'metrics.json').read_text())
    assert rows[-1]['step']==4
    assert np.isfinite(summary['final_test_loss'])
    assert 0<=summary['final_test_accuracy']<=1

def test_channel_noise_pairing_and_independence(tmp_path):
    from exp13.strategies import save
    c=configuration(True); p=SimpleNamespace(horizon=4,min_sep=4,max_participations=1)
    dest=tmp_path/'strategies'; dest.mkdir()
    strategy=banded.ColumnNormalizedBanded.default(4,4)
    for name in ('C_m','C_v'): save(dest/f'{name}.npz',strategy,{})
    outputs=[]
    for method in ('iid_ime','bandmf_ime_sep'):
        priv=channels(c,p,method,tmp_path,jax.random.key(17))
        g=jnp.zeros(64); state=tuple(x.init(g) for x in priv); rows=[]
        for _ in range(4):
            first,second,state=private_inputs(g,state,priv,method)
            rows.append(np.stack([first,second]))
        outputs.append(np.stack(rows))
    C=np.asarray(strategy.materialize())
    np.testing.assert_allclose(np.einsum('ts,scd->tcd',C,outputs[1]),outputs[0],atol=2e-6)
    assert abs(np.corrcoef(outputs[0][:,0].ravel(),outputs[0][:,1].ravel())[0,1])<.2


def test_private_second_query_is_square_of_mean():
    import optax
    identity=optax.identity()
    g=jnp.array([-.2,.3]); state=(identity.init(g),identity.init(g))
    first,second,_=private_inputs(g,state,(identity,identity),'iid_ime')
    np.testing.assert_array_equal(first,g)
    np.testing.assert_allclose(second,g*g)

def test_aggregate_complete_paired_runs(tmp_path, monkeypatch):
    import json
    import exp13.aggregate as module
    monkeypatch.setattr(module,'output',lambda:tmp_path)
    monkeypatch.setattr(module,'utility_plot',lambda *args:None)
    for index,method in enumerate(METHODS):
        for seed in range(10):
            dest=tmp_path/'training'/f'{method}_seed{seed}'
            dest.mkdir(parents=True)
            record=dict(method=method,seed=seed,**{k:index+seed/10 for k in module.METRICS})
            (dest/'summary.json').write_text(json.dumps(record))
    module.aggregate()
    paired=json.loads((tmp_path/'paired_differences.json').read_text())
    assert len(paired)==10*5
    selected=next(r for r in paired if r['comparison']=='bandmf_ime_sep - bandmf_single_m')
    assert selected['mean']==pytest.approx(2)
    assert selected['n']==10
    assert selected['se']==pytest.approx(0,abs=1e-14)
    assert len(json.loads((tmp_path/'paired_seed_differences.json').read_text()))==500
    (tmp_path/'training/nonprivate_adam_seed0/summary.json').unlink()
    with pytest.raises(FileNotFoundError): module.aggregate()

def test_smoke_pairing():
    import json
    from exp13.common import HERE
    records=[json.loads((HERE/'results_smoke/training'/f'{m}_seed0/metadata.json').read_text()) for m in METHODS]
    for field in ('schedule_sha256','pretrained_sha256','initial_test_metrics'):
        assert all(record[field]==records[0][field] for record in records)
