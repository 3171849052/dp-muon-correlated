from types import SimpleNamespace
import itertools
import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_privacy.matrix_factorization import banded
from exp13.workloads import ema
from exp13.common import configuration, jobs, METHODS, GPUS
from exp13.strategies import fit
from exp13.privacy import calibration
from exp13.adam import init, moments, square_mean, channels, private_inputs
from exp13.replay import squared_error_sum, theoretical_state_mse


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


def test_final_methods_and_seed_config():
    assert METHODS==('nonprivate_adam','iid_adam','bandmf_single_m','iid_ime','bandmf_ime_sep_raw','bandmf_ime_sep_vbc','bandmf_ime_sep_bc')
    assert all(not method.endswith('_abs') and 'relu' not in method for method in METHODS)
    assert configuration().seeds==[0,1,2]


def test_calibration():
    c=configuration(); p=SimpleNamespace(max_participations=5)
    v=calibration(c,p,'iid_ime')
    assert v['a1']==1/512
    assert v['a2']==1023/512**2
    assert v['mu1']==pytest.approx(v['mu']/np.sqrt(2))
    assert v['mu2']==pytest.approx(v['mu']/np.sqrt(2))
    assert v['mu1']**2+v['mu2']**2==pytest.approx(v['mu']**2)
    assert v['sigma1']==pytest.approx(v['a1']*np.sqrt(p.max_participations)/v['mu1'])
    assert v['sigma2']==pytest.approx(v['a2']*np.sqrt(p.max_participations)/v['mu2'])
    single=calibration(c,p,'bandmf_single_m')
    assert single['mu1']==pytest.approx(single['mu'])
    assert single['sigma1']==pytest.approx(v['sigma1']/np.sqrt(2))


def test_square_after_mean():
    per_example=jnp.array([[1.,0.],[-1.,0.]])
    result=square_mean(per_example.mean(0))
    np.testing.assert_array_equal(result,[0,0])
    assert not np.array_equal(result,(per_example**2).mean(0))


def test_abs_readout_and_unprojected_raw_recurrence():
    c=configuration(True); x=jnp.array([1.,2.,3.])
    state=init(x)
    state,read=moments(state,x,jnp.array([-2.,0.,3.]),c,method='iid_ime')
    assert float(read[1][0])<0
    np.testing.assert_array_equal(read[2],jnp.abs(read[1]))
    np.testing.assert_array_equal(state[2],(1-c.beta2)*jnp.array([-2.,0.,3.]))
    prior_raw=state[2]
    state2,read2=moments(state,x,jnp.zeros(3),c,method='iid_ime')
    np.testing.assert_allclose(state2[2],c.beta2*prior_raw,rtol=1e-6)
    assert float(state2[2][0])<0
    np.testing.assert_array_equal(read2[2],jnp.abs(read2[1]))
    ordinary,_=moments(init(x),x,x,c,method='iid_adam')
    doubled,_=moments(init(x),x,2*x,c,method='iid_adam')
    np.testing.assert_allclose(doubled[2],2*ordinary[2])


def test_uncorrected_first_and_second_state_mse():
    clean_first=jnp.array([[1.,2.],[3.,4.]])
    private_first=jnp.array([[2.,2.],[1.,6.]])
    clean_second=jnp.array([[1.,4.],[9.,16.]])
    private_second=jnp.array([[2.,2.],[1.,20.]])
    first=float(squared_error_sum(private_first,clean_first))/clean_first.size
    second=float(squared_error_sum(private_second,clean_second))/clean_second.size
    assert first==pytest.approx(np.mean((np.asarray(private_first)-np.asarray(clean_first))**2))
    assert second==pytest.approx(np.mean((np.asarray(private_second)-np.asarray(clean_second))**2))


def test_theoretical_iid_workload_mse_matches_dense_identity():
    beta=.6; sigma=.7; horizon=5
    identity=banded.ColumnNormalizedBanded.default(horizon,1)
    H=np.asarray(ema(beta).materialize(horizon))
    expected=sigma**2*np.mean(np.sum(H**2,axis=1))
    assert theoretical_state_mse(identity,beta,sigma)==pytest.approx(expected,rel=1e-6)


def test_bandmf_theoretical_state_mse_uses_fitted_strategy_objective():
    c=configuration(True); p=SimpleNamespace(horizon=8,min_sep=4,max_participations=2)
    strategy,_=fit(c.beta1,c,p)
    objective=float(jnp.mean(banded.per_query_error(strategy,A=ema(c.beta1))))
    sigma=.23
    assert theoretical_state_mse(strategy,c.beta1,sigma)==pytest.approx(sigma**2*objective,rel=1e-6)


def test_gpu_job_partition_from_config():
    c=configuration()
    assert GPUS==(1,2,3)
    assert c.gpus==[1,2,3]
    expected=list(itertools.product(METHODS,c.seeds))
    partitions={gpu:jobs(gpu) for gpu in GPUS}
    assigned=[item for gpu in GPUS for item in partitions[gpu]]
    assert len(expected)==21
    assert all(len(partitions[gpu])==7 for gpu in GPUS)
    assert len(set(assigned))==21
    assert set(assigned)==set(expected)
    assert all(partitions[gpu]==expected[GPUS.index(gpu)::len(GPUS)] for gpu in GPUS)


def test_adam_matches_optax():
    import optax
    from exp13.adam import optimizer
    c=configuration(True); g=jnp.array([.1,-.2]); params=jnp.ones(2)
    opt=optimizer(c); state=opt.init(params); raw=init(params)
    for i in range(1,5):
        updates,state=opt.update(g*i,state)
        raw,read=moments(raw,g*i,(g*i)**2,c,method='iid_adam')
        np.testing.assert_allclose(-c.learning_rate*read[3],updates,rtol=2e-5)


def test_channel_noise_pairing_and_independence(tmp_path):
    from exp13.strategies import save
    c=configuration(True); p=SimpleNamespace(horizon=4,min_sep=4,max_participations=1)
    dest=tmp_path/'strategies'; dest.mkdir()
    strategy=banded.ColumnNormalizedBanded.default(4,1)
    for name in ('C_m_raw','C_m_bc','C_v_raw','C_v_bc'): save(dest/f'{name}.npz',strategy,{})
    outputs=[]
    for method in ('iid_ime','bandmf_ime_sep_raw','bandmf_ime_sep_vbc','bandmf_ime_sep_bc'):
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
    import exp13.aggregate as module
    monkeypatch.setattr(module,'output',lambda *args:tmp_path)
    monkeypatch.setattr(module,'utility_plot',lambda *args:None)
    c=configuration()
    for index,method in enumerate(METHODS):
        for seed in c.seeds:
            dest=tmp_path/'training'/f'{method}_seed{seed}'
            dest.mkdir(parents=True)
            record=dict(method=method,seed=seed,**{k:index+seed/10 for k in module.METRICS})
            (dest/'summary.json').write_text(json.dumps(record))
    module.aggregate()
    paired=json.loads((tmp_path/'paired_differences.json').read_text())
    assert len(paired)==len(module.PAIRS)*len(module.METRICS)
    selected=next(r for r in paired if r['comparison']=='bandmf_ime_sep_bc - iid_ime')
    assert selected['mean']==pytest.approx(3)
    assert selected['n']==3
    assert selected['se']==pytest.approx(0,abs=1e-14)
    assert len(json.loads((tmp_path/'paired_seed_differences.json').read_text()))==len(module.PAIRS)*3*len(module.METRICS)
    (tmp_path/'training/nonprivate_adam_seed0/summary.json').unlink()
    with pytest.raises(FileNotFoundError): module.aggregate()
