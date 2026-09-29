import inspect
import json
from dataclasses import dataclass
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax_privacy.matrix_factorization import banded
from exp13.common import configuration, METHODS, HERE
from exp13.strategies import STRATEGIES, fit, save, validate
from exp13.workloads import ema
from exp13.replay import theoretical_state_mse
from exp13.adam import clean_step, IME_METHODS, moments, init
from exp13.collect_trajectory import sample_coordinates

@dataclass
class Participation:
    horizon: int = 8
    min_sep: int = 4
    max_participations: int = 2

@pytest.mark.parametrize('beta',[.9,.999])
@pytest.mark.parametrize('corrected',[False,True])
def test_streaming_workload_and_theory(beta,corrected):
    t,s=np.indices((8,8))
    raw=np.where(t>=s,(1-beta)*beta**(t-s),0)
    expected=raw/(1-beta**(t[:,0]+1))[:,None] if corrected else raw
    H=np.asarray(ema(beta,corrected).materialize(8))
    np.testing.assert_allclose(H,expected,rtol=3e-5,atol=2e-7)
    c=configuration(True)
    strategy,_=fit(beta,c,Participation(),corrected)
    expected_mse=.3**2*np.mean(np.sum((expected@np.linalg.inv(strategy.materialize()))**2,axis=1))
    assert theoretical_state_mse(strategy,beta,.3,corrected)==pytest.approx(expected_mse,rel=4e-5)

def test_artifacts_constraints_and_mismatch(tmp_path):
    c=configuration(True); p=Participation()
    assert STRATEGIES==('C_m_raw','C_m_bc','C_v_raw','C_v_bc')
    (tmp_path/'strategies').mkdir()
    for name in STRATEGIES:
        beta=c.beta1 if '_m_' in name else c.beta2
        strategy,meta=fit(beta,c,p,name.endswith('_bc'))
        assert meta['sensitivity_squared']==2
        save(tmp_path/'strategies'/f'{name}.npz',strategy,meta)
    validate(tmp_path,c,p)
    c.learning_rate*=2
    with pytest.raises(ValueError,match='mismatch'): validate(tmp_path,c,p)
    c.learning_rate/=2
    (tmp_path/'strategies/C_v_bc.npz').unlink()
    with pytest.raises(FileNotFoundError): validate(tmp_path,c,p)

def test_coordinate_sampling():
    c=configuration()
    a=sample_coordinates(5500000,c.replay_num_coordinates,c.replay_coordinate_seed)
    np.testing.assert_array_equal(a,sample_coordinates(5500000,len(a),1203))
    assert len(a)==len(np.unique(a))==262144
    assert c.replay_quantile_coordinates==8192
    assert c.replay_draws==3 and c.replay_chunk==32768

def test_stage_separation_and_bounded_statistics():
    from exp13 import replay, collect_trajectory
    source=inspect.getsource(replay)
    assert 'collect(' not in source
    assert 'open_memmap' not in source
    assert 'absolute_directions.npy' not in source and 'denominators.npy' not in source
    assert 'temporary_shape=(c.replay_draws,p.horizon,c.replay_quantile_coordinates)' in source
    assert 'jnp.sum(denominator<threshold)' in source
    assert 'step_coordinate_count=c.replay_draws*gradients.shape[1]' in source
    assert 'magnitude[selected],denominator[selected]' in source
    assert 'shape=(p.horizon,len(indices))' in inspect.getsource(collect_trajectory)
    shell=(HERE/'run_training.sh').read_text()
    for forbidden in ('fit_strategies.py','collect_trajectory.py','replay.py','numerical.py'):
        assert forbidden not in shell
    assert 'validate_strategies.py' in shell

class ToyModel:
    def apply(self,params,x):
        return x@params

def test_clean_adam_whole_batch(monkeypatch):
    import exp13.full_training as training
    def forbidden(*args,**kwargs):
        raise AssertionError('clean Adam must not clip or calibrate')
    monkeypatch.setattr(training,'make_clipped_gradient_query',forbidden)
    monkeypatch.setattr(training,'calibration',forbidden)
    c=configuration(True); model=ToyModel()
    params=jnp.array([[.2,-.1],[.7,.3]])
    batch={'image':jnp.array([[1.,2.],[-2.,1.],[3.,-1.]]),'label':jnp.array([0,1,1])}
    update,opt=clean_step(c,model); state=opt.init(params)
    reference=params; reference_state=opt.init(params)
    for _ in range(3):
        params,state=update(params,state,batch)
        grad=jax.grad(lambda w: optax.softmax_cross_entropy_with_integer_labels(batch['image']@w,batch['label']).mean())(reference)
        u,reference_state=opt.update(grad,reference_state,reference)
        reference=optax.apply_updates(reference,u)
        np.testing.assert_allclose(params,reference,rtol=1e-6)
    source=inspect.getsource(training.train)
    branch=source.split("if method == 'nonprivate_adam':")[1].split('elif')[0]
    assert 'calibration(' not in branch and 'query(' not in branch

def test_canonical_iid():
    from exp13.privacy import iid_calibration
    from dp_muon.privacy.nonamplified import calibrate_nonamplified_iid
    import exp13.full_training as training
    c=configuration(); p=Participation()
    actual=iid_calibration(c,p)
    expected=calibrate_nonamplified_iid(epsilon=c.epsilon,delta=c.delta,clip_norm=c.clip_norm,
        normalize_by=float(c.batch_size),adjacency=c.adjacency,max_participations=p.max_participations)
    assert actual==expected
    assert actual.iid_noise_std==pytest.approx(actual.noise_multiplier*c.clip_norm/c.batch_size*np.sqrt(p.max_participations))
    source=inspect.getsource(training.train)
    assert 'make_nonamplified_dpadamw_train_step(' in source
    assert 'epsilon_spent_for_iid_prefix(' in source
    assert 'weight_decay=0' in source and 'sampling_amplification=False' in source

@pytest.mark.parametrize('method',IME_METHODS)
def test_all_ime_abs_and_privacy(method):
    from exp13.privacy import calibration
    c=configuration(); p=Participation(); g=jnp.ones(4)
    state,read=moments(init(g),g,-g,c,method=method)
    assert np.all(np.asarray(state[2])<0)
    np.testing.assert_array_equal(read[2],jnp.abs(read[1]))
    assert calibration(c,p,method)==calibration(c,p,'iid_ime')
    assert calibration(c,p,method)['amplification'] is False
    assert c.adam_eps==1e-8

def test_cross_eval_complete(tmp_path,monkeypatch):
    import exp13.fit_strategies as module
    monkeypatch.setattr(module,'output',lambda smoke:tmp_path)
    monkeypatch.setattr(module,'contract',lambda c,smoke:Participation())
    module.main(True)
    rows=json.loads((tmp_path/'cross_eval.json').read_text())
    assert len(rows)==16
    assert {(r['workload'],r['strategy']) for r in rows}=={
        (w,s) for w in ('H_m_raw','H_m_bc','H_v_raw','H_v_bc') for s in STRATEGIES}

def test_gpu_statistics_reduce_all_coordinates_not_quantile_subset(tmp_path):
    from exp13.replay import make_runner, THRESHOLDS
    from exp13.adam import channels, private_inputs
    c=configuration(True); p=Participation(horizon=4,min_sep=4,max_participations=1)
    # Small inputs plus a small noise scale exercise the denominator thresholds.
    c.clip_norm=1e-7
    g=jnp.array([[1e-10,2e-10,3e-10,4e-10]]*4)
    key=jax.random.key(123)
    values,mins,maxes,mags,dens=make_runner(c,p,'iid_ime',tmp_path)(g,key,jnp.array([1,3]))
    priv=channels(c,p,'iid_ime',tmp_path,key)
    noise=tuple(x.init(g[0]) for x in priv)
    state=init(g[0]); reference=init(g[0])
    for t in range(4):
        first,second,noise=private_inputs(g[t],noise,priv,'iid_ime')
        state,read=moments(state,first,second,c,method='iid_ime')
        reference,clean=moments(reference,g[t],g[t]**2,c,method='nonprivate_adam')
        den=np.sqrt(np.asarray(read[2]))+c.adam_eps
        mag=np.abs(np.asarray(read[3]))
        np.testing.assert_allclose(dens[t],den[[1,3]],rtol=2e-5)
        np.testing.assert_allclose(mags[t],mag[[1,3]],rtol=2e-5)
        np.testing.assert_allclose(values[t,8:],[np.sum(den<x) for x in THRESHOLDS])
        assert values[t,-1]==4  # Two quantile coordinates, four threshold coordinates.
        assert mins[t]==pytest.approx(den.min(),rel=2e-5)
        assert maxes[t]==pytest.approx(mag.max(),rel=2e-5)
        assert values[t,6]==pytest.approx(float(jnp.sum((read[3]-clean[3])**2)),rel=2e-5)

def test_replay_missing_trajectory_fails_without_collection(tmp_path,monkeypatch):
    import exp13.replay as module
    monkeypatch.setattr(module,'output',lambda smoke:tmp_path)
    monkeypatch.setattr(module,'contract',lambda c,smoke:Participation())
    monkeypatch.setattr(module,'validate',lambda *args:None)
    with pytest.raises(FileNotFoundError,match='collect step'):
        module.replay(True)
    assert not (tmp_path/'trajectory').exists()

def test_iid_epoch_privacy_is_prefix_accounted():
    from exp13.privacy import iid_calibration
    from dp_muon.privacy.nonamplified import epsilon_spent_for_iid_prefix
    c=configuration(); p=Participation(horizon=488,min_sep=97,max_participations=5)
    cal=iid_calibration(c,p)
    values=[epsilon_spent_for_iid_prefix(prefix_steps=t,horizon=p.horizon,min_sep=p.min_sep,
        max_participations=p.max_participations,calibration=cal) for t in (97,194,291,388,488)]
    assert values==sorted(values)
    assert values[0]<values[-1]
    assert values[-1]==pytest.approx(c.epsilon,rel=1e-5)

def test_nonprivate_training_never_builds_clipped_query(tmp_path,monkeypatch):
    import exp13.full_training as module
    import exp13.common as common
    def forbidden(*args,**kwargs):
        raise AssertionError('nonprivate path called clipping or privacy calibration')
    monkeypatch.setattr(module,'make_clipped_gradient_query',forbidden)
    monkeypatch.setattr(module,'calibration',forbidden)
    monkeypatch.setattr(module,'iid_calibration',forbidden)
    monkeypatch.setattr(module,'validate',lambda *args:None)
    c=configuration(True)
    monkeypatch.setattr(module,'configuration',lambda smoke:c)
    monkeypatch.setattr(module,'output',lambda *args:tmp_path)
    monkeypatch.setattr(common,'contract',lambda *args:Participation())
    x=np.ones((16,2),dtype=np.float32); y=np.zeros(16,dtype=np.int32)
    params=jnp.array([[.1,.2],[.3,.4]])
    monkeypatch.setattr(module,'dataset',lambda *args:(x,y))
    monkeypatch.setattr(module,'iter_logical_batches',lambda x,y,schedule:[{'image':x,'label':y}])
    monkeypatch.setattr(module,'ViTTiny',ToyModel)
    monkeypatch.setattr(module,'load_pretrained_vit_tiny',lambda *args,**kwargs:params)
    monkeypatch.setattr(module,'evaluate_classifier_metrics',lambda *args,**kwargs:dict(test_loss=1.,test_accuracy=.5))
    module.train('nonprivate_adam',0,True)
    meta=json.loads((tmp_path/'training/nonprivate_adam_seed0/metadata.json').read_text())
    assert meta['privacy'] is None
    assert meta['clipping'] is False and meta['noise'] is False
    assert meta['optimizer']=='clean_adam'
