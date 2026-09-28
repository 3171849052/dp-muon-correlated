import socket
import numpy as np
from exp12.common import METHODS,configuration,dataset,contract,schedule,calibration,jobs
from dp_muon.privacy.nonamplified import epsilon_spent_for_iid_prefix,epsilon_spent_for_bandinv_prefix

def test_shared_schedule_privacy(c,p):
    schedules=[schedule(c,p,64,3) for m in METHODS]
    for s in schedules[1:]: np.testing.assert_array_equal(s,schedules[0])
    for m in METHODS:
        cal=calibration(c,p,m)
        assert (cal.epsilon,cal.delta,cal.adjacency)==(3.,1e-5,'add_remove')
    assert c.weight_decay==0 and c.optimizer=='adam'

def test_jobs():
    all_jobs=[j for gpu in range(4) for j in jobs(gpu)]
    assert len(set(all_jobs))==40
    assert all(len(jobs(g))==10 for g in range(4))

def test_smoke_no_network(monkeypatch):
    def forbidden(*a,**kw): raise AssertionError('network access')
    monkeypatch.setattr(socket.socket,'connect',forbidden)
    import dp_muon.data.cifar10 as module
    monkeypatch.setattr(module,'urlretrieve',forbidden)
    c=configuration(True)
    x,y=dataset(c,True)
    assert len(x)==64
    assert contract(c,True).horizon==4

def test_formal_configuration():
    c=configuration(); p=contract(c)
    assert c.epochs==5 and c.batch_size==512 and c.microbatch_size==256
    assert c.fit_steps==1000 and c.K_fit==256 and c.K_eval==4096
    assert p.horizon==5*50000//512

def test_jitted_streaming_pairing():
    import jax
    import jax.numpy as jnp
    from dp_muon.bandinvmf.noise import init_bandinv_noise_state,sample_bandinv_noise
    template={'a':jnp.zeros(3),'b':jnp.zeros((2,2))}
    sample=jax.jit(sample_bandinv_noise)
    key=jax.random.key(4)
    one=init_bandinv_noise_state(template,1)
    four=init_bandinv_noise_state(template,4)
    k1=k4=key
    for _ in range(5):
        n1,one,k1=sample(k1,one,jnp.array([1.]),jnp.array(.02))
        n4,four,k4=sample(k4,four,jnp.array([1.,0.,0.,0.]),jnp.array(.02))
        for a,b in zip(jax.tree.leaves(n1),jax.tree.leaves(n4)):
            np.testing.assert_array_equal(a,b)

def test_equal_full_privacy_guarantees(c,p):
    import jax.numpy as jnp
    from exp12.strategies import normalize
    for method in METHODS:
        cal=calibration(c,p,method)
        args=dict(prefix_steps=p.horizon,horizon=p.horizon,min_sep=p.min_sep,
                  max_participations=p.max_participations,calibration=cal)
        if method=='dp_adam':
            epsilon=epsilon_spent_for_iid_prefix(**args)
        else:
            coef=normalize(jnp.array([1.,-.3,.02,.01]),p)
            epsilon=epsilon_spent_for_bandinv_prefix(**args,noising_coef=coef)
        np.testing.assert_allclose(epsilon,c.epsilon,rtol=1e-8)
