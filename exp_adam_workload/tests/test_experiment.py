import jax
import numpy as np
import pytest
from exp_adam_workload.core import (NAMES, configuration, workload, linear_energy,
    jobs, energy_rows)
from exp_adam_workload.aggregate import stats
from dp_muon.optim import (decayed_prefix_sum_workload_coef,
    adam_first_moment_workload_matrix)
from exp2.run import _lower_toeplitz, adamw_perturbations


@pytest.fixture
def config():
    return configuration(True)[0]


def test_sgd(config):
    np.testing.assert_array_equal(workload('sgd', 8, config), np.tril(np.ones((8,8))))
    expected = _lower_toeplitz(np.asarray(decayed_prefix_sum_workload_coef(
        8, config.learning_rate, config.weight_decay)), 8)
    np.testing.assert_array_equal(workload('sgd_wd', 8, config), expected)


@pytest.mark.parametrize('name', NAMES)
def test_shape_causality(name, config):
    A = workload(name, 9, config)
    assert A.shape == (9,9)
    assert np.isfinite(A).all()
    np.testing.assert_array_equal(np.triu(A,1), 0)


@pytest.mark.parametrize('wd', [0., .01])
def test_momentum_impulses(config, wd):
    expected = np.zeros((12,12))
    for source in range(12):
        m, theta = 0., 0.
        for step in range(12):
            m = config.beta1*m + (1-config.beta1)*(step==source)
            theta = (1-config.learning_rate*wd)*theta + m
            expected[step,source] = theta
    name = 'momentum_wd' if wd else 'momentum'
    np.testing.assert_allclose(workload(name,12,config), expected, rtol=1e-6)
    # No bias correction: first diagonal entry must be 1-beta1.
    assert workload(name,12,config)[0,0] == pytest.approx(1-config.beta1)


@pytest.mark.parametrize('name,wd', [('adam_momentum',0.), ('adam_momentum_wd',.01)])
def test_adam_helper(name, wd, config):
    with jax.default_matmul_precision('highest'):
        expected = np.asarray(adam_first_moment_workload_matrix(
            8, config.beta1, config.learning_rate, wd))/config.learning_rate
    np.testing.assert_array_equal(workload(name,8,config), expected)


def test_matched_marginal():
    D = _lower_toeplitz(np.array([2.,-1.,.5]), 7)
    A = np.tril(np.arange(49).reshape(7,7)/49 + .1)
    covariance = D @ D.T
    corr, iid = linear_energy(A,D)
    np.testing.assert_allclose(corr, np.diag(A @ covariance @ A.T))
    np.testing.assert_allclose(iid, np.diag(A @ np.diag(np.diag(covariance)) @ A.T))
    np.testing.assert_allclose(linear_energy(np.eye(7),D)[0], np.diag(covariance))
    np.testing.assert_allclose(*linear_energy(np.eye(7),D))


def test_partition():
    whole = [(name, seed) for name in NAMES for seed in range(10)]
    pieces = [jobs(i,4) for i in range(4)]
    assert [len(p) for p in pieces] == [15]*4
    assert set(sum(pieces,[])) == set(whole)
    assert len(set(sum(pieces,[]))) == 60
    for i,piece in enumerate(pieces):
        assert piece == whole[i::4]


def test_windows():
    rows = list(energy_rows(np.ones(488),np.ones(488)*2,97))
    assert rows[0]['end_step'] == 97
    assert rows[2]['start_step'] == 98
    assert rows[4]['correlated_energy'] == 488
    assert all(r['ratio']==.5 for r in rows)


def test_replay_matches_direct_adamw(config):
    import jax.numpy as jnp
    import optax
    g = np.random.default_rng(0).normal(size=(7,2,2))*.1
    noise = np.random.default_rng(1).normal(size=(1,7,2,2))*.03
    params = {f: getattr(config,f) for f in ('learning_rate','beta1','beta2','eps','weight_decay')}
    _, delta = adamw_perturbations(g, noise, **params)
    optimizer = optax.adamw(config.learning_rate, b1=config.beta1,
        b2=config.beta2, eps=config.eps, weight_decay=config.weight_decay)
    paths=[]
    for gradients in (g, g+noise[0]):
        theta=jnp.ones((2,2)); state=optimizer.init(theta); path=[]
        for gradient in gradients:
            updates,state=optimizer.update(jnp.asarray(gradient),state,theta)
            theta=optax.apply_updates(theta,updates); path.append(np.asarray(theta))
        paths.append(np.array(path))
    np.testing.assert_allclose(delta[0],paths[1]-paths[0],atol=4e-7)


def test_stats():
    result = stats(np.arange(10))
    assert result['mean'] == 4.5
    assert result['se'] == pytest.approx(np.std(np.arange(10),ddof=1)/np.sqrt(10))
    assert result['ci95_low'] < result['mean'] < result['ci95_high']


def test_independent_calibration(config):
    from types import SimpleNamespace
    from exp_adam_workload.core import mechanism
    def strategy(sensitivity):
        return SimpleNamespace(horizon=4, noising_coef=np.array([1.,-.4]),
                               sensitivity_squared=sensitivity, objective=1.)
    D1, marginal1, c1 = mechanism(strategy(1.),config)
    D2, marginal2, c2 = mechanism(strategy(4.),config)
    assert c2['calibrated_noise_stddev'] == pytest.approx(2*c1['calibrated_noise_stddev'])
    np.testing.assert_allclose(D2, 2*D1)
    np.testing.assert_allclose(marginal2, 4*marginal1)


def test_aggregate_paired_seeds(tmp_path, monkeypatch):
    import json
    import exp_adam_workload.aggregate as module
    monkeypatch.setattr(module, 'bars', lambda *args: None)
    for i,name in enumerate(NAMES):
        for seed in range(10):
            directory = tmp_path / 'runs' / name / f'seed{seed}'
            directory.mkdir(parents=True)
            record = dict(strategy=name, seed=seed,
                          **{m: seed+i for m in module.METRICS})
            (directory/'result.json').write_text(json.dumps(record))
    module.aggregate(tmp_path)
    rows=json.loads((tmp_path/'utility_paired_differences.json').read_text())
    assert len(rows) == 15*len(module.METRICS)
    assert rows[0]['mean'] == -1
    assert rows[0]['std'] == 0  # shared seed variation cancels in paired differences
    (tmp_path/'runs'/'sgd'/'seed0'/'result.json').unlink()
    with pytest.raises(FileNotFoundError):
        module.aggregate(tmp_path)
